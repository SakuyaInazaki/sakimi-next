use candle_core::{Result, Tensor, Device, DType};
use candle_nn::ops;

/// ShortConvolution layer for Gated DeltaNet
///
/// 1D depthwise convolution with kernel_size=4, used for capturing local patterns
/// in the Q, K, V projections before the main DeltaNet computation.
///
/// This is a causal convolution where each position only sees previous positions.
/// Uses depthwise convolution (each channel has its own filter).
#[derive(Clone)]
pub struct ShortConvolution {
    // Convolution kernel: (kernel_size, channels)
    // Each channel has its own filter of size kernel_size
    weight: Tensor,
    kernel_size: usize,
    channels: usize,
}

impl ShortConvolution {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, device: &Device) -> Result<Self> {
        // For depthwise convolution, in_channels == out_channels
        assert_eq!(in_channels, out_channels, "ShortConvolution requires in_channels == out_channels for depthwise conv");

        // Initialize weight with small std
        // Weight shape: (channels, kernel_size) for easier processing
        let std = (2.0 / (in_channels * kernel_size) as f64).sqrt();
        let weight = Tensor::randn(0.0, std, (in_channels, kernel_size), device)?
            .to_dtype(DType::F32)?;

        Ok(Self {
            weight,
            kernel_size,
            channels: in_channels,
        })
    }

    /// Forward pass - causal 1D depthwise convolution
    ///
    /// Input shape: (batch_size, seq_len, channels)
    /// Output shape: (batch_size, seq_len, channels)
    ///
    /// For each channel and each position t, computes:
    ///   output[t] = sum_{i=0}^{kernel_size-1} input[t-i] * weight[i]
    /// where input[t-i] = 0 for t-i < 0 (causal padding)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, channels) = x.dims3()?;

        // Handle short sequences
        if seq_len == 0 {
            return Ok(x.clone());
        }

        // For efficiency, we'll process each time step and batch together
        // Build output by iterating through sequence
        let mut output_rows = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // For each position t, we look back up to kernel_size positions
            // Extract relevant slice: x[:, max(0,t-k+1):t+1, :]
            let start = t.saturating_sub(self.kernel_size - 1);
            let end = t + 1;
            let slice_len = end - start;

            // Get slice: (B, slice_len, C)
            let x_slice = x.narrow(1, start, slice_len)?;

            // Pad with zeros if needed (when t < kernel_size)
            let mut padded_slices = Vec::new();
            if start == 0 && slice_len < self.kernel_size {
                // Need zero padding at the beginning
                let pad_len = self.kernel_size - slice_len;
                let zero_shape = &[batch_size, pad_len, channels];
                let zeros = Tensor::zeros(zero_shape.as_ref(), DType::F32, x.device())?;
                padded_slices.push(zeros);
            }

            // Add actual slice (need to reverse to match causal order)
            // For position t with lookback k: we want [x[t-k+1], ..., x[t]]
            // which corresponds to weight[:, k-1], ..., weight[:, 0]
            let mut slice_vec = Vec::with_capacity(slice_len);
            for i in 0..slice_len {
                // Take in reverse order: position t-i comes first in weight
                slice_vec.push(x_slice.narrow(1, slice_len - 1 - i, 1)?);
            }
            let x_slice_reversed = Tensor::cat(&slice_vec, 1)?;
            padded_slices.push(x_slice_reversed);

            // Concatenate and apply weights
            let padded = Tensor::cat(&padded_slices, 1)?;  // (B, K, C)
            let padded_transposed = padded.transpose(1, 2)?;  // (B, C, K)

            // Reshape weight for broadcasting: (1, C, K)
            let weight_broadcast = self.weight.reshape(&[1, channels, self.kernel_size])?;

            // Element-wise multiply and sum over kernel dimension
            let weighted = padded_transposed.broadcast_mul(&weight_broadcast)?;  // (B, C, K)
            let sum_k = weighted.sum_keepdim(2)?;  // (B, C, 1)
            let sum_k = sum_k.squeeze(2)?;  // (B, C)

            output_rows.push(sum_k);
        }

        // Stack results: seq_len tensors of shape (B, C)
        // Stack along dim 0 to get (seq_len, B, C), then transpose to (B, seq_len, C)
        let stacked = Tensor::cat(&output_rows, 0)?;  // (seq_len * B, C)
        let stacked = stacked.reshape(&[seq_len, batch_size, channels])?;
        stacked.transpose(0, 1)  // (B, seq_len, C)
    }

    /// Get the weight tensor (for optimizer)
    pub fn get_weight(&self) -> Tensor {
        self.weight.clone()
    }
}

/// FusedRMSNormSwishGate for Gated DeltaNet output
///
/// Combines RMS normalization, Swish activation, and gating in one layer.
/// This is used as o_norm in the official Gaged DeltaNet implementation.
///
/// Formula: output = RMSNorm(x) * (swish(gate_input) + gate_param)
/// where swish(z) = z * sigmoid(z)
#[derive(Clone)]
pub struct FusedRMSNormSwishGate {
    weight: Tensor,  // (d_model,) - RMSNorm weight
    gate: Tensor,    // (d_model,) - learnable gate bias
    eps: f32,
    d_model: usize,
}

impl FusedRMSNormSwishGate {
    pub fn new(d_model: usize, device: &Device) -> Result<Self> {
        // Initialize weight to ones, gate to zeros
        let weight = Tensor::ones(&[d_model], DType::F32, device)?;
        let gate = Tensor::zeros(&[d_model], DType::F32, device)?;

        Ok(Self {
            weight,
            gate,
            eps: 1e-5,
            d_model,
        })
    }

    /// Forward pass: RMSNorm + Swish gating
    ///
    /// Input shape: (batch_size, seq_len, d_model)
    /// Output shape: (batch_size, seq_len, d_model)
    ///
    /// Formula:
    ///   normalized = RMSNorm(x) = x / RMS(x) * weight
    ///   gate_value = swish(gate_input) + gate_param
    ///   output = normalized * gate_value
    pub fn forward(&self, x: &Tensor, gate_input: &Tensor) -> Result<Tensor> {
        let (b, l, d) = x.dims3()?;

        // RMS normalization
        let x_flat = x.reshape(&[b * l, d])?;
        // sum of squares: (b*l, 1)
        let sum_sq = x_flat.sqr()?.sum_keepdim(1)?;
        // divide by d and add eps
        let d_tensor = Tensor::new(d as f32, x_flat.device())?;
        let eps_tensor = Tensor::new(self.eps, x_flat.device())?;
        let rms = sum_sq.broadcast_div(&d_tensor)?.broadcast_add(&eps_tensor)?.sqrt()?;
        let x_norm = x_flat.broadcast_div(&rms)?;

        // Apply RMSNorm weight
        let x_norm = x_norm.broadcast_mul(&self.weight)?;

        // Compute gate: swish(gate_input) + gate_param
        // swish(z) = z * sigmoid(z)
        let gate_flat = gate_input.reshape(&[b * l, d])?;
        let gate_sigmoid = ops::sigmoid(&gate_flat)?;
        let gate_swish = gate_flat.broadcast_mul(&gate_sigmoid)?;

        // Add learnable gate bias
        let gated = gate_swish.broadcast_add(&self.gate)?;

        // Apply gate to normalized input
        let output = x_norm.broadcast_mul(&gated)?;

        output.reshape(&[b, l, d])
    }

    /// Get weight tensor (for optimizer)
    pub fn get_weight(&self) -> Tensor {
        self.weight.clone()
    }

    /// Get gate parameter tensor (for optimizer)
    pub fn get_gate(&self) -> Tensor {
        self.gate.clone()
    }
}
