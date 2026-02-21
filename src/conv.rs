use candle_core::{DType, Device, Result, Tensor};

use crate::activation::apply_hidden_act;
use crate::trainable::make_trainable;

/// ShortConvolution layer for Gated DeltaNet
///
/// 1D depthwise convolution (kernel size from config), used for capturing local patterns
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
}

impl ShortConvolution {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        init_std: f64,
        device: &Device,
    ) -> Result<Self> {
        // For depthwise convolution, in_channels == out_channels
        assert_eq!(
            in_channels, out_channels,
            "ShortConvolution requires in_channels == out_channels for depthwise conv"
        );

        // Weight shape: (channels, kernel_size) for easier depthwise processing.
        let weight = Tensor::randn(0f32, init_std as f32, (in_channels, kernel_size), device)?
            .to_dtype(DType::F32)?;
        let weight = make_trainable(weight)?;

        Ok(Self {
            weight,
            kernel_size,
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
        let (output, _) = self.forward_with_state(x, None)?;
        Ok(output)
    }

    /// Forward with optional cached convolution state.
    ///
    /// - `x`: (B, L, C)
    /// - `state`: (B, C, K) containing previous K tokens per channel
    /// Returns:
    /// - output: (B, L, C)
    /// - new_state: (B, C, K)
    pub fn forward_with_state(
        &self,
        x: &Tensor,
        state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_with_state_fast(x, state)
    }

    /// Reference implementation kept for debugging and parity checks.
    ///
    /// This computes causal depthwise convolution using explicit sliding windows.
    pub fn forward_with_state_reference(
        &self,
        x: &Tensor,
        state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len, channels) = x.dims3()?;
        if seq_len == 0 {
            let empty_state = if let Some(s) = state {
                s.clone()
            } else {
                Tensor::zeros(
                    &[batch_size, channels, self.kernel_size],
                    DType::F32,
                    x.device(),
                )?
            };
            return Ok((x.clone(), empty_state));
        }

        let history = if let Some(s) = state {
            s.clone()
        } else {
            Tensor::zeros(
                &[batch_size, channels, self.kernel_size],
                DType::F32,
                x.device(),
            )?
        };

        let x_t = x.transpose(1, 2)?; // (B, C, L)
        let history_cat = Tensor::cat(&[&history, &x_t], 2)?; // (B, C, K+L)

        let weight_broadcast = self.weight.reshape(&[1, channels, self.kernel_size])?;
        let mut output_rows = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Window ending at current step with fixed kernel size.
            let window = history_cat.narrow(2, t + 1, self.kernel_size)?; // (B, C, K)
            let weighted = window.broadcast_mul(&weight_broadcast)?;
            let sum_k = weighted.sum_keepdim(2)?.squeeze(2)?; // (B, C)
            output_rows.push(sum_k);
        }

        let stacked = Tensor::cat(&output_rows, 0)? // (L*B, C)
            .reshape(&[seq_len, batch_size, channels])?
            .transpose(0, 1)?; // (B, L, C)

        let new_state_start = self.kernel_size + seq_len - self.kernel_size;
        let new_state = history_cat.narrow(2, new_state_start, self.kernel_size)?;
        Ok((stacked, new_state))
    }

    /// Optimized implementation backed by Candle's grouped `conv1d`.
    ///
    /// Depthwise convolution is encoded as:
    /// - input: `(B, C, L)`
    /// - kernel: `(C, 1, K)`
    /// - groups: `C`
    fn forward_with_state_fast(
        &self,
        x: &Tensor,
        state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, seq_len, channels) = x.dims3()?;
        if seq_len == 0 {
            let empty_state = if let Some(s) = state {
                s.clone()
            } else {
                Tensor::zeros(
                    &[batch_size, channels, self.kernel_size],
                    DType::F32,
                    x.device(),
                )?
            };
            return Ok((x.clone(), empty_state));
        }

        let history = if let Some(s) = state {
            s.clone()
        } else {
            Tensor::zeros(
                &[batch_size, channels, self.kernel_size],
                DType::F32,
                x.device(),
            )?
        };

        let x_t = x.transpose(1, 2)?; // (B, C, L)
        let history_cat = Tensor::cat(&[&history, &x_t], 2)?; // (B, C, K+L)

        // Depthwise conv kernel format for grouped conv1d: (C_out=C, C_in/groups=1, K).
        let kernel = self.weight.reshape(&[channels, 1, self.kernel_size])?;
        let conv_all = history_cat.conv1d(&kernel, 0, 1, 1, channels)?; // (B, C, L+1)
        let conv = conv_all.narrow(2, 1, seq_len)?; // causal alignment
        let output = conv.transpose(1, 2)?; // (B, L, C)

        let new_state = history_cat.narrow(2, seq_len, self.kernel_size)?;
        Ok((output, new_state))
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Get the weight tensor (for optimizer)
    pub fn get_weight(&self) -> Tensor {
        self.weight.clone()
    }
}

/// FusedRMSNormSwishGate for Gated DeltaNet output
///
/// Combines RMS normalization and SiLU gating in one layer.
/// This matches the official Qwen3-Next `RMSNormGated` behavior:
///   output = RMSNorm(x) * silu(gate_input)
///
/// Unlike the previous version, there is no extra learned gate bias term.
#[derive(Clone)]
pub struct FusedRMSNormSwishGate {
    weight: Tensor, // (d_model,) - RMSNorm weight
    eps: f32,
    hidden_act: String,
}

impl FusedRMSNormSwishGate {
    pub fn new(d_model: usize, eps: f64, hidden_act: &str, device: &Device) -> Result<Self> {
        // Initialize RMSNorm weight to ones (official behavior).
        let weight = Tensor::ones(&[d_model], DType::F32, device)?;
        let weight = make_trainable(weight)?;

        Ok(Self {
            weight,
            eps: eps as f32,
            hidden_act: hidden_act.to_string(),
        })
    }

    /// Forward pass: RMSNorm + SiLU gating.
    ///
    /// Supports any rank >= 2 as long as the last dim is the hidden dim.
    pub fn forward(&self, x: &Tensor, gate_input: &Tensor) -> Result<Tensor> {
        let x_dims = x.dims().to_vec();
        let d = *x_dims.last().ok_or_else(|| {
            candle_core::Error::Msg("RMSNormGated input must have at least 1 dimension".to_string())
        })?;
        let n = x_dims[..x_dims.len().saturating_sub(1)]
            .iter()
            .product::<usize>();

        // Flatten all leading dims for per-token normalization.
        let x_flat = x.reshape(&[n, d])?;
        let sum_sq = x_flat.sqr()?.sum_keepdim(1)?;
        let d_tensor = Tensor::new(d as f32, x_flat.device())?;
        let eps_tensor = Tensor::new(self.eps, x_flat.device())?;
        let rms = sum_sq
            .broadcast_div(&d_tensor)?
            .broadcast_add(&eps_tensor)?
            .sqrt()?;
        let x_norm = x_flat.broadcast_div(&rms)?;
        let x_norm = x_norm.broadcast_mul(&self.weight)?;

        // Gate path uses SiLU exactly as in official implementation.
        let gate_flat = gate_input.reshape(&[n, d])?;
        let gate_act = apply_hidden_act(&gate_flat, &self.hidden_act)?;
        let output = x_norm.broadcast_mul(&gate_act)?;

        output.reshape(x_dims.as_slice())
    }

    /// Get weight tensor (for optimizer)
    pub fn get_weight(&self) -> Tensor {
        self.weight.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, D};

    #[test]
    fn test_short_conv_fast_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let conv = ShortConvolution::new(4, 4, 4, 0.02, &device)?;

        let x = Tensor::randn(0.0, 1.0, (2, 6, 4), &device)?.to_dtype(DType::F32)?;
        let state = Tensor::randn(0.0, 1.0, (2, 4, 4), &device)?.to_dtype(DType::F32)?;

        let (fast_out, fast_state) = conv.forward_with_state(&x, Some(&state))?;
        let (ref_out, ref_state) = conv.forward_with_state_reference(&x, Some(&state))?;

        let out_diff = fast_out
            .broadcast_sub(&ref_out)?
            .abs()?
            .reshape(&[2 * 6 * 4])?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        let state_diff = fast_state
            .broadcast_sub(&ref_state)?
            .abs()?
            .reshape(&[2 * 4 * 4])?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;

        assert!(
            out_diff < 1e-5,
            "fast/reference conv output diff {}",
            out_diff
        );
        assert!(
            state_diff < 1e-6,
            "fast/reference conv state diff {}",
            state_diff
        );
        Ok(())
    }
}
