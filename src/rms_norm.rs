use candle_core::{Result, Tensor, D, DType};

/// Zero-Centered RMSNorm (Gemma style)
///
/// Qwen3-Next uses Zero-Centered RMSNorm [7] to improve training stability.
///
/// Standard RMSNorm: output = x / RMS(x) * weight
/// Zero-Centered: output = (RMSNorm(x) - mean(RMSNorm(x))) * weight
///
/// This helps prevent norm weights from growing unbounded.
#[derive(Clone)]
pub struct RMSNorm {
    pub weight: Tensor,
    eps: f64,
    d_model: usize,
}

impl RMSNorm {
    pub fn new(weight: Tensor, d_model: usize) -> Self {
        Self {
            weight,
            eps: 1e-6,
            d_model,
        }
    }

    pub fn from_device(d_model: usize, device: &candle_core::Device) -> Result<Self> {
        let weight = Tensor::ones(&[d_model], DType::F32, device)?;
        Ok(Self::new(weight, d_model))
    }

    /// Forward pass with zero-centering (Gemma style)
    ///
    /// Input shape: (batch_size, seq_len, d_model)
    /// Output shape: (batch_size, seq_len, d_model)
    ///
    /// Zero-Centered RMSNorm formula (per Gemma paper [7]):
    ///   1. normalized = x / RMS(x) * weight
    ///   2. output = normalized - mean(normalized, dim=-1, keepdim=True)
    ///
    /// This differs from standard RMSNorm by centering the output,
    /// which prevents norm weights from growing unbounded.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Step 1: Compute RMS: sqrt(mean(x^2) + eps)
        let x_squared = x.sqr()?;
        let mean_sq = x_squared.mean_keepdim(D::Minus1)?;
        let rms_tensor = Tensor::new(self.eps as f32, x.device())?;
        let mean_sq = mean_sq.broadcast_add(&rms_tensor)?;
        let rms = mean_sq.sqrt()?;

        // Step 2: Normalize: x / RMS
        let normalized = x.broadcast_div(&rms)?;

        // Step 3: Scale by weight (this is the RMSNorm output)
        let scaled = normalized.broadcast_mul(&self.weight)?;

        // Step 4: Zero-center: subtract mean along hidden dimension
        // This is the KEY difference from standard RMSNorm
        let mean = scaled.mean_keepdim(D::Minus1)?;
        let centered = scaled.broadcast_sub(&mean)?;

        Ok(centered)
    }
}
