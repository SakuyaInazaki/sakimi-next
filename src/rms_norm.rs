use candle_core::{DType, Result, Tensor, D};

use crate::trainable::make_trainable;

/// RMSNorm variants used in Sakimi-Next.
///
/// - Decoder layer norms follow Qwen3-Next/Gemma3 style: `norm(x) * (1 + weight)` with weight initialized to 0.
/// - Q/K norms in attention use the same zero-centered form in official Qwen3-Next.
/// - `from_device_standard` is kept for optional experiments.
#[derive(Clone)]
pub struct RMSNorm {
    pub weight: Tensor,
    eps: f64,
    add_unit_offset: bool,
}

impl RMSNorm {
    pub fn new(weight: Tensor, add_unit_offset: bool, eps: f64) -> Self {
        Self {
            weight,
            eps,
            add_unit_offset,
        }
    }

    /// Qwen3-Next/Gemma3 style RMSNorm: `norm(x) * (1 + weight)`.
    pub fn from_device(d_model: usize, device: &candle_core::Device) -> Result<Self> {
        Self::from_device_with_eps(d_model, device, 1e-6)
    }

    pub fn from_device_with_eps(
        d_model: usize,
        device: &candle_core::Device,
        eps: f64,
    ) -> Result<Self> {
        let weight = Tensor::zeros(&[d_model], DType::F32, device)?;
        let weight = make_trainable(weight)?;
        Ok(Self::new(weight, true, eps))
    }

    /// Standard RMSNorm: `norm(x) * weight`.
    pub fn from_device_standard(d_model: usize, device: &candle_core::Device) -> Result<Self> {
        Self::from_device_standard_with_eps(d_model, device, 1e-6)
    }

    pub fn from_device_standard_with_eps(
        d_model: usize,
        device: &candle_core::Device,
        eps: f64,
    ) -> Result<Self> {
        let weight = Tensor::ones(&[d_model], DType::F32, device)?;
        let weight = make_trainable(weight)?;
        Ok(Self::new(weight, false, eps))
    }

    /// Forward pass for RMSNorm.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute RMS: sqrt(mean(x^2) + eps)
        let x_squared = x.sqr()?;
        let mean_sq = x_squared.mean_keepdim(D::Minus1)?;
        let rms_tensor = Tensor::new(self.eps as f32, x.device())?;
        let mean_sq = mean_sq.broadcast_add(&rms_tensor)?;
        let rms = mean_sq.sqrt()?;

        // Normalize: x / RMS
        let normalized = x.broadcast_div(&rms)?;

        // Scale by either weight or (1 + weight), depending on the variant.
        let scaled_weight = if self.add_unit_offset {
            let dim = self.weight.dims1()?;
            let one = Tensor::ones(&[dim], DType::F32, self.weight.device())?;
            self.weight.broadcast_add(&one)?
        } else {
            self.weight.clone()
        };

        normalized.broadcast_mul(&scaled_weight)
    }
}
