use candle_core::{Result, Tensor, Device, DType};
use candle_nn::ops;

use crate::Config;

/// SwiGLU Feed-Forward Network (FFN) layer
///
/// Qwen3-Next uses SwiGLU activation instead of standard GELU.
/// This is the same architecture used in LLaMA, Mistral, and modern LLMs.
///
/// SwiGLU formula:
///   FFN(x) = (xW_gate ⊙ swish(xW_up)) W_down
///   where swish(x) = x * sigmoid(x)
///   and ⊙ denotes element-wise multiplication
///
/// Architecture:
/// - w_gate: (d_model, intermediate_size) - gate projection
/// - w_up: (d_model, intermediate_size) - up projection
/// - w_down: (intermediate_size, d_model) - down projection
/// - Usually intermediate_size = (8/3) * d_model or 3 * d_model for SwiGLU
#[derive(Clone)]
pub struct FFN {
    pub w_gate: Tensor,
    pub w_up: Tensor,
    pub w_down: Tensor,
    d_model: usize,
    intermediate_size: usize,
    device: Device,
}

impl FFN {
    pub fn new(cfg: &Config, device: &Device) -> Result<Self> {
        let d_model = cfg.d_model;
        let intermediate_size = cfg.intermediate_size;

        // SwiGLU uses similar initialization to GELU
        let std = (2.0 / d_model as f64).sqrt();

        // w_gate: (d_model, intermediate_size)
        let w_gate = Tensor::randn(0.0, std, (d_model, intermediate_size), device)?
            .to_dtype(DType::F32)?;

        // w_up: (d_model, intermediate_size)
        let w_up = Tensor::randn(0.0, std, (d_model, intermediate_size), device)?
            .to_dtype(DType::F32)?;

        // w_down: (intermediate_size, d_model)
        let w_down = Tensor::randn(0.0, std, (intermediate_size, d_model), device)?
            .to_dtype(DType::F32)?;

        Ok(Self {
            w_gate,
            w_up,
            w_down,
            d_model,
            intermediate_size,
            device: device.clone(),
        })
    }

    /// Forward pass with SwiGLU activation
    ///
    /// Input shape: (batch_size, seq_len, d_model)
    /// Output shape: (batch_size, seq_len, d_model)
    ///
    /// Computation:
    ///   gate = swish(x @ w_gate) = (x @ w_gate) * sigmoid(x @ w_gate)
    ///   up = x @ w_up
    ///   hidden = gate ⊙ up  (element-wise multiplication)
    ///   output = hidden @ w_down
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, d) = x.dims3()?;

        // Reshape to 2D for matrix multiplication
        let x_2d = x.reshape(&[b * l, d])?;

        // Compute gate projection: (B*L, I)
        let gate = x_2d.matmul(&self.w_gate)?;

        // Compute up projection: (B*L, I)
        let up = x_2d.matmul(&self.w_up)?;

        // Apply SwiGLU: swish(gate) * up
        // swish(x) = x * sigmoid(x)
        let gate_sigmoid = ops::sigmoid(&gate)?;
        let gate_swish = gate.broadcast_mul(&gate_sigmoid)?;
        let hidden = gate_swish.broadcast_mul(&up)?;

        // Down projection: (B*L, I) @ (I, D) -> (B*L, D)
        let output = hidden.matmul(&self.w_down)?;

        // Reshape back to 3D: (B, L, D)
        output.reshape(&[b, l, d])
    }
}
