use candle_core::{Error, Result, Tensor};

/// Numerically stable sigmoid from primitive ops.
///
/// Keep this backend-agnostic so CUDA builds without a dedicated sigmoid kernel
/// (e.g. candle 0.8 on some GPUs) still work.
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let half = Tensor::new(0.5f32, x.device())?;
    let one = Tensor::new(1.0f32, x.device())?;
    // sigmoid(x) = 0.5 * (tanh(0.5 * x) + 1)
    let half_x = x.broadcast_mul(&half)?;
    let tanh_half_x = half_x.tanh()?;
    let shifted = tanh_half_x.broadcast_add(&one)?;
    shifted.broadcast_mul(&half)
}

/// SiLU / Swish activation implemented via primitive ops.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let sig = sigmoid(x)?;
    x.broadcast_mul(&sig)
}

/// Apply a HuggingFace-style hidden activation by name.
pub fn apply_hidden_act(x: &Tensor, hidden_act: &str) -> Result<Tensor> {
    match hidden_act {
        "silu" | "swish" => silu(x),
        "gelu" => x.gelu(),
        "relu" => x.relu(),
        _ => Err(Error::Msg(format!("unsupported hidden_act: {hidden_act}"))),
    }
}
