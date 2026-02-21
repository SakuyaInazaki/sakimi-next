use candle_core::{Result, Tensor, Var};

/// Convert a tensor into a trainable tensor (backed by `Var` storage).
///
/// This guarantees optimizers built via `Var::from_tensor` share the same
/// underlying storage with the model tensor.
pub fn make_trainable(tensor: Tensor) -> Result<Tensor> {
    let var = Var::from_tensor(&tensor)?;
    Ok(var.as_tensor().clone())
}
