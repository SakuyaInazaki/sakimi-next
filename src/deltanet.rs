use candle_core::{DType, Device, Result, Tensor};
use rand::Rng;

use crate::activation::apply_hidden_act;
use crate::activation::sigmoid;
use crate::cache::Qwen3NextDynamicCache;
use crate::conv::{FusedRMSNormSwishGate, ShortConvolution};
use crate::trainable::make_trainable;
use crate::Config;

/// Gated DeltaNet aligned with the official Qwen3-Next recurrence path.
///
/// Key points:
/// - `in_proj_qkvz` jointly projects q/k/v/z.
/// - `in_proj_ba` projects beta and a.
/// - depthwise short convolution on concatenated qkv.
/// - recurrent gated-delta-rule update with `g = -exp(A_log) * softplus(a + dt_bias)`.
/// - RMSNormGated before `out_proj`.
#[derive(Clone)]
pub struct GatedDeltaNet {
    /// Input projection for q, k, v, z.
    pub in_proj_qkvz: Tensor,
    /// Input projection for beta and a.
    pub in_proj_ba: Tensor,
    /// Discretization bias.
    pub dt_bias: Tensor,
    /// Decay parameter in log space.
    pub a_log: Tensor,
    /// Output projection.
    pub out_proj: Tensor,

    conv1d: ShortConvolution,
    norm: FusedRMSNormSwishGate,

    d_model: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    hidden_act: String,
    use_fast_kernels: bool,
    device: Device,
}

impl GatedDeltaNet {
    pub fn new(cfg: &Config, device: &Device) -> Result<Self> {
        let d_model = cfg.d_model;
        let num_v_heads = cfg.linear_num_value_heads;
        let num_k_heads = cfg.linear_num_key_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;

        if num_v_heads % num_k_heads != 0 {
            return Err(candle_core::Error::Msg(format!(
                "num_v_heads ({}) must be divisible by num_k_heads ({})",
                num_v_heads, num_k_heads
            )));
        }

        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = 2 * key_dim + value_dim;

        let std = cfg.initializer_range;
        let in_proj_qkvz = Tensor::randn(
            0f32,
            std as f32,
            (d_model, 2 * key_dim + 2 * value_dim),
            device,
        )?
        .to_dtype(DType::F32)?;
        let in_proj_qkvz = make_trainable(in_proj_qkvz)?;
        let in_proj_ba = Tensor::randn(0f32, std as f32, (d_model, 2 * num_v_heads), device)?
            .to_dtype(DType::F32)?;
        let in_proj_ba = make_trainable(in_proj_ba)?;
        let out_proj = Tensor::randn(0f32, std as f32, (value_dim, d_model), device)?
            .to_dtype(DType::F32)?;
        let out_proj = make_trainable(out_proj)?;

        let dt_bias = Tensor::ones(&[num_v_heads], DType::F32, device)?;
        let dt_bias = make_trainable(dt_bias)?;

        let mut rng = rand::thread_rng();
        let mut a_init = Vec::with_capacity(num_v_heads);
        for _ in 0..num_v_heads {
            let v = rng.gen_range(0.0f32..16.0f32);
            a_init.push(v.ln());
        }
        let a_log = Tensor::new(a_init.as_slice(), device)?;
        let a_log = make_trainable(a_log)?;

        let conv1d = ShortConvolution::new(
            conv_dim,
            conv_dim,
            cfg.linear_conv_kernel_dim,
            cfg.initializer_range,
            device,
        )?;
        let norm =
            FusedRMSNormSwishGate::new(head_v_dim, cfg.rms_norm_eps, &cfg.hidden_act, device)?;

        Ok(Self {
            in_proj_qkvz,
            in_proj_ba,
            dt_bias,
            a_log,
            out_proj,
            conv1d,
            norm,
            d_model,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            hidden_act: cfg.hidden_act.clone(),
            use_fast_kernels: cfg.use_fast_kernels,
            device: device.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_cache(x, None, 0, None)
    }

    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: Option<&mut Qwen3NextDynamicCache>,
        layer_idx: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hidden_states = apply_mask_to_padding_states(x, attention_mask)?;
        let (batch_size, seq_len, d_model) = hidden_states.dims3()?;
        let x_2d = hidden_states.reshape(&[batch_size * seq_len, d_model])?;

        let mixed_qkvz = x_2d.matmul(&self.in_proj_qkvz)?.reshape(&[
            batch_size,
            seq_len,
            2 * self.key_dim + 2 * self.value_dim,
        ])?;
        let mixed_ba =
            x_2d.matmul(&self.in_proj_ba)?
                .reshape(&[batch_size, seq_len, 2 * self.num_v_heads])?;

        let (query, key, value, z, beta, a) =
            self.fix_query_key_value_ordering(&mixed_qkvz, &mixed_ba)?;

        let query_flat = query.reshape(&[batch_size, seq_len, self.key_dim])?;
        let key_flat = key.reshape(&[batch_size, seq_len, self.key_dim])?;
        let value_flat = value.reshape(&[batch_size, seq_len, self.value_dim])?;

        let mixed_qkv = Tensor::cat(&[&query_flat, &key_flat, &value_flat], 2)?;
        let mut cache = cache;
        let prev_conv_state = cache.as_deref_mut().and_then(|c| c.conv_state(layer_idx));
        let (mixed_qkv, new_conv_state) = if self.use_fast_kernels {
            self.conv1d
                .forward_with_state(&mixed_qkv, prev_conv_state.as_ref())?
        } else {
            self.conv1d
                .forward_with_state_reference(&mixed_qkv, prev_conv_state.as_ref())?
        };
        if let Some(cache) = cache.as_deref_mut() {
            cache.set_conv_state(layer_idx, new_conv_state)?;
        }
        let mixed_qkv = apply_hidden_act(&mixed_qkv, &self.hidden_act)?;

        let query = mixed_qkv.narrow(2, 0, self.key_dim)?.reshape(&[
            batch_size,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ])?;
        let key = mixed_qkv.narrow(2, self.key_dim, self.key_dim)?.reshape(&[
            batch_size,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ])?;
        let value = mixed_qkv
            .narrow(2, 2 * self.key_dim, self.value_dim)?
            .reshape(&[batch_size, seq_len, self.num_v_heads, self.head_v_dim])?;

        let beta = sigmoid(&beta)?;
        let dt_bias = self.dt_bias.reshape(&[1, 1, self.num_v_heads])?;
        let a_plus_bias = a.broadcast_add(&dt_bias)?;
        let g = self.a_log.exp()?.reshape(&[1, 1, self.num_v_heads])?;
        let g = g.broadcast_mul(&softplus(&a_plus_bias)?)?;
        let g = g.affine(-1.0, 0.0)?;

        let repeat_factor = self.num_v_heads / self.num_k_heads;
        let query = if repeat_factor > 1 {
            self.repeat_heads(query, repeat_factor)?
        } else {
            query
        };
        let key = if repeat_factor > 1 {
            self.repeat_heads(key, repeat_factor)?
        } else {
            key
        };

        let prev_recurrent_state = cache
            .as_deref_mut()
            .and_then(|c| c.recurrent_state(layer_idx));
        let (core_attn_out, final_recurrent_state) = if self.use_fast_kernels {
            self.recurrent_gated_delta_rule_fast(
                &query,
                &key,
                &value,
                &g,
                &beta,
                prev_recurrent_state.as_ref(),
            )?
        } else {
            self.recurrent_gated_delta_rule_reference(
                &query,
                &key,
                &value,
                &g,
                &beta,
                prev_recurrent_state.as_ref(),
            )?
        };
        if let Some(cache) = cache.as_deref_mut() {
            cache.set_recurrent_state(layer_idx, final_recurrent_state)?;
        }
        let core_attn_out = self.norm.forward(&core_attn_out, &z)?;

        let core_attn_out_2d = core_attn_out.reshape(&[batch_size * seq_len, self.value_dim])?;
        core_attn_out_2d
            .matmul(&self.out_proj)?
            .reshape(&[batch_size, seq_len, self.d_model])
    }

    fn fix_query_key_value_ordering(
        &self,
        mixed_qkvz: &Tensor,
        mixed_ba: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let (b, l, _) = mixed_qkvz.dims3()?;
        let v_per_group = self.num_v_heads / self.num_k_heads;
        let grouped_v_dim = v_per_group * self.head_v_dim;

        let qkvz = mixed_qkvz.reshape(&[
            b,
            l,
            self.num_k_heads,
            2 * self.head_k_dim + 2 * grouped_v_dim,
        ])?;
        let ba = mixed_ba.reshape(&[b, l, self.num_k_heads, 2 * v_per_group])?;

        let query = qkvz.narrow(3, 0, self.head_k_dim)?;
        let key = qkvz.narrow(3, self.head_k_dim, self.head_k_dim)?;
        let value = qkvz
            .narrow(3, 2 * self.head_k_dim, grouped_v_dim)?
            .reshape(&[b, l, self.num_v_heads, self.head_v_dim])?;
        let z = qkvz
            .narrow(3, 2 * self.head_k_dim + grouped_v_dim, grouped_v_dim)?
            .reshape(&[b, l, self.num_v_heads, self.head_v_dim])?;

        let beta = ba
            .narrow(3, 0, v_per_group)?
            .reshape(&[b, l, self.num_v_heads])?;
        let a = ba
            .narrow(3, v_per_group, v_per_group)?
            .reshape(&[b, l, self.num_v_heads])?;

        Ok((query, key, value, z, beta, a))
    }

    fn repeat_heads(&self, x: Tensor, repeat_factor: usize) -> Result<Tensor> {
        let (_, _, n_heads, _) = x.dims4()?;
        let mut repeated = Vec::with_capacity(n_heads * repeat_factor);
        for i in 0..n_heads {
            let head = x.narrow(2, i, 1)?;
            for _ in 0..repeat_factor {
                repeated.push(head.clone());
            }
        }
        Tensor::cat(&repeated, 2)
    }

    /// Optimized recurrent gated-delta rule:
    /// - keeps the required recurrence over sequence length
    /// - vectorizes across batch and heads
    /// - preserves gradients for `g` and `beta` (no scalar extraction)
    fn recurrent_gated_delta_rule_fast(
        &self,
        query: &Tensor,                 // (B, L, H, K)
        key: &Tensor,                   // (B, L, H, K)
        value: &Tensor,                 // (B, L, H, V)
        g: &Tensor,                     // (B, L, H)
        beta: &Tensor,                  // (B, L, H)
        initial_state: Option<&Tensor>, // (B, H, K, V)
    ) -> Result<(Tensor, Tensor)> {
        let query = l2_normalize_last(query)?;
        let key = l2_normalize_last(key)?;

        let (batch_size, seq_len, num_heads, k_dim) = key.dims4()?;
        let (_, _, _, v_dim) = value.dims4()?;

        let scale = Tensor::new(1.0f32 / (k_dim as f32).sqrt(), query.device())?;
        let query = query.broadcast_mul(&scale)?;

        let mut state = if let Some(initial_state) = initial_state {
            initial_state.clone()
        } else {
            Tensor::zeros(
                &[batch_size, num_heads, k_dim, v_dim],
                DType::F32,
                query.device(),
            )?
        };

        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let q_t = query
                .narrow(1, t, 1)?
                .reshape(&[batch_size, num_heads, k_dim])?;
            let k_t = key
                .narrow(1, t, 1)?
                .reshape(&[batch_size, num_heads, k_dim])?;
            let v_t = value
                .narrow(1, t, 1)?
                .reshape(&[batch_size, num_heads, v_dim])?;

            let g_t = g.narrow(1, t, 1)?.reshape(&[batch_size, num_heads, 1, 1])?;
            let beta_t = beta.narrow(1, t, 1)?.reshape(&[batch_size, num_heads, 1])?;

            state = state.broadcast_mul(&g_t.exp()?)?;

            let k_t_exp = k_t.reshape(&[batch_size, num_heads, k_dim, 1])?;
            let kv_mem = state.broadcast_mul(&k_t_exp)?.sum_keepdim(2)?.squeeze(2)?; // (B, H, V)
            let delta = v_t.broadcast_sub(&kv_mem)?.broadcast_mul(&beta_t)?;

            let update =
                k_t_exp.broadcast_mul(&delta.reshape(&[batch_size, num_heads, 1, v_dim])?)?;
            state = state.broadcast_add(&update)?;

            let y_t = state
                .broadcast_mul(&q_t.reshape(&[batch_size, num_heads, k_dim, 1])?)?
                .sum_keepdim(2)?
                .squeeze(2)?; // (B, H, V)
            outputs.push(y_t.reshape(&[batch_size, 1, num_heads, v_dim])?);
        }

        let output_refs: Vec<&Tensor> = outputs.iter().collect();
        let output = Tensor::cat(output_refs.as_slice(), 1)?; // (B, L, H, V)
        Ok((output, state))
    }

    /// Reference implementation kept for parity checks.
    fn recurrent_gated_delta_rule_reference(
        &self,
        query: &Tensor,                 // (B, L, H, K)
        key: &Tensor,                   // (B, L, H, K)
        value: &Tensor,                 // (B, L, H, V)
        g: &Tensor,                     // (B, L, H)
        beta: &Tensor,                  // (B, L, H)
        initial_state: Option<&Tensor>, // (B, H, K, V)
    ) -> Result<(Tensor, Tensor)> {
        let query = l2_normalize_last(query)?;
        let key = l2_normalize_last(key)?;

        let (_, _, _, k_dim) = key.dims4()?;
        let scale = Tensor::new(1.0f32 / (k_dim as f32).sqrt(), &self.device)?;
        let query = query.broadcast_mul(&scale)?;

        let (batch_size, seq_len, num_heads, k_dim) = key.dims4()?;
        let (_, _, _, v_dim) = value.dims4()?;
        let mut all_outputs = Vec::with_capacity(batch_size);
        let mut all_final_states = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let mut batch_outputs = Vec::with_capacity(num_heads);
            let mut batch_final_states = Vec::with_capacity(num_heads);
            for head_idx in 0..num_heads {
                let q_bh = query
                    .narrow(0, batch_idx, 1)?
                    .narrow(2, head_idx, 1)?
                    .reshape(&[seq_len, k_dim])?;
                let k_bh = key
                    .narrow(0, batch_idx, 1)?
                    .narrow(2, head_idx, 1)?
                    .reshape(&[seq_len, k_dim])?;
                let v_bh = value
                    .narrow(0, batch_idx, 1)?
                    .narrow(2, head_idx, 1)?
                    .reshape(&[seq_len, v_dim])?;
                let g_bh = g
                    .narrow(0, batch_idx, 1)?
                    .narrow(2, head_idx, 1)?
                    .reshape(&[seq_len])?;
                let beta_bh = beta
                    .narrow(0, batch_idx, 1)?
                    .narrow(2, head_idx, 1)?
                    .reshape(&[seq_len])?;

                let mut state = if let Some(initial_state) = initial_state {
                    initial_state
                        .narrow(0, batch_idx, 1)?
                        .narrow(1, head_idx, 1)?
                        .reshape(&[k_dim, v_dim])?
                } else {
                    Tensor::zeros(&[k_dim, v_dim], DType::F32, &self.device)?
                };
                let mut head_outputs = Vec::with_capacity(seq_len);

                for t in 0..seq_len {
                    let q_t = q_bh.narrow(0, t, 1)?.reshape(&[k_dim])?;
                    let k_t = k_bh.narrow(0, t, 1)?.reshape(&[k_dim])?;
                    let v_t = v_bh.narrow(0, t, 1)?.reshape(&[v_dim])?;
                    let g_t = g_bh.narrow(0, t, 1)?.reshape(&[])?.to_scalar::<f32>()?;
                    let beta_t = beta_bh.narrow(0, t, 1)?.reshape(&[])?.to_scalar::<f32>()?;

                    let g_exp = Tensor::new(g_t.exp(), &self.device)?;
                    state = state.broadcast_mul(&g_exp)?;

                    let kv_mem = k_t
                        .reshape(&[1, k_dim])?
                        .matmul(&state)?
                        .reshape(&[v_dim])?;
                    let delta = v_t.broadcast_sub(&kv_mem)?;
                    let beta_t = Tensor::new(beta_t, &self.device)?;
                    let delta = delta.broadcast_mul(&beta_t)?;

                    let update = k_t
                        .reshape(&[k_dim, 1])?
                        .matmul(&delta.reshape(&[1, v_dim])?)?;
                    state = state.broadcast_add(&update)?;

                    let y_t = q_t
                        .reshape(&[1, k_dim])?
                        .matmul(&state)?
                        .reshape(&[v_dim])?;
                    head_outputs.push(y_t);
                }

                let head_out = Tensor::cat(&head_outputs, 0)?.reshape(&[seq_len, v_dim])?;
                batch_outputs.push(head_out);
                batch_final_states.push(state);
            }

            let stacked = Tensor::cat(&batch_outputs, 0)?
                .reshape(&[num_heads, seq_len, v_dim])?
                .transpose(0, 1)?;
            all_outputs.push(stacked);

            let final_state =
                Tensor::cat(&batch_final_states, 0)?.reshape(&[num_heads, k_dim, v_dim])?;
            all_final_states.push(final_state);
        }

        let stacked = Tensor::cat(&all_outputs, 0)?;
        let output = stacked.reshape(&[batch_size, seq_len, num_heads, v_dim])?;
        let final_state =
            Tensor::cat(&all_final_states, 0)?.reshape(&[batch_size, num_heads, k_dim, v_dim])?;
        Ok((output, final_state))
    }

    pub fn get_conv_weight(&self) -> Tensor {
        self.conv1d.get_weight()
    }

    pub fn get_norm_weight(&self) -> Tensor {
        self.norm.get_weight()
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    x.exp()?.affine(1.0, 1.0)?.log()
}

fn apply_mask_to_padding_states(
    hidden_states: &Tensor,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    if let Some(mask) = attention_mask {
        let (batch_size, seq_len, _hidden) = hidden_states.dims3()?;
        let (mask_b, mask_l) = mask.dims2()?;
        if batch_size > 1 && seq_len > 1 && batch_size == mask_b && seq_len == mask_l {
            let mask = mask
                .to_dtype(hidden_states.dtype())?
                .reshape(&[batch_size, seq_len, 1])?;
            return hidden_states.broadcast_mul(&mask);
        }
    }
    Ok(hidden_states.clone())
}

fn l2_normalize_last(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let d = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("tensor must have rank >= 1".to_string()))?;
    let n = dims[..dims.len().saturating_sub(1)]
        .iter()
        .product::<usize>();

    let x_flat = x.reshape(&[n, d])?;
    let sum_sq = x_flat.sqr()?.sum_keepdim(1)?;
    let eps = Tensor::new(1e-6f32, x.device())?;
    let norm = sum_sq.broadcast_add(&eps)?.sqrt()?;
    let normalized = x_flat.broadcast_div(&norm)?;
    normalized.reshape(dims.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use candle_core::{DType, Device, D};

    #[test]
    fn test_recurrent_fast_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = Config::tiny_5m();
        cfg.use_fast_kernels = true;
        let net = GatedDeltaNet::new(&cfg, &device)?;

        let b = 2;
        let l = 4;
        let h = net.num_v_heads;
        let k_dim = net.head_k_dim;
        let v_dim = net.head_v_dim;

        let q = Tensor::randn(0.0, 1.0, (b, l, h, k_dim), &device)?.to_dtype(DType::F32)?;
        let k = Tensor::randn(0.0, 1.0, (b, l, h, k_dim), &device)?.to_dtype(DType::F32)?;
        let v = Tensor::randn(0.0, 1.0, (b, l, h, v_dim), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0.0, 1.0, (b, l, h), &device)?.to_dtype(DType::F32)?;
        let beta = Tensor::randn(0.0, 1.0, (b, l, h), &device)?.to_dtype(DType::F32)?;

        let (fast_out, fast_state) =
            net.recurrent_gated_delta_rule_fast(&q, &k, &v, &g, &beta, None)?;
        let (ref_out, ref_state) =
            net.recurrent_gated_delta_rule_reference(&q, &k, &v, &g, &beta, None)?;

        let out_diff = fast_out
            .broadcast_sub(&ref_out)?
            .abs()?
            .reshape(&[b * l * h * v_dim])?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        let state_diff = fast_state
            .broadcast_sub(&ref_state)?
            .abs()?
            .reshape(&[b * h * k_dim * v_dim])?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;

        assert!(out_diff < 1e-4, "fast/reference output diff {}", out_diff);
        assert!(
            state_diff < 1e-4,
            "fast/reference recurrent state diff {}",
            state_diff
        );
        Ok(())
    }
}
