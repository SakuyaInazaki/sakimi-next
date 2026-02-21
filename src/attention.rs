use candle_core::{DType, Result, Tensor, D};
use candle_nn::ops;
use std::cell::Cell;

use crate::activation::sigmoid;
use crate::cache::Qwen3NextDynamicCache;
use crate::trainable::make_trainable;
use crate::{rms_norm::RMSNorm, rope::RoPE, Config};

/// Gated multi-head attention with GQA.
///
/// Official Qwen3-Next behavior:
/// - `q_proj` emits query + gate.
/// - q/k per-head RMSNorm.
/// - RoPE on q/k.
/// - gate is sigmoid and applied before `o_proj`.
#[derive(Clone)]
pub struct GatedAttention {
    /// Combined Q+gate projection: (d_model, 2 * (n_heads * head_dim))
    pub q_proj: Tensor,
    /// K projection: (d_model, n_kv_heads * head_dim)
    pub k_proj: Tensor,
    /// V projection: (d_model, n_kv_heads * head_dim)
    pub v_proj: Tensor,
    /// Output projection: (n_heads * head_dim, d_model)
    pub o_proj: Tensor,
    /// Optional attention biases (official `attention_bias`).
    pub q_bias: Option<Tensor>,
    pub k_bias: Option<Tensor>,
    pub v_bias: Option<Tensor>,
    pub o_bias: Option<Tensor>,
    /// Q norm on head dim.
    pub q_norm: RMSNorm,
    /// K norm on head dim.
    pub k_norm: RMSNorm,
    n_heads: usize,
    n_kv_heads: usize,
    n_groups: usize,
    head_dim: usize,
    attn_hidden_size: usize,
    attention_dropout: f32,
    training: Cell<bool>,
    rope: RoPE,
}

impl GatedAttention {
    pub fn new(cfg: &Config, device: &candle_core::Device) -> Result<Self> {
        let d_model = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.kv_heads;
        let head_dim = cfg.head_dim;
        let n_groups = n_heads / n_kv_heads;
        let attn_hidden_size = n_heads * head_dim;

        let std = cfg.initializer_range;

        let q_proj = Tensor::randn(0f32, std as f32, (d_model, 2 * attn_hidden_size), device)?
            .to_dtype(DType::F32)?;
        let q_proj = make_trainable(q_proj)?;
        let k_proj = Tensor::randn(0f32, std as f32, (d_model, n_kv_heads * head_dim), device)?
            .to_dtype(DType::F32)?;
        let k_proj = make_trainable(k_proj)?;
        let v_proj = Tensor::randn(0f32, std as f32, (d_model, n_kv_heads * head_dim), device)?
            .to_dtype(DType::F32)?;
        let v_proj = make_trainable(v_proj)?;
        let o_proj =
            Tensor::randn(0f32, std as f32, (attn_hidden_size, d_model), device)?
                .to_dtype(DType::F32)?;
        let o_proj = make_trainable(o_proj)?;

        let (q_bias, k_bias, v_bias, o_bias) = if cfg.attention_bias {
            (
                Some(make_trainable(Tensor::zeros(
                    &[2 * attn_hidden_size],
                    DType::F32,
                    device,
                )?)?),
                Some(make_trainable(Tensor::zeros(
                    &[n_kv_heads * head_dim],
                    DType::F32,
                    device,
                )?)?),
                Some(make_trainable(Tensor::zeros(
                    &[n_kv_heads * head_dim],
                    DType::F32,
                    device,
                )?)?),
                Some(make_trainable(Tensor::zeros(
                    &[d_model],
                    DType::F32,
                    device,
                )?)?),
            )
        } else {
            (None, None, None, None)
        };

        let q_norm = RMSNorm::from_device_with_eps(head_dim, device, cfg.rms_norm_eps)?;
        let k_norm = RMSNorm::from_device_with_eps(head_dim, device, cfg.rms_norm_eps)?;
        let rope = RoPE::new(
            head_dim,
            cfg.max_seq_len,
            cfg.partial_rotary_factor,
            cfg.rope_theta,
        );

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_bias,
            k_bias,
            v_bias,
            o_bias,
            q_norm,
            k_norm,
            n_heads,
            n_kv_heads,
            n_groups,
            head_dim,
            attn_hidden_size,
            attention_dropout: cfg.attention_dropout as f32,
            training: Cell::new(false),
            rope,
        })
    }

    /// Match official behavior: attention dropout is applied only during training.
    pub fn set_training(&self, training: bool) {
        self.training.set(training);
    }

    pub fn forward(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        self.forward_with_cache(x, offset, None, 0, None)
    }

    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        offset: usize,
        cache: Option<&mut Qwen3NextDynamicCache>,
        layer_idx: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, d_model) = x.dims3()?;
        let x_2d = x.reshape(&[batch_size * seq_len, d_model])?;

        // q_and_gate: (B, L, n_heads, 2*head_dim)
        let q_and_gate =
            self.linear_with_optional_bias(&x_2d, &self.q_proj, self.q_bias.as_ref())?;
        let q_and_gate =
            q_and_gate.reshape(&[batch_size, seq_len, self.n_heads, 2 * self.head_dim])?;
        let q = q_and_gate.narrow(3, 0, self.head_dim)?;
        let gate = q_and_gate
            .narrow(3, self.head_dim, self.head_dim)?
            .reshape(&[batch_size, seq_len, self.attn_hidden_size])?;
        let gate = sigmoid(&gate)?;

        let k = self.linear_with_optional_bias(&x_2d, &self.k_proj, self.k_bias.as_ref())?;
        let k = k.reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = self.linear_with_optional_bias(&x_2d, &self.v_proj, self.v_bias.as_ref())?;
        let v = v.reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let (q, k) = self.rope.forward(&q, &k, offset)?;

        let (k, v) = if let Some(cache) = cache {
            cache.update_attention(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        let k = self.repeat_kv(k, self.n_groups)?;
        let v = self.repeat_kv(v, self.n_groups)?;

        let attn = self.scaled_dot_product_attention(&q, &k, &v, offset, attention_mask)?;
        let attn = attn.reshape(&[batch_size, seq_len, self.attn_hidden_size])?;

        let gated = attn.broadcast_mul(&gate)?;
        let gated_2d = gated.reshape(&[batch_size * seq_len, self.attn_hidden_size])?;
        let output =
            self.linear_with_optional_bias(&gated_2d, &self.o_proj, self.o_bias.as_ref())?;
        output.reshape(&[batch_size, seq_len, d_model])
    }

    fn linear_with_optional_bias(
        &self,
        x_2d: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let y = x_2d.matmul(weight)?;
        if let Some(bias) = bias {
            y.broadcast_add(bias)
        } else {
            Ok(y)
        }
    }

    fn scaled_dot_product_attention(
        &self,
        q: &Tensor, // (B, L, n_heads, head_dim)
        k: &Tensor, // (B, K, n_heads, head_dim)
        v: &Tensor, // (B, K, n_heads, head_dim)
        query_offset: usize,
        attention_mask: Option<&Tensor>, // (B, K), 1 for valid tokens.
    ) -> Result<Tensor> {
        let (b, q_len, n_heads, head_dim) = q.dims4()?;
        let (_, k_len, _, _) = k.dims4()?;
        let mut all_outputs = Vec::with_capacity(b);

        for batch_idx in 0..b {
            let mut batch_outputs = Vec::with_capacity(n_heads);

            for head_idx in 0..n_heads {
                let q_bh = q
                    .narrow(0, batch_idx, 1)?
                    .narrow(1, 0, q_len)?
                    .narrow(2, head_idx, 1)?
                    .reshape(&[q_len, head_dim])?;

                let k_bh = k
                    .narrow(0, batch_idx, 1)?
                    .narrow(1, 0, k_len)?
                    .narrow(2, head_idx, 1)?
                    .reshape(&[k_len, head_dim])?;

                let v_bh = v
                    .narrow(0, batch_idx, 1)?
                    .narrow(1, 0, k_len)?
                    .narrow(2, head_idx, 1)?
                    .reshape(&[k_len, head_dim])?;

                let scores = q_bh.matmul(&k_bh.t()?)?;
                let scale = Tensor::new((head_dim as f32).sqrt(), scores.device())?;
                let scores = scores.broadcast_div(&scale)?;
                let scores = self.apply_causal_mask_2d(&scores, q_len, k_len, query_offset)?;
                let scores =
                    self.apply_padding_mask_2d(&scores, attention_mask, batch_idx, k_len, q_len)?;
                let mut attn_weights = ops::softmax(&scores, D::Minus1)?;
                if self.training.get() && self.attention_dropout > 0.0 {
                    attn_weights = ops::dropout(&attn_weights, self.attention_dropout)?;
                }
                let head_out = attn_weights.matmul(&v_bh)?;
                batch_outputs.push(head_out);
            }

            let stacked = Tensor::cat(&batch_outputs, 0)?
                .reshape(&[n_heads, q_len, head_dim])?
                .transpose(0, 1)?;
            all_outputs.push(stacked);
        }

        let stacked = Tensor::cat(&all_outputs, 0)?;
        stacked.reshape(&[b, q_len, n_heads, head_dim])
    }

    fn apply_causal_mask_2d(
        &self,
        scores: &Tensor,
        query_len: usize,
        key_len: usize,
        query_offset: usize,
    ) -> Result<Tensor> {
        let mut mask = vec![0.0f32; query_len * key_len];
        for qi in 0..query_len {
            let q_pos = query_offset + qi;
            for kj in 0..key_len {
                if kj > q_pos {
                    mask[qi * key_len + kj] = f32::NEG_INFINITY;
                }
            }
        }

        let mask_tensor =
            Tensor::new(mask.as_slice(), scores.device())?.reshape(&[query_len, key_len])?;
        scores.broadcast_add(&mask_tensor)
    }

    fn apply_padding_mask_2d(
        &self,
        scores: &Tensor,
        attention_mask: Option<&Tensor>,
        batch_idx: usize,
        key_len: usize,
        query_len: usize,
    ) -> Result<Tensor> {
        let Some(attention_mask) = attention_mask else {
            return Ok(scores.clone());
        };

        let (mask_b, mask_l) = attention_mask.dims2()?;
        if mask_b <= batch_idx || mask_l != key_len {
            return Ok(scores.clone());
        }

        let mask_row = attention_mask
            .narrow(0, batch_idx, 1)?
            .reshape(&[key_len])?
            .to_dtype(DType::F32)?;
        let mask_vals = mask_row.to_vec1::<f32>()?;
        let mut mask = vec![0.0f32; query_len * key_len];
        for k in 0..key_len {
            if mask_vals[k] <= 0.0 {
                for q in 0..query_len {
                    mask[q * key_len + k] = f32::NEG_INFINITY;
                }
            }
        }

        let mask_tensor =
            Tensor::new(mask.as_slice(), scores.device())?.reshape(&[query_len, key_len])?;
        scores.broadcast_add(&mask_tensor)
    }

    fn repeat_kv(&self, x: Tensor, n_groups: usize) -> Result<Tensor> {
        let (_, _, n_kv_heads, _) = x.dims4()?;
        if n_groups == 1 {
            return Ok(x);
        }

        let mut repeated = Vec::with_capacity(n_kv_heads * n_groups);
        for i in 0..n_kv_heads {
            let head = x.narrow(2, i, 1)?;
            for _ in 0..n_groups {
                repeated.push(head.clone());
            }
        }
        Tensor::cat(&repeated, 2)
    }
}
