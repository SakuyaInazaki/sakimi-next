use candle_core::{Result, Tensor, D, Device, DType};
use candle_nn::ops;

use crate::{Config, rope::RoPE, rms_norm::RMSNorm};

/// Gated Multi-Head Attention with GQA
///
/// Qwen3-Next uses [2]:
/// 1. Output gating mechanism to reduce low-rank issues and attention sink
/// 2. Grouped Query Attention (GQA) - multiple Q share K/V
/// 3. Head dim = 256 (increased from 128) - we use smaller for memory constraints
/// 4. RoPE on first 25% of dimensions only
///
/// Gated Attention formula (per paper [2]):
///   attn_output = softmax(Q @ K^T / sqrt(d)) @ V @ O
///   gate = sigmoid(gate_proj(x))
///   output = LayerNorm(attn_output) * gate
///   final = output
///
/// The gate mechanism helps eliminate attention sink and massive activation issues.
/// Note: The simplified version uses only gate_proj, not both gate_proj and up_proj.
#[derive(Clone)]
pub struct GatedAttention {
    // Q projection: (d_model, n_heads * head_dim)
    pub q_proj: Tensor,
    // K projection: (d_model, n_kv_heads * head_dim)
    pub k_proj: Tensor,
    // V projection: (d_model, n_kv_heads * head_dim)
    pub v_proj: Tensor,
    // Output projection: (n_heads * head_dim, d_model)
    pub o_proj: Tensor,
    // Gate projection: (d_model, n_heads * head_dim) - for output gating
    pub gate_proj: Tensor,
    // Internal normalization for gated output
    pub gate_norm: RMSNorm,
    n_heads: usize,
    n_kv_heads: usize,
    n_groups: usize,  // n_heads / n_kv_heads
    head_dim: usize,
    d_model: usize,
    rope: RoPE,
    device: Device,
}

impl GatedAttention {
    pub fn new(cfg: &Config, device: &Device) -> Result<Self> {
        let d_model = cfg.d_model;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.kv_heads;
        let head_dim = d_model / n_heads;
        let n_groups = n_heads / n_kv_heads;

        // GELU activation -> use Kaiming init
        let std = (2.0 / d_model as f64).sqrt();

        // Q: (d_model, n_heads * head_dim) = (d_model, d_model)
        let q_proj = Tensor::randn(0.0, std, (d_model, d_model), device)?
            .to_dtype(DType::F32)?;

        // K: (d_model, n_kv_heads * head_dim)
        let k_proj = Tensor::randn(0.0, std, (d_model, n_kv_heads * head_dim), device)?
            .to_dtype(DType::F32)?;

        // V: (d_model, n_kv_heads * head_dim)
        let v_proj = Tensor::randn(0.0, std, (d_model, n_kv_heads * head_dim), device)?
            .to_dtype(DType::F32)?;

        // O: (n_heads * head_dim, d_model) = (d_model, d_model)
        let o_proj = Tensor::randn(0.0, std, (d_model, d_model), device)?
            .to_dtype(DType::F32)?;

        // Gate: (d_model, d_model) - output gate mechanism
        // Following Gated Attention paper [2], the gate modulates the attention output
        let gate_proj = Tensor::randn(0.0, std, (d_model, d_model), device)?
            .to_dtype(DType::F32)?;

        // Internal normalization for gated output
        let gate_norm = RMSNorm::from_device(d_model, device)?;

        // RoPE: applied to first 25% of head_dim
        let rope = RoPE::new(head_dim, cfg.max_seq_len, 0.25);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            gate_norm,
            n_heads,
            n_kv_heads,
            n_groups,
            head_dim,
            d_model,
            rope,
            device: device.clone(),
        })
    }

    /// Forward pass with GQA and output gating (per Gated Attention paper [2])
    ///
    /// Input shape: (batch_size, seq_len, d_model)
    /// Output shape: (batch_size, seq_len, d_model)
    ///
    /// Gated Attention formula (per paper [2]):
    ///   1. Compute standard attention: attn = softmax(Q @ K^T / sqrt(d)) @ V @ O
    ///   2. Compute gate: gate = sigmoid(gate_proj(x))
    ///   3. Apply normalization to attention output
    ///   4. Modulate: output = norm(attn) * gate
    ///
    /// This eliminates attention sink and massive activation issues.
    pub fn forward(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (batch_size, seq_len, d_model) = x.dims3()?;

        // Reshape to 2D for projection
        let x_2d = x.reshape(&[batch_size * seq_len, d_model])?;

        // Project to Q, K, V
        let q = x_2d.matmul(&self.q_proj)?.reshape(&[batch_size, seq_len, self.n_heads, self.head_dim])?;
        let k = x_2d.matmul(&self.k_proj)?.reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = x_2d.matmul(&self.v_proj)?.reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?;

        // Apply RoPE to Q and K
        let (q, k) = self.rope.forward(&q, &k, offset)?;

        // GQA: repeat K/V for each head in a group
        let k = self.repeat_kv(k, self.n_groups)?;  // (B, L, n_heads, head_dim)
        let v = self.repeat_kv(v, self.n_groups)?;  // (B, L, n_heads, head_dim)

        // Scaled dot-product attention (includes O projection)
        let attn_out = self.scaled_dot_product_attention(&q, &k, &v)?;

        // Compute output gate (per Gated Attention paper [2])
        // gate = sigmoid(gate_proj(x))
        let gate = x_2d.matmul(&self.gate_proj)?
            .reshape(&[batch_size, seq_len, d_model])?;
        let gate = ops::sigmoid(&gate)?;

        // Normalize attention output (helps with stability)
        let attn_norm = self.gate_norm.forward(&attn_out)?;

        // Apply gating: norm(attn) * gate (element-wise)
        // This is the key "Gated Attention" mechanism from paper [2]
        let gated_out = attn_norm.broadcast_mul(&gate)?;

        Ok(gated_out)
    }

    /// Scaled dot-product attention with causal mask
    ///
    /// Simplified implementation that processes each head separately
    /// Returns attention output BEFORE final projection (gating is applied first)
    fn scaled_dot_product_attention(
        &self,
        q: &Tensor,  // (B, L, n_heads, head_dim)
        k: &Tensor,  // (B, L, n_heads, head_dim)
        v: &Tensor,  // (B, L, n_heads, head_dim)
    ) -> Result<Tensor> {
        let (b, l, n_heads, head_dim) = q.dims4()?;

        // Process each batch and head separately
        let mut all_outputs = Vec::with_capacity(b);

        for batch_idx in 0..b {
            let mut batch_outputs = Vec::with_capacity(n_heads);

            for head_idx in 0..n_heads {
                // Get q, k, v for this batch and head
                let q_bh = q.narrow(0, batch_idx, 1)?.narrow(1, 0, l)?.narrow(2, head_idx, 1)?;
                let q_bh = q_bh.reshape(&[l, head_dim])?;  // (L, head_dim)

                let k_bh = k.narrow(0, batch_idx, 1)?.narrow(1, 0, l)?.narrow(2, head_idx, 1)?;
                let k_bh = k_bh.reshape(&[l, head_dim])?;  // (L, head_dim)

                let v_bh = v.narrow(0, batch_idx, 1)?.narrow(1, 0, l)?.narrow(2, head_idx, 1)?;
                let v_bh = v_bh.reshape(&[l, head_dim])?;  // (L, head_dim)

                // Compute attention scores: q @ k^T
                let scores = q_bh.matmul(&k_bh.t()?)?;  // (L, L)

                // Scale
                let scale = (head_dim as f32).sqrt();
                let scale_tensor = Tensor::new(scale, scores.device())?;
                let scores = scores.broadcast_div(&scale_tensor)?;

                // Apply causal mask
                let scores = self.apply_causal_mask_2d(&scores, l)?;

                // Softmax
                let attn_weights = ops::softmax(&scores, D::Minus1)?;  // (L, L)

                // Apply to values: attn @ v
                let head_out = attn_weights.matmul(&v_bh)?;  // (L, head_dim)

                batch_outputs.push(head_out);
            }

            // Stack heads: (n_heads, L, head_dim) -> (L, n_heads, head_dim)
            let stacked = Tensor::cat(&batch_outputs, 0)?;
            let stacked = stacked.reshape(&[n_heads, l, head_dim])?;
            let stacked = stacked.transpose(0, 1)?;  // (L, n_heads, head_dim)
            all_outputs.push(stacked);
        }

        // Stack batches: (B, L, n_heads, head_dim)
        let stacked = Tensor::cat(&all_outputs, 0)?;
        let output = stacked.reshape(&[b, l, n_heads, head_dim])?;

        // Merge heads: (B, L, n_heads * head_dim)
        let out_2d = output.reshape(&[b * l, n_heads * head_dim])?;
        let projected = out_2d.matmul(&self.o_proj)?.reshape(&[b, l, self.d_model])?;

        Ok(projected)
    }

    /// Apply causal mask to 2D scores tensor
    fn apply_causal_mask_2d(&self, scores: &Tensor, seq_len: usize) -> Result<Tensor> {
        let mut mask = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }

        let mask_tensor = Tensor::new(mask.as_slice(), scores.device())?
            .reshape(&[seq_len, seq_len])?;

        scores.broadcast_add(&mask_tensor)
    }

    /// Repeat K/V for GQA
    ///
    /// Input: (B, L, n_kv_heads, head_dim)
    /// Output: (B, L, n_heads, head_dim)
    fn repeat_kv(&self, x: Tensor, n_groups: usize) -> Result<Tensor> {
        let (b, l, n_kv_heads, head_dim) = x.dims4()?;

        if n_groups == 1 {
            return Ok(x);
        }

        // Repeat each KV head n_groups times
        let mut repeated = Vec::with_capacity(n_kv_heads * n_groups);
        for i in 0..n_kv_heads {
            let head = x.narrow(2, i, 1)?;  // (B, L, 1, head_dim)
            for _ in 0..n_groups {
                repeated.push(head.clone());
            }
        }

        let cat_result = Tensor::cat(&repeated, 2)?;  // (B, L, n_kv_heads * n_groups, head_dim)
        Ok(cat_result)
    }
}
