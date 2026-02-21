use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::attention::GatedAttention;
use crate::cache::Qwen3NextDynamicCache;
use crate::deltanet::GatedDeltaNet;
use crate::ffn::FFN;
use crate::mtp::{MTPConfig, MultiTokenPrediction};
use crate::rms_norm::RMSNorm;
use crate::trainable::make_trainable;
use crate::Config;

/// A single hybrid layer in Qwen3-Next
///
/// Each layer uses either Gated DeltaNet or Gated Attention, followed by standard FFN.
/// The ratio is approximately 3:1 DeltaNet:Attention.
#[derive(Clone)]
pub struct HybridLayer {
    norm1: RMSNorm,
    norm2: RMSNorm,
    deltanet: Option<GatedDeltaNet>,
    attention: Option<GatedAttention>,
    ffn: FFN,
    use_deltanet: bool,
}

impl HybridLayer {
    pub fn new(device: &Device, cfg: &Config, use_deltanet: bool) -> Result<Self> {
        let d_model = cfg.d_model;

        // RMSNorm weights
        let norm1 = RMSNorm::from_device_with_eps(d_model, device, cfg.rms_norm_eps)?;
        let norm2 = RMSNorm::from_device_with_eps(d_model, device, cfg.rms_norm_eps)?;

        let deltanet = if use_deltanet {
            Some(GatedDeltaNet::new(cfg, device)?)
        } else {
            None
        };

        // Gated Attention (only for non-DeltaNet layers)
        let attention = if !use_deltanet {
            Some(GatedAttention::new(cfg, device)?)
        } else {
            None
        };

        let ffn = FFN::new(cfg, device)?;

        Ok(Self {
            norm1,
            norm2,
            deltanet,
            attention,
            ffn,
            use_deltanet,
        })
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
        // Pre-norm: x + Layer(Norm(x))
        let normed = self.norm1.forward(x)?;

        // DeltaNet or Gated Attention
        let attn_out = if self.use_deltanet {
            self.deltanet.as_ref().unwrap().forward_with_cache(
                &normed,
                cache,
                layer_idx,
                attention_mask,
            )?
        } else {
            self.attention.as_ref().unwrap().forward_with_cache(
                &normed,
                offset,
                cache,
                layer_idx,
                attention_mask,
            )?
        };

        let x = (x + attn_out)?;

        // Standard FFN
        let normed = self.norm2.forward(&x)?;
        let ffn_out = self.ffn.forward(&normed)?;
        x + ffn_out
    }
}

/// Mini Qwen3-Next model
///
/// Architecture:
/// - Token Embedding
/// - n_layers HybridLayer (~3:1 ratio of DeltaNet:Attention)
/// - Final RMSNorm (Zero-Centered)
/// - LM Head (tied with embedding weights for efficiency)
/// - Optional: Multi-Token Prediction (MTP) module for faster inference
#[derive(Clone)]
pub struct MiniQwenNext {
    embed: Tensor, // Manual embedding tensor
    layers: Vec<HybridLayer>,
    norm_f: RMSNorm,
    lm_head: Tensor,
    mtp: Option<MultiTokenPrediction>, // Optional MTP module
    config: Config,
    device: Device,
    vocab_size: usize,
}

impl MiniQwenNext {
    pub fn new(_vb: &VarBuilder, device: &Device, cfg: Config) -> Result<Self> {
        cfg.validate();

        let embeddings = Tensor::randn(
            0f32,
            cfg.initializer_range as f32,
            (cfg.vocab_size, cfg.d_model),
            device,
        )?
        .to_dtype(DType::F32)?;
        let embeddings = make_trainable(embeddings)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);

        for layer_type in &cfg.layer_types {
            let use_deltanet = layer_type == "linear_attention";
            layers.push(HybridLayer::new(device, &cfg, use_deltanet)?);
        }

        let norm_f = RMSNorm::from_device_with_eps(cfg.d_model, device, cfg.rms_norm_eps)?;

        // lm_head: (vocab_size, d_model) - will transpose during forward
        // Tie with embedding weights for efficiency
        let lm_head = embeddings.clone();

        // MTP is disabled by default - requires special training
        let mtp = None;

        Ok(Self {
            embed: embeddings,
            layers,
            norm_f,
            lm_head,
            mtp,
            config: cfg.clone(),
            device: device.clone(),
            vocab_size: cfg.vocab_size,
        })
    }

    /// Create model with Multi-Token Prediction enabled
    ///
    /// This is a separate constructor for models trained with MTP.
    /// MTP requires specific training data and loss computation.
    pub fn with_mtp(
        _vb: &VarBuilder,
        device: &Device,
        cfg: Config,
        mtp_config: MTPConfig,
    ) -> Result<Self> {
        cfg.validate();

        let embeddings = Tensor::randn(
            0f32,
            cfg.initializer_range as f32,
            (cfg.vocab_size, cfg.d_model),
            device,
        )?
        .to_dtype(DType::F32)?;
        let embeddings = make_trainable(embeddings)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);

        for layer_type in &cfg.layer_types {
            let use_deltanet = layer_type == "linear_attention";
            layers.push(HybridLayer::new(device, &cfg, use_deltanet)?);
        }

        let norm_f = RMSNorm::from_device_with_eps(cfg.d_model, device, cfg.rms_norm_eps)?;
        let lm_head = embeddings.clone();

        // Create MTP module if enabled
        let mtp = if mtp_config.enabled {
            Some(MultiTokenPrediction::new(&cfg, mtp_config, device)?)
        } else {
            None
        };

        Ok(Self {
            embed: embeddings,
            layers,
            norm_f,
            lm_head,
            mtp,
            config: cfg.clone(),
            device: device.clone(),
            vocab_size: cfg.vocab_size,
        })
    }

    /// Forward pass
    ///
    /// Input shape: (batch_size, seq_len)
    /// Output shape: (batch_size, seq_len, vocab_size)
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_with_attention_mask(input_ids, None)
    }

    /// Forward pass with optional 2D attention mask `(B, L)`.
    pub fn forward_with_attention_mask(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Embed tokens: (batch, seq) -> (batch, seq, d_model)
        let mut x = self.embed_forward(input_ids)?;

        // Pass through layers
        for layer in &self.layers {
            x = layer.forward_with_cache(&x, 0, None, 0, attention_mask)?;
        }

        // Final norm
        let x = self.norm_f.forward(&x)?;

        // LM Head projection
        let (b, l, d) = x.dims3()?;
        let x_2d = x.reshape(&[b * l, d])?;
        // lm_head: (vocab, d_model), need to transpose
        let lm_head_t = self.lm_head.t()?;
        x_2d.matmul(&lm_head_t)?.reshape(&[b, l, self.vocab_size])
    }

    /// Forward pass with dynamic cache for autoregressive decoding.
    ///
    /// `input_ids` can be either a prompt chunk (L > 1) or single-step token (L = 1).
    pub fn forward_with_cache(
        &self,
        input_ids: &Tensor,
        cache: &mut Qwen3NextDynamicCache,
    ) -> Result<Tensor> {
        let (_, _l) = input_ids.dims2()?;
        let offset = cache.seen_tokens();

        let mut x = self.embed_forward(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_with_cache(&x, offset, Some(cache), layer_idx, None)?;
        }

        let x = self.norm_f.forward(&x)?;
        let (b, l, d) = x.dims3()?;
        let x_2d = x.reshape(&[b * l, d])?;
        let lm_head_t = self.lm_head.t()?;
        let logits = x_2d.matmul(&lm_head_t)?.reshape(&[b, l, self.vocab_size])?;

        cache.advance(l);
        Ok(logits)
    }

    /// Create an empty dynamic cache for decoding.
    pub fn create_cache(&self) -> Qwen3NextDynamicCache {
        Qwen3NextDynamicCache::new(&self.config)
    }

    /// Embedding lookup
    fn embed_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, l) = input_ids.dims2()?;

        // Flatten and gather embeddings
        let input_ids_flat = input_ids.reshape(&[b * l])?;
        let embeddings = self.embed.index_select(&input_ids_flat, 0)?;
        embeddings.reshape(&[b, l, self.config.d_model])
    }

    /// Get configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Set training/eval mode for submodules that have mode-dependent behavior.
    ///
    /// Currently this controls attention dropout.
    pub fn set_training(&self, training: bool) {
        for layer in &self.layers {
            if let Some(ref attn) = layer.attention {
                attn.set_training(training);
            }
        }
    }

    /// Count total parameters
    ///
    /// Returns an accurate parameter count matching the actual model architecture.
    pub fn count_params(&self) -> Result<usize> {
        let d = self.config.d_model;
        let vocab = self.config.vocab_size;
        let layers = self.config.n_layers;
        let intermediate = self.config.intermediate_size;
        let kv_heads = self.config.kv_heads;
        let kernel_size = self.config.linear_conv_kernel_dim;
        let head_dim = self.config.head_dim;

        // Embedding: vocab * d_model
        let embed = vocab * d;

        // Count DeltaNet vs Attention layers
        let deltanet_layers = self.config.n_linear_layers();
        let attn_layers = self.config.n_full_attention_layers();

        // Official-style linear-attention dimensions from config.
        let linear_k_heads = self.config.linear_num_key_heads;
        let linear_v_heads = self.config.linear_num_value_heads;
        let head_k_dim = self.config.linear_key_head_dim;
        let head_v_dim = self.config.linear_value_head_dim;
        let key_dim = linear_k_heads * head_k_dim;
        let value_dim = linear_v_heads * head_v_dim;
        let conv_dim = 2 * key_dim + value_dim;

        // DeltaNet params per layer.
        let deltanet_in_proj_qkvz = d * (2 * key_dim + 2 * value_dim);
        let deltanet_in_proj_ba = d * (2 * linear_v_heads);
        let deltanet_discretization = 2 * linear_v_heads; // dt_bias + a_log
        let deltanet_out_proj = value_dim * d;
        let deltanet_conv = conv_dim * kernel_size;
        let deltanet_norm = head_v_dim;
        let deltanet_params_per_layer = deltanet_in_proj_qkvz
            + deltanet_in_proj_ba
            + deltanet_discretization
            + deltanet_out_proj
            + deltanet_conv
            + deltanet_norm;
        let deltanet_params = deltanet_layers * deltanet_params_per_layer;

        // GatedAttention params per layer.
        let attn_hidden = self.config.attention_hidden_size();
        let kv_dim = kv_heads * head_dim;
        let attn_q_proj = d * (2 * attn_hidden);
        let attn_kv_proj = 2 * d * kv_dim;
        let attn_o_proj = attn_hidden * d;
        let attn_qk_norm = 2 * head_dim;
        let attn_bias = if self.config.attention_bias {
            2 * attn_hidden + 2 * kv_dim + d
        } else {
            0
        };
        let attn_params =
            attn_layers * (attn_q_proj + attn_kv_proj + attn_o_proj + attn_qk_norm + attn_bias);

        // SwiGLU FFN params per layer: 3 * d_model * intermediate_size
        // (w_gate, w_up, w_down - each is d_model x intermediate_size)
        let ffn_params = layers * 3 * d * intermediate;

        // Layer norms: 2 per layer + 1 final = 2*layers + 1, each is d
        let norm_params = (2 * layers + 1) * d;

        let total = embed + deltanet_params + attn_params + ffn_params + norm_params;

        Ok(total)
    }

    /// Get all trainable tensors (for optimizer)
    pub fn get_tensors(&self) -> Vec<Tensor> {
        let mut tensors = Vec::new();

        // Embedding
        tensors.push(self.embed.clone());

        // Layers
        for layer in &self.layers {
            // RMSNorm weights
            tensors.push(layer.norm1.weight.clone());
            tensors.push(layer.norm2.weight.clone());

            // DeltaNet or Attention
            if let Some(ref deltanet) = layer.deltanet {
                tensors.push(deltanet.in_proj_qkvz.clone());
                tensors.push(deltanet.in_proj_ba.clone());
                tensors.push(deltanet.dt_bias.clone());
                tensors.push(deltanet.a_log.clone());
                tensors.push(deltanet.out_proj.clone());
                tensors.push(deltanet.get_conv_weight());
                tensors.push(deltanet.get_norm_weight());
            }

            if let Some(ref attn) = layer.attention {
                tensors.push(attn.q_proj.clone());
                tensors.push(attn.k_proj.clone());
                tensors.push(attn.v_proj.clone());
                tensors.push(attn.o_proj.clone());
                if let Some(ref q_bias) = attn.q_bias {
                    tensors.push(q_bias.clone());
                }
                if let Some(ref k_bias) = attn.k_bias {
                    tensors.push(k_bias.clone());
                }
                if let Some(ref v_bias) = attn.v_bias {
                    tensors.push(v_bias.clone());
                }
                if let Some(ref o_bias) = attn.o_bias {
                    tensors.push(o_bias.clone());
                }
                tensors.push(attn.q_norm.weight.clone());
                tensors.push(attn.k_norm.weight.clone());
            }

            // FFN (SwiGLU: w_gate, w_up, w_down)
            tensors.push(layer.ffn.w_gate.clone());
            tensors.push(layer.ffn.w_up.clone());
            tensors.push(layer.ffn.w_down.clone());
        }

        // Final norm
        tensors.push(self.norm_f.weight.clone());

        // LM head is tied with embedding, don't count twice

        // Note: MTP parameters would be added here when fully implemented

        tensors
    }

    /// Check if MTP is enabled for this model
    pub fn has_mtp(&self) -> bool {
        self.mtp.as_ref().map(|m| m.is_enabled()).unwrap_or(false)
    }

    /// Get MTP module reference (if available)
    pub fn mtp(&self) -> Option<&MultiTokenPrediction> {
        self.mtp.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor, D};

    #[test]
    fn test_forward_with_cache_matches_full_forward() -> Result<()> {
        let device = Device::Cpu;
        let cfg = Config::tiny_5m();
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MiniQwenNext::new(&vb, &device, cfg)?;

        let input_ids = Tensor::new([1u32, 2, 3, 4].as_slice(), &device)?.reshape(&[1, 4])?;
        let full_logits = model.forward(&input_ids)?;

        let mut cache = model.create_cache();
        let mut step_logits = Vec::new();
        for t in 0..4 {
            let token_step = input_ids.narrow(1, t, 1)?;
            let logits = model.forward_with_cache(&token_step, &mut cache)?;
            step_logits.push(logits);
        }
        let step_refs: Vec<&Tensor> = step_logits.iter().collect();
        let cached_logits = Tensor::cat(step_refs.as_slice(), 1)?;

        let diff = full_logits
            .broadcast_sub(&cached_logits)?
            .abs()?
            .reshape(&[4 * model.config.vocab_size])?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-4, "cache and full forward mismatch: {}", diff);
        Ok(())
    }
}
