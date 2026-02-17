use candle_core::{Result, Tensor, Device, DType};
use candle_nn::VarBuilder;

use crate::Config;
use crate::deltanet::GatedDeltaNet;
use crate::ffn::FFN;
use crate::attention::GatedAttention;
use crate::rms_norm::RMSNorm;
use crate::mtp::{MultiTokenPrediction, MTPConfig};

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
    d_model: usize,
}

impl HybridLayer {
    pub fn new(device: &Device, cfg: &Config, use_deltanet: bool) -> Result<Self> {
        let d_model = cfg.d_model;

        // RMSNorm weights
        let norm1 = RMSNorm::from_device(d_model, device)?;
        let norm2 = RMSNorm::from_device(d_model, device)?;

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
            d_model,
        })
    }

    pub fn forward(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        // Pre-norm: x + Layer(Norm(x))
        let normed = self.norm1.forward(x)?;

        // DeltaNet or Gated Attention
        let attn_out = if self.use_deltanet {
            self.deltanet.as_ref().unwrap().forward(&normed)?
        } else {
            self.attention.as_ref().unwrap().forward(&normed, offset)?
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
    embed: Tensor,  // Manual embedding tensor
    layers: Vec<HybridLayer>,
    norm_f: RMSNorm,
    lm_head: Tensor,
    mtp: Option<MultiTokenPrediction>,  // Optional MTP module
    config: Config,
    device: Device,
    vocab_size: usize,
}

impl MiniQwenNext {
    pub fn new(_vb: &VarBuilder, device: &Device, cfg: Config) -> Result<Self> {
        cfg.validate();

        // Initialize embeddings with small std (0.02)
        let embeddings = Tensor::randn(0.0, 0.02, (cfg.vocab_size, cfg.d_model), device)?
            .to_dtype(DType::F32)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);

        for i in 0..cfg.n_layers {
            // ~3:1 ratio - use DeltaNet for most layers
            // Every 4th layer is Gated Attention
            let use_deltanet = (i % 4) != 3;

            layers.push(HybridLayer::new(device, &cfg, use_deltanet)?);
        }

        let norm_f = RMSNorm::from_device(cfg.d_model, device)?;

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

        let embeddings = Tensor::randn(0.0, 0.02, (cfg.vocab_size, cfg.d_model), device)?
            .to_dtype(DType::F32)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);

        for i in 0..cfg.n_layers {
            let use_deltanet = (i % 4) != 3;
            layers.push(HybridLayer::new(device, &cfg, use_deltanet)?);
        }

        let norm_f = RMSNorm::from_device(cfg.d_model, device)?;
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
        let (b, l) = input_ids.dims2()?;

        // Embed tokens: (batch, seq) -> (batch, seq, d_model)
        let mut x = self.embed_forward(input_ids)?;

        // Pass through layers
        for layer in &self.layers {
            x = layer.forward(&x, 0)?;
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

    /// Count total parameters
    ///
    /// Returns an accurate parameter count matching the actual model architecture.
    pub fn count_params(&self) -> Result<usize> {
        let d = self.config.d_model;
        let vocab = self.config.vocab_size;
        let layers = self.config.n_layers;
        let intermediate = self.config.intermediate_size;
        let n_heads = self.config.n_heads;
        let kernel_size = 4;  // ShortConvolution kernel size

        // Embedding: vocab * d_model
        let embed = vocab * d;

        // Count DeltaNet vs Attention layers
        let deltanet_layers = (layers * 3 + 3) / 4;  // ~75%
        let attn_layers = layers - deltanet_layers;

        // DeltaNet params per layer (v_dim = 2*d_model)
        let v_dim = 2 * d;
        let head_dim = d / n_heads;
        let state_dim = v_dim / n_heads;
        // Projections: Q(d*d) + K(d*d) + V(d*v_dim) + G(d*v_dim) + O(v_dim*d)
        let deltanet_projs = 2 * d * d + 2 * d * v_dim + v_dim * d;
        // Head-to-state projection: head_dim * state_dim (LEARNABLE - this was a bug before)
        let deltanet_h2s_proj = head_dim * state_dim;
        // Gates: B + A + Lambda, each is d*n_heads
        let deltanet_gates = 3 * d * n_heads;
        // Conv weights: 3 * d * kernel_size (Q,K) + v_dim * kernel_size (V)
        let deltanet_conv = 2 * d * kernel_size + v_dim * kernel_size;
        // Output norm: weight + gate, each is v_dim
        let deltanet_norm = 2 * v_dim;
        let deltanet_params_per_layer = deltanet_projs + deltanet_h2s_proj + deltanet_gates + deltanet_conv + deltanet_norm;
        let deltanet_params = deltanet_layers * deltanet_params_per_layer;

        // GatedAttention params per layer
        // Q,K,V,O, Gate_proj: each is d*d
        // gate_norm: d
        let attn_params = attn_layers * (5 * d * d + d);

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
                // Main projections
                tensors.push(deltanet.q_proj.clone());
                tensors.push(deltanet.k_proj.clone());
                tensors.push(deltanet.v_proj.clone());
                // Gate projections (including new a_proj and g_proj)
                tensors.push(deltanet.b_proj.clone());
                tensors.push(deltanet.a_proj.clone());
                tensors.push(deltanet.lambda_proj.clone());
                tensors.push(deltanet.g_proj.clone());
                tensors.push(deltanet.o_proj.clone());
                // Head-to-state projection (LEARNABLE - fixes the bug)
                tensors.push(deltanet.head_to_state_proj.clone());

                // ShortConvolution weights
                tensors.push(deltanet.get_q_conv_weight());
                tensors.push(deltanet.get_k_conv_weight());
                tensors.push(deltanet.get_v_conv_weight());

                // Output normalization (FusedRMSNormSwishGate)
                tensors.push(deltanet.get_o_norm_weight());
                tensors.push(deltanet.get_o_norm_gate());
            }

            if let Some(ref attn) = layer.attention {
                tensors.push(attn.q_proj.clone());
                tensors.push(attn.k_proj.clone());
                tensors.push(attn.v_proj.clone());
                tensors.push(attn.o_proj.clone());
                tensors.push(attn.gate_proj.clone());
                tensors.push(attn.gate_norm.weight.clone());
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
