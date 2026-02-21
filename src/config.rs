use serde::{Deserialize, Serialize};

/// Configuration for Mini Qwen-Next model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Vocabulary size
    pub vocab_size: usize,

    /// Number of layers (must be multiple of 4 for 3:1 DeltaNet:Attention ratio)
    pub n_layers: usize,

    /// Hidden dimension
    pub d_model: usize,

    /// Number of attention heads (for Attention layers)
    pub n_heads: usize,

    /// Number of KV heads (for GQA)
    pub kv_heads: usize,

    /// Attention head dimension (official-style explicit field).
    pub head_dim: usize,

    /// Intermediate size for FFN (usually 4 * d_model)
    pub intermediate_size: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Hidden activation function (official default: silu).
    pub hidden_act: String,

    /// Initialization std for weight matrices.
    pub initializer_range: f64,

    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,

    /// Whether to use bias in attention projections.
    pub attention_bias: bool,

    /// Attention dropout probability.
    pub attention_dropout: f64,

    /// RoPE base theta.
    pub rope_theta: f64,

    /// Fraction of head_dim for rotary embedding.
    pub partial_rotary_factor: f64,

    /// Layer types per decoder layer ("linear_attention" or "full_attention")
    pub layer_types: Vec<String>,

    /// Linear attention: conv kernel size
    pub linear_conv_kernel_dim: usize,

    /// Linear attention: key head dim
    pub linear_key_head_dim: usize,

    /// Linear attention: value head dim
    pub linear_value_head_dim: usize,

    /// Linear attention: number of key heads
    pub linear_num_key_heads: usize,

    /// Linear attention: number of value heads
    pub linear_num_value_heads: usize,

    /// Enable optimized kernel paths for DeltaNet/conv when available in this runtime.
    #[serde(default = "default_use_fast_kernels")]
    pub use_fast_kernels: bool,
}

fn default_use_fast_kernels() -> bool {
    true
}

impl Config {
    fn default_layer_types(n_layers: usize, full_attention_interval: usize) -> Vec<String> {
        (0..n_layers)
            .map(|i| {
                if (i + 1) % full_attention_interval == 0 {
                    "full_attention".to_string()
                } else {
                    "linear_attention".to_string()
                }
            })
            .collect()
    }

    /// Tiny ~10M parameters model (REDESIGNED with proper head_dim)
    ///
    /// Redesigned following Qwen3-Next architecture:
    /// - head_dim = 64 (increased from 32 for better representation)
    /// - Uses GQA with 4:1 ratio for efficiency
    /// - 3:1 DeltaNet:Attention hybrid architecture
    ///
    /// Parameter breakdown:
    /// - Embedding: 8192 × 256 = 2.10M
    /// - LM Head: tied with embedding
    /// - Layers (6):
    ///   - 4 DeltaNet + 2 Attention layers
    ///   - FFN per layer: 2 × (256 × 768) = 0.39M × 6 = 2.34M
    ///   - DeltaNet/Attention per layer: ~1.1M each = 6.6M
    /// Total: ~10M parameters
    ///
    /// Parameter distribution: ~20% Embedding, ~80% Model Body
    pub fn tiny_10m() -> Self {
        let n_layers = 8;
        let d_model = 256;
        let n_heads = 4;
        let kv_heads = 1;
        let head_dim = d_model / n_heads;
        let linear_num_key_heads = n_heads;
        let linear_num_value_heads = 2 * n_heads;
        Self {
            vocab_size: 8192, // Better Chinese token efficiency while keeping ~10M params!
            n_layers,         // 8 layers: 6 DeltaNet + 2 Attention (exactly 3:1 ratio)
            d_model,          // Hidden dimension
            n_heads,          // head_dim = 256/4 = 64 (improved from 32!)
            kv_heads,         // GQA 4:1 (MQA - all Q share one K/V)
            head_dim,
            intermediate_size: 768, // 3 * d_model (reduced to fit budget)
            max_seq_len: 2048,
            hidden_act: "silu".to_string(),
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            attention_dropout: 0.0,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.25,
            layer_types: Self::default_layer_types(n_layers, 4),
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: head_dim,
            linear_value_head_dim: head_dim,
            linear_num_key_heads,
            linear_num_value_heads,
            use_fast_kernels: true,
        }
    }

    /// Tiny ~10M parameters model (OLD - for reference)
    ///
    /// Old configuration with 32K vocabulary - NOT RECOMMENDED
    /// 89% of parameters are in embedding, leaving no capacity for learning.
    /// Use tiny_10m() instead.
    #[deprecated(note = "Use tiny_10m() instead - this has 89% params in embedding")]
    pub fn tiny_10m_old() -> Self {
        let n_layers = 4;
        let d_model = 128;
        let n_heads = 4;
        let kv_heads = 1;
        let head_dim = d_model / n_heads;
        let linear_num_key_heads = n_heads;
        let linear_num_value_heads = 2 * n_heads;
        Self {
            vocab_size: 32000,
            n_layers,
            d_model,
            n_heads,
            kv_heads,
            head_dim,
            intermediate_size: 512,
            max_seq_len: 2048,
            hidden_act: "silu".to_string(),
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            attention_dropout: 0.0,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.25,
            layer_types: Self::default_layer_types(n_layers, 4),
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: head_dim,
            linear_value_head_dim: head_dim,
            linear_num_key_heads,
            linear_num_value_heads,
            use_fast_kernels: true,
        }
    }

    /// Experimental ~5M parameters model (ultra-small)
    ///
    /// For testing and experiments only:
    /// - vocab_size: 2048
    /// - 4 layers, d_model=128
    /// Total: ~5M parameters
    pub fn tiny_5m() -> Self {
        let n_layers = 4;
        let d_model = 128;
        let n_heads = 4;
        let kv_heads = 1;
        let head_dim = d_model / n_heads;
        let linear_num_key_heads = n_heads;
        let linear_num_value_heads = 2 * n_heads;
        Self {
            vocab_size: 8192,
            n_layers, // 4 layers: 3 DeltaNet + 1 Attention (exactly 3:1 ratio)
            d_model,
            n_heads,
            kv_heads,
            head_dim,
            intermediate_size: 512,
            max_seq_len: 1024,
            hidden_act: "silu".to_string(),
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            attention_dropout: 0.0,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.25,
            layer_types: Self::default_layer_types(n_layers, 4),
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: head_dim,
            linear_value_head_dim: head_dim,
            linear_num_key_heads,
            linear_num_value_heads,
            use_fast_kernels: true,
        }
    }

    /// Small ~50M parameters model (no MoE)
    ///
    /// head_dim = 512/8 = 64
    pub fn small_50m() -> Self {
        let n_layers = 12;
        let d_model = 512;
        let n_heads = 8;
        let kv_heads = 2;
        let head_dim = d_model / n_heads;
        let linear_num_key_heads = n_heads;
        let linear_num_value_heads = 2 * n_heads;
        Self {
            vocab_size: 50257,
            n_layers, // 3 groups of 4 (9 DeltaNet + 3 Attention)
            d_model,
            n_heads,  // head_dim = 64
            kv_heads, // GQA 8:2 = 4:1
            head_dim,
            intermediate_size: 2048, // 4 * d_model
            max_seq_len: 2048,
            hidden_act: "silu".to_string(),
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            attention_dropout: 0.0,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.25,
            layer_types: Self::default_layer_types(n_layers, 4),
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: head_dim,
            linear_value_head_dim: head_dim,
            linear_num_key_heads,
            linear_num_value_heads,
            use_fast_kernels: true,
        }
    }

    /// Medium ~100M parameters model
    ///
    /// head_dim = 128 (closer to Qwen3-Next's 256)
    pub fn medium_100m() -> Self {
        let n_layers = 16;
        let d_model = 768;
        let n_heads = 6;
        let kv_heads = 2;
        let head_dim = d_model / n_heads;
        let linear_num_key_heads = n_heads;
        let linear_num_value_heads = 2 * n_heads;
        Self {
            vocab_size: 50257,
            n_layers, // 4 groups of 4 (12 DeltaNet + 4 Attention)
            d_model,
            n_heads,  // head_dim = 128
            kv_heads, // GQA 6:2 = 3:1
            head_dim,
            intermediate_size: 3072, // 4 * d_model
            max_seq_len: 4096,
            hidden_act: "silu".to_string(),
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            attention_dropout: 0.0,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.25,
            layer_types: Self::default_layer_types(n_layers, 4),
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: head_dim,
            linear_value_head_dim: head_dim,
            linear_num_key_heads,
            linear_num_value_heads,
            use_fast_kernels: true,
        }
    }

    /// Validate configuration
    pub fn validate(&self) {
        assert!(
            self.n_heads % self.kv_heads == 0,
            "n_heads must be divisible by kv_heads"
        );
        assert!(self.n_heads > 0 && self.kv_heads > 0, "heads must be > 0");
        let head_dim = self.head_dim;
        assert!(
            head_dim >= 32,
            "head_dim ({}) should be at least 32",
            head_dim
        );
        assert!(self.head_dim > 0, "head_dim must be > 0");
        assert!(
            self.partial_rotary_factor > 0.0 && self.partial_rotary_factor <= 1.0,
            "partial_rotary_factor must be in (0, 1]"
        );
        assert!(self.rope_theta > 0.0, "rope_theta must be > 0");
        assert!(
            (0.0..1.0).contains(&self.attention_dropout),
            "attention_dropout must be in [0, 1)"
        );
        assert!(
            matches!(self.hidden_act.as_str(), "silu" | "swish" | "gelu" | "relu"),
            "unsupported hidden_act: {}",
            self.hidden_act
        );

        assert_eq!(
            self.layer_types.len(),
            self.n_layers,
            "layer_types length must equal n_layers"
        );
        for t in &self.layer_types {
            assert!(
                t == "linear_attention" || t == "full_attention",
                "invalid layer type: {}",
                t
            );
        }

        assert!(
            self.linear_num_value_heads % self.linear_num_key_heads == 0,
            "linear_num_value_heads must be divisible by linear_num_key_heads"
        );
        assert!(
            self.linear_num_key_heads > 0 && self.linear_num_value_heads > 0,
            "linear_num_key_heads and linear_num_value_heads must be > 0"
        );
        assert!(
            self.linear_key_head_dim > 0 && self.linear_value_head_dim > 0,
            "linear_key_head_dim and linear_value_head_dim must be > 0"
        );
        assert!(
            self.linear_conv_kernel_dim > 0,
            "linear_conv_kernel_dim must be > 0"
        );
    }

    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Attention projection width (`num_heads * head_dim`).
    pub fn attention_hidden_size(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Number of linear-attention layers.
    pub fn n_linear_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| t.as_str() == "linear_attention")
            .count()
    }

    /// Number of full-attention layers.
    pub fn n_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| t.as_str() == "full_attention")
            .count()
    }
}
