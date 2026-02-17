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

    /// DeltaNet state dimension
    pub d_state: usize,

    /// Intermediate size for FFN (usually 4 * d_model)
    pub intermediate_size: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl Config {
    /// Tiny ~10M parameters model (REDESIGNED with proper head_dim)
    ///
    /// Redesigned following Qwen3-Next architecture:
    /// - head_dim = 64 (increased from 32 for better representation)
    /// - Uses GQA with 4:1 ratio for efficiency
    /// - 3:1 DeltaNet:Attention hybrid architecture
    ///
    /// Parameter breakdown:
    /// - Embedding: 4096 × 256 = 1.05M
    /// - LM Head: tied with embedding
    /// - Layers (6):
    ///   - 4 DeltaNet + 2 Attention layers
    ///   - FFN per layer: 2 × (256 × 768) = 0.39M × 6 = 2.34M
    ///   - DeltaNet/Attention per layer: ~1.1M each = 6.6M
    /// Total: ~10M parameters
    ///
    /// Parameter distribution: ~10% Embedding, ~90% Model Body
    pub fn tiny_10m() -> Self {
        Self {
            vocab_size: 4096,      // Reduced from 32K to save params!
            n_layers: 8,           // 8 layers: 6 DeltaNet + 2 Attention (exactly 3:1 ratio)
            d_model: 256,          // Hidden dimension
            n_heads: 4,            // head_dim = 256/4 = 64 (improved from 32!)
            kv_heads: 1,           // GQA 4:1 (MQA - all Q share one K/V)
            d_state: 64,
            intermediate_size: 768,   // 3 * d_model (reduced to fit budget)
            max_seq_len: 2048,
        }
    }

    /// Tiny ~10M parameters model (OLD - for reference)
    ///
    /// Old configuration with 32K vocabulary - NOT RECOMMENDED
    /// 89% of parameters are in embedding, leaving no capacity for learning.
    /// Use tiny_10m() instead.
    #[deprecated(note = "Use tiny_10m() instead - this has 89% params in embedding")]
    pub fn tiny_10m_old() -> Self {
        Self {
            vocab_size: 32000,
            n_layers: 4,
            d_model: 128,
            n_heads: 4,
            kv_heads: 1,
            d_state: 64,
            intermediate_size: 512,
            max_seq_len: 2048,
        }
    }

    /// Experimental ~5M parameters model (ultra-small)
    ///
    /// For testing and experiments only:
    /// - vocab_size: 2048
    /// - 4 layers, d_model=128
    /// Total: ~5M parameters
    pub fn tiny_5m() -> Self {
        Self {
            vocab_size: 2048,
            n_layers: 4,           // 4 layers: 3 DeltaNet + 1 Attention (exactly 3:1 ratio)
            d_model: 128,
            n_heads: 4,
            kv_heads: 1,
            d_state: 32,
            intermediate_size: 512,
            max_seq_len: 1024,
        }
    }

    /// Small ~50M parameters model (no MoE)
    ///
    /// head_dim = 512/8 = 64
    pub fn small_50m() -> Self {
        Self {
            vocab_size: 50257,
            n_layers: 12,         // 3 groups of 4 (9 DeltaNet + 3 Attention)
            d_model: 512,
            n_heads: 8,           // head_dim = 64
            kv_heads: 2,          // GQA 8:2 = 4:1
            d_state: 64,
            intermediate_size: 2048,  // 4 * d_model
            max_seq_len: 2048,
        }
    }

    /// Medium ~100M parameters model
    ///
    /// head_dim = 128 (closer to Qwen3-Next's 256)
    pub fn medium_100m() -> Self {
        Self {
            vocab_size: 50257,
            n_layers: 16,         // 4 groups of 4 (12 DeltaNet + 4 Attention)
            d_model: 768,
            n_heads: 6,           // head_dim = 128
            kv_heads: 2,          // GQA 6:2 = 3:1
            d_state: 64,
            intermediate_size: 3072,  // 4 * d_model
            max_seq_len: 4096,
        }
    }

    /// Validate configuration
    pub fn validate(&self) {
        // n_layers must be multiple of 4 to maintain exact 3:1 DeltaNet:Attention ratio
        // Each group of 4 layers has: 3 DeltaNet + 1 Attention
        assert!(self.n_layers % 4 == 0, "n_layers must be multiple of 4 for proper 3:1 DeltaNet:Attention ratio");
        assert!(self.n_heads % self.kv_heads == 0, "n_heads must be divisible by kv_heads");
        assert!(self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads");

        // head_dim should be at least 64 for good representation (Qwen3-Next uses 256)
        let head_dim = self.head_dim();
        assert!(head_dim >= 32, "head_dim ({}) should be at least 32", head_dim);
    }

    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Number of layer groups (each group has 4 layers: 3 DeltaNet + 1 Attention)
    pub fn n_groups(&self) -> usize {
        self.n_layers / 4
    }
}
