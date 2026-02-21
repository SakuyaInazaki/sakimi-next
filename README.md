# Sakimi-Next

A minimal Rust implementation of Qwen3-Next architecture with ~10M parameters.

## Overview

Sakimi-Next is a compact implementation of the [Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd) architecture, featuring:

- **Hybrid 3:1 DeltaNet:Attention layers** - Core innovation from Qwen3-Next
- **Gated DeltaNet (official-style recurrence)** - `beta/g/A_log/dt_bias` path with short convolution
- **Fast kernel path (Rust/Candle)** - grouped `conv1d` + vectorized recurrent updates
- **Gated Attention** - official `q_proj -> (query, gate)` path
- **Qwen3-Next/Gemma3-style RMSNorm** - `norm(x) * (1 + weight)` for decoder norms
- **RoPE (25%)** - Rotary Position Embedding on first 25% of dimensions only
- **SwiGLU FFN** - Modern activation function
- **GQA** - Grouped Query Attention for efficiency

## Architecture

```
Input → Embedding → [HybridLayer × n_layers] → FinalNorm → LMHead → Output

Each HybridLayer:
  x = x + DeltaNetOrAttention(Norm1(x))
  x = x + FFN(Norm2(x))

Layer ratio (3:1):
  - 3 × Gated DeltaNet layers
  - 1 × Gated Attention layer
```

## Configuration Presets

| Config | Parameters | vocab | layers | d_model | heads | Description |
|--------|------------|-------|--------|---------|-------|-------------|
| `tiny_10m` | ~10M | 4K | 8 | 256 | 4/1 | Recommended |
| `tiny_5m` | ~5M | 2K | 4 | 128 | 4/1 | Ultra-small |
| `small_50m` | ~50M | 50K | 12 | 512 | 8/2 | Small model |

## Installation

```bash
# Clone the repository
git clone https://github.com/SakuyaInazaki/sakimi-next.git
cd sakimi-next

# Build
cargo build --release
```

## Usage

```bash
# Run test forward pass
cargo run --release -- --config tiny_10m --mode test

# Prepare training data
cargo run --release -- --config tiny_10m --mode prepare_data --train-data path/to/train.txt

# Train
cargo run --release -- --config tiny_10m --mode train --steps 10000 --learning-rate 1e-4

# Generate (TODO)
cargo run --release -- --config tiny_10m --mode generate
```

## Tencent Cloud GPU

If you want to train on Tencent Cloud GPU CVM, use:

```bash
bash scripts/train_tencent_gpu.sh
```

Quickstart details:

- `docs/tencent-cloud-quickstart.md`
- `scripts/train_nvidia_gpu.sh` (common NVIDIA cloud training entry)
- `scripts/switch_candle_backend.sh` (switch `cuda` / `metal` / `cpu`)

## Implementation Details

### Gated DeltaNet

Implements the core Qwen3-Next gated-delta path in a small-model setting:

- Joint `q/k/v/z` projection + `b/a` projection
- ShortConvolution on concatenated QKV (kernel_size=4)
- `g = -exp(A_log) * softplus(a + dt_bias)` discretization path
- Recurrent gated-delta update used in the official implementation
- RMSNormGated (`RMSNorm(x) * silu(z)`) before output projection

### Stability Features

From Qwen3-Next paper:

- **Qwen3-Next/Gemma3-style RMSNorm** for decoder norms
- **Q/K RMSNorm + gated attention output path**
- **25% RoPE** for attention heads

### Scope

This repository targets compact models and does not yet include:

- MoE routing/expert layers from the full Qwen3-Next release
- External CUDA kernel stack (`flash-linear-attention`, `causal-conv1d`) from the official ecosystem
- Full MTP training/inference pipeline

## References

- [Qwen3-Next Blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd)
- [Gated Delta Networks (NVlabs)](https://github.com/NVlabs/GatedDeltaNet)
- [Gemma Team (Zero-Centered RMSNorm)](https://github.com/google/gemma_pytorch)

## License

MIT

## Author

SakuyaInazaki
