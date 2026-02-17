# Sakimi-Next

A minimal Rust implementation of Qwen3-Next architecture with ~10M parameters.

## Overview

Sakimi-Next is a compact implementation of the [Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd) architecture, featuring:

- **Hybrid 3:1 DeltaNet:Attention layers** - Core innovation from Qwen3-Next
- **Gated DeltaNet** - Full NVlabs specification with all gates (B, A, Lambda, G)
- **Gated Attention** - Output gating mechanism to reduce attention sink
- **Zero-Centered RMSNorm** - Gemma-style normalization for stability
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

## Implementation Details

### Gated DeltaNet

Implements the full specification from [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet):

- Q, K, V projections with ShortConvolution (kernel_size=4)
- Lambda (decay): learnable time-variant decay
- A (alpha): independent forget gate
- B (gate): output gating
- G (gate): output gate projection
- Numerically stable log-space scan algorithm
- Head-to-state learnable projection

### Stability Features

From Qwen3-Next paper:

- **Zero-Centered RMSNorm**: Prevents norm weights from growing unbounded
- **Weight decay on norms**: Unlike QK-Norm approach, all parameters get weight decay
- **Output gating**: Eliminates attention sink and massive activation issues
- **25% RoPE**: Improves extrapolation to longer sequences

## References

- [Qwen3-Next Blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd)
- [Gated Delta Networks (NVlabs)](https://github.com/NVlabs/GatedDeltaNet)
- [Gemma Team (Zero-Centered RMSNorm)](https://github.com/google/gemma_pytorch)

## License

MIT

## Author

SakuyaInazaki
