# Sakimi-Next

Minimal Rust implementation of a Qwen3-Next-style tiny model.

## Current Target (Active)

This repository is currently aligned to **one main training target**:

- Config: `tiny_10m`
- Parameters: ~10.5M
- Vocab size: 8192
- Layers: 8 (6 DeltaNet + 2 Attention, 3:1 pattern)
- Hidden size: 256
- Heads: 4 (KV heads: 1)
- FFN size: 768
- RoPE partial rotary factor: 0.25
- Fast kernels: enabled by default

## Implemented Components

- Hybrid 3:1 DeltaNet:Attention stack
- Gated DeltaNet recurrent path (`A_log/dt_bias` + short convolution)
- Gated Attention path (`q_proj -> query + gate`)
- Qwen3-Next/Gemma3-style decoder RMSNorm (`norm(x) * (1 + weight)`)
- Q/K RMSNorm and RoPE on partial head dimensions
- SwiGLU FFN
- GQA-style KV sharing
- Training / resume / checkpoint save
- Local generation from checkpoint

## Build

```bash
git clone https://github.com/SakuyaInazaki/sakimi-next.git
cd sakimi-next
cargo build --release
```

## Basic Workflow

### 1) Prepare tokenizer + tokenized data

```bash
cargo run --release -- \
  --config tiny_10m \
  --mode prepare_data \
  --train-data data/train.txt \
  --tokenizer data/tokenizer.json
```

### 2) Train

```bash
cargo run --release -- \
  --config tiny_10m \
  --mode train \
  --steps 30000 \
  --batch-size 8 \
  --seq-len 128 \
  --save-every 250 \
  --train-data data/train.txt \
  --tokenizer data/tokenizer.json
```

### 3) Resume training

```bash
cargo run --release -- \
  --config tiny_10m \
  --mode train \
  --resume-from checkpoints/step_020500.safetensors \
  --steps 30000 \
  --batch-size 8 \
  --seq-len 128 \
  --save-every 250 \
  --train-data data/train.txt \
  --tokenizer data/tokenizer.json
```

### 4) Generate with trained checkpoint

```bash
cargo run --release -- \
  --config tiny_10m \
  --mode generate \
  --tokenizer data/tokenizer.json \
  --resume-from checkpoints/step_020500.safetensors \
  --prompt "介绍一下哈基米" \
  --max-new-tokens 120
```

## Notes

- This project focuses on compact dense models (no MoE routing).
- Official external CUDA kernel stack is not required for basic training/inference here.

## References

- [Qwen3-Next Blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd)
- [Gated Delta Networks](https://github.com/NVlabs/GatedDeltaNet)
- [Gemma (RMSNorm variant)](https://github.com/google/gemma_pytorch)

## License

MIT
