# Qwen3-Next Full Audit (Official Comparison)

Date: 2026-02-18

## Official baselines used

- QwenLM repo metadata / release notes:
  - https://github.com/QwenLM/Qwen3-Next
- Reference implementation used for line-by-line behavior:
  - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py
  - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/configuration_qwen3_next.py

Local revision for `transformers` during this audit:
- `398780d3b659d648a74ce33ac1240708dc9c05cb`

## Already aligned in this repo

- Config now supports official linear-attention fields and per-layer `layer_types`.
  - `src/config.rs`
- Config now also supports official attention/RoPE knobs:
  - `head_dim`, `attention_bias`, `attention_dropout`
  - `rope_theta`, `partial_rotary_factor`
  - `hidden_act`, `initializer_range`, `rms_norm_eps`
  - `src/config.rs`
- Decoder now selects `linear_attention` / `full_attention` strictly from `layer_types`.
  - `src/model.rs`
- Attention path now follows official gated format:
  - `q_proj` emits query+gate
  - q/k per-head RMSNorm (zero-centered form)
  - explicit `head_dim`-based projections (`q/k/v/o`)
  - optional attention biases
  - configurable RoPE (`theta`, partial rotary factor)
  - gate applied before `o_proj`
  - `src/attention.rs`
- DeltaNet path now uses official parameterization:
  - `in_proj_qkvz`, `in_proj_ba`, `dt_bias`, `a_log`, `out_proj`
  - `g = -exp(A_log) * softplus(a + dt_bias)`
  - grouped q/k/v/z unpacking logic
  - configurable hidden activation + initializer range
  - padding-state masking (`apply_mask_to_padding_states` equivalent)
  - `src/deltanet.rs`
- Gated norm for DeltaNet output now follows official RMSNormGated shape/semantics.
  - `src/conv.rs`
- Dynamic cache path has been added for decoding:
  - attention KV cache
  - linear-attention conv state + recurrent state
  - beam-search style `reorder_cache`
  - cache-aware model forward and generate loop
  - `src/cache.rs`, `src/attention.rs`, `src/deltanet.rs`, `src/model.rs`, `src/main.rs`
- CE loss gather bug fixed (flattened token indexing).
  - `src/training.rs`
- Tokenizer vocab build now keeps unique tokens and always fills to target `vocab_size`.
  - avoids train-time `tokenizer vocab != config vocab` drift on small datasets
  - `src/tokenizer.rs`, `src/data.rs`
- Parameter counting logic updated to new official-style fields.
  - `src/model.rs`

## Remaining differences vs official (full list by module)

### 1) Decoder architecture scope

- Official includes sparse MoE in decoder layers; this repo uses dense FFN only.
  - Official: `Qwen3NextSparseMoeBlock` in decoder path.
  - Here: `FFN` only.

### 2) Cache / generation path

- Dynamic cache and beam reorder are implemented.
- Still missing compared to HF:
  - full HF `Cache`/generation mixin interface compatibility
  - full mask-size API wiring (`get_mask_sizes`-style flow per backend)

### 3) DeltaNet execution kernels

- Official supports fast kernels (`flash-linear-attention`, `causal-conv1d`) and keeps torch fallback.
- This repo now has an optimized fast path in Rust/Candle:
  - grouped `conv1d` depthwise short-conv path (replaces per-step window loops)
  - vectorized recurrent gated-delta updates over batch/head dimensions
  - runtime switch via config/CLI (`use_fast_kernels`, `--disable-fast-kernels`)
- Remaining gap vs official:
  - no external CUDA kernels from FLA/causal-conv1d yet

### 4) Attention implementation details

- Official can dispatch to backend attention interfaces.
- This repo uses explicit looped causal attention path.
- Train/eval-aware attention dropout has now been aligned:
  - dropout is applied only in training mode
  - trainer enables training mode for attention layers
  - default inference/test path keeps dropout disabled

### 5) Mask semantics

- Official applies padding mask handling in linear-attention path (`apply_mask_to_padding_states`).
- This repo now mirrors that masking in DeltaNet internals.
- Dense-path public API now supports optional 2D `attention_mask` in forward.
- Remaining gap: full HF `create_causal_mask` + `cache_position` semantics across all decode/backends.

### 6) Config surface area

- Official MoE/router config set is still intentionally not implemented here.
- Non-MoE config surface is now mostly aligned for dense-path behavior.

### 7) Runtime/infra differences

- Official stack integrates pretrained loading, generation mixins, multi-backend attention, serving ecosystem.
- This repo is standalone training/inference code, not HF-compatible checkpoint loader.

### 8) Numerical/performance notes

- Current `tiny_5m` preset prints ~1.5M params with new counting logic.
- Preset names (`tiny_5m`, `tiny_10m`) no longer match actual counted params after recent structural alignment and should be re-labeled.
- Cache correctness sanity-check:
  - new unit test validates `forward_with_cache` token-by-token outputs match full `forward`
  - `src/model.rs` test `test_forward_with_cache_matches_full_forward`

## Verification status after modifications

- `cargo fmt`: pass
- `cargo check`: pass
- `cargo test`: 9 passed
- `cargo run -- --config tiny_5m --mode test --batch-size 1 --seq-len 8`: pass
- `cargo run -- --config tiny_5m --mode train --steps 2 --batch-size 2 --seq-len 16 --train-data data/train.txt`: pass
- `cargo run -- --config tiny_5m --mode train --steps 1 --batch-size 2 --seq-len 16 --train-data data/train.txt`: pass
- `cargo run -- --config tiny_5m --disable-fast-kernels --mode train --steps 1 --batch-size 2 --seq-len 16 --train-data data/train.txt`: pass
