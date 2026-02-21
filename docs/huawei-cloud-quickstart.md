# Huawei Cloud Quickstart (GPU Training)

This project can be trained directly on Huawei Cloud GPU instances (ECS GPU or ModelArts Lite Server).

## 1) Create a GPU instance

- OS: Ubuntu 22.04 (recommended)
- GPU: NVIDIA (single-card is enough for `tiny_10m`)
- Make sure NVIDIA driver + CUDA runtime are available (`nvidia-smi` must work)

## 2) Clone project

```bash
git clone https://github.com/SakuyaInazaki/sakimi-next.git
cd sakimi-next
```

## 3) Run one-command training

```bash
bash scripts/train_huawei_gpu.sh
```

What this script does:

1. Verifies GPU (`nvidia-smi`)
2. Installs Rust toolchain if missing
3. Switches Candle backend to CUDA
4. Runs `cargo check --release`
5. Runs `prepare_data` if `data/tokenizer.json` or `data/tokens.bin` is missing
6. Starts training and writes logs to `logs/`

## 4) Common overrides

```bash
CONFIG=tiny_10m \
STEPS=50000 \
BATCH_SIZE=32 \
SEQ_LEN=512 \
LEARNING_RATE=1e-4 \
TRAIN_DATA=data/synth/train_ai_hq_bal_300k.txt \
PRINT_EVERY=50 \
SAVE_EVERY=1000 \
bash scripts/train_huawei_gpu.sh
```

## 5) Backend switching

Use this helper script:

```bash
bash scripts/switch_candle_backend.sh cuda
bash scripts/switch_candle_backend.sh metal
bash scripts/switch_candle_backend.sh cpu
```
