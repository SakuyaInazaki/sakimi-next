# Tencent Cloud Quickstart (GPU Training)

This project can be trained directly on Tencent Cloud GPU CVM.

## 1) Create a GPU CVM

- OS: Ubuntu 22.04 (recommended)
- GPU: NVIDIA (single card is enough for `tiny_10m`)
- Ensure NVIDIA driver and CUDA runtime are ready (`nvidia-smi` works)

## 2) Clone project

```bash
git clone https://github.com/SakuyaInazaki/sakimi-next.git
cd sakimi-next
```

## 3) One-command training

```bash
bash scripts/train_tencent_gpu.sh
```

This command will:

1. Check GPU (`nvidia-smi`)
2. Install Rust if needed
3. Switch Candle backend to CUDA
4. Run `cargo check --release`
5. Run `prepare_data` when tokenized files are missing
6. Start training and save logs in `logs/`

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
bash scripts/train_tencent_gpu.sh
```

## 5) Backend switching

```bash
bash scripts/switch_candle_backend.sh cuda
bash scripts/switch_candle_backend.sh metal
bash scripts/switch_candle_backend.sh cpu
```
