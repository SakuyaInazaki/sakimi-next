#!/usr/bin/env bash
set -euo pipefail

# One-command training entry for any NVIDIA GPU cloud VM
# (Tencent Cloud / Huawei Cloud / other CUDA-compatible providers).
#
# Example:
#   bash scripts/train_nvidia_gpu.sh
#
# Optional environment overrides:
#   CONFIG=tiny_5m
#   STEPS=20000
#   BATCH_SIZE=16
#   SEQ_LEN=512
#   LEARNING_RATE=1e-4
#   TRAIN_DATA=data/synth/train_ai_hq_bal_300k.txt
#   PRINT_EVERY=50
#   SAVE_EVERY=1000
#   DISABLE_FAST_KERNELS=0

CONFIG="${CONFIG:-tiny_5m}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRAIN_DATA="${TRAIN_DATA:-data/synth/train_ai_hq_bal_300k.txt}"
PRINT_EVERY="${PRINT_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
DISABLE_FAST_KERNELS="${DISABLE_FAST_KERNELS:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This machine does not look like an NVIDIA GPU instance."
  exit 1
fi

if [[ ! -f "Cargo.toml" ]]; then
  echo "please run this script from project root (Cargo.toml not found)"
  exit 1
fi

if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
  echo "== Install base packages (if missing) =="
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y curl git build-essential pkg-config libssl-dev perl ca-certificates
else
  echo "== Skip apt-get install (not root or apt-get unavailable) =="
fi

echo "== GPU =="
nvidia-smi

echo "== Rust toolchain =="
if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
fi
cargo --version
rustc --version

echo "== Switch backend to CUDA =="
bash scripts/switch_candle_backend.sh cuda

echo "== Build check =="
cargo check --release

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "training data not found: ${TRAIN_DATA}"
  exit 1
fi

if [[ ! -f "data/tokenizer.json" || ! -f "data/tokens.bin" ]]; then
  echo "== Prepare tokenized data =="
  cargo run --release -q -- \
    --config "${CONFIG}" \
    --device cuda \
    --mode prepare_data \
    --seq-len "${SEQ_LEN}" \
    --train-data "${TRAIN_DATA}"
fi

mkdir -p logs checkpoints
TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/train_${CONFIG}_cuda_s${STEPS}_b${BATCH_SIZE}_l${SEQ_LEN}_${TS}.log"

CMD=(cargo run --release -q -- \
  --config "${CONFIG}" \
  --device cuda \
  --mode train \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --learning-rate "${LEARNING_RATE}" \
  --print-every "${PRINT_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  --train-data "${TRAIN_DATA}")

if [[ "${DISABLE_FAST_KERNELS}" == "1" ]]; then
  CMD+=(--disable-fast-kernels)
fi

echo "== Start training =="
echo "log: ${LOG_PATH}"
printf '%q ' "${CMD[@]}"
echo

/usr/bin/time -p "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"

echo "== Done =="
echo "checkpoints: $(pwd)/checkpoints"
echo "log file:    $(pwd)/${LOG_PATH}"
