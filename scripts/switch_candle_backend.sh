#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/switch_candle_backend.sh cuda
#   scripts/switch_candle_backend.sh metal
#   scripts/switch_candle_backend.sh cpu

BACKEND="${1:-}"
if [[ -z "${BACKEND}" ]]; then
  echo "usage: $0 <cuda|metal|cpu>"
  exit 1
fi

case "${BACKEND}" in
  cuda)
    FEATURES='["cuda"]'
    ;;
  metal)
    FEATURES='["accelerate", "metal"]'
    ;;
  cpu)
    FEATURES='["accelerate"]'
    ;;
  *)
    echo "unsupported backend: ${BACKEND}"
    echo "supported: cuda, metal, cpu"
    exit 1
    ;;
esac

if [[ ! -f "Cargo.toml" ]]; then
  echo "Cargo.toml not found in current directory: $(pwd)"
  exit 1
fi

if command -v perl >/dev/null 2>&1; then
  perl -i.bak -pe "s#^candle-core\\s*=.*#candle-core = { version = \"0.9\", features = ${FEATURES} }#g" Cargo.toml
  rm -f Cargo.toml.bak
else
  # Fallback for minimal cloud images without perl.
  sed -i'' -E "s#^candle-core\\s*=.*#candle-core = { version = \"0.9\", features = ${FEATURES} }#g" Cargo.toml
fi

echo "switched candle backend to: ${BACKEND}"
if command -v rg >/dev/null 2>&1; then
  rg -n "^candle-core\\s*=" Cargo.toml
else
  grep -n "^candle-core\\s*=" Cargo.toml
fi
