#!/usr/bin/env bash
set -euo pipefail

# Tencent Cloud GPU entrypoint.
# Same behavior as train_nvidia_gpu.sh, kept for convenience.
exec bash "$(dirname "$0")/train_nvidia_gpu.sh" "$@"
