#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper kept for existing docs/commands.
exec bash "$(dirname "$0")/train_nvidia_gpu.sh" "$@"
