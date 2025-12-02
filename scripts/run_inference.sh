#!/bin/bash
# Quick inference script for Flow Matching LM
#
# Usage:
#   ./scripts/run_inference.sh "Your prompt here"
#   ./scripts/run_inference.sh "def fibonacci(" --num_steps 64
#   ./scripts/run_inference.sh --interactive

set -e

cd "$(dirname "$0")/.."

# Default checkpoint
CHECKPOINT="${CHECKPOINT:-checkpoint-1500}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/flow_matching_qwen_1.7b}"

# Check if first arg is a flag or prompt
if [[ "$1" == --* ]]; then
    # First arg is a flag, pass all args directly
    python scripts/inference.py \
        --checkpoint "$CHECKPOINT" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        "$@"
else
    # First arg is a prompt
    PROMPT="$1"
    shift
    python scripts/inference.py \
        --checkpoint "$CHECKPOINT" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --prompt "$PROMPT" \
        "$@"
fi

