#!/bin/bash
# Flow Matching LM Pre-training Script
# Designed for 4x H100 GPUs
#
# Usage:
#   ./scripts/train_pretrain.sh                   # Without code datasets
#   ./scripts/train_pretrain.sh --include_stack_v1  # RECOMMENDED: With The Stack v1 (has actual code content)
#   ./scripts/train_pretrain.sh --include_stack_v2  # With The Stack v2 (requires AWS credentials)

set -e

# Configuration
CONFIG_FILE="${CONFIG_FILE:-training/configs/pretrain_h100x4.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/flow_matching_qwen_1.7b}"
NUM_GPUS="${NUM_GPUS:-4}"

# Environment setup
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_PROJECT="${WANDB_PROJECT:-flow-matching-lm}"

# Check if accelerate is available
if ! command -v accelerate &> /dev/null; then
    echo "Installing accelerate..."
    pip install accelerate
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Flow Matching LM Pre-training"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

# Launch training with accelerate
accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    --multi_gpu \
    training/train_h100.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

echo "Training complete!"

