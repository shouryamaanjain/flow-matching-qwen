#!/bin/bash
# Flow Matching LM Mid-training Script
# Continues from pre-trained checkpoint with code-focused data

set -e

# Configuration
CONFIG_FILE="${CONFIG_FILE:-training/configs/midtrain_h100x4.yaml}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-./checkpoints/flow_matching_qwen_1.7b/checkpoint-210000}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/flow_matching_qwen_1.7b_midtrain}"
NUM_GPUS="${NUM_GPUS:-4}"

# Environment setup
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_PROJECT="${WANDB_PROJECT:-flow-matching-lm}"

echo "=========================================="
echo "Flow Matching LM Mid-training"
echo "=========================================="
echo "Pretrain checkpoint: $PRETRAIN_CHECKPOINT"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training with accelerate
accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    --multi_gpu \
    training/train_h100.py \
    --config "$CONFIG_FILE" \
    --resume_from_checkpoint "$PRETRAIN_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

echo "Mid-training complete!"

