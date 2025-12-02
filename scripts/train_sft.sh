#!/bin/bash
# Flow Matching LM Post-Training (SFT) Script
# Run on 4x H100 GPUs

set -e

# Configuration
CONFIG_FILE="training/configs/sft_h100x4.yaml"
NUM_GPUS=4
MASTER_PORT=29500

# Check for mid-trained checkpoint
MIDTRAIN_CHECKPOINT="${MIDTRAIN_CHECKPOINT:-./checkpoints/flow_matching_qwen_1.7b_midtrain/checkpoint-50000}"
if [ ! -d "$MIDTRAIN_CHECKPOINT" ]; then
    echo "Error: Mid-trained checkpoint not found at $MIDTRAIN_CHECKPOINT"
    echo "Please set MIDTRAIN_CHECKPOINT environment variable to the correct path."
    exit 1
fi

echo "=========================================="
echo "Flow Matching LM Post-Training (SFT)"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $MIDTRAIN_CHECKPOINT"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

# Activate environment if needed
# source /path/to/venv/bin/activate

# Run training with accelerate
accelerate launch \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --main_process_port=$MASTER_PORT \
    training/train_h100.py \
    --config $CONFIG_FILE \
    --model_name_or_path "$MIDTRAIN_CHECKPOINT" \
    --training_mode sft

echo "=========================================="
echo "SFT training completed!"
echo "Checkpoint saved to: ./checkpoints/flow_matching_qwen_1.7b_instruct"
echo "=========================================="

