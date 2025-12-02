#!/bin/bash
# Flow Matching LM Evaluation Script

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-./checkpoints/flow_matching_qwen_1.7b_midtrain/checkpoint-50000}"
BENCHMARK="${BENCHMARK:-humaneval}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_results}"
NUM_STEPS="${NUM_STEPS:-32}"
TEMPERATURE="${TEMPERATURE:-0.0}"

echo "=========================================="
echo "Flow Matching LM Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Benchmark: $BENCHMARK"
echo "Num Steps: $NUM_STEPS"
echo "Temperature: $TEMPERATURE"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
python evaluation/eval_code.py \
    --model_path "$MODEL_PATH" \
    --benchmark "$BENCHMARK" \
    --num_steps "$NUM_STEPS" \
    --temperature "$TEMPERATURE" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

echo "Evaluation complete! Results in $OUTPUT_DIR"

