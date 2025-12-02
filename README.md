# Flow Matching Language Model (FlowMatchingLM)

A Discrete Flow Matching based language model that converts Qwen-3-1.7B from autoregressive to flow matching generation.

## Overview

This project implements a Flow Matching LLM by adapting the methodology from CoDA (Coding via Diffusion Adaptation), but replacing the masked diffusion objective with Discrete Flow Matching (DFM) using Meta's flow_matching library.

### Key Features

- **Discrete Flow Matching**: Uses probability paths and ODE-based sampling instead of iterative denoising
- **Bidirectional Generation**: Non-autoregressive, parallel token generation
- **Retrofit Approach**: Converts pretrained Qwen-3-1.7B weights to flow matching
- **H100 Optimized**: Training infrastructure designed for 4x H100 GPUs with FSDP

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Original: Qwen-3-1.7B (AR)                       │
│  - Causal attention (is_causal=True)                                │
│  - Next-token prediction                                            │
│  - Sequential generation                                            │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│              Target: Qwen-3-1.7B Flow Matching LLM                  │
│  - Bidirectional attention (is_causal=False)                        │
│  - Posterior prediction p(x_0|x_t,t)                                │
│  - Parallel generation via discrete ODE solver                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
cd flow-matching-qwen

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
flow-matching-qwen/
├── FlowMatchingLM/           # Model implementation
│   ├── model_config.py       # FlowMatchingConfig
│   ├── modeling_fm.py        # FlowMatchingLanguageModel
│   ├── modeling_utils.py     # TimeEmbedding, DiscreteFlowPath
│   └── generation_utils.py   # FlowMatchingSampler
├── training/                 # Training infrastructure
│   ├── train_h100.py         # Main training script
│   ├── fm_trainer.py         # FlowMatchingTrainer class
│   └── configs/              # Training configurations
├── evaluation/               # Evaluation scripts
│   └── eval_code.py          # HumanEval/MBPP evaluation
├── scripts/                  # Launch scripts
│   ├── train_pretrain.sh     # Pre-training launcher
│   ├── train_midtrain.sh     # Mid-training launcher
│   └── evaluate.sh           # Evaluation launcher
└── requirements.txt
```

## Key Differences from CoDA

| Component | CoDA (Diffusion) | Flow Matching (This Project) |
|-----------|------------------|------------------------------|
| Forward process | `transition()` with sigma masking | `DiscreteFlowPath` with scheduler |
| Time variable | sigma ∈ [eps, 1] | t ∈ [0, 1] with PolynomialConvexScheduler |
| Loss function | CE × (1/sigma) weighting | Cross-entropy on posteriors |
| Sampling | Iterative denoising | Discrete ODE solver |
| Time conditioning | sigma passed to model | Sinusoidal t-embedding |

## Training

Following the same 3-stage training pipeline as CoDA:

| Stage | Data | Steps | Time (4x H100) |
|-------|------|-------|----------------|
| **Pre-training** | ~180B tokens (web + code + math) | 210K | ~12 days |
| **Mid-training** | ~20B tokens (code-focused) | 50K | ~3 days |
| **Post-training (SFT)** | Instruction data | 10K | ~1 day |

### Pre-training Datasets

Modern, streamable replacements for the CoDA mixture (same token weights):

| Dataset | HuggingFace Path | Tokens Used | Category |
|---------|------------------|-------------|----------|
| dclm-baseline-1.0 | `mlfoundations/dclm-baseline-1.0` | 60.17B | Text (Web) |
| OpenWebMath | `open-web-math/open-web-math` | 12.98B | Math |
| ArXiv Summaries | `ccdv/arxiv-summarization` | 9.18B | Academic |
| Wikipedia (English 20231101) | `wikimedia/wikipedia` | 5.41B | Encyclopedia |
| MathPile Text | `zwhe99/mathpile-text` | 3.24B | Math |

**Code Datasets (choose one):**

| Dataset | HuggingFace Path | Content? | Requirements |
|---------|------------------|----------|--------------|
| **Stack v1 (RECOMMENDED)** | `bigcode/the-stack-dedup` | ✅ Yes | HF login only |
| Stack v2 | `bigcode/the-stack-v2` | ❌ File IDs only | HF + AWS credentials |

*The Stack v1 includes actual code content directly. Stack v2 only has file IDs - requires downloading from Software Heritage S3.*

### Stage 1: Pre-training (210K steps)

```bash
# Without code datasets (~91B tokens from text/math/academic data)
./scripts/train_pretrain.sh

# RECOMMENDED: Include The Stack v1 (has actual code content, just needs HF login)
./scripts/train_pretrain.sh --include_stack_v1

# Or with custom config
CONFIG_FILE=training/configs/pretrain_h100x4.yaml \
NUM_GPUS=4 \
./scripts/train_pretrain.sh --include_stack_v1

# Alternative: Include Stack v2 (requires AWS credentials for content)
./scripts/train_pretrain.sh --include_stack_v2
```

### Stage 2: Mid-training (50K steps)

```bash
# Continue from pre-trained checkpoint
PRETRAIN_CHECKPOINT=./checkpoints/flow_matching_qwen_1.7b/checkpoint-210000 \
./scripts/train_midtrain.sh
```

### Stage 3: Post-training / SFT (10K steps)

```bash
# Continue from mid-trained checkpoint
MIDTRAIN_CHECKPOINT=./checkpoints/flow_matching_qwen_1.7b_midtrain/checkpoint-50000 \
./scripts/train_sft.sh
```

### Training Configuration

Edit `training/configs/pretrain_h100x4.yaml`:

```yaml
# Key parameters
max_steps: 210000
global_batch_size: 128
learning_rate: 3.0e-4
sequence_length: 8192

# DFM parameters
scheduler_type: "polynomial_convex"
scheduler_n: 1.0

# Datasets (see config for full list)
include_stack_v2: false  # Set to true if you have credentials
```

### Setting Up Code Datasets

#### Option A: The Stack v1 (RECOMMENDED)

The Stack v1 includes actual code content directly - much easier to use!

1. Login to HuggingFace:
   ```bash
   huggingface-cli login
   ```
2. Accept the license: https://huggingface.co/datasets/bigcode/the-stack-dedup
3. Enable in training:
   ```bash
   ./scripts/train_pretrain.sh --include_stack_v1
   ```

#### Option B: The Stack v2 (Advanced)

The Stack v2 is larger (~900B tokens vs ~200B) but **only contains file IDs**, not actual code content. Content must be downloaded from Software Heritage S3.

1. Accept the license: https://huggingface.co/datasets/bigcode/the-stack-v2
2. Contact datasets@softwareheritage.org for bulk download access
3. Set up AWS credentials:
   ```bash
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   ```
4. Enable in training:
   ```bash
   ./scripts/train_pretrain.sh --include_stack_v2
   ```

## Inference

### Basic Generation

```python
from FlowMatchingLM import FlowMatchingLanguageModel
from FlowMatchingLM.generation_utils import FlowMatchingSampler, FlowMatchingGenerationConfig
from transformers import AutoTokenizer

# Load model
model = FlowMatchingLanguageModel.from_pretrained("./checkpoints/model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# Create sampler
config = FlowMatchingGenerationConfig(
    num_steps=32,
    temperature=0.7,
    top_p=0.95,
    mask_token_id=151669,
)
sampler = FlowMatchingSampler(model, config)

# Generate
prompt = tokenizer("def fibonacci(n):", return_tensors="pt")
output = sampler.generate(prompt_ids=prompt.input_ids, max_length=256)
print(tokenizer.decode(output[0]))
```

### Code Infilling

```python
# Fill in between prefix and suffix
prefix = tokenizer("def sort(arr):\n    ", return_tensors="pt")
suffix = tokenizer("\n    return arr", return_tensors="pt")
output = sampler.infill(prefix.input_ids, suffix.input_ids, infill_length=50)
```

## Evaluation

```bash
# Evaluate on HumanEval
MODEL_PATH=./checkpoints/model \
BENCHMARK=humaneval \
./scripts/evaluate.sh

# Evaluate on MBPP
MODEL_PATH=./checkpoints/model \
BENCHMARK=mbpp \
./scripts/evaluate.sh
```

## Hardware Requirements

- **Minimum**: 4x H100 80GB (or equivalent ~320GB VRAM total)
- **Recommended**: 8x H100 80GB for faster training
- **Training Time**: ~12 days for 210K steps on 4x H100

## Estimated Training Cost

```
Tokens per step: 8192 × 128 = ~1M tokens
Total tokens: 210K steps × 1M = ~210B tokens
H100 throughput: ~200K tokens/sec (4 GPUs)
Time: ~12 days
```

## Citation

```bibtex
@misc{flowmatchinglm2025,
  title={Flow Matching Language Model: Converting AR to Flow Matching},
  year={2025},
}
```

## Acknowledgments

- [CoDA](https://github.com/Salesforce/CoDA) - Diffusion adaptation methodology
- [Meta Flow Matching](https://github.com/facebookresearch/flow_matching) - DFM library
- [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B) - Base model architecture

