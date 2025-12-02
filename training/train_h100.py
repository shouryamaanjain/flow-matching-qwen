#!/usr/bin/env python3
"""
Main training script for Flow Matching Language Model on H100 GPUs.

This script handles:
- Model initialization from Qwen3 or from checkpoint
- Data loading and preprocessing
- Distributed training setup (FSDP)
- Training loop execution

Usage:
    # Single GPU
    python train_h100.py --config configs/pretrain_h100x4.yaml

    # Multi-GPU with accelerate
    accelerate launch --num_processes 4 train_h100.py --config configs/pretrain_h100x4.yaml
    
    # With DeepSpeed
    accelerate launch --config_file accelerate_config.yaml train_h100.py --config configs/pretrain_h100x4.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from FlowMatchingLM import FlowMatchingConfig, FlowMatchingLanguageModel
from training.fm_trainer import FlowMatchingTrainer, FlowMatchingTrainingConfig
from data.data_loader import PretrainDataset, create_pretrain_dataloader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> FlowMatchingTrainingConfig:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return FlowMatchingTrainingConfig(**config_dict)


def create_model(config: FlowMatchingTrainingConfig) -> FlowMatchingLanguageModel:
    """
    Create FlowMatchingLanguageModel.
    
    Options:
    1. Initialize from pretrained Qwen3 (default)
    2. Load from checkpoint
    3. Initialize from scratch
    """
    # Create model config
    model_config = FlowMatchingConfig(
        # Use Qwen3-1.7B defaults
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=40960,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        
        # Token IDs
        mask_token_id=config.mask_token_id,
        
        # DFM parameters
        scheduler_type=config.scheduler_type,
        scheduler_n=config.scheduler_n,
        sampling_eps=config.sampling_eps,
    )
    
    if config.resume_from_checkpoint:
        # Load from checkpoint
        logger.info(f"Loading model from checkpoint: {config.resume_from_checkpoint}")
        model = FlowMatchingLanguageModel.from_pretrained(
            config.resume_from_checkpoint,
            config=model_config,
        )
    else:
        # Initialize from Qwen3
        logger.info(f"Initializing from Qwen3: {config.model_name_or_path}")
        try:
            model = FlowMatchingLanguageModel.from_pretrained_qwen3(
                config.model_name_or_path,
                config=model_config,
            )
        except Exception as e:
            logger.warning(f"Could not load Qwen3 weights: {e}")
            logger.info("Initializing model from scratch")
            model = FlowMatchingLanguageModel(model_config)
    
    # Log model info
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {n_params:,} parameters ({n_params/1e9:.2f}B)")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Flow Matching LM on H100")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_h100x4.yaml",
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Override model name or path from config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max steps from config",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default="pretrain",
        choices=["pretrain", "midtrain", "sft"],
        help="Training mode (pretrain, midtrain, or sft)",
    )
    parser.add_argument(
        "--include_stack_v1",
        action="store_true",
        help="Include The Stack v1 (RECOMMENDED - has actual code content, just needs HF login)",
    )
    parser.add_argument(
        "--include_stack_v2",
        action="store_true",
        help="Include The Stack v2 (requires AWS credentials for content download)",
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using default configuration")
        config = FlowMatchingTrainingConfig()
    
    # Apply command line overrides
    if args.model_name_or_path:
        config.model_name_or_path = args.model_name_or_path
    if args.output_dir:
        config.checkpoint_dir = args.output_dir
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.seed:
        config.seed = args.seed
    
    # Log configuration
    logger.info("Training configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    
    # Create model
    model = create_model(config)
    
    # Create training dataset based on mode
    logger.info(f"Creating dataset for {args.training_mode} mode...")
    
    include_stack_v1 = args.include_stack_v1 or getattr(config, 'include_stack_v1', False)
    include_stack_v2 = args.include_stack_v2 or getattr(config, 'include_stack_v2', False)
    
    if args.training_mode in ["pretrain", "midtrain"]:
        # Use CoDA pre-training datasets
        train_dataset = PretrainDataset(
            tokenizer=tokenizer,
            max_length=config.sequence_length,
            include_stack_v1=include_stack_v1,
            include_stack_v2=include_stack_v2,
            seed=config.seed,
        )
        logger.info(f"Created pre-training dataset (include_stack_v1={include_stack_v1}, include_stack_v2={include_stack_v2})")
    else:
        # SFT mode - use instruction dataset
        from datasets import load_dataset
        
        dataset_name = getattr(config, 'dataset_name', None) or "HuggingFaceH4/ultrachat_200k"
        logger.info(f"Loading SFT dataset: {dataset_name}")
        
        sft_dataset = load_dataset(dataset_name, split="train_sft", streaming=True)
        
        # Wrap in a simple dataset class for tokenization
        class SFTDataset(torch.utils.data.IterableDataset):
            def __init__(self, hf_dataset, tokenizer, max_length):
                self.dataset = hf_dataset
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __iter__(self):
                for example in self.dataset:
                    # Format as conversation
                    messages = example.get("messages", [])
                    text = ""
                    for msg in messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                    
                    tokens = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_length,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    
                    yield {
                        "input_ids": tokens["input_ids"].squeeze(0),
                        "attention_mask": tokens["attention_mask"].squeeze(0),
                    }
        
        train_dataset = SFTDataset(sft_dataset, tokenizer, config.sequence_length)
        logger.info(f"Created SFT dataset")
    
    # Create trainer
    trainer = FlowMatchingTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

