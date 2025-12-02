"""
Flow Matching Trainer for H100 GPUs

This module implements the training infrastructure for Flow Matching Language Models
on H100 GPUs using FSDP (Fully Sharded Data Parallel) for distributed training.

Key features:
- FSDP with ZeRO-3 style sharding for memory efficiency
- Mixed precision training (bf16)
- Gradient accumulation
- WandB logging
- Checkpoint saving and resuming
"""

import os
import math
import logging
import shutil
from typing import Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from timeit import default_timer as timer

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import set_seed
from transformers import (
    AutoTokenizer,
    get_scheduler,
    PreTrainedTokenizerBase,
)

import wandb

logger = logging.getLogger(__name__)


@dataclass
class FlowMatchingTrainingConfig:
    """
    Configuration for Flow Matching training.
    
    This replaces CoDA's Hydra config with a dataclass-based config
    optimized for H100 training.
    """
    # Model
    model_name_or_path: str = "Qwen/Qwen3-1.7B"
    mask_token_id: int = 151669
    
    # DFM parameters
    scheduler_type: str = "polynomial_convex"
    scheduler_n: float = 1.0
    sampling_eps: float = 1e-3
    
    # Training
    max_steps: int = 210000
    global_batch_size: int = 128
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    sequence_length: int = 8192
    
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # LR Scheduler
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 2000
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_steps: int = 5000
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 2  # Only keep this many checkpoints (saves disk space)
    
    # Logging
    logging_steps: int = 10
    wandb_project: str = "flow-matching-lm"
    run_name: Optional[str] = None
    
    # Distributed
    mixed_precision: str = "bf16"
    fsdp_config: Optional[Dict] = None
    
    # Data
    dataset_name: Optional[str] = None
    data_dir: Optional[str] = None
    datasets: Optional[list] = None  # Dataset configurations from YAML
    include_stack_v1: bool = False   # Include Stack v1 code datasets (RECOMMENDED)
    include_stack_v2: bool = False   # Include Stack v2 code datasets (requires AWS)
    num_workers: int = 0             # Number of data loading workers (0 avoids HF rate limits)
    streaming: bool = True           # Use streaming for large datasets
    
    # Misc
    seed: int = 42
    
    def __post_init__(self):
        # Calculate gradient accumulation if needed
        if self.fsdp_config is None:
            self.fsdp_config = {
                "auto_wrap_policy": "transformer_based_wrap",
                "backward_prefetch": "BACKWARD_PRE",
                "forward_prefetch": True,
                "sharding_strategy": "FULL_SHARD",
                "state_dict_type": "sharded_state_dict",
                "limit_all_gathers": True,
                "use_orig_params": True,
                "activation_checkpointing": True,
            }


class FlowMatchingTrainer:
    """
    Trainer for Flow Matching Language Model on H100 GPUs.
    
    This trainer implements the training loop for discrete flow matching,
    handling:
    - Distributed training with FSDP
    - Mixed precision (bf16)
    - Gradient accumulation
    - Logging and checkpointing
    
    Key differences from CoDA's trainer:
    - Uses DFM probability path instead of sigma-based masking
    - No dsigma loss weighting
    - Uses Accelerate for distributed training instead of torch_xla
    """
    
    def _build_fsdp_plugin(self) -> FullyShardedDataParallelPlugin:
        """Create an Accelerate FSDP plugin from the config dictionary."""
        cfg = self.config.fsdp_config or {}
        
        def _normalize_str(value: Optional[str], default: str) -> str:
            if value is None:
                return default
            return value.lower()
        
        return FullyShardedDataParallelPlugin(
            sharding_strategy=cfg.get("sharding_strategy", "FULL_SHARD"),
            backward_prefetch=cfg.get("backward_prefetch", "BACKWARD_PRE"),
            forward_prefetch=cfg.get("forward_prefetch", True),
            auto_wrap_policy=_normalize_str(cfg.get("auto_wrap_policy"), "transformer_based_wrap"),
            state_dict_type=cfg.get("state_dict_type", "sharded_state_dict"),
            limit_all_gathers=cfg.get("limit_all_gathers", True),
            use_orig_params=cfg.get("use_orig_params", True),
            activation_checkpointing=cfg.get("activation_checkpointing", True),
        )
    
    def __init__(
        self,
        model: nn.Module,
        config: FlowMatchingTrainingConfig,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._step_time_ema: Optional[float] = None
        
        # Initialize accelerator for distributed training
        # Note: Create accelerator first, then check is_main_process
        fsdp_plugin = self._build_fsdp_plugin()
        
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_dir=config.checkpoint_dir,
            fsdp_plugin=fsdp_plugin,
        )
        
        # Set random seed
        set_seed(config.seed)
        
        # Store model
        self.model = model
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Prepare for distributed training
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Initialize logging (only on main process)
        if self.is_main_process:
            self._init_wandb()
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.accelerator.is_main_process
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer."""
        # Separate weight decay for different parameter groups
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler."""
        return get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
        )
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.is_main_process:
            return
        
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.run_name,
            config={
                "model": self.config.model_name_or_path,
                "max_steps": self.config.max_steps,
                "global_batch_size": self.config.global_batch_size,
                "learning_rate": self.config.learning_rate,
                "scheduler_type": self.config.scheduler_type,
                "scheduler_n": self.config.scheduler_n,
                "sequence_length": self.config.sequence_length,
            },
        )
    
    def _get_train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_batch_size,
            shuffle=not isinstance(self.train_dataset, IterableDataset),
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step.
        
        This is the core of Flow Matching training:
        1. Sample t ~ Uniform(eps, 1)
        2. Corrupt input using probability path: x_t = path.sample(x_0, t)
        3. Forward pass: logits = model(x_t, t)
        4. Compute cross-entropy loss on masked positions
        
        Args:
            batch: Dictionary with 'input_ids' and optionally 'attention_mask'
            
        Returns:
            Loss tensor
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        src_mask = batch.get("src_mask", None)
        
        # Forward pass through model (handles t sampling and corruption internally)
        logits, loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            src_mask=src_mask,
            training_mode="pretrain",
        )
        
        return loss
    
    def train(self):
        """
        Main training loop.
        
        Runs training for max_steps, handling:
        - Gradient accumulation
        - Mixed precision
        - Logging
        - Checkpointing
        """
        if self.is_main_process:
            logger.info("Starting training")
            logger.info(f"  Max steps: {self.config.max_steps}")
            logger.info(f"  Global batch size: {self.config.global_batch_size}")
            logger.info(f"  Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        
        # Create dataloader
        train_dataloader = self._get_train_dataloader()
        train_dataloader = self.accelerator.prepare(train_dataloader)
        train_iterator = iter(train_dataloader)
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)
        
        self.model.train()
        
        for step in range(self.global_step, self.config.max_steps):
            step_start_time = timer()
            
            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                self.epoch += 1
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
            
            # Training step with gradient accumulation
            with self.accelerator.accumulate(self.model):
                loss = self.train_step(batch)
                self.accelerator.backward(loss)
                
                # Calculate gradient norm BEFORE clipping (for monitoring)
                grad_norm = None
                if self.accelerator.sync_gradients:
                    grad_norm = self._get_grad_norm()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            self.global_step = step + 1
            step_end_time = timer()
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self._log_metrics(
                    loss=loss.detach().item(),
                    step_time=step_end_time - step_start_time,
                    grad_norm=grad_norm,
                )
            
            # Checkpointing
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()
        
        # Final checkpoint
        self._save_checkpoint()
        
        if self.is_main_process:
            logger.info("Training complete!")
            wandb.finish()
    
    def _get_grad_norm(self) -> float:
        """Calculate total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _log_metrics(self, loss: float, step_time: float, grad_norm: float = None):
        """Log training metrics."""
        if not self.is_main_process:
            return
        
        lr = self.lr_scheduler.get_last_lr()[0]
        tokens_per_step = (
            self.config.global_batch_size * 
            self.config.sequence_length
        )
        
        if self._step_time_ema is None:
            self._step_time_ema = step_time
        else:
            self._step_time_ema = 0.9 * self._step_time_ema + 0.1 * step_time
        
        steps_remaining = max(self.config.max_steps - self.global_step, 0)
        eta_seconds = steps_remaining * self._step_time_ema
        progress = self.global_step / self.config.max_steps
        
        metrics = {
            "train/loss": loss,
            "train/ppl": math.exp(min(loss, 20)),  # Cap to avoid overflow
            "train/learning_rate": lr,
            "train/step_time": step_time * 1000,  # ms
            "train/tokens_per_second": tokens_per_step / step_time,
            "train/epoch": self.epoch,
            "train/step": self.global_step,
            "train/progress": progress,
            "train/eta_minutes": eta_seconds / 60.0,
        }
        
        # Add gradient norm if available
        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm
        
        # Log message with grad_norm
        grad_norm_str = f", grad_norm={grad_norm:.2f}" if grad_norm is not None else ""
        logger.info(
            f"Step {self.global_step}/{self.config.max_steps} "
            f"({progress*100:.2f}%): loss={loss:.4f}, lr={lr:.2e}{grad_norm_str}, "
            f"step_time={step_time*1000:.1f}ms, ETA {self._format_eta(eta_seconds)}"
        )
        
        wandb.log(metrics, step=self.global_step)
    
    @staticmethod
    def _format_eta(seconds: float) -> str:
        if seconds <= 0 or math.isinf(seconds):
            return "0s"
        minutes, sec = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes and len(parts) < 2:
            parts.append(f"{minutes}m")
        if not parts:
            parts.append(f"{sec}s")
        return " ".join(parts)
    
    def _save_checkpoint(self):
        """Save training checkpoint and cleanup old ones."""
        checkpoint_dir = Path(self.config.checkpoint_dir) / f"checkpoint-{self.global_step}"
        
        self.accelerator.wait_for_everyone()
        
        # Save model and optimizer state
        self.accelerator.save_state(checkpoint_dir)
        
        # Save additional training state
        if self.is_main_process:
            state = {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "config": self.config.__dict__,
            }
            torch.save(state, checkpoint_dir / "training_state.pt")
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # Cleanup old checkpoints to save disk space
            self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoint_base = Path(self.config.checkpoint_dir)
        save_total_limit = getattr(self.config, 'save_total_limit', 2)
        
        if save_total_limit <= 0:
            return  # Keep all checkpoints
        
        # Find all checkpoint directories
        checkpoints = []
        for item in checkpoint_base.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append((step, item))
                except (ValueError, IndexError):
                    continue
        
        # Sort by step number (newest first)
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Delete old checkpoints beyond the limit
        for step, ckpt_path in checkpoints[save_total_limit:]:
            try:
                logger.info(f"Deleting old checkpoint: {ckpt_path}")
                shutil.rmtree(ckpt_path)
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {ckpt_path}: {e}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model and optimizer state
        self.accelerator.load_state(checkpoint_path)
        
        # Load additional training state
        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            
            if self.is_main_process:
                logger.info(f"Resumed from checkpoint at step {self.global_step}")


def create_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    config: FlowMatchingTrainingConfig,
) -> DataLoader:
    """
    Create training dataloader.
    
    This function handles loading and preprocessing the dataset.
    Supports both HuggingFace datasets and custom data formats.
    """
    from datasets import load_dataset
    
    # Load dataset
    if config.dataset_name:
        dataset = load_dataset(config.dataset_name, split="train")
    elif config.data_dir:
        dataset = load_dataset("json", data_dir=config.data_dir, split="train")
    else:
        raise ValueError("Either dataset_name or data_dir must be provided")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.sequence_length,
            padding="max_length",
            return_tensors="pt",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    tokenized_dataset.set_format(type="torch")
    
    return DataLoader(
        tokenized_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

