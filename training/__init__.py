"""
Training infrastructure for Flow Matching Language Model.

This package provides training utilities for training FlowMatchingLM
on H100 GPUs using FSDP.
"""

from .fm_trainer import FlowMatchingTrainer, FlowMatchingTrainingConfig, create_dataloaders

__all__ = [
    "FlowMatchingTrainer",
    "FlowMatchingTrainingConfig",
    "create_dataloaders",
]

