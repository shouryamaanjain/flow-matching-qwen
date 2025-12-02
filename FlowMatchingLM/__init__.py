"""
FlowMatchingLM - Discrete Flow Matching Language Model

This package implements a Flow Matching based language model by adapting
the Qwen-3-1.7B architecture from autoregressive to discrete flow matching,
following the methodology inspired by CoDA but using Meta's flow_matching library.
"""

from .model_config import FlowMatchingConfig
from .modeling_fm import FlowMatchingLanguageModel, FlowMatchingModel
from .modeling_utils import TimeEmbedding, DiscreteFlowPath
from .generation_utils import FlowMatchingSampler, FlowMatchingGenerationConfig

__all__ = [
    "FlowMatchingConfig",
    "FlowMatchingLanguageModel",
    "FlowMatchingModel",
    "TimeEmbedding",
    "DiscreteFlowPath",
    "FlowMatchingSampler",
    "FlowMatchingGenerationConfig",
]

__version__ = "0.1.0"

