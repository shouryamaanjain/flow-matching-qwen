"""
FlowMatchingConfig - Configuration for Discrete Flow Matching Language Model

Adapted from CoDA's CoDAConfig, replacing masked diffusion parameters with
Discrete Flow Matching (DFM) specific parameters.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging
from typing import Optional, List, Union

logger = logging.get_logger(__name__)


class FlowMatchingConfig(PretrainedConfig):
    """
    Configuration class for FlowMatchingLanguageModel.
    
    This configuration extends the Qwen3 architecture config with parameters
    specific to Discrete Flow Matching (DFM) for text generation.
    
    Key differences from CoDA's diffusion config:
    - Replaces sigma-based sampling with t-based probability paths
    - Uses scheduler_type and scheduler_n instead of sampling_eps
    - Removes dsigma weighting parameters
    - Adds num_sampling_steps for ODE solver
    
    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Dimension of hidden representations
        intermediate_size: Dimension of FFN intermediate layer
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads for GQA
        head_dim: Dimension of each attention head
        hidden_act: Activation function
        max_position_embeddings: Maximum sequence length
        initializer_range: Std for weight initialization
        rms_norm_eps: Epsilon for RMS normalization
        use_cache: Whether to use KV cache
        rope_theta: Base for rotary position embeddings
        rope_scaling: RoPE scaling configuration
        attention_bias: Whether to use bias in attention
        attention_dropout: Dropout rate for attention
        attention_kernel: Type of attention kernel (flash_attention, etc.)
        
        # Token IDs
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        mask_token_id: Mask token ID for DFM corruption
        
        # DFM-specific parameters (replaces CoDA's diffusion params)
        scheduler_type: Type of probability path scheduler
            - "polynomial_convex": PolynomialConvexScheduler (default)
            - "linear": Linear interpolation
            - "cosine": Cosine schedule
        scheduler_n: Polynomial degree for polynomial_convex scheduler
        num_sampling_steps: Number of steps for discrete ODE solver
        time_embedding_dim: Dimension of time embeddings (defaults to hidden_size)
        time_embedding_type: Type of time embedding ("sinusoidal" or "learned")
    """
    
    model_type = "FlowMatchingLM"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # Qwen3 architecture parameters
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 40960,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        use_sliding_window: bool = False,
        sliding_window: Optional[int] = None,
        max_window_layers: int = 28,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attention_kernel: str = "flash_attention",
        attn_implementation: str = "eager",  # Use "eager" for custom models
        tie_word_embeddings: bool = True,
        
        # Token IDs
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        mask_token_id: int = 151669,
        
        # DFM-specific parameters (NEW - replaces CoDA's diffusion params)
        scheduler_type: str = "polynomial_convex",
        scheduler_n: float = 1.0,
        num_sampling_steps: int = 32,
        time_embedding_dim: Optional[int] = None,
        time_embedding_type: str = "sinusoidal",
        
        # Training parameters
        sampling_eps: float = 1e-3,  # Minimum t value to avoid numerical issues
        
        **kwargs,
    ):
        # Store Qwen3 architecture parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_kernel = attention_kernel
        self.attn_implementation = attn_implementation
        self._attn_implementation = attn_implementation  # Required by transformers
        
        # Handle num_key_value_heads for GQA
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        
        # Validate RoPE configuration
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        
        # Token IDs
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        
        # DFM-specific parameters
        self.scheduler_type = scheduler_type
        self.scheduler_n = scheduler_n
        self.num_sampling_steps = num_sampling_steps
        self.time_embedding_dim = time_embedding_dim or hidden_size
        self.time_embedding_type = time_embedding_type
        self.sampling_eps = sampling_eps
        
        # Validate DFM parameters
        if scheduler_type not in ["polynomial_convex", "linear", "cosine"]:
            raise ValueError(
                f"Invalid scheduler_type: {scheduler_type}. "
                f"Must be one of: polynomial_convex, linear, cosine"
            )
        if scheduler_n <= 0:
            raise ValueError(f"scheduler_n must be positive, got {scheduler_n}")
        if num_sampling_steps <= 0:
            raise ValueError(f"num_sampling_steps must be positive, got {num_sampling_steps}")
        
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    @classmethod
    def from_qwen3_config(cls, qwen3_config, **dfm_kwargs):
        """
        Create a FlowMatchingConfig from an existing Qwen3 configuration.
        
        This is useful for converting a pretrained Qwen3 model to a
        Flow Matching model while preserving the architecture.
        
        Args:
            qwen3_config: A Qwen3 configuration or dictionary
            **dfm_kwargs: DFM-specific parameters to override defaults
            
        Returns:
            FlowMatchingConfig: Configuration for Flow Matching model
        """
        if hasattr(qwen3_config, "to_dict"):
            config_dict = qwen3_config.to_dict()
        else:
            config_dict = dict(qwen3_config)
        
        # Remove parameters that shouldn't be passed
        config_dict.pop("model_type", None)
        config_dict.pop("transformers_version", None)
        
        # Update with DFM-specific parameters
        config_dict.update(dfm_kwargs)
        
        return cls(**config_dict)

