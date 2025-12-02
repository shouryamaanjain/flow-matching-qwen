"""
FlowMatchingLanguageModel - Discrete Flow Matching Language Model

This module implements a Flow Matching based language model by adapting
the Qwen3 architecture. It replaces CoDA's masked diffusion objective with
Discrete Flow Matching (DFM) using probability paths and the MixturePathGeneralizedKL loss.

Key differences from CoDA's modeling_coda.py:
1. Uses TimeEmbedding for t-conditioning instead of sigma
2. Uses DiscreteFlowPath for corruption instead of transition()
3. Uses DFM loss instead of dsigma-weighted cross-entropy
4. Bidirectional attention (is_causal=False)
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import logging

from .model_config import FlowMatchingConfig
from .modeling_utils import (
    HomogeneousSequential,
    RopeScaling,
    default_rope_frequencies,
    apply_rotary_pos_emb,
    TimeEmbedding,
    DiscreteFlowPath,
)

logger = logging.get_logger(__name__)


class FlowMatchingRMSNorm(nn.Module):
    """RMS Normalization layer."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class FlowMatchingMLP(nn.Module):
    """Feed-forward MLP with SwiGLU activation."""
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class FlowMatchingAttention(nn.Module):
    """
    Multi-headed attention with bidirectional attention for Flow Matching.
    
    Key difference from AR attention: is_causal=False for bidirectional context.
    """

    def __init__(self, config: FlowMatchingConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended."
            )
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        
        # CRITICAL: Bidirectional attention for flow matching (not causal like AR)
        self.is_causal = False

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=getattr(config, "attention_bias", False),
        )
        
        # QK normalization
        self.q_norm = FlowMatchingRMSNorm(
            self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.k_norm = FlowMatchingRMSNorm(
            self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Apply QK normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose for attention computation
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat KV for GQA
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention using scaled dot-product attention
        # Note: is_causal=False for bidirectional attention
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,  # CRITICAL: Bidirectional attention for flow matching
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class FlowMatchingRotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    
    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        scaling: Optional[RopeScaling] = None,
    ):
        super().__init__()
        if scaling is None:
            inv_freq = default_rope_frequencies(head_dim, theta=rope_theta)
        else:
            raise NotImplementedError("RoPE scaling is not implemented")
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class FlowMatchingDecoderLayer(nn.Module):
    """Single transformer decoder layer for Flow Matching."""
    
    def __init__(self, config: FlowMatchingConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = FlowMatchingAttention(config=config, layer_idx=layer_idx)
        self.mlp = FlowMatchingMLP(config)
        self.input_layernorm = FlowMatchingRMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.post_attention_layernorm = FlowMatchingRMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Feed Forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class FlowMatchingModel(PreTrainedModel):
    """
    Transformer backbone for Flow Matching Language Model.
    
    This is the core transformer model without the LM head, adapted from
    Qwen3 architecture with bidirectional attention for flow matching.
    """
    
    config_class = FlowMatchingConfig
    _supports_sdpa = False  # Disable SDPA check for custom model

    def __init__(self, config: FlowMatchingConfig):
        # Ensure eager attention is used
        config._attn_implementation = "eager"
        super().__init__(config=config)
        self.vocab_size = config.vocab_size
        self.padding_idx = getattr(config, "pad_token_id", None)
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        
        # Time embedding for t-conditioning (NEW for Flow Matching)
        self.time_embed = TimeEmbedding(
            hidden_size=config.time_embedding_dim,
            embedding_type=config.time_embedding_type,
        )
        
        # Project time embedding to hidden size if dimensions differ
        if config.time_embedding_dim != config.hidden_size:
            self.time_proj = nn.Linear(config.time_embedding_dim, config.hidden_size)
        else:
            self.time_proj = nn.Identity()
        
        # Transformer layers
        self.layers = HomogeneousSequential(
            *[
                FlowMatchingDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        
        # Final normalization
        self.norm = FlowMatchingRMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

        # Rotary position embeddings
        rope_scaling = getattr(config, "rope_scaling", None)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        if rope_scaling is not None:
            rope_scaling = RopeScaling(**rope_scaling)
        self.rotary_emb = FlowMatchingRotaryEmbedding(
            head_dim=head_dim, rope_theta=self.rope_theta, scaling=rope_scaling
        )
        
        self.post_init()

    def _init_weights(self, module: nn.Module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor,
        t: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer backbone.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            t: Time values of shape (batch_size,) for conditioning, values in [0, 1]
            attention_mask: Optional attention mask
            
        Returns:
            Hidden states of shape (batch_size, seq_len, hidden_size)
        """
        # Get token embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length, _ = inputs_embeds.shape

        # Add time embedding conditioning (NEW for Flow Matching)
        if t is not None:
            time_emb = self.time_embed(t)  # (batch_size, time_embedding_dim)
            time_emb = self.time_proj(time_emb)  # (batch_size, hidden_size)
            # Add time embedding to all positions
            inputs_embeds = inputs_embeds + time_emb.unsqueeze(1)

        # Position IDs
        position_ids = torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0).float()

        # For bidirectional attention, no causal mask needed
        # Only apply attention_mask if provided (e.g., for padding)
        causal_mask = None
        if attention_mask is not None:
            # Convert to attention mask format
            causal_mask = attention_mask[:, None, None, :]

        hidden_states = inputs_embeds

        # Compute rotary position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Pass through transformer layers
        hidden_states = self.layers(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class FlowMatchingLanguageModel(PreTrainedModel):
    """
    Flow Matching Language Model for discrete text generation.
    
    This model adapts the Qwen3 architecture for Discrete Flow Matching (DFM),
    replacing the autoregressive objective with a flow-based generative approach.
    
    Key components:
    - FlowMatchingModel: Transformer backbone with bidirectional attention
    - TimeEmbedding: Conditions the model on time t ∈ [0, 1]
    - DiscreteFlowPath: Defines the corruption process for training
    - LM head: Projects hidden states to vocabulary logits
    
    Training:
    - Sample t ~ Uniform(eps, 1)
    - Corrupt input tokens using probability path: x_t = path.sample(x_0, t)
    - Predict clean tokens: logits = model(x_t, t)
    - Compute cross-entropy loss on masked positions
    
    Inference:
    - Start with x_T = mask tokens
    - Iteratively solve discrete ODE from t=1 to t=0
    - Output clean tokens x_0
    """
    
    config_class = FlowMatchingConfig
    base_model_prefix = "model"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["FlowMatchingDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: FlowMatchingConfig):
        super().__init__(config)
        self.config = config
        self.model = FlowMatchingModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mask_token_id = config.mask_token_id
        
        # Initialize probability path for training
        self.probability_path = DiscreteFlowPath(
            vocab_size=config.vocab_size,
            mask_token_id=config.mask_token_id,
            scheduler_type=config.scheduler_type,
            scheduler_n=config.scheduler_n,
            sampling_eps=config.sampling_eps,
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_embeds(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Get input embeddings."""
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        t: Optional[torch.Tensor] = None,
        x_t: Optional[torch.LongTensor] = None,
        src_mask: Optional[torch.BoolTensor] = None,
        training_mode: str = "pretrain",
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Forward pass for Flow Matching Language Model.
        
        Args:
            input_ids: Clean token IDs (x_0) of shape (batch_size, seq_len)
            labels: Target token IDs (same as input_ids for DFM)
            attention_mask: Attention mask
            t: Pre-sampled time values. If None, sampled during training.
            x_t: Pre-corrupted tokens. If None, computed during training.
            src_mask: Source mask for SFT (positions that shouldn't be masked)
            training_mode: "pretrain" or "sft"
            
        Returns:
            Tuple of (logits, loss) where loss is None during inference
        """
        # Inference mode - no corruption, just forward pass
        if not self.training:
            if t is None:
                t = torch.zeros(input_ids.shape[0], device=input_ids.device)
            hidden_states = self.model(input_ids=input_ids, t=t, attention_mask=attention_mask)
            logits = self.lm_head(hidden_states)
            return logits, None

        # Training mode
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Sample time t if not provided
        if t is None:
            sampling_eps = self.config.sampling_eps
            t = (1 - sampling_eps) * torch.rand(batch_size, device=device) + sampling_eps
        
        # Create maskable_mask based on training mode
        if src_mask is not None:
            # SFT mode: don't mask source/instruction tokens
            maskable_mask = ~src_mask
        else:
            # Pretrain mode: all tokens are maskable
            maskable_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Corrupt tokens along probability path if not provided
        if x_t is None:
            x_t = self.probability_path.sample(input_ids, t, maskable_mask=maskable_mask)
        
        # Get loss mask (positions where loss is computed)
        loss_mask = self.probability_path.get_loss_mask(x_t)
        
        # Forward pass with time conditioning
        hidden_states = self.model(input_ids=x_t, t=t, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        # Compute loss on masked positions
        # Shift logits and labels for next-token prediction style
        # Note: Unlike AR, we predict the clean token at each position
        logits_shifted = logits[..., :-1, :].contiguous()
        loss_mask_shifted = loss_mask[..., 1:].contiguous()
        target_ids = input_ids[..., 1:].contiguous()
        
        # Compute cross-entropy loss
        loss = self.loss_fn(
            logits_shifted.reshape(-1, logits_shifted.shape[-1]),
            target_ids.reshape(-1)
        ).reshape(target_ids.shape[0], -1)
        
        # Mask out positions that shouldn't contribute to loss
        loss = loss.masked_fill(~loss_mask_shifted, 0)
        
        # Apply dsigma weighting (like CoDA) to normalize loss across time values
        # dsigma = 1/t weights low-t samples (few masks) more heavily
        # This is crucial for stable training as it balances the loss contribution
        # across different masking ratios
        dsigma = 1.0 / (t + 1e-8)  # 1/t weighting, avoid div by zero
        loss = (dsigma[:, None] * loss).sum() / (batch_size * seq_len)
        
        return logits, loss

    @classmethod
    def from_pretrained_qwen3(
        cls,
        qwen3_model_name_or_path: str,
        config: Optional[FlowMatchingConfig] = None,
        **kwargs
    ):
        """
        Load a FlowMatchingLanguageModel from a pretrained Qwen3 model.
        
        This method:
        1. Loads the Qwen3 weights
        2. Creates a FlowMatchingConfig if not provided
        3. Initializes the Flow Matching model with Qwen3 weights
        4. Adds the time embedding layers (randomly initialized)
        
        Args:
            qwen3_model_name_or_path: Path or HF model name for Qwen3
            config: Optional FlowMatchingConfig. If None, created from Qwen3 config.
            **kwargs: Additional arguments passed to from_pretrained
            
        Returns:
            FlowMatchingLanguageModel with Qwen3 weights
        """
        from transformers import AutoConfig, AutoModelForCausalLM
        
        # Load Qwen3 config and model
        qwen3_config = AutoConfig.from_pretrained(qwen3_model_name_or_path)
        qwen3_model = AutoModelForCausalLM.from_pretrained(qwen3_model_name_or_path, **kwargs)
        
        # Create FlowMatchingConfig from Qwen3 config
        if config is None:
            config = FlowMatchingConfig.from_qwen3_config(qwen3_config)
        
        # Initialize FlowMatchingLanguageModel
        fm_model = cls(config)
        
        # Copy weights from Qwen3 to FlowMatching model
        # This maps the Qwen3 state dict to our model structure
        qwen3_state_dict = qwen3_model.state_dict()
        fm_state_dict = fm_model.state_dict()
        
        # Map Qwen3 weights to FlowMatching model
        # The architecture is the same except for time embeddings
        for name, param in qwen3_state_dict.items():
            # Convert Qwen3 naming to our naming
            new_name = name.replace("model.", "model.")
            if new_name in fm_state_dict:
                if fm_state_dict[new_name].shape == param.shape:
                    fm_state_dict[new_name] = param
                else:
                    logger.warning(f"Shape mismatch for {new_name}: {fm_state_dict[new_name].shape} vs {param.shape}")
        
        fm_model.load_state_dict(fm_state_dict, strict=False)
        
        logger.info(f"Loaded Qwen3 weights from {qwen3_model_name_or_path}")
        logger.info("Time embedding layers are randomly initialized")
        
        return fm_model

