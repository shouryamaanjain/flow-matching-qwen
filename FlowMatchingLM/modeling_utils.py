"""
Modeling utilities for Flow Matching Language Model

This module contains:
- TimeEmbedding: Sinusoidal time embeddings for t-conditioning
- DiscreteFlowPath: Probability path for discrete token corruption
- Utility functions adapted from CoDA
"""

import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class HomogeneousSequential(nn.Sequential):
    """
    HomogenousSequential is a sequential container that requires all child modules
    to be of the same type and have matching input/output shapes.
    
    Adapted from CoDA's implementation.
    """

    repeated_layer: type

    def __init__(self, *args: nn.Module) -> None:
        super().__init__(*args)
        types = set(type(module) for module in args)
        assert len(types) == 1, f"All modules must be of the same type. Got {types}"
        self.repeated_layer = types.pop()

    def forward(self, *input, **broadcasted_inputs):
        for module in self:
            input = module(*splat(input), **broadcasted_inputs)
        return input


def splat(input):
    """Helper function for HomogeneousSequential."""
    if not isinstance(input, (list, tuple)):
        input = (input,)
    return input


@dataclass(kw_only=True)
class RopeScaling:
    """RoPE scaling parameters for extended context."""
    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_context_len: int = 8192


def default_rope_frequencies(
    head_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Compute default RoPE frequencies."""
    return 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    position_ids: Optional[torch.Tensor] = None, 
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for discrete flow matching.
    
    This module generates positional-style embeddings for the time variable t ∈ [0, 1].
    The embeddings are used to condition the model on the current position along
    the probability path during training and sampling.
    
    Args:
        hidden_size: Dimension of the output embeddings
        max_period: Maximum period for sinusoidal functions (default: 10000)
        embedding_type: Type of embedding ("sinusoidal" or "learned")
    
    The sinusoidal embeddings follow the formulation:
        emb[2i] = sin(t * 10000^(-2i/d))
        emb[2i+1] = cos(t * 10000^(-2i/d))
    
    This is similar to positional embeddings in transformers but applied to
    the continuous time variable t instead of discrete positions.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        max_period: int = 10000,
        embedding_type: str = "sinusoidal"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_period = max_period
        self.embedding_type = embedding_type
        
        if embedding_type == "sinusoidal":
            # Precompute the frequency bands
            half_dim = hidden_size // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
            )
            self.register_buffer("freqs", freqs)
            
            # Optional projection layer for scaling
            self.proj = nn.Linear(hidden_size, hidden_size)
            
        elif embedding_type == "learned":
            # Learned embedding with MLP
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generate time embeddings.
        
        Args:
            t: Time values of shape (batch_size,) or (batch_size, 1), values in [0, 1]
            
        Returns:
            Time embeddings of shape (batch_size, hidden_size)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch_size, 1)
        
        if self.embedding_type == "sinusoidal":
            # Scale t to get different frequencies
            # t: (batch_size, 1), freqs: (half_dim,)
            args = t * self.freqs.unsqueeze(0) * self.max_period
            
            # Compute sin and cos embeddings
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            
            # Handle odd hidden_size
            if self.hidden_size % 2 == 1:
                emb = F.pad(emb, (0, 1), mode="constant", value=0)
            
            # Cast to match projection layer dtype (handles bf16 inference)
            emb = emb.to(self.proj.weight.dtype)
            
            # Project to final dimension
            emb = self.proj(emb)
            
        else:  # learned
            emb = self.mlp(t)
        
        return emb


class DiscreteFlowPath:
    """
    Discrete probability path for Flow Matching on tokens.
    
    This class implements the forward corruption process for discrete flow matching,
    replacing CoDA's transition() function. It defines a probability path that
    interpolates between the data distribution (clean tokens) and a noise
    distribution (masked tokens) as t goes from 0 to 1.
    
    The corruption probability follows a scheduler:
    - At t=0: x_t = x_0 (clean data)
    - At t=1: x_t = mask_token_id (fully corrupted)
    - At intermediate t: tokens are masked with probability p(t) from scheduler
    
    Args:
        vocab_size: Size of the vocabulary
        mask_token_id: Token ID to use for masking
        scheduler_type: Type of scheduler ("polynomial_convex", "linear", "cosine")
        scheduler_n: Polynomial degree for polynomial_convex scheduler
        sampling_eps: Minimum t value to avoid numerical issues
    """
    
    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        scheduler_type: str = "polynomial_convex",
        scheduler_n: float = 1.0,
        sampling_eps: float = 1e-3,
    ):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.scheduler_type = scheduler_type
        self.scheduler_n = scheduler_n
        self.sampling_eps = sampling_eps
    
    def get_corruption_prob(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the corruption probability for a given time t.
        
        The corruption probability defines how likely each token is to be
        replaced with the mask token at time t.
        
        Args:
            t: Time values of shape (batch_size,) or (batch_size, 1), values in [0, 1]
            
        Returns:
            Corruption probabilities of same shape as t
        """
        if self.scheduler_type == "polynomial_convex":
            # Polynomial convex scheduler: p(t) = t^n
            # At t=0, p=0 (no corruption); at t=1, p=1 (full corruption)
            corruption_prob = t ** self.scheduler_n
            
        elif self.scheduler_type == "linear":
            # Linear scheduler: p(t) = t
            corruption_prob = t
            
        elif self.scheduler_type == "cosine":
            # Cosine scheduler: p(t) = 1 - cos(t * pi/2)
            # Smoother transition, slower at endpoints
            corruption_prob = 1 - torch.cos(t * math.pi / 2)
            
        else:
            raise ValueError(f"Unknown scheduler_type: {self.scheduler_type}")
        
        return corruption_prob
    
    def sample(
        self, 
        x_0: torch.Tensor, 
        t: torch.Tensor,
        maskable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_t along the probability path given clean tokens x_0 and time t.
        
        This replaces CoDA's transition() function with a DFM-style probability path.
        
        Args:
            x_0: Clean token IDs of shape (batch_size, seq_len)
            t: Time values of shape (batch_size,), values in [0, 1]
            maskable_mask: Optional boolean mask of shape (batch_size, seq_len)
                          indicating which positions can be masked. If None,
                          all positions are maskable.
                          
        Returns:
            x_t: Corrupted tokens of shape (batch_size, seq_len)
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device
        
        # Get corruption probability for each sample
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch_size, 1)
        corruption_prob = self.get_corruption_prob(t)  # (batch_size, 1)
        
        # Sample which tokens to corrupt
        # Each token is independently masked with probability corruption_prob
        random_vals = torch.rand(batch_size, seq_len, device=device)
        should_mask = random_vals < corruption_prob
        
        # Apply maskable_mask if provided
        if maskable_mask is not None:
            should_mask = should_mask & maskable_mask
        
        # Create corrupted tokens
        x_t = torch.where(should_mask, self.mask_token_id, x_0)
        
        return x_t
    
    def get_target_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the target posterior distribution p(x_0 | x_t, t).
        
        For discrete flow matching with mask-based corruption:
        - If x_t[i] != mask_token_id: p(x_0[i] = x_t[i] | x_t, t) = 1
        - If x_t[i] == mask_token_id: p(x_0[i] | x_t, t) is the true data token
        
        This is used to compute the training loss.
        
        Args:
            x_0: Clean token IDs of shape (batch_size, seq_len)
            x_t: Corrupted token IDs of shape (batch_size, seq_len)
            t: Time values of shape (batch_size,)
            
        Returns:
            Target token IDs of shape (batch_size, seq_len) - just x_0 for x-prediction
        """
        # For x-prediction (probability denoiser), the target is simply x_0
        # The model learns to predict the clean token at each position
        return x_0
    
    def get_loss_mask(
        self,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the mask indicating which positions should contribute to the loss.
        
        In DFM, we typically compute loss on all positions, but we can also
        focus on masked positions like in CoDA.
        
        Args:
            x_t: Corrupted token IDs of shape (batch_size, seq_len)
            
        Returns:
            Loss mask of shape (batch_size, seq_len), True where loss is computed
        """
        # Compute loss on masked positions (like CoDA)
        # This focuses learning on the denoising task
        return x_t == self.mask_token_id


class SchedulerWrapper:
    """
    Wrapper to provide a consistent interface for different schedulers.
    
    This can wrap Meta's flow_matching schedulers or our custom implementation.
    """
    
    def __init__(self, scheduler_type: str = "polynomial_convex", scheduler_n: float = 1.0):
        self.scheduler_type = scheduler_type
        self.scheduler_n = scheduler_n
        
        # Try to use Meta's flow_matching library if available
        self._meta_scheduler = None
        try:
            from flow_matching.path.scheduler import PolynomialConvexScheduler
            if scheduler_type == "polynomial_convex":
                self._meta_scheduler = PolynomialConvexScheduler(n=scheduler_n)
        except ImportError:
            pass
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Get corruption probability at time t."""
        if self._meta_scheduler is not None:
            return self._meta_scheduler(t)
        
        # Fallback implementation
        if self.scheduler_type == "polynomial_convex":
            return t ** self.scheduler_n
        elif self.scheduler_type == "linear":
            return t
        elif self.scheduler_type == "cosine":
            return 1 - torch.cos(t * math.pi / 2)
        else:
            raise ValueError(f"Unknown scheduler_type: {self.scheduler_type}")


def prefix_input_ids(
    input_ids: torch.Tensor, 
    maskable_mask: torch.Tensor, 
    apply_prefix: torch.Tensor
) -> torch.Tensor:
    """
    Apply prefix masking - make the prefix unmaskable.
    
    Adapted from CoDA's implementation for curriculum learning.
    
    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)
        maskable_mask: Boolean mask of shape (batch_size, seq_len)
        apply_prefix: Boolean tensor of shape (batch_size,) indicating which
                     samples should have prefix applied
                     
    Returns:
        Updated maskable_mask with prefix positions set to False
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Generate random prefix lengths for all batch items
    prefix_lengths = torch.randint(1, seq_len, (batch_size,), device=device)
    
    # Create position indices
    position_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Create prefix mask: True where position < prefix_length
    prefix_mask = position_indices < prefix_lengths.unsqueeze(1)
    
    # Apply prefix masking only to selected samples
    maskable_mask = maskable_mask & ~(apply_prefix.unsqueeze(1) & prefix_mask)
    
    return maskable_mask


def truncate_input_ids(
    input_ids: torch.Tensor, 
    apply_truncate: torch.Tensor, 
    pad_token_id: int
) -> torch.Tensor:
    """
    Truncate input at random position and fill with pad token.
    
    Adapted from CoDA's implementation for curriculum learning.
    
    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)
        apply_truncate: Boolean tensor of shape (batch_size,) indicating which
                       samples should be truncated
        pad_token_id: Token ID to use for padding
        
    Returns:
        Truncated input_ids with suffix replaced by pad_token_id
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Generate random truncation positions
    truncate_positions = torch.randint(1, seq_len, (batch_size,), device=device)
    
    # Create position indices
    position_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Create truncate mask: True where position >= truncate_position
    truncate_mask = position_indices >= truncate_positions.unsqueeze(1)
    
    # Apply truncation only to selected samples
    input_ids = torch.where(
        apply_truncate.unsqueeze(1) & truncate_mask, 
        pad_token_id, 
        input_ids
    )
    
    return input_ids

