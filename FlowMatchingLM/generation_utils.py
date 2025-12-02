"""
Generation utilities for Flow Matching Language Model

This module implements sampling/generation for discrete flow matching,
including the FlowMatchingSampler with discrete ODE solver.

Key differences from CoDA's generation:
- Uses discrete ODE solver instead of iterative denoising
- Can achieve high quality in fewer steps
- Supports various sampling strategies
"""

from typing import Optional, List, Union, Callable
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FlowMatchingGenerationConfig:
    """
    Configuration for Flow Matching generation.
    
    Args:
        num_steps: Number of discrete ODE solver steps (default: 32)
        temperature: Sampling temperature (default: 1.0)
        top_k: Top-k filtering (default: None, no filtering)
        top_p: Top-p (nucleus) filtering (default: None, no filtering)
        mask_token_id: Token ID for mask (required)
        solver_type: Type of discrete solver ("euler", "heun", "adaptive")
        guidance_scale: Classifier-free guidance scale (default: 1.0, no guidance)
    """
    num_steps: int = 32
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    mask_token_id: int = 151669
    solver_type: str = "euler"
    guidance_scale: float = 1.0


class FlowMatchingSampler:
    """
    Sampler for Flow Matching Language Model using discrete ODE solver.
    
    This replaces CoDA's iterative denoising with a more principled
    discrete flow matching approach that can achieve high quality
    in fewer steps.
    
    The sampling process:
    1. Initialize x_T with mask tokens (t=1)
    2. Iteratively solve: x_{t-dt} = x_t + dt * v(x_t, t)
       where v is derived from the model's predicted posterior
    3. Return x_0 (clean tokens)
    
    Args:
        model: FlowMatchingLanguageModel
        config: FlowMatchingGenerationConfig
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[FlowMatchingGenerationConfig] = None,
    ):
        self.model = model
        self.config = config or FlowMatchingGenerationConfig(
            mask_token_id=model.config.mask_token_id
        )
        self.mask_token_id = self.config.mask_token_id
        self.vocab_size = model.config.vocab_size
    
    def _get_time_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """
        Generate time schedule from t=1 to t=0.
        
        Returns time points for discrete ODE solver steps.
        """
        # Linear schedule from 1 to 0
        return torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    
    def _apply_top_k_filtering(
        self, 
        logits: torch.Tensor, 
        top_k: int
    ) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        if top_k <= 0:
            return logits
        
        # Remove tokens with probability less than the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
        return logits
    
    def _apply_top_p_filtering(
        self, 
        logits: torch.Tensor, 
        top_p: float
    ) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        if top_p >= 1.0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")
        return logits
    
    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample tokens from logits with optional filtering.
        
        Args:
            logits: Logits of shape (batch_size, seq_len, vocab_size)
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            
        Returns:
            Sampled tokens of shape (batch_size, seq_len)
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply filtering
        if top_k is not None and top_k > 0:
            logits = self._apply_top_k_filtering(logits, top_k)
        if top_p is not None and top_p < 1.0:
            logits = self._apply_top_p_filtering(logits, top_p)
        
        # Sample from categorical distribution
        probs = F.softmax(logits, dim=-1)
        
        # Reshape for sampling
        batch_size, seq_len, vocab_size = probs.shape
        probs_flat = probs.view(-1, vocab_size)
        
        # Sample
        tokens_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        tokens = tokens_flat.view(batch_size, seq_len)
        
        return tokens
    
    def _euler_step(
        self,
        x_t: torch.Tensor,
        t: float,
        dt: float,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform one Euler step of the discrete ODE solver.
        
        The discrete ODE for flow matching:
        x_{t-dt} = x_t + dt * v(x_t, t)
        
        For discrete tokens, we interpret this as:
        1. Get model prediction p(x_0 | x_t, t)
        2. Sample new tokens based on mixing x_t and predicted x_0
        
        Args:
            x_t: Current tokens at time t
            t: Current time
            dt: Time step (negative for t -> 0)
            prompt_mask: Positions to keep fixed (prompt)
            
        Returns:
            x_{t+dt}: Updated tokens
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device
        
        # Get model prediction
        t_tensor = torch.full((batch_size,), t, device=device)
        
        with torch.no_grad():
            logits, _ = self.model(input_ids=x_t, t=t_tensor)
        
        # Compute probability of unmasking at this step
        # As t decreases, we gradually unmask more tokens
        unmask_prob = abs(dt)  # Probability of unmasking at this step
        
        # Get mask of currently masked positions
        is_masked = x_t == self.mask_token_id
        
        # Sample which masked positions to unmask
        unmask_these = is_masked & (torch.rand_like(is_masked.float()) < unmask_prob)
        
        # Sample new tokens for positions being unmasked
        new_tokens = self._sample_from_logits(
            logits,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )
        
        # Update x_t: unmask selected positions
        x_new = torch.where(unmask_these, new_tokens, x_t)
        
        # Keep prompt positions fixed
        if prompt_mask is not None:
            x_new = torch.where(prompt_mask, x_t, x_new)
        
        return x_new
    
    def _confidence_based_step(
        self,
        x_t: torch.Tensor,
        t: float,
        dt: float,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform one step using confidence-based unmasking (like MaskGIT).
        
        Instead of random unmasking, unmask tokens with highest confidence.
        Optimized: fully vectorized, no Python loops or .item() calls.
        
        Args:
            x_t: Current tokens at time t
            t: Current time
            dt: Time step
            prompt_mask: Positions to keep fixed
            
        Returns:
            Updated tokens
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device
        
        # Get model prediction
        t_tensor = torch.full((batch_size,), t, device=device)
        
        with torch.no_grad():
            logits, _ = self.model(input_ids=x_t, t=t_tensor)
        
        # Get probabilities and max confidence
        probs = F.softmax(logits / self.config.temperature, dim=-1)
        max_probs, predicted_tokens = probs.max(dim=-1)
        
        # Get mask of currently masked positions
        is_masked = x_t == self.mask_token_id
        
        # Calculate number of tokens to unmask this step (vectorized)
        num_masked = is_masked.sum(dim=-1, keepdim=True).float()
        num_to_unmask = (num_masked * abs(dt)).clamp(min=1)
        
        # Set confidence of non-masked positions to -inf
        confidence = max_probs.clone()
        confidence[~is_masked] = float("-inf")
        
        # OPTIMIZED: Vectorized threshold-based unmasking (no Python loop!)
        # Sort by confidence descending
        sorted_conf, sorted_idx = confidence.sort(dim=-1, descending=True)
        
        # Create position indices [0, 1, 2, ...] and compare to threshold
        position_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Unmask positions where rank < num_to_unmask AND is currently masked
        unmask_mask = (position_idx < num_to_unmask) & (sorted_conf > float("-inf"))
        
        # Scatter back to original positions
        unmask_mask_original = torch.zeros_like(is_masked)
        unmask_mask_original.scatter_(1, sorted_idx, unmask_mask)
        
        # Unmask selected positions
        x_new = torch.where(unmask_mask_original, predicted_tokens, x_t)
        
        # Keep prompt positions fixed
        if prompt_mask is not None:
            x_new = torch.where(prompt_mask, x_t, x_new)
        
        return x_new
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Optional[torch.Tensor] = None,
        max_length: int = 256,
        batch_size: int = 1,
        num_steps: Optional[int] = None,
        use_confidence: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Generate tokens using discrete flow matching.
        
        Args:
            prompt_ids: Optional prompt tokens of shape (batch_size, prompt_len)
            max_length: Maximum sequence length to generate
            batch_size: Batch size (used if prompt_ids is None)
            num_steps: Number of solver steps (overrides config)
            use_confidence: Whether to use confidence-based unmasking
            verbose: Whether to print progress
            
        Returns:
            Generated tokens of shape (batch_size, max_length)
        """
        device = next(self.model.parameters()).device
        num_steps = num_steps or self.config.num_steps
        
        # Initialize with prompt or all mask tokens
        if prompt_ids is not None:
            batch_size = prompt_ids.shape[0]
            prompt_len = prompt_ids.shape[1]
            
            # Create sequence with prompt + mask tokens
            x_t = torch.full((batch_size, max_length), self.mask_token_id, device=device)
            x_t[:, :prompt_len] = prompt_ids
            
            # Create prompt mask (True for prompt positions)
            prompt_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
            prompt_mask[:, :prompt_len] = True
        else:
            # All positions are mask tokens
            x_t = torch.full((batch_size, max_length), self.mask_token_id, device=device)
            prompt_mask = None
        
        # Get time schedule
        time_schedule = self._get_time_schedule(num_steps, device)
        
        # Select step function
        step_fn = self._confidence_based_step if use_confidence else self._euler_step
        
        # Pre-compute time schedule on CPU to avoid repeated GPU syncs
        time_list = time_schedule.tolist()
        
        # Iteratively solve discrete ODE from t=1 to t=0
        for i in range(num_steps):
            t = time_list[i]
            t_next = time_list[i + 1]
            dt = t_next - t  # Negative, moving towards 0
            
            x_t = step_fn(x_t, t, dt, prompt_mask)
            
            if verbose and (i + 1) % 10 == 0:
                num_masked = (x_t == self.mask_token_id).sum().item()
                print(f"Step {i + 1}/{num_steps}, t={t:.3f}, masked tokens: {num_masked}")
        
        # Final step: replace any remaining mask tokens
        if (x_t == self.mask_token_id).any():
            t_tensor = torch.zeros(batch_size, device=device)
            logits, _ = self.model(input_ids=x_t, t=t_tensor)
            final_tokens = self._sample_from_logits(
                logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )
            x_t = torch.where(x_t == self.mask_token_id, final_tokens, x_t)
        
        return x_t
    
    @torch.no_grad()
    def generate_with_prefix(
        self,
        prefix_ids: torch.Tensor,
        suffix_length: int,
        num_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate tokens to continue a prefix.
        
        Args:
            prefix_ids: Prefix tokens of shape (batch_size, prefix_len)
            suffix_length: Number of tokens to generate after prefix
            num_steps: Number of solver steps
            **kwargs: Additional arguments for generate()
            
        Returns:
            Generated tokens including prefix
        """
        max_length = prefix_ids.shape[1] + suffix_length
        return self.generate(
            prompt_ids=prefix_ids,
            max_length=max_length,
            num_steps=num_steps,
            **kwargs
        )
    
    @torch.no_grad()
    def infill(
        self,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        infill_length: int,
        num_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate tokens to fill between prefix and suffix.
        
        Args:
            prefix_ids: Prefix tokens of shape (batch_size, prefix_len)
            suffix_ids: Suffix tokens of shape (batch_size, suffix_len)
            infill_length: Number of tokens to generate in the middle
            num_steps: Number of solver steps
            **kwargs: Additional arguments
            
        Returns:
            Full sequence with infilled middle
        """
        batch_size = prefix_ids.shape[0]
        device = prefix_ids.device
        prefix_len = prefix_ids.shape[1]
        suffix_len = suffix_ids.shape[1]
        total_length = prefix_len + infill_length + suffix_len
        
        # Create sequence with prefix, masks, suffix
        x_t = torch.full((batch_size, total_length), self.mask_token_id, device=device)
        x_t[:, :prefix_len] = prefix_ids
        x_t[:, prefix_len + infill_length:] = suffix_ids
        
        # Create mask for fixed positions (prefix and suffix)
        prompt_mask = torch.ones((batch_size, total_length), dtype=torch.bool, device=device)
        prompt_mask[:, prefix_len:prefix_len + infill_length] = False
        
        num_steps = num_steps or self.config.num_steps
        time_schedule = self._get_time_schedule(num_steps, device)
        
        # Use confidence-based stepping for infilling
        for i in range(num_steps):
            t = time_schedule[i].item()
            t_next = time_schedule[i + 1].item()
            dt = t_next - t
            
            x_t = self._confidence_based_step(x_t, t, dt, prompt_mask)
        
        return x_t


class DLMGenerationMixin:
    """
    Mixin class for generation utilities, compatible with HuggingFace interface.
    
    This provides a generate() method that uses FlowMatchingSampler internally.
    """
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_length: int = 256,
        num_steps: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate tokens using flow matching.
        
        Args:
            input_ids: Optional prompt tokens
            max_length: Maximum sequence length
            num_steps: Number of solver steps
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            **kwargs: Additional arguments
            
        Returns:
            Generated tokens
        """
        config = FlowMatchingGenerationConfig(
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            mask_token_id=self.config.mask_token_id,
        )
        
        sampler = FlowMatchingSampler(self, config)
        
        return sampler.generate(
            prompt_ids=input_ids,
            max_length=max_length,
            num_steps=num_steps,
            **kwargs
        )

