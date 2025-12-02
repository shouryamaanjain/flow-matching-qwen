#!/usr/bin/env python3
"""
Inference script for Flow Matching Language Model

Loads a checkpoint and runs generation using the FlowMatchingSampler.

Usage:
    # Basic generation from checkpoint-1500
    python scripts/inference.py --checkpoint checkpoint-1500 --prompt "def fibonacci("

    # With custom generation params
    python scripts/inference.py \
        --checkpoint checkpoint-1500 \
        --prompt "The quick brown fox" \
        --max_length 128 \
        --num_steps 64 \
        --temperature 0.8 \
        --top_p 0.9
    
    # Infill mode (fill between prefix and suffix)
    python scripts/inference.py \
        --checkpoint checkpoint-1500 \
        --prefix "def add(a, b):" \
        --suffix "return result" \
        --infill_length 32

    # Interactive mode
    python scripts/inference.py --checkpoint checkpoint-1500 --interactive
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer
from torch.distributed.checkpoint import load_state_dict
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from FlowMatchingLM.model_config import FlowMatchingConfig
from FlowMatchingLM.modeling_fm import FlowMatchingLanguageModel
from FlowMatchingLM.generation_utils import (
    FlowMatchingSampler,
    FlowMatchingGenerationConfig,
)


def consolidate_fsdp_checkpoint(checkpoint_dir: Path, output_path: Path) -> Path:
    """
    Consolidate FSDP sharded checkpoint into a single file.
    
    Args:
        checkpoint_dir: Path to checkpoint directory with pytorch_model_fsdp_0/
        output_path: Path to save consolidated checkpoint
        
    Returns:
        Path to consolidated checkpoint
    """
    if output_path.exists():
        print(f"Using existing consolidated checkpoint: {output_path}")
        return output_path
    
    fsdp_dir = checkpoint_dir / "pytorch_model_fsdp_0"
    if not fsdp_dir.exists():
        raise FileNotFoundError(f"FSDP checkpoint not found at {fsdp_dir}")
    
    print(f"Consolidating FSDP checkpoint from {fsdp_dir}...")
    print("This may take a minute...")
    
    # Use PyTorch's distributed checkpoint utilities
    dcp_to_torch_save(str(fsdp_dir), str(output_path))
    
    print(f"Consolidated checkpoint saved to {output_path}")
    return output_path


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[FlowMatchingLanguageModel, AutoTokenizer]:
    """
    Load model and tokenizer from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory or consolidated .pt file
        device: Device to load model on
        dtype: Model dtype
        
    Returns:
        Tuple of (model, tokenizer)
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Determine checkpoint location
    if checkpoint_path.is_dir():
        # FSDP sharded checkpoint - need to consolidate
        consolidated_path = checkpoint_path / "model_consolidated.pt"
        consolidate_fsdp_checkpoint(checkpoint_path, consolidated_path)
        state_dict_path = consolidated_path
    else:
        state_dict_path = checkpoint_path
    
    print(f"Loading model from {state_dict_path}...")
    
    # Load state dict
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    
    # FIX: Unwrap nested FSDP checkpoint structure if present
    # FSDP checkpoints can wrap the entire state_dict under a 'model' key
    if 'model' in state_dict and len(state_dict) == 1 and isinstance(state_dict['model'], dict):
        print("Unwrapping nested FSDP checkpoint structure...")
        state_dict = state_dict['model']
    
    # Create config (Qwen3-1.7B architecture)
    config = FlowMatchingConfig(
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
        mask_token_id=151669,
        num_sampling_steps=32,
    )
    
    # Create model
    print("Initializing model...")
    model = FlowMatchingLanguageModel(config)
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")
    
    # Move to device and dtype
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    print(f"Model loaded successfully on {device} with dtype {dtype}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Load tokenizer (prefer local cache to avoid network calls)
    print("Loading tokenizer...")
    script_dir = Path(__file__).parent.parent
    local_tokenizer_path = script_dir / "tokenizer_cache"
    
    if local_tokenizer_path.exists():
        print(f"Using local tokenizer from {local_tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(local_tokenizer_path),
            trust_remote_code=True,
            padding_side="left",
            local_files_only=True,
        )
    else:
        print("Local tokenizer not found, downloading from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-1.7B",
            trust_remote_code=True,
            padding_side="left",
        )
    
    # Add mask token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate(
    model: FlowMatchingLanguageModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 256,
    num_steps: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_confidence: bool = True,
    verbose: bool = True,
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: FlowMatchingLanguageModel
        tokenizer: Tokenizer
        prompt: Input prompt
        max_length: Maximum total sequence length
        num_steps: Number of ODE solver steps
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Top-p filtering
        use_confidence: Use confidence-based unmasking
        verbose: Print progress
        
    Returns:
        Generated text
    """
    device = next(model.parameters()).device
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]
    
    if prompt_len >= max_length:
        print(f"Warning: prompt length ({prompt_len}) >= max_length ({max_length})")
        max_length = prompt_len + 64
    
    # Create sampler
    gen_config = FlowMatchingGenerationConfig(
        num_steps=num_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        mask_token_id=model.config.mask_token_id,
    )
    sampler = FlowMatchingSampler(model, gen_config)
    
    # Generate
    tokens_to_generate = max_length - prompt_len
    if verbose:
        print(f"\nGenerating {tokens_to_generate} tokens with {num_steps} steps...")
    
    # Time the generation
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.perf_counter()
    
    output_ids = sampler.generate(
        prompt_ids=prompt_ids,
        max_length=max_length,
        num_steps=num_steps,
        use_confidence=use_confidence,
        verbose=verbose,
    )
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed_time = time.perf_counter() - start_time
    
    # Calculate tokens/sec
    tokens_per_sec = tokens_to_generate / elapsed_time if elapsed_time > 0 else 0
    
    if verbose:
        print(f"\n⏱️  Generation time: {elapsed_time:.3f}s | Tokens/sec: {tokens_per_sec:.1f}")
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


def infill(
    model: FlowMatchingLanguageModel,
    tokenizer: AutoTokenizer,
    prefix: str,
    suffix: str,
    infill_length: int = 32,
    num_steps: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    verbose: bool = True,
) -> str:
    """
    Infill text between prefix and suffix.
    
    Args:
        model: FlowMatchingLanguageModel
        tokenizer: Tokenizer  
        prefix: Text before the infill region
        suffix: Text after the infill region
        infill_length: Number of tokens to generate in the middle
        num_steps: Number of ODE solver steps
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Top-p filtering
        verbose: Print progress
        
    Returns:
        Complete text with infilled middle
    """
    device = next(model.parameters()).device
    
    # Tokenize
    prefix_ids = tokenizer.encode(prefix, return_tensors="pt", add_special_tokens=False).to(device)
    suffix_ids = tokenizer.encode(suffix, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Create sampler
    gen_config = FlowMatchingGenerationConfig(
        num_steps=num_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        mask_token_id=model.config.mask_token_id,
    )
    sampler = FlowMatchingSampler(model, gen_config)
    
    # Generate infill
    if verbose:
        print(f"\nInfilling {infill_length} tokens between prefix and suffix...")
    
    output_ids = sampler.infill(
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        infill_length=infill_length,
        num_steps=num_steps,
    )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


def interactive_mode(
    model: FlowMatchingLanguageModel,
    tokenizer: AutoTokenizer,
    max_length: int = 256,
    num_steps: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
):
    """
    Interactive generation mode.
    """
    print("\n" + "=" * 60)
    print("Flow Matching LM Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  /quit or /exit - Exit interactive mode")
    print("  /steps N       - Set number of solver steps")
    print("  /temp T        - Set temperature")
    print("  /length L      - Set max generation length")
    print("  /help          - Show this help")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not prompt:
            continue
        
        # Handle commands
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()
            
            if cmd in ["/quit", "/exit"]:
                print("Exiting...")
                break
            elif cmd == "/help":
                print("Commands: /quit, /exit, /steps N, /temp T, /length L, /help")
            elif cmd == "/steps" and len(parts) > 1:
                num_steps = int(parts[1])
                print(f"Solver steps set to {num_steps}")
            elif cmd == "/temp" and len(parts) > 1:
                temperature = float(parts[1])
                print(f"Temperature set to {temperature}")
            elif cmd == "/length" and len(parts) > 1:
                max_length = int(parts[1])
                print(f"Max length set to {max_length}")
            else:
                print(f"Unknown command: {cmd}")
            continue
        
        # Generate
        try:
            output = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=max_length,
                num_steps=num_steps,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                verbose=True,
            )
            print(f"\n{'─' * 60}")
            print(output)
            print(f"{'─' * 60}")
        except Exception as e:
            print(f"Error during generation: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Flow Matching LM Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint-1500",
        help="Checkpoint name (e.g., checkpoint-1500) or full path",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/flow_matching_qwen_1.7b",
        help="Base directory containing checkpoints",
    )
    
    # Generation mode
    parser.add_argument("--prompt", type=str, help="Input prompt for generation")
    parser.add_argument("--prefix", type=str, help="Prefix for infill mode")
    parser.add_argument("--suffix", type=str, help="Suffix for infill mode")
    parser.add_argument("--infill_length", type=int, default=32, help="Infill length in tokens")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--num_steps", type=int, default=32, help="Number of ODE solver steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p filtering")
    parser.add_argument("--no_confidence", action="store_true", help="Disable confidence-based unmasking")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if os.path.isabs(args.checkpoint) or os.path.exists(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Available checkpoints in {args.checkpoint_dir}:")
        if os.path.exists(args.checkpoint_dir):
            for item in sorted(os.listdir(args.checkpoint_dir)):
                if item.startswith("checkpoint"):
                    print(f"  - {item}")
        sys.exit(1)
    
    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model, tokenizer = load_model(checkpoint_path, args.device, dtype)
    
    # Run generation
    if args.interactive:
        interactive_mode(
            model=model,
            tokenizer=tokenizer,
            max_length=args.max_length,
            num_steps=args.num_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    elif args.prefix and args.suffix:
        # Infill mode
        output = infill(
            model=model,
            tokenizer=tokenizer,
            prefix=args.prefix,
            suffix=args.suffix,
            infill_length=args.infill_length,
            num_steps=args.num_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"\n{'=' * 60}")
        print("Infilled text:")
        print(f"{'=' * 60}")
        print(output)
    elif args.prompt:
        # Standard generation
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            num_steps=args.num_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_confidence=not args.no_confidence,
        )
        print(f"\n{'=' * 60}")
        print("Generated text:")
        print(f"{'=' * 60}")
        print(output)
    else:
        print("Error: Must specify --prompt, --prefix/--suffix, or --interactive")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

