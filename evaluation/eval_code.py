#!/usr/bin/env python3
"""
Code generation evaluation script for Flow Matching LM.

This script evaluates the model on:
- HumanEval
- MBPP
- EvalPlus (enhanced variants)

Usage:
    python eval_code.py --model_path ./checkpoints/checkpoint-210000 --benchmark humaneval
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from FlowMatchingLM import FlowMatchingConfig, FlowMatchingLanguageModel
from FlowMatchingLM.generation_utils import FlowMatchingSampler, FlowMatchingGenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_humaneval() -> List[Dict]:
    """Load HumanEval benchmark."""
    try:
        from human_eval.data import read_problems
        return list(read_problems().values())
    except ImportError:
        logger.warning("human_eval not installed. Install with: pip install human-eval")
        return []


def load_mbpp() -> List[Dict]:
    """Load MBPP benchmark."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("mbpp", split="test")
        return list(dataset)
    except Exception as e:
        logger.warning(f"Could not load MBPP: {e}")
        return []


def generate_code(
    model: FlowMatchingLanguageModel,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    num_steps: int = 32,
    temperature: float = 0.0,
    num_samples: int = 1,
) -> List[str]:
    """
    Generate code completions for a prompt.
    
    Args:
        model: FlowMatchingLanguageModel
        tokenizer: Tokenizer
        prompt: Code prompt to complete
        max_length: Maximum generation length
        num_steps: Number of flow matching steps
        temperature: Sampling temperature
        num_samples: Number of samples to generate
        
    Returns:
        List of generated code completions
    """
    device = next(model.parameters()).device
    
    # Tokenize prompt
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    
    # Create sampler
    config = FlowMatchingGenerationConfig(
        num_steps=num_steps,
        temperature=temperature if temperature > 0 else 0.1,  # Avoid division by zero
        top_k=40 if temperature > 0 else None,
        top_p=0.95 if temperature > 0 else None,
        mask_token_id=model.config.mask_token_id,
    )
    sampler = FlowMatchingSampler(model, config)
    
    completions = []
    for _ in range(num_samples):
        # Expand prompt for batch
        prompt_batch = prompt_ids.expand(1, -1)
        
        # Generate
        output_ids = sampler.generate(
            prompt_ids=prompt_batch,
            max_length=max_length,
            num_steps=num_steps,
            use_confidence=True,
        )
        
        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract completion (remove prompt)
        completion = output_text[len(tokenizer.decode(prompt_ids[0], skip_special_tokens=True)):]
        completions.append(completion)
    
    return completions


def evaluate_humaneval(
    model: FlowMatchingLanguageModel,
    tokenizer,
    num_samples_per_task: int = 1,
    max_length: int = 512,
    num_steps: int = 32,
    temperature: float = 0.0,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Evaluate on HumanEval benchmark.
    
    Args:
        model: FlowMatchingLanguageModel
        tokenizer: Tokenizer
        num_samples_per_task: Number of samples per task (for pass@k)
        max_length: Maximum generation length
        num_steps: Number of flow matching steps
        temperature: Sampling temperature
        output_path: Path to save results
        
    Returns:
        Evaluation results
    """
    problems = load_humaneval()
    if not problems:
        return {"error": "HumanEval not available"}
    
    logger.info(f"Evaluating on {len(problems)} HumanEval problems")
    
    samples = []
    for problem in tqdm(problems, desc="Generating"):
        prompt = problem["prompt"]
        task_id = problem["task_id"]
        
        completions = generate_code(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            num_steps=num_steps,
            temperature=temperature,
            num_samples=num_samples_per_task,
        )
        
        for completion in completions:
            samples.append({
                "task_id": task_id,
                "completion": completion,
            })
    
    # Save samples
    if output_path:
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        logger.info(f"Saved samples to {output_path}")
    
    # Run evaluation
    try:
        from human_eval.evaluation import evaluate_functional_correctness
        results = evaluate_functional_correctness(output_path or "humaneval_samples.jsonl")
        logger.info(f"HumanEval Results: {results}")
        return results
    except Exception as e:
        logger.warning(f"Could not run evaluation: {e}")
        return {"samples": len(samples)}


def evaluate_mbpp(
    model: FlowMatchingLanguageModel,
    tokenizer,
    num_samples_per_task: int = 1,
    max_length: int = 512,
    num_steps: int = 32,
    temperature: float = 0.0,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Evaluate on MBPP benchmark.
    """
    problems = load_mbpp()
    if not problems:
        return {"error": "MBPP not available"}
    
    logger.info(f"Evaluating on {len(problems)} MBPP problems")
    
    samples = []
    for problem in tqdm(problems, desc="Generating"):
        prompt = problem["text"] + "\n\n" + problem["code"].split("\n")[0]
        task_id = problem.get("task_id", len(samples))
        
        completions = generate_code(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=max_length,
            num_steps=num_steps,
            temperature=temperature,
            num_samples=num_samples_per_task,
        )
        
        for completion in completions:
            samples.append({
                "task_id": task_id,
                "prompt": prompt,
                "completion": completion,
            })
    
    if output_path:
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        logger.info(f"Saved samples to {output_path}")
    
    return {"samples": len(samples)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flow Matching LM on code generation")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["humaneval", "mbpp", "all"],
        default="humaneval",
        help="Benchmark to evaluate",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per task (for pass@k)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=32,
        help="Number of flow matching steps",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = FlowMatchingLanguageModel.from_pretrained(args.model_path)
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Run evaluation
    results = {}
    
    if args.benchmark in ["humaneval", "all"]:
        humaneval_results = evaluate_humaneval(
            model=model,
            tokenizer=tokenizer,
            num_samples_per_task=args.num_samples,
            max_length=args.max_length,
            num_steps=args.num_steps,
            temperature=args.temperature,
            output_path=str(output_dir / "humaneval_samples.jsonl"),
        )
        results["humaneval"] = humaneval_results
    
    if args.benchmark in ["mbpp", "all"]:
        mbpp_results = evaluate_mbpp(
            model=model,
            tokenizer=tokenizer,
            num_samples_per_task=args.num_samples,
            max_length=args.max_length,
            num_steps=args.num_steps,
            temperature=args.temperature,
            output_path=str(output_dir / "mbpp_samples.jsonl"),
        )
        results["mbpp"] = mbpp_results
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    for benchmark, res in results.items():
        print(f"\n{benchmark.upper()}:")
        for key, value in res.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

