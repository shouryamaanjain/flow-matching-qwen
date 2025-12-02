"""
Data loading utilities for Flow Matching LM pre-training.

Default mixture (updated after dataset compatibility audit):
- dclm-baseline-1.0 (60.17B tokens, text/web)
- OpenWebMath (12.98B tokens, math/web)
- ArXiv Summaries (`ccdv/arxiv-summarization`) (~9B tokens, academic prose)
- Wikipedia EN 20231101 (5.41B tokens, encyclopedia)
- MathPile Text (`zwhe99/mathpile-text`) (~3B tokens, math reasoning)

These, plus Stack v1 code splits, preserve CoDA’s token balance (~180B) without relying
on deprecated dataset scripts.
"""

import os
import ast
import logging
from typing import Optional, Dict, Any, List, Iterator, Union
from dataclasses import dataclass, field
from functools import partial

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, interleave_datasets, IterableDataset as HFIterableDataset
from transformers import PreTrainedTokenizer
from huggingface_hub import HfFolder


logger = logging.getLogger(__name__)

# Improve Hugging Face Hub resiliency for long-running streaming jobs
os.environ.setdefault("HF_ALLOW_CODE_EXECUTION", "1")
os.environ.setdefault("HF_HUB_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_MAX_RETRIES", "10")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1"))


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    path: str
    subset: Optional[str] = None  # Hugging Face config name (e.g., "20231101.en")
    data_dir: Optional[str] = None  # Directory inside the repo (e.g., "data/python" for Stack v1)
    split: str = "train"
    text_column: str = "text"
    weight: float = 1.0  # Sampling weight in mixture
    streaming: bool = True
    trust_remote_code: Optional[bool] = False
    requires_auth: bool = False


# CoDA Pre-training Dataset Configurations
PRETRAIN_DATASETS = [
    DatasetConfig(
        name="dclm-baseline",
        path="mlfoundations/dclm-baseline-1.0",
        text_column="text",
        weight=60.17,  # Based on token count
    ),
    DatasetConfig(
        name="openwebmath",
        path="open-web-math/open-web-math",
        text_column="text",
        weight=12.98,
    ),
    DatasetConfig(
        name="arxiv-summarization",
        path="ccdv/arxiv-summarization",
        text_column="article",
        weight=9.18,
    ),
    DatasetConfig(
        name="wikipedia-en",
        path="wikimedia/wikipedia",
        subset="20231101.en",
        text_column="text",
        weight=5.41,
    ),
    DatasetConfig(
        name="mathpile",
        path="zwhe99/mathpile-text",
        text_column="text",
        weight=3.24,
    ),
]

# ============================================================================
# CODE DATASETS OPTIONS
# ============================================================================
# 
# RECOMMENDED: The Stack v1 (bigcode/the-stack-dedup)
# - Contains actual code content directly in 'content' field
# - No AWS credentials needed, just HuggingFace login
# - ~200B tokens, 3TB deduplicated
# - Usage: load_dataset("bigcode/the-stack-dedup", data_dir="data/python", split="train")
#
# ALTERNATIVE: The Stack v2 (bigcode/the-stack-v2)
# - Only contains file IDs (blob_id), NOT actual content
# - Requires: HF agreement + AWS credentials from Software Heritage/INRIA
# - ~900B tokens, but need S3 download for content
# - Much more complex setup
# ============================================================================

# The Stack v1 (RECOMMENDED - includes actual content, no AWS needed)
STACK_V1_DATASETS = [
    DatasetConfig(
        name="stack-v1-python",
        path="bigcode/the-stack-dedup",
        data_dir="data/python",  # v1 uses data/{language} format
        text_column="content",  # Actual code content included!
        weight=50.75,
        requires_auth=True,
    ),
    DatasetConfig(
        name="stack-v1-javascript",
        path="bigcode/the-stack-dedup",
        data_dir="data/javascript",
        text_column="content",
        weight=20.0,
        requires_auth=True,
    ),
    DatasetConfig(
        name="stack-v1-java",
        path="bigcode/the-stack-dedup",
        data_dir="data/java",
        text_column="content",
        weight=15.0,
        requires_auth=True,
    ),
    DatasetConfig(
        name="stack-v1-cpp",
        path="bigcode/the-stack-dedup",
        data_dir="data/cpp",
        text_column="content",
        weight=10.0,
        requires_auth=True,
    ),
]

# The Stack v2 (requires AWS credentials for content download from S3)
# WARNING: Only contains file IDs, not actual code content!
STACK_V2_DATASETS = [
    DatasetConfig(
        name="stack-v2-python",
        path="bigcode/the-stack-v2",
        subset="Python",
        text_column="blob_id",  # Only file IDs - need S3 download for content
        weight=50.75,
        trust_remote_code=True,
        requires_auth=True,
    ),
    DatasetConfig(
        name="stack-v2-smol",
        path="bigcode/the-stack-v2-train-smol-ids",
        text_column="blob_id",  # Only file IDs - need S3 download for content
        weight=38.22,
        trust_remote_code=True,
        requires_auth=True,
    ),
]


def load_single_dataset(
    config: DatasetConfig,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: int = 8192,
) -> HFIterableDataset:
    """Load a single dataset with optional tokenization."""
    
    logger.info(f"Loading dataset: {config.name} from {config.path}")
    
    try:
        load_kwargs = {
            "split": config.split,
            "streaming": config.streaming,
        }
        if config.subset:
            load_kwargs["name"] = config.subset
        if config.data_dir:
            load_kwargs["data_dir"] = config.data_dir
        if config.trust_remote_code is not None:
            load_kwargs["trust_remote_code"] = config.trust_remote_code
        
        if config.requires_auth:
            token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or HfFolder().get_token()
            if token is None:
                raise EnvironmentError(
                    f"Dataset '{config.path}' is gated and requires a Hugging Face access token.\n"
                    "Please run `huggingface-cli login` (or set HUGGINGFACE_HUB_TOKEN) "
                    "from a shell on this machine, accept the dataset terms in your browser, "
                    "and then rerun training."
                )
            load_kwargs["token"] = token
        
        try:
            ds = load_dataset(
                config.path,
                **load_kwargs,
            )
        except TypeError as e:
            message = str(e)
            if "unexpected keyword argument 'trust_remote_code'" in message:
                load_kwargs.pop("trust_remote_code", None)
                ds = load_dataset(config.path, **load_kwargs)
            elif "unexpected keyword argument 'token'" in message:
                load_kwargs.pop("token", None)
                ds = load_dataset(config.path, **load_kwargs)
            else:
                raise
        
        # Normalize schema to a single "text" column to avoid feature mismatches
        column_names = []
        if hasattr(ds, "column_names") and ds.column_names is not None:
            column_names = list(ds.column_names)
        columns_to_remove = [col for col in column_names if col != config.text_column]
        
        def extract_text(example):
            text = example.get(config.text_column)
            if text is None:
                # Fall back to other common keys
                text = example.get("text") or example.get("content") or ""
            if isinstance(text, (bytes, bytearray)):
                text = text.decode("utf-8", errors="ignore")
            if isinstance(text, str):
                stripped = text.strip()
                if stripped.startswith("b'") or stripped.startswith('b"'):
                    try:
                        text_bytes = ast.literal_eval(stripped)
                        if isinstance(text_bytes, (bytes, bytearray)):
                            text = text_bytes.decode("utf-8", errors="ignore")
                    except Exception:
                        pass
            else:
                text = str(text)
            return {"text": text}
        
        ds = ds.map(
            extract_text,
            remove_columns=columns_to_remove if columns_to_remove else None,
        )
        
        return ds
        
    except Exception as e:
        logger.warning(f"Failed to load dataset {config.name}: {e}")
        return None


def create_pretrain_mixture(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 8192,
    include_stack_v1: bool = False,
    include_stack_v2: bool = False,
    seed: int = 42,
) -> HFIterableDataset:
    """
    Create a weighted mixture of pre-training datasets.
    
    Args:
        tokenizer: Tokenizer for processing text
        max_length: Maximum sequence length
        include_stack_v1: Whether to include The Stack v1 (RECOMMENDED - has content)
        include_stack_v2: Whether to include The Stack v2 (requires AWS credentials)
        seed: Random seed for shuffling
        
    Returns:
        Interleaved dataset with weighted sampling
    """
    datasets_to_load = PRETRAIN_DATASETS.copy()
    
    # RECOMMENDED: Stack v1 - includes actual content, only needs HF login
    if include_stack_v1:
        datasets_to_load.extend(STACK_V1_DATASETS)
        logger.info(
            "Including The Stack v1 datasets (RECOMMENDED):\n"
            "- Contains actual code content directly\n"
            "- No AWS credentials needed\n"
            "- Just login with: huggingface-cli login\n"
            "See: https://huggingface.co/datasets/bigcode/the-stack-dedup"
        )
    
    # Stack v2 - requires AWS for content download (NOT RECOMMENDED unless you have credentials)
    if include_stack_v2:
        datasets_to_load.extend(STACK_V2_DATASETS)
        logger.warning(
            "Including The Stack v2 datasets (requires AWS credentials):\n"
            "WARNING: Stack v2 only contains file IDs, NOT actual code content!\n"
            "You need AWS credentials to download content from Software Heritage S3.\n"
            "Consider using --include_stack_v1 instead (has actual content).\n"
            "See: https://huggingface.co/datasets/bigcode/the-stack-v2"
        )
    
    # Load all datasets
    loaded_datasets = []
    dataset_names = []
    weights = []
    
    for config in datasets_to_load:
        ds = load_single_dataset(config, tokenizer, max_length)
        if ds is not None:
            loaded_datasets.append(ds)
            dataset_names.append(config.name)
            weights.append(config.weight)
            logger.info(f"✓ Loaded {config.name} with weight {config.weight}")
        else:
            logger.warning(f"✗ Skipped {config.name}")
    
    if not loaded_datasets:
        raise ValueError("No datasets were loaded successfully!")
    
    # Normalize weights to probabilities
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    
    logger.info(f"Creating mixture with {len(loaded_datasets)} datasets")
    logger.info(f"Sampling probabilities: {dict(zip(dataset_names, probabilities))}")
    
    # Interleave datasets with weighted sampling
    mixed_dataset = interleave_datasets(
        loaded_datasets,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy="all_exhausted",  # Continue until all datasets are exhausted
    )
    
    return mixed_dataset


class PretrainDataset(IterableDataset):
    """
    PyTorch IterableDataset wrapper for pre-training.
    
    Handles tokenization, padding, and batching.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 8192,
        include_stack_v1: bool = False,
        include_stack_v2: bool = False,
        seed: int = 42,
        num_workers: int = 0,
        worker_id: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_stack_v1 = include_stack_v1
        self.include_stack_v2 = include_stack_v2
        self.seed = seed
        self.num_workers = num_workers
        self.worker_id = worker_id
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self._dataset = None
    
    def _get_dataset(self):
        """Lazy load dataset."""
        if self._dataset is None:
            self._dataset = create_pretrain_mixture(
                self.tokenizer,
                self.max_length,
                self.include_stack_v1,
                self.include_stack_v2,
                self.seed + self.worker_id,
            )
        return self._dataset
    
    def _tokenize_and_chunk(self, examples: Dict[str, List]) -> Iterator[Dict[str, torch.Tensor]]:
        """Tokenize text and create fixed-length chunks."""
        
        # Get text column (could be 'text', 'content', etc.)
        texts = examples.get("text", examples.get("content", []))
        if not texts:
            return
        
        # Tokenize
        for text in texts:
            if not text or not isinstance(text, str):
                continue
                
            tokens = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
            
            # Create chunks of max_length
            for i in range(0, len(tokens), self.max_length):
                chunk_tokens = tokens[i:i + self.max_length]
                
                # Skip very short chunks
                if len(chunk_tokens) < 128:
                    continue
                
                valid_len = min(len(chunk_tokens), self.max_length)
                
                # Pad if necessary
                if valid_len < self.max_length:
                    chunk_tokens = chunk_tokens + [self.tokenizer.pad_token_id] * (self.max_length - valid_len)
                
                attention = [True] * valid_len + [False] * (self.max_length - valid_len)
                
                yield {
                    "input_ids": torch.tensor(chunk_tokens, dtype=torch.long),
                    "attention_mask": torch.tensor(attention, dtype=torch.bool),
                }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over tokenized chunks."""
        dataset = self._get_dataset()
        
        # Handle multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split dataset across workers
            dataset = dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )
        
        # Process in batches for efficiency
        batch_size = 100
        batch = []
        
        for example in dataset:
            batch.append(example)
            
            if len(batch) >= batch_size:
                yield from self._tokenize_and_chunk({"text": [b.get("text", b.get("content", "")) for b in batch]})
                batch = []
        
        # Process remaining
        if batch:
            yield from self._tokenize_and_chunk({"text": [b.get("text", b.get("content", "")) for b in batch]})


def create_pretrain_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 8192,
    include_stack_v1: bool = False,
    include_stack_v2: bool = False,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """
    Create DataLoader for pre-training.
    
    Args:
        tokenizer: Tokenizer for processing text
        batch_size: Batch size per device
        max_length: Maximum sequence length
        include_stack_v1: Whether to include The Stack v1 (RECOMMENDED - has actual content)
        include_stack_v2: Whether to include The Stack v2 (requires AWS for content)
        num_workers: Number of data loading workers
        seed: Random seed
        
    Returns:
        DataLoader for pre-training
    """
    dataset = PretrainDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        include_stack_v1=include_stack_v1,
        include_stack_v2=include_stack_v2,
        seed=seed,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# Simple test
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    logging.basicConfig(level=logging.INFO)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    
    # Create dataloader (without Stack v2 for testing)
    dataloader = create_pretrain_dataloader(
        tokenizer=tokenizer,
        batch_size=2,
        max_length=2048,
        include_stack_v2=False,
        num_workers=0,
    )
    
    # Test iteration
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        
        if i >= 2:
            break
    
    print("Data loading test passed!")

