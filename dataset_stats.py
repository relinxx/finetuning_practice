"""dataset_stats.py

Purpose:
- Analyze a dataset (CSV/JSON/JSONL) and print comprehensive statistics.
- Useful for understanding dataset composition before training.

Usage:
  python dataset_stats.py --data path/to/dataset.jsonl
  python dataset_stats.py --data path/to/dataset.csv --text_column instruction

Notes:
- Computes text lengths, token counts, and basic distributions.
- Helps identify potential issues (e.g., very long sequences).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from config import DEFAULT_MAX_SEQ_LENGTH
from logging_utils import setup_logging
from utils import load_rows

logger = logging.getLogger(__name__)


def compute_text_stats(texts: list[str], name: str) -> dict[str, Any]:
    """Compute basic statistics for a list of texts."""
    if not texts:
        return {"count": 0, "avg_length": 0, "median_length": 0, "max_length": 0}

    lengths = [len(text) for text in texts]
    return {
        "count": len(texts),
        "avg_length": sum(lengths) / len(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2],
        "max_length": max(lengths),
        "min_length": min(lengths),
    }


def compute_token_stats(texts: list[str], tokenizer: Any = None) -> dict[str, Any]:
    """Compute token-level statistics if tokenizer is available."""
    if not tokenizer:
        return {}

    try:
        # Tokenize all texts
        tokens = tokenizer(texts, return_length=True, add_special_tokens=True)
        lengths = tokens["length"]

        return {
            "avg_tokens": sum(lengths) / len(lengths),
            "median_tokens": sorted(lengths)[len(lengths) // 2],
            "max_tokens": max(lengths),
            "min_tokens": min(lengths),
        }
    except Exception as e:
        logger.warning("Could not compute token stats: %s", e)
        return {}


def print_stats(stats: dict[str, Any], title: str) -> None:
    """Pretty print statistics."""
    print(f"\n{title}")
    print("=" * len(title))

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Analyze dataset statistics",
        epilog="""
Examples:
  python dataset_stats.py --data dataset.jsonl
  python dataset_stats.py --data dataset.csv --text_column instruction
  python dataset_stats.py --data dataset.jsonl --load_tokenizer unsloth/llama-3.1-8b-unsloth-bnb-4bit
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset file (CSV/JSON/JSONL)",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="instruction",
        help="Column to analyze for text stats (default: instruction)",
    )
    parser.add_argument(
        "--load_tokenizer",
        type=str,
        help="Optional: HF tokenizer ID to compute token statistics",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Max sequence length for warnings (default: {DEFAULT_MAX_SEQ_LENGTH})",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset file not found: {data_path}")

    logger.info("Loading dataset: %s", data_path)

    try:
        rows = load_rows(str(data_path))
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        raise SystemExit("Dataset loading failed")

    if not rows:
        raise SystemExit("Dataset is empty")

    logger.info("Loaded %d rows", len(rows))

    # Extract text columns
    instructions = []
    inputs = []
    outputs = []

    for row in rows:
        instructions.append(str(row.get("instruction", "")))
        inputs.append(str(row.get("input", "")))
        outputs.append(str(row.get("output", "")))

    # Compute basic text stats
    inst_stats = compute_text_stats(instructions, "instruction")
    input_stats = compute_text_stats(inputs, "input")
    output_stats = compute_text_stats(outputs, "output")

    print_stats(inst_stats, "Instruction Statistics")
    print_stats(input_stats, "Input Statistics")
    print_stats(output_stats, "Output Statistics")

    # Combined text (instruction + input + output)
    combined_texts = []
    for inst, inp, out in zip(instructions, inputs, outputs):
        combined = f"{inst} {inp} {out}".strip()
        combined_texts.append(combined)

    combined_stats = compute_text_stats(combined_texts, "combined")
    print_stats(combined_stats, "Combined Text Statistics")

    # Token statistics if tokenizer provided
    tokenizer = None
    if args.load_tokenizer:
        try:
            from transformers import AutoTokenizer

            logger.info("Loading tokenizer: %s", args.load_tokenizer)
            tokenizer = AutoTokenizer.from_pretrained(args.load_tokenizer)
        except Exception as e:
            logger.error("Failed to load tokenizer: %s", e)

    if tokenizer:
        token_stats = compute_token_stats(combined_texts, tokenizer)
        if token_stats:
            print_stats(token_stats, "Token Statistics")

            # Warnings for sequence length
            if token_stats.get("max_tokens", 0) > args.max_seq_length:
                logger.warning(
                    "Max tokens (%d) exceeds max_seq_length (%d). Consider filtering or increasing limit.",
                    token_stats["max_tokens"],
                    args.max_seq_length,
                )

    # Additional analysis
    print("\nDataset Analysis")
    print("================")

    # Check for empty fields
    empty_instructions = sum(1 for x in instructions if not x.strip())
    empty_inputs = sum(1 for x in inputs if not x.strip())
    empty_outputs = sum(1 for x in outputs if not x.strip())

    print(f"Empty instructions: {empty_instructions}")
    print(f"Empty inputs: {empty_inputs}")
    print(f"Empty outputs: {empty_outputs}")

    # Most common words (basic)
    if instructions:
        all_words = []
        for text in instructions:
            all_words.extend(text.lower().split())

        word_counts = Counter(all_words)
        print(f"\nTop 10 words in instructions: {word_counts.most_common(10)}")

    logger.info("Dataset analysis complete")


if __name__ == "__main__":
    main()