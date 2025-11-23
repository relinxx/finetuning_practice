"""prepare_dataset.py

Purpose:
- Loads training data from CSV, JSON, or JSONL with fields:
    instruction, input, output
- Formats examples into a consistent instruction-following text.
- Tokenizes the text with the model tokenizer.
- Saves a Hugging Face dataset to disk for fast reuse during training.

Why this matters for VRAM:
- Longer sequences increase VRAM (roughly proportional to context length).
- Tokenization + truncation here ensures training won't unexpectedly exceed max length.

Usage:
  python prepare_dataset.py --data data/train.jsonl --out_dir artifacts/dataset
  python prepare_dataset.py --data data/train.csv --out_dir artifacts/dataset --val_ratio 0.05

Input formats:
- CSV: columns instruction,input,output (input can be empty)
- JSON: either a list[dict] or a dict with key "data" holding list[dict]
- JSONL: one JSON object per line
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL_ID,
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
)
from logging_utils import setup_logging

logger = logging.getLogger(__name__)


def load_rows(path: Path) -> list[dict[str, str]]:
    """Load examples from CSV/JSON/JSONL into a list of dicts with validation.

    Validates that each row has non-empty 'instruction' and 'output' fields.
    Fails fast with clear error messages on invalid data.
    """
    if not path.exists():
        raise SystemExit(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_num, r in enumerate(reader, start=2):  # CSV rows start at 1, but header is 1
                try:
                    instruction = (r.get("instruction") or "").strip()
                    input_text = (r.get("input") or "").strip()
                    output = (r.get("output") or "").strip()

                    if not instruction:
                        raise ValueError(f"Row {row_num}: 'instruction' is empty or missing")
                    if not output:
                        raise ValueError(f"Row {row_num}: 'output' is empty or missing")

                    rows.append({
                        "instruction": instruction,
                        "input": input_text,
                        "output": output,
                    })
                except ValueError as e:
                    raise SystemExit(f"Validation error in {path}: {e}")
        return rows

    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        instruction = str(obj.get("instruction", "")).strip()
                        input_text = str(obj.get("input", "")).strip()
                        output = str(obj.get("output", "")).strip()

                        if not instruction:
                            raise ValueError(f"Line {line_num}: 'instruction' is empty or missing")
                        if not output:
                            raise ValueError(f"Line {line_num}: 'output' is empty or missing")

                        rows.append({
                            "instruction": instruction,
                            "input": input_text,
                            "output": output,
                        })
                    except (json.JSONDecodeError, ValueError) as e:
                        raise SystemExit(f"Validation error in {path} at line {line_num}: {e}")
            return rows

        # .json
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid JSON in {path}: {e}")

        if isinstance(obj, dict) and "data" in obj:
            obj = obj["data"]

        if not isinstance(obj, list):
            raise SystemExit(f"JSON must be a list[dict] or a dict with key 'data' = list[dict] in {path}.")

        rows = []
        for idx, r in enumerate(obj, start=1):
            try:
                instruction = str(r.get("instruction", "")).strip()
                input_text = str(r.get("input", "")).strip()
                output = str(r.get("output", "")).strip()

                if not instruction:
                    raise ValueError(f"Item {idx}: 'instruction' is empty or missing")
                if not output:
                    raise ValueError(f"Item {idx}: 'output' is empty or missing")

                rows.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                })
            except (TypeError, ValueError) as e:
                raise SystemExit(f"Validation error in {path} at item {idx}: {e}")
        return rows

    raise ValueError(f"Unsupported file type: {suffix} (use .csv, .json, or .jsonl)")


def build_text(instruction: str, user_input: str, output: str) -> str:
    """Create a simple instruction-following sample.

    We keep this template minimal and stable; changing templates changes training behavior.
    """
    instruction = instruction.strip()
    user_input = user_input.strip()
    output = output.strip()

    # If 'input' is empty, keep it out of the prompt to reduce tokens.
    if user_input:
        prompt = f"### Instruction\n{instruction}\n\n### Input\n{user_input}\n\n### Response\n"
    else:
        prompt = f"### Instruction\n{instruction}\n\n### Response\n"
    return prompt + output


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Prepare/tokenize dataset for Unsloth fine-tuning",
        epilog="""
Examples:
  python prepare_dataset.py --data data/train.jsonl --out_dir artifacts/dataset
  python prepare_dataset.py --data data/train.csv --out_dir artifacts/dataset --val_ratio 0.05 --max_seq_length 1024
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input dataset file (CSV, JSON, or JSONL). Must contain 'instruction', 'input' (optional), and 'output' fields.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Model ID for tokenizer loading. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"Output directory for tokenized dataset. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Fraction of data for validation split (0.0 to 1.0). Default: {DEFAULT_VAL_RATIO}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for shuffling/splitting. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Maximum sequence length for tokenization. Default: {DEFAULT_MAX_SEQ_LENGTH}",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(data_path)
    if not rows:
        raise SystemExit("No valid rows found after validation. Ensure data has 'instruction' and 'output' fields.")

    logger.info("Loaded %s examples", len(rows))

    # Tokenization does not need the model or GPU. Avoid loading weights to prevent VRAM spikes.
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def to_text(example: dict[str, str]) -> dict[str, str]:
        return {
            "text": build_text(example["instruction"], example.get("input", ""), example["output"])
        }

    ds = Dataset.from_list(rows).map(to_text, remove_columns=["instruction", "input", "output"])

    # Tokenize (with truncation) to control context length and memory.
    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tok = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        # SFT-style labels: predict next token over full sequence.
        tok["labels"] = tok["input_ids"].copy()
        return tok

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])

    if 0.0 < args.val_ratio < 1.0:
        split = ds_tok.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
        dsd = DatasetDict({"train": split["train"], "validation": split["test"]})
    else:
        dsd = DatasetDict({"train": ds_tok})

    dsd.save_to_disk(str(out_dir))
    logger.info("Saved tokenized dataset to: %s", out_dir)
    logger.info("Fields: %s", dsd["train"].column_names)


if __name__ == "__main__":
    main()
