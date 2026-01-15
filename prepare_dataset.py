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
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def load_rows(path: Path) -> list[dict[str, str]]:
    """Load examples from CSV/JSON/JSONL into a list of dicts."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(
                    {
                        "instruction": (r.get("instruction") or "").strip(),
                        "input": (r.get("input") or "").strip( ),
                        "output": (r.get("output") or "").strip(),
                    }
                )
        return rows

    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    rows.append(
                        {
                            "instruction": str(obj.get("instruction", "")).strip(),
                            "input": str(obj.get("input", "")).strip(),
                            "output": str(obj.get("output", "")).strip(),
                        }
                    )
            return rows

        # .json
        obj: Any
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict) and "data" in obj:
            obj = obj["data"]

        if not isinstance(obj, list):
            raise ValueError("JSON must be a list[dict] or a dict with key 'data' = list[dict].")

        rows = []
        for r in obj:
            rows.append(
                {
                    "instruction": str(r.get("instruction", "")).strip(),
                    "input": str(r.get("input", "")).strip(),
                    "output": str(r.get("output", "")).strip(),
                }
            )
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
    parser = argparse.ArgumentParser(description="Prepare/tokenize dataset for Unsloth fine-tuning")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV/JSON/JSONL")
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/llama-3.1-8b-unsloth-bnb-4bit",
        help="Model id used to load the tokenizer.",
    )
    parser.add_argument("--out_dir", type=str, default="artifacts/dataset")
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(data_path)
    rows = [r for r in rows if r["instruction"] and r["output"]]
    if not rows:
        raise SystemExit("No valid rows found. Ensure 'instruction' and 'output' are non-empty.")

    print(f"Loaded {len(rows)} examples")

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
    print(f"Saved tokenized dataset to: {out_dir}")
    print("Fields:", dsd["train"].column_names)


if __name__ == "__main__":
    main()
