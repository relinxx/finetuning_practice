"""Shared utilities for the project."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

import torch


class Example(TypedDict):
    instruction: str
    input: str
    output: str


def run(cmd: list[str], *, check: bool = True) -> int:
    """Run a command and stream output to the console."""
    print(f"\n[cmd] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def print_vram(tag: str) -> None:
    """Print torch-reported VRAM stats."""
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return

    idx = 0
    props = torch.cuda.get_device_properties(idx)
    total = props.total_memory / (1024**3)
    alloc = torch.cuda.memory_allocated(idx) / (1024**3)
    res = torch.cuda.memory_reserved(idx) / (1024**3)
    print(f"[{tag}] VRAM total={total:.2f}GB alloc={alloc:.2f}GB reserved={res:.2f}GB")
    if res / max(total, 1e-6) > 0.92:
        print(f"[{tag}] WARNING: reserved VRAM >92% — reduce seq length/batch/grad_accum.")


def try_nvidia_smi() -> None:
    """Print basic GPU + VRAM info via nvidia-smi if available."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        print("nvidia-smi not found on PATH (this is OK if NVIDIA drivers/tools are not installed).")
        return

    # Query only what we need to keep output short and readable.
    run(
        [
            exe,
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )

def post_install_gpu_check(python_exe: Path) -> None:
    """Import torch and print CUDA/GPU details, including VRAM."""
    code = r"""
import os
import torch

print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    idx = 0
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / (1024**3)
    print('GPU:', props.name)
    print('Total VRAM (GB):', round(total_gb, 2))

    # Force a small allocation to ensure CUDA context is created.
    x = torch.empty((1024, 1024), device='cuda')
    del x
    torch.cuda.synchronize()

    allocated_gb = torch.cuda.memory_allocated(idx) / (1024**3)
    reserved_gb = torch.cuda.memory_reserved(idx) / (1024**3)
    print('VRAM allocated by torch (GB):', round(allocated_gb, 2))
    print('VRAM reserved by torch (GB):', round(reserved_gb, 2))

    # Helpful environment variables for memory fragmentation and tokenizer behavior.
    print('\nRecommended env vars:')
    print('  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128')
    print('  TOKENIZERS_PARALLELISM=false')
else:
    print('No CUDA device detected by torch. Check NVIDIA driver + CUDA-compatible torch wheel.')
"""
    run([str(python_exe), "-c", code], check=False)

def load_rows(path: Path) -> list[Example]:
    """Load examples from CSV/JSON/JSONL into a list of dicts with validation.

    Validates that each row has non-empty 'instruction' and 'output' fields.
    Fails fast with clear error messages on invalid data.
    """
    if not path.exists():
        raise SystemExit(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        import csv

        rows: list[Example] = []
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