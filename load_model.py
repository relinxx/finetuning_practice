"""load_model.py

Purpose:
- Loads a 4-bit quantized base model via Unsloth.
- Attaches a LoRA adapter configuration (QLoRA-style fine-tuning on 4-bit weights).
- Prints VRAM usage checkpoints to help keep an RTX 5070-class GPU (~8–12GB) safe.

Usage:
  python load_model.py
  python load_model.py --model unsloth/llama-3.1-8b-unsloth-bnb-4bit --max_seq_length 2048

Notes:
- Requires `unsloth` and a CUDA-capable PyTorch installation.
- If VRAM is near the limit, reduce `--max_seq_length` or choose a smaller model.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess

import torch


def print_vram(tag: str) -> None:
    """Print torch-reported VRAM stats."""
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return

    idx = 0
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / (1024**3)
    alloc_gb = torch.cuda.memory_allocated(idx) / (1024**3)
    res_gb = torch.cuda.memory_reserved(idx) / (1024**3)

    print(f"[{tag}] GPU: {props.name}")
    print(f"[{tag}] Total VRAM: {total_gb:.2f} GB")
    print(f"[{tag}] torch allocated: {alloc_gb:.2f} GB")
    print(f"[{tag}] torch reserved:  {res_gb:.2f} GB")

    # Simple heuristic warning.
    if res_gb / max(total_gb, 1e-6) > 0.90:
        print(f"[{tag}] WARNING: VRAM reserved >90% of total. Consider lowering seq length/batch size.")


def try_nvidia_smi() -> None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return
    subprocess.run(
        [
            exe,
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a 4-bit Unsloth model and print VRAM usage")
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/llama-3.1-8b-unsloth-bnb-4bit",
        help="HF model id (4-bit recommended for 8–12GB GPUs).",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    args = parser.parse_args()

    # Recommended env vars to reduce fragmentation and noisy tokenizer threading.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try_nvidia_smi()

    print_vram("before_load")

    from unsloth import FastLanguageModel  # imported after env vars

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # let Unsloth pick bf16/fp16
        load_in_4bit=True,
    )

    print_vram("after_base_load")

    # Attach LoRA adapters (parameter-efficient fine-tuning). This is the QLoRA pattern when
    # the base weights are loaded in 4-bit.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # helps fit longer context in limited VRAM
        random_state=3407,
    )

    print_vram("after_lora_attach")

    # Print trainable params to confirm PEFT.
    trainable, total = 0, 0
    for p in model.parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    pct = 100.0 * trainable / max(total, 1)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    # Quick smoke generation (kept tiny to avoid surprise VRAM spikes).
    prompt = "Write one short sentence about GPU memory." 
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.7,
        )
    print("\nSample output:\n", tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
