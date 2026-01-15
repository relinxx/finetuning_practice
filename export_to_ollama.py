"""export_to_ollama.py

Purpose:
- Loads the base 4-bit model + a trained LoRA adapter.
- Merges adapters into a full model (HF folder).
- Attempts to export to GGUF (preferred for Ollama).
- Generates an Ollama Modelfile and can optionally call `ollama create`.

Important constraints:
- GGUF export support depends on your installed Unsloth version and platform.
  If GGUF export is not available, this script will still produce a merged HF model
  directory and print instructions for converting with llama.cpp.

Usage:
  python export_to_ollama.py --lora_dir artifacts/lora --out_dir artifacts/ollama --ollama_name mymodel

Then:
  ollama create mymodel -f artifacts/ollama/Modelfile
  ollama run mymodel "Hello!"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import torch


def run(cmd: list[str], *, check: bool = False) -> int:
    print(f"\n[cmd] {' '.join(cmd)}")
    p = subprocess.run(cmd, check=False)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a fine-tuned model to Ollama")
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/llama-3.1-8b-unsloth-bnb-4bit",
        help="Base model id",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="artifacts/lora",
        help="Directory containing trained LoRA adapter (output of train_finetune.py)",
    )
    parser.add_argument("--out_dir", type=str, default="artifacts/ollama")
    parser.add_argument("--ollama_name", type=str, default="finetuned-llama")
    parser.add_argument(
        "--gguf_quant",
        type=str,
        default="q4_k_m",
        help="GGUF quantization method (if supported by exporter).",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="If set, runs `ollama create` after writing Modelfile.",
    )

    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    lora_dir = Path(args.lora_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_dir = out_dir / "merged_hf"
    merged_dir.mkdir(parents=True, exist_ok=True)

    gguf_dir = out_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base + LoRA... (this may use significant VRAM)")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Load adapter weights.
    try:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(lora_dir))
    except Exception as e:
        raise SystemExit(
            "Failed to load LoRA adapter. Ensure --lora_dir points to a PEFT adapter folder.\n"
            f"Error: {repr(e)}"
        )

    # Merge into a full model folder. This can spike memory; if it fails, we still proceed to
    # attempt GGUF export directly (some exporters merge internally).
    print("\nMerging adapter into base model (HF format)...")
    merged_ok = False
    try:
        if hasattr(model, "save_pretrained_merged"):
            model.save_pretrained_merged(
                str(merged_dir),
                tokenizer,
                save_method="merged_16bit",
            )
        elif hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            merged.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
        else:
            raise RuntimeError("Model does not support merging (no save_pretrained_merged or merge_and_unload)")
        merged_ok = True
        print(f"Merged HF model saved to: {merged_dir}")
    except Exception as e:
        print("Merge failed (often due to RAM/VRAM constraints).")
        print("Error:", repr(e))
        print("Continuing with GGUF export attempt anyway.")

    # Try GGUF export via Unsloth if available.
    gguf_path = None
    print("\nAttempting GGUF export...")
    try:
        if not hasattr(model, "save_pretrained_gguf"):
            raise RuntimeError("This Unsloth/PEFT stack does not expose save_pretrained_gguf")

        # Many Unsloth versions provide save_pretrained_gguf.
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=args.gguf_quant,
        )

        # Heuristic: pick the first .gguf file generated.
        candidates = sorted(gguf_dir.glob("*.gguf"))
        if candidates:
            gguf_path = candidates[0]
            print(f"GGUF written to: {gguf_path}")
        else:
            print("GGUF export completed but no .gguf found; check output directory.")

    except Exception as e:
        print("GGUF export not available / failed on this setup.")
        print("Error:", repr(e))

        if merged_ok:
            print(
                "\nFallback option: convert merged HF model to GGUF using llama.cpp (recommended in WSL2/Linux).\n"
                "Example (from a llama.cpp checkout):\n"
                "  python convert_hf_to_gguf.py --outtype f16 --outfile model.gguf "
                f"{merged_dir}\n"
                "Then quantize (example):\n"
                "  ./llama-quantize model.gguf model-q4_k_m.gguf q4_k_m\n"
            )

    # Write Modelfile for Ollama.
    modelfile_path = out_dir / "Modelfile"
    if gguf_path is not None:
        from_line = f"FROM ./{gguf_path.relative_to(out_dir).as_posix()}"
    else:
        # Still write a Modelfile placeholder.
        from_line = "# FROM ./gguf/your-model.gguf"

    modelfile = "\n".join(
        [
            from_line,
            "",
            f"# name: {args.ollama_name}",
            "# description: Fine-tuned with Unsloth QLoRA/LoRA on a custom instruction dataset.",
            "",
            "PARAMETER temperature 0.7",
            "PARAMETER top_p 0.9",
            "PARAMETER num_ctx 2048",
            "",
            "# You can add a SYSTEM prompt if desired:",
            "# SYSTEM You are a helpful assistant.",
            "",
        ]
    )

    modelfile_path.write_text(modelfile, encoding="utf-8")
    print(f"\nWrote Ollama Modelfile: {modelfile_path}")

    # Optionally create the model in Ollama.
    if args.create:
        if not shutil.which("ollama"):
            print("ollama CLI not found on PATH. Install Ollama and ensure `ollama` is available.")
        else:
            run(["ollama", "create", args.ollama_name, "-f", str(modelfile_path)], check=False)
            run(["ollama", "run", args.ollama_name, "Hello!"], check=False)

    # Final note on GPU usage.
    if torch.cuda.is_available():
        print(
            "\nNote: Ollama GPU usage is managed by Ollama itself. "
            "If you see CPU-only inference, verify your Ollama installation supports CUDA on your OS."
        )


if __name__ == "__main__":
    main()
