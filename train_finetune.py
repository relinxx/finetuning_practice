"""train_finetune.py

Purpose:
- Fine-tune a 4-bit base model using LoRA adapters (QLoRA-style) via Unsloth + TRL SFTTrainer.
- Logs training loss (and optional validation loss).
- Saves the LoRA adapter to disk, and optionally saves a merged full-precision HF model directory.

Designed for ~8–12GB VRAM GPUs (like an RTX 5070-class):
- 4-bit base weights
- LoRA adapters only (tiny trainable fraction)
- gradient checkpointing
- conservative batch size + gradient accumulation

Usage:
  python train_finetune.py --dataset_dir artifacts/dataset --out_dir artifacts/lora

  # With evaluation:
  python train_finetune.py --dataset_dir artifacts/dataset --out_dir artifacts/lora --do_eval

  # Optionally merge adapters into a full model folder (useful before GGUF export):
  python train_finetune.py --dataset_dir artifacts/dataset --out_dir artifacts/lora --merge_out artifacts/merged_model
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import TrainingArguments


def print_vram(tag: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a 4-bit model with Unsloth QLoRA")
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/llama-3.1-8b-unsloth-bnb-4bit",
        help="4-bit base model id",
    )
    parser.add_argument("--dataset_dir", type=str, default="artifacts/dataset")
    parser.add_argument("--out_dir", type=str, default="artifacts/lora")
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # Safe defaults for 8–12GB VRAM.
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)

    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=200)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument(
        "--merge_out",
        type=str,
        default="",
        help="If set, saves a merged full HF model directory here (before GGUF export).",
    )

    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dsd = load_from_disk(str(dataset_dir))
    train_ds = dsd["train"]
    eval_ds = dsd.get("validation") if args.do_eval and "validation" in dsd else None

    print("Train size:", len(train_ds))
    if eval_ds is not None:
        print("Val size:", len(eval_ds))

    from unsloth import FastLanguageModel, is_bfloat16_supported

    print_vram("before_load")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

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
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print_vram("after_model")

    # TRL SFTTrainer works well for instruction-following fine-tuning.
    from trl import SFTTrainer

    fp16 = not is_bfloat16_supported()
    bf16 = is_bfloat16_supported()

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],  # keep it simple; add "wandb" if you want
        fp16=fp16,
        bf16=bf16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps,
        do_eval=eval_ds is not None,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
    )

    # Dataset is already tokenized (input_ids/attention_mask/labels). We pass tokenizer so
    # the trainer can handle padding/collation.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        max_seq_length=args.max_seq_length,
        packing=False,  # safer for small VRAM; set True for higher throughput if you have headroom
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving LoRA adapter + tokenizer...")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print_vram("after_train")

    if args.merge_out:
        merge_out = Path(args.merge_out)
        merge_out.mkdir(parents=True, exist_ok=True)

        # This creates a full model folder (still HF format). You can convert this to GGUF later.
        # Merging can temporarily increase RAM/VRAM usage; do it only if you have headroom.
        print("\nMerging LoRA into base model and saving full model...")
        try:
            if hasattr(model, "save_pretrained_merged"):
                model.save_pretrained_merged(
                    str(merge_out),
                    tokenizer,
                    save_method="merged_16bit",  # safe default
                )
            else:
                # Fallback for environments where Unsloth helper is unavailable.
                if hasattr(model, "merge_and_unload"):
                    merged = model.merge_and_unload()
                    merged.save_pretrained(str(merge_out))
                    tokenizer.save_pretrained(str(merge_out))
                else:
                    raise RuntimeError("Model does not support merging (no save_pretrained_merged or merge_and_unload)")

            print(f"Saved merged model to: {merge_out}")
        except Exception as e:
            print("Merge failed (often due to memory limits or unsupported method on your setup).")
            print("Error:", repr(e))
            print("You can still export by loading base + adapter and merging during export.")


if __name__ == "__main__":
    main()
