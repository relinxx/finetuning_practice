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
import logging
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import TrainingArguments

from config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_STEPS,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOGGING_STEPS,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LORA_DIR,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL_ID,
    DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    DEFAULT_SAVE_STEPS,
    DEFAULT_WARMUP_STEPS,
)
from logging_utils import setup_logging

logger = logging.getLogger(__name__)


def print_vram(tag: str) -> None:
    if not torch.cuda.is_available():
        logger.info("[%s] CUDA not available", tag)
        return
    idx = 0
    props = torch.cuda.get_device_properties(idx)
    total = props.total_memory / (1024**3)
    alloc = torch.cuda.memory_allocated(idx) / (1024**3)
    res = torch.cuda.memory_reserved(idx) / (1024**3)
    logger.info("[%s] VRAM total=%.2fGB alloc=%.2fGB reserved=%.2fGB", tag, total, alloc, res)
    if res / max(total, 1e-6) > 0.92:
        logger.warning("[%s] reserved VRAM >92%% - reduce seq length/batch/grad_accum.", tag)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Fine-tune a 4-bit model with Unsloth QLoRA",
        epilog="""
Examples:
  python train_finetune.py --dataset_dir artifacts/dataset --out_dir artifacts/lora
  python train_finetune.py --dataset_dir artifacts/dataset --out_dir artifacts/lora --do_eval --merge_out artifacts/merged_model
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"4-bit base model ID. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory with tokenized dataset (from prepare_dataset.py). Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_LORA_DIR,
        help=f"Output directory for LoRA adapter. Default: {DEFAULT_LORA_DIR}",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Maximum sequence length. Default: {DEFAULT_MAX_SEQ_LENGTH}",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate. Default: {DEFAULT_LEARNING_RATE}",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs. Default: {DEFAULT_EPOCHS}",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
        help=f"Batch size per device. Default: {DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE}",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help=f"Gradient accumulation steps. Default: {DEFAULT_GRADIENT_ACCUMULATION_STEPS}",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help=f"Warmup steps. Default: {DEFAULT_WARMUP_STEPS}",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=DEFAULT_LOGGING_STEPS,
        help=f"Logging interval. Default: {DEFAULT_LOGGING_STEPS}",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=DEFAULT_SAVE_STEPS,
        help=f"Checkpoint save interval. Default: {DEFAULT_SAVE_STEPS}",
    )

    parser.add_argument("--do_eval", action="store_true", help="Enable validation loss logging if validation split exists.")
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=DEFAULT_EVAL_STEPS,
        help=f"Evaluation interval. Default: {DEFAULT_EVAL_STEPS}",
    )

    # LoRA parameters
    parser.add_argument(
        "--lora_r",
        type=int,
        default=DEFAULT_LORA_R,
        help=f"LoRA rank. Default: {DEFAULT_LORA_R}",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=DEFAULT_LORA_ALPHA,
        help=f"LoRA alpha. Default: {DEFAULT_LORA_ALPHA}",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=DEFAULT_LORA_DROPOUT,
        help=f"LoRA dropout. Default: {DEFAULT_LORA_DROPOUT}",
    )

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

    logger.info("Train size: %s", len(train_ds))
    if eval_ds is not None:
        logger.info("Val size: %s", len(eval_ds))

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

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving LoRA adapter + tokenizer...")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print_vram("after_train")

    if args.merge_out:
        merge_out = Path(args.merge_out)
        merge_out.mkdir(parents=True, exist_ok=True)

        # This creates a full model folder (still HF format). You can convert this to GGUF later.
        # Merging can temporarily increase RAM/VRAM usage; do it only if you have headroom.
        logger.info("Merging LoRA into base model and saving full model...")
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

            logger.info("Saved merged model to: %s", merge_out)
        except Exception as e:
            logger.warning(
                "Merge failed (often due to memory limits or unsupported method on your setup)."
            )
            logger.error("Error: %s", repr(e))
            logger.info("You can still export by loading base + adapter and merging during export.")


if __name__ == "__main__":
    main()
