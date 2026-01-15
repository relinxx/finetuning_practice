"""setup_env.py

Purpose:
- Creates an optional virtual environment.
- Installs required packages for Unsloth + PyTorch + Hugging Face + Ollama.
- Runs post-install GPU checks and prints VRAM.

Notes:
- Unsloth is primarily supported on Linux (including WSL2). It may not work on native
  Windows Python environments.
- RTX 5070 is assumed to be an NVIDIA CUDA-capable GPU; actual support depends on
  your driver/CUDA stack.

Usage examples:
  python setup_env.py --venv .venv
  .\.venv\Scripts\Activate.ps1
  python setup_env.py --post_check_only

  # For PyTorch CUDA wheels (example: cu121):
  python setup_env.py --torch_index_url https://download.pytorch.org/whl/cu121
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, check: bool = True) -> int:
    """Run a command and stream output to the console."""
    print(f"\n[cmd] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def is_wsl() -> bool:
    """Best-effort detection for WSL."""
    if platform.system().lower() != "linux":
        return False
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def ensure_venv(venv_dir: Path) -> Path:
    """Create a venv if missing; returns python executable inside the venv."""
    venv_dir = venv_dir.resolve()
    if not venv_dir.exists():
        print(f"Creating venv at: {venv_dir}")
        run([sys.executable, "-m", "venv", str(venv_dir)])

    if platform.system().lower() == "windows":
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        python_exe = venv_dir / "bin" / "python"

    if not python_exe.exists():
        raise SystemExit(f"Could not find venv python at: {python_exe}")
    return python_exe


def pip_install(python_exe: Path, packages: list[str], *, extra_args: list[str] | None = None) -> None:
    """Install packages into the interpreter referenced by python_exe."""
    extra_args = extra_args or []
    run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(python_exe), "-m", "pip", "install", *extra_args, *packages])


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup environment for Unsloth fine-tuning + Ollama")
    parser.add_argument(
        "--venv",
        type=str,
        default="",
        help="Optional venv directory to create/use (recommended).",
    )
    parser.add_argument(
        "--torch_index_url",
        type=str,
        default="https://download.pytorch.org/whl/cu121",
        help="PyTorch wheel index URL for CUDA builds (example: cu121).",
    )
    parser.add_argument(
        "--skip_unsloth",
        action="store_true",
        help="Skip installing unsloth (useful on native Windows where it may fail).",
    )
    parser.add_argument(
        "--post_check_only",
        action="store_true",
        help="Only run GPU checks (assumes packages already installed).",
    )

    args = parser.parse_args()

    print("System:", platform.platform())
    if platform.system().lower() == "windows":
        print(
            "\nImportant: Unsloth is typically used on Linux/WSL2. "
            "If installs fail, use WSL2 Ubuntu and run these scripts there."
        )
    elif is_wsl():
        print("Detected WSL: good option for CUDA-based fine-tuning on Windows.")

    print("\nGPU check via nvidia-smi (if available):")
    try_nvidia_smi()

    if args.venv:
        python_exe = ensure_venv(Path(args.venv))
        print(f"\nUsing venv python: {python_exe}")
    else:
        python_exe = Path(sys.executable)
        print(f"\nUsing current python: {python_exe}")

    if args.post_check_only:
        post_install_gpu_check(python_exe)
        return

    # Core training + HF stack.
    common = [
        "transformers>=4.46.0",
        "datasets>=2.19.0",
        "accelerate>=0.33.0",
        "peft>=0.12.0",
        "trl>=0.10.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece",
        "protobuf",
        "safetensors",
        "huggingface_hub>=0.24.0",
        "pandas",
        "pyarrow",
        "tqdm",
        "rich",
    ]

    # Ollama integration: Python client + requests.
    ollama = [
        "ollama>=0.3.0",
        "requests>=2.31.0",
    ]

    # Install PyTorch first from the chosen CUDA wheel index.
    torch_pkgs = ["torch", "torchvision", "torchaudio"]
    print("\nInstalling PyTorch CUDA wheels...")
    pip_install(python_exe, torch_pkgs, extra_args=["--index-url", args.torch_index_url])

    # Install the rest.
    print("\nInstalling Hugging Face / training dependencies...")
    pip_install(python_exe, common)
    print("\nInstalling Ollama Python client...")
    pip_install(python_exe, ollama)

    if not args.skip_unsloth:
        print("\nInstalling Unsloth...")
        pip_install(python_exe, ["unsloth"])
    else:
        print("\nSkipping Unsloth install (--skip_unsloth set).")

    print("\nPost-install GPU verification:")
    post_install_gpu_check(python_exe)

    print("\nNext steps:")
    if args.venv and platform.system().lower() == "windows":
        venv_dir = Path(args.venv).resolve()
        print(f"  Activate venv (PowerShell): {venv_dir}\\Scripts\\Activate.ps1")
    elif args.venv:
        venv_dir = Path(args.venv).resolve()
        print(f"  Activate venv (bash): source {venv_dir}/bin/activate")

    print("  Run: python load_model.py")


if __name__ == "__main__":
    main()
