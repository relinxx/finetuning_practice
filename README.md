# finetuning_practice

Mini-project: QLoRA/LoRA fine-tuning with Unsloth + export to Ollama.

This repo contains 6 standalone scripts:
- [setup_env.py](setup_env.py)
- [load_model.py](load_model.py)
- [prepare_dataset.py](prepare_dataset.py)
- [train_finetune.py](train_finetune.py)
- [export_to_ollama.py](export_to_ollama.py)
- [inference_ollama.py](inference_ollama.py)

## 0) Platform notes (Windows + RTX 5070)

- Ollama runs on Windows.
- Unsloth fine-tuning is most reliable on Linux (including WSL2 Ubuntu on Windows).
- If you try native Windows Python and Unsloth install/import fails, use WSL2 for the fine-tuning steps.

## 1) Create & activate environment

### Option A (recommended on Windows): WSL2 Ubuntu

Create and activate a venv:
- `python3 -m venv .venv`
- `source .venv/bin/activate`

Run the installer:
- `python setup_env.py --venv .venv`

### Option B: Native Windows (PowerShell)

Create and activate a venv:
- `python -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`

Run the installer:
- `python setup_env.py --venv .venv`

If PyTorch CUDA wheels mismatch your system, pick a different CUDA index (example):
- `python setup_env.py --venv .venv --torch_index_url https://download.pytorch.org/whl/cu121`

Recommended environment variables (helps avoid CUDA memory fragmentation):
- PowerShell:
	- `$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"`
	- `$env:TOKENIZERS_PARALLELISM="false"`
- bash:
	- `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
	- `export TOKENIZERS_PARALLELISM=false`

## 2) Confirm model fits in VRAM

This loads a 4-bit base model and prints VRAM checkpoints.

- `python load_model.py`

If VRAM is near the limit:
- Reduce `--max_seq_length` (e.g. 1024)
- Choose a smaller base model

## 3) Prepare dataset (CSV/JSON/JSONL)

Input data must contain:
- `instruction` (string)
- `input` (string; may be empty)
- `output` (string)

Example JSONL row:
```json
{"instruction":"Summarize this", "input":"Text...", "output":"Summary..."}
```

Prepare + tokenize (saves to `artifacts/dataset`):
- `python prepare_dataset.py --data path/to/train.jsonl --out_dir artifacts/dataset --val_ratio 0.02`

Notes:
- Tokenization happens on CPU; it does not need the GPU.
- Keep `--max_seq_length` conservative for 8–12GB VRAM.

## 4) Fine-tune (QLoRA/LoRA)

Train with safe defaults for ~8–12GB VRAM:
- `python train_finetune.py --dataset_dir artifacts/dataset --out_dir artifacts/lora`

Enable validation loss logging (if you created a validation split):
- `python train_finetune.py --dataset_dir artifacts/dataset --out_dir artifacts/lora --do_eval`

If you hit CUDA OOM:
- Lower `--max_seq_length`
- Keep `--per_device_train_batch_size 1`
- Increase `--gradient_accumulation_steps` instead of batch size

Optional: save a merged HF model directory (can require more RAM/VRAM temporarily):
- `python train_finetune.py --dataset_dir artifacts/dataset --out_dir artifacts/lora --merge_out artifacts/merged_model`

## 5) Export to Ollama

Install Ollama separately (outside Python), and ensure `ollama` works:
- `ollama --version`

Export and write an Ollama `Modelfile`:
- `python export_to_ollama.py --lora_dir artifacts/lora --out_dir artifacts/ollama --ollama_name finetuned-llama`

Then create the Ollama model:
- `ollama create finetuned-llama -f artifacts/ollama/Modelfile`

Test it:
- `ollama run finetuned-llama "Hello!"`

Notes:
- If GGUF export fails on your platform/Unsloth version, the script will still produce a merged HF model folder
	at `artifacts/ollama/merged_hf` and print a llama.cpp conversion fallback.

## 6) Interactive inference via Ollama

- `python inference_ollama.py --model finetuned-llama`

Commands inside the chat:
- `/reset` clears history
- `/exit` quits
