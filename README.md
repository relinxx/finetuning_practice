# finetuning_practice

Mini-project: QLoRA/LoRA fine-tuning with Unsloth + export to Ollama.

This repo contains 7 standalone scripts:
- [setup_env.py](setup_env.py)
- [load_model.py](load_model.py)
- [dataset_stats.py](dataset_stats.py)
- [prepare_dataset.py](prepare_dataset.py)
- [train_finetune.py](train_finetune.py)
- [export_to_ollama.py](export_to_ollama.py)
- [inference_ollama.py](inference_ollama.py)

## Project workflow (quick map)

End-to-end flow (inputs -> outputs):
- [setup_env.py](setup_env.py): installs dependencies and checks GPU
- [load_model.py](load_model.py): sanity-checks model loading + VRAM usage
- [dataset_stats.py](dataset_stats.py): analyzes raw dataset statistics and distributions
- [prepare_dataset.py](prepare_dataset.py): raw dataset -> tokenized HF dataset at `artifacts/dataset`
- [train_finetune.py](train_finetune.py): tokenized dataset -> LoRA adapter at `artifacts/lora`
- [export_to_ollama.py](export_to_ollama.py): LoRA adapter -> merged HF model and optional GGUF at `artifacts/ollama`
- [inference_ollama.py](inference_ollama.py): chat client for the Ollama model

Data/artefact locations (defaults):
- Input dataset: CSV/JSON/JSONL with `instruction`, `input`, `output`
- Tokenized dataset: `artifacts/dataset`
- LoRA adapter: `artifacts/lora`
- Merged model: `artifacts/ollama/merged_hf`
- GGUF export (if supported): `artifacts/ollama/gguf/*.gguf`
- Ollama Modelfile: `artifacts/ollama/Modelfile`

Script arguments (high level):
- Common model id: `unsloth/llama-3.1-8b-unsloth-bnb-4bit`
- Sequence length: `--max_seq_length` in [prepare_dataset.py](prepare_dataset.py) and [train_finetune.py](train_finetune.py)
- VRAM safety: keep batch size small; scale with `--gradient_accumulation_steps`

Defults are centrad in [confg.y](cnfi.p). Overrd vi CLI argumets as needed.

Shared utilities are in [utils.py](utils.py)

Run any script with `--help` for detaild usage and examples.

## Testing

Run unit tests with
- `python -m pytest test_prepare_dataset.py -v`

(Requires pytest in environment; install via `pip install pytest`)

## Type Checking

Run type checking with
- `python -m mypy prepare_dataset.py --ignore-missing-imports`

(Requires mypy; install via `pip install mypy`)


## 0) Platform notes (Windows + RTX 5070)

- Ollama runs on Windows
- Unsloth fine-tuning is most reliable on Linux (including WSL2 Ubuntu on Windows)
- If you try native Windows Python and Unsloth install/import fails, use WSL2 for the fine-tuning steps

## 1) Create & activate environment

### Option A (recommended on Windows): WSL2 Ubuntu

Create and activate a venv
- `python3 -m venv .venv`
- `source .venv/bin/activate`

Run the installer:
- `python setup_env.py --venv .venv`

### Option B: Native Windows (PowerShell)

Create and activate a ven
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

## 3) Analyze dataset statistics

Get insights into your dataset before tokenization:
- `python dataset_stats.py --data path/to/train.jsonl`

With tokenizer for token counts:
- `python dataset_stats.py --data path/to/train.jsonl --load_tokenizer unsloth/llama-3.1-8b-unsloth-bnb-4bit`

This helps identify:
- Text length distributions
- Potential sequence length issues
- Empty or problematic samples

## 4) Prepare dataset (CSV/JSON/JSONL)

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

## 5) Fine-tune (QLoRA/LoRA)

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

## 6) Export to Ollama

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

## 7) Interactive inference via Ollama

- `python inference_ollama.py --model finetuned-llama`

Commands inside the chat:
- `/reset` clears history
- `/exit` quits

## Enhancement Plan Progress

This project follows a 20-iteration enhancement plan to improve code quality, testing, documentation, and reliability:

### Completed Iterations
1. ✅ **Project audit** - Reviewed all scripts and identified improvement areas
2. ✅ **Logging setup** - Added centralized logging with timestamps and levels
3. ✅ **Config centralization** - Moved all defaults to config.py for consistency
4. ✅ **CLI polish** - Enhanced argument parsing and help text
5. ✅ **Dataset validation** - Added input validation and error messages
6. ✅ **Unit tests** - Created test_prepare_dataset.py with pytest coverage
7. ✅ **Type hints** - Added mypy-compatible type annotations
8. ✅ **Utilities refactoring** - Consolidated shared functions in utils.py
9. ✅ **Error handling** - Added try-except blocks for network/file I/O operations
10. ✅ **Dataset stats report** - Created dataset_stats.py for dataset analysis

### Upcoming Iterations
11. Reproducibility (seeds, determinism)
12. Caching (model/tokenizer downloads)
13. Training improvements (early stopping, learning rate scheduling)
14. Evaluation metrics (BLEU, ROUGE, perplexity)
15. Export checks (model validation)
16. Inference benchmarking (latency, throughput)
17. Documentation updates (API docs, troubleshooting)
18. Linting (black, flake8, pre-commit)
19. CI/CD setup (GitHub Actions)
20. Final review and optimization

Current status: **50% complete** (10/20 iterations)
