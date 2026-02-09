"""inference_ollama.py

Purpose:
- Simple interactive chat loop using a locally-created Ollama model.
- Sends prompts to the Ollama server and streams responses.

Usage:
  # First ensure you've created the model:
  #   ollama create finetuned-llama -f artifacts/ollama/Modelfile
  # Then run:
  python inference_ollama.py --model finetuned-llama

Notes:
- Ollama must be running (default: http://localhost:11434).
- GPU utilization is controlled by Ollama; this script just calls the API.
"""

from __future__ import annotations

import argparse
import logging

import ollama

from logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Interactive inference via Ollama")
    parser.add_argument("--model", type=str, default="finetuned-llama")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.")
    parser.add_argument("--num_ctx", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    logger.info("Ollama interactive chat")
    logger.info("Model: %s", args.model)
    logger.info("Type /exit to quit, /reset to clear chat history.\n")

    messages = [{"role": "system", "content": args.system}]

    while True:
        user = input("You> ").strip()
        if not user:
            continue
        if user.lower() in {"/exit", "/quit"}:
            break
        if user.lower() == "/reset":
            messages = [{"role": "system", "content": args.system}]
            logger.info("(history cleared)\n")
            continue

        messages.append({"role": "user", "content": user})

        # Stream tokens from Ollama for a responsive experience.
        print("Assistant> ", end="", flush=True)
        response_text = ""
        try:
            stream = ollama.chat(
                model=args.model,
                messages=messages,
                stream=True,
                options={
                    "num_ctx": args.num_ctx,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
            )
            for part in stream:
                token = part.get("message", {}).get("content", "")
                if token:
                    response_text += token
                    print(token, end="", flush=True)
            print("\n")
        except Exception as e:
            logger.error("Error talking to Ollama: %s", repr(e))
            logger.info("- Ensure Ollama is installed and running")
            logger.info("- Ensure the model exists: ollama list")
            logger.info("- Try: ollama run %s", args.model)
            # Remove the last user message so we can retry cleanly.
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
