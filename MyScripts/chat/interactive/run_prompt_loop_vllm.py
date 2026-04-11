#!/usr/bin/env python3
"""
Interactive prompt loop for fast local prompt iteration with vLLM.

Run this script on an allocated GPU compute node (interactive PBS session).
The model is loaded once, then each Enter key reruns with the latest prompt file content.
"""

import argparse
import datetime as dt
import os
from pathlib import Path

from vllm import LLM, SamplingParams


MODEL_PATH_MAP = {
    "2_7b": "/rds/general/user/yl9422/home/files/models/Llama-2-7b-chat-hf",
    "2_13b": "/rds/general/user/yl9422/home/files/models/Llama-2-13b-chat-hf",
    "3_8b_instruct": "/rds/general/user/yl9422/home/files/models/Meta-Llama-3-8B-Instruct",
    "3_8b": "/rds/general/user/yl9422/home/files/models/Meta-Llama-3-8B",
    "deepseek": "/rds/general/user/yl9422/home/files/models/DeepSeek-R1-Distill-Qwen-7B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive one-prompt loop with vLLM")
    parser.add_argument(
        "--model-type",
        default="3_8b",
        choices=["2_7b", "2_13b", "3_8b_instruct", "3_8b", "deepseek"],
        help="Preset model type mapped to an absolute local model path",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional absolute path override; if set, model-type mapping is ignored",
    )
    parser.add_argument(
        "--prompt-file",
        default="/rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/chat/prompts/example_prompt.txt",
        help="Absolute path to plain text prompt file",
    )
    parser.add_argument(
        "--output-dir",
        default="/rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/chat/outputs",
        help="Directory to store latest and timestamped responses",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional engine max_model_len override",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Optional engine gpu_memory_utilization override",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to vLLM model loader when required by model",
    )
    return parser.parse_args()


def resolve_model_path(model_type: str, model_path_override: str | None) -> str:
    if model_path_override:
        return model_path_override

    model_path = MODEL_PATH_MAP.get(model_type)
    if model_path is None:
        raise ValueError(
            "[!] Invalid model type. Please choose from: 2_7b, 2_13b, 3_8b_instruct, 3_8b, and deepseek"
        )
    return model_path


def read_prompt(prompt_file: Path) -> str:
    text = prompt_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return text


def save_outputs(output_dir: Path, prompt: str, reply: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("PBS_JOBID", "nojob").split(".")[0]

    latest_path = output_dir / "latest_reply.txt"
    history_path = output_dir / f"reply_{job_id}_{timestamp}.txt"

    latest_path.write_text(reply + "\n", encoding="utf-8")
    history_path.write_text(
        "=== PROMPT ===\n"
        + prompt
        + "\n\n=== REPLY ===\n"
        + reply
        + "\n",
        encoding="utf-8",
    )
    return latest_path, history_path


def main() -> None:
    args = parse_args()

    prompt_file = Path(args.prompt_file)
    output_dir = Path(args.output_dir)
    model_path = resolve_model_path(args.model_type, args.model_path)

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
    }

    # DeepSeek-R1 defaults to very large context length; cap it for single-prompt
    # interactive usage so KV cache initialization is stable on 1 GPU.
    if args.model_type == "deepseek":
        llm_kwargs["max_model_len"] = 8192 if args.max_model_len is None else args.max_model_len
        llm_kwargs["gpu_memory_utilization"] = 0.92 if args.gpu_memory_utilization is None else args.gpu_memory_utilization
    else:
        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len
        if args.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    print("[i] Loading model once. This may take some time...")
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    print("[i] Model loaded.")
    print(f"[i] Model type : {args.model_type}")
    print(f"[i] Model path : {model_path}")
    if "max_model_len" in llm_kwargs:
        print(f"[i] max_model_len: {llm_kwargs['max_model_len']}")
    if "gpu_memory_utilization" in llm_kwargs:
        print(f"[i] gpu_mem_util: {llm_kwargs['gpu_memory_utilization']}")
    print(f"[i] Prompt file: {prompt_file}")
    print(f"[i] Output dir : {output_dir}")

    round_id = 1
    while True:
        print(f"\n{'=' * 20} ROUND {round_id} {'=' * 20}")

        try:
            prompt = read_prompt(prompt_file)
        except Exception as exc:
            print(f"[x] Cannot read prompt: {exc}")
            user_cmd = input("[Enter] retry, or type q to quit: ").strip().lower()
            if user_cmd == "q":
                break
            continue

        result = llm.generate([prompt], sampling)
        reply = result[0].outputs[0].text.strip()

        latest_path, history_path = save_outputs(output_dir, prompt, reply)

        print("\n=== REPLY ===\n")
        print(reply)
        print("\n[i] Saved latest : " + str(latest_path))
        print("[i] Saved history: " + str(history_path))

        user_cmd = input("\n[Enter] rerun after editing prompt file, or type q to quit: ").strip().lower()
        if user_cmd == "q":
            break
        round_id += 1


if __name__ == "__main__":
    main()
