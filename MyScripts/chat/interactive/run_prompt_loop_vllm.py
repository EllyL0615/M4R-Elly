#!/usr/bin/env python3
"""
Interactive multi-model prompt loop for fast local prompt iteration with vLLM.

Run this script on an allocated GPU compute node (interactive PBS session).
Several models are loaded once and kept resident in GPU memory at the same time.
Each round reads the latest prompt file content; the first line of the prompt
file selects which resident model answers this round, e.g.:

    #model: 3_1_8b
    <your prompt from here on>

Editing only the first line lets you switch between models instantly, with no
model reloading.
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
    "3_1_8b": "/rds/general/user/yl9422/home/files/models/Llama-3.1-8B",
    "3_2_1b": "/rds/general/user/yl9422/home/files/models/Llama-3.2-1B",
    "3_2_3b": "/rds/general/user/yl9422/home/files/models/Llama-3.2-3B",
    "deepseek": "/rds/general/user/yl9422/home/files/models/DeepSeek-R1-Distill-Qwen-7B",
}

# Conservative per-model fraction of TOTAL GPU memory each resident vLLM engine
# may use (weights + KV cache + activations). The sum must stay below 1.0 so all
# models can coexist on one GPU. Override at runtime with --gpu-frac.
DEFAULT_GPU_FRAC = {
    "3_1_8b": 0.45,
    "3_2_3b": 0.25,
    "3_2_1b": 0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive multi-model prompt loop with vLLM")
    parser.add_argument(
        "--models",
        default="3_1_8b,3_2_3b,3_2_1b",
        help="Comma-separated model types to load once and keep resident",
    )
    parser.add_argument(
        "--gpu-frac",
        default=None,
        help=(
            "Optional per-model gpu_memory_utilization override, e.g. "
            '"3_1_8b=0.45,3_2_3b=0.25,3_2_1b=0.15". Missing models fall back to defaults.'
        ),
    )
    parser.add_argument(
        "--prompt-file",
        default="/rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/chat/prompts/linear_probe.txt",
        help="Absolute path to plain text prompt file (first line selects the model)",
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
        default=4096,
        help="Engine max_model_len applied to every model (smaller -> smaller KV cache)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to vLLM model loader when required by model",
    )
    return parser.parse_args()


def resolve_model_path(model_type: str) -> str:
    model_path = MODEL_PATH_MAP.get(model_type)
    if model_path is None:
        raise ValueError(
            f"[!] Invalid model type '{model_type}'. Choose from: {', '.join(MODEL_PATH_MAP)}"
        )
    return model_path


def parse_gpu_frac(models: list[str], override: str | None) -> dict[str, float]:
    frac = {m: DEFAULT_GPU_FRAC.get(m, 0.30) for m in models}
    if override:
        for item in override.split(","):
            item = item.strip()
            if not item:
                continue
            name, _, value = item.partition("=")
            name = name.strip()
            if name not in frac:
                raise ValueError(f"[!] --gpu-frac references unloaded model '{name}'")
            frac[name] = float(value)
    return frac


def parse_prompt_with_model(prompt_file: Path, available: set[str]) -> tuple[str, str]:
    """Return (model_name, prompt_body) parsed from the prompt file.

    The first line must look like ``#model: <name>``. Raises ValueError on a
    missing/invalid directive or an empty prompt body.
    """
    text = prompt_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"Prompt file is empty: {prompt_file}")

    first = lines[0].strip()
    marker = first.lstrip("#").strip()
    if not marker.lower().startswith("model:"):
        raise ValueError(
            "First line must be a model directive like '#model: 3_1_8b'. "
            f"Available: {', '.join(sorted(available))}"
        )

    model_name = marker.split(":", 1)[1].strip()
    if model_name not in available:
        raise ValueError(
            f"Model '{model_name}' is not loaded. Available: {', '.join(sorted(available))}"
        )

    body = "\n".join(lines[1:]).strip()
    if not body:
        raise ValueError("Prompt body (everything after the first line) is empty.")
    return model_name, body


def save_outputs(output_dir: Path, model_name: str, prompt: str, reply: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("PBS_JOBID", "nojob").split(".")[0]

    latest_path = output_dir / "latest_reply.txt"
    history_path = output_dir / f"reply_{model_name}_{job_id}_{timestamp}.txt"

    latest_path.write_text(reply + "\n", encoding="utf-8")
    history_path.write_text(
        f"=== MODEL ===\n{model_name}\n\n"
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
    model_types = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_types:
        raise ValueError("--models must list at least one model type")

    gpu_frac = parse_gpu_frac(model_types, args.gpu_frac)

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    # Resolve and validate every model path up front before loading anything.
    paths = {}
    for name in model_types:
        path = resolve_model_path(name)
        if not Path(path).exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        paths[name] = path

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Load each model once, serially, into a resident dict.
    llms = {}
    total_frac = sum(gpu_frac[name] for name in model_types)
    print(f"[i] Loading {len(model_types)} models once (sum gpu_frac = {total_frac:.2f}).")
    if total_frac >= 1.0:
        print("[!] WARNING: sum of gpu_frac >= 1.0; this will likely OOM on a single GPU.")
    for name in model_types:
        print(f"[i] Loading '{name}' (gpu_frac={gpu_frac[name]}) from {paths[name]} ...")
        llms[name] = LLM(
            model=paths[name],
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=gpu_frac[name],
            max_model_len=args.max_model_len,
        )

    available = set(llms)
    print("\n[i] All models loaded and resident.")
    print(f"[i] Models     : {', '.join(model_types)}")
    print(f"[i] max_model_len: {args.max_model_len}")
    print(f"[i] Prompt file: {prompt_file}")
    print(f"[i] Output dir : {output_dir}")
    print("[i] Set the first line of the prompt file to '#model: <name>' to pick a model.")

    round_id = 1
    while True:
        print(f"\n{'=' * 20} ROUND {round_id} {'=' * 20}")

        try:
            model_name, prompt = parse_prompt_with_model(prompt_file, available)
        except Exception as exc:
            print(f"[x] Cannot read prompt: {exc}")
            user_cmd = input("[Enter] retry, or type q to quit: ").strip().lower()
            if user_cmd == "q":
                break
            continue

        print(f"[i] This round uses model: {model_name}")
        result = llms[model_name].generate([prompt], sampling)
        reply = result[0].outputs[0].text.strip()

        latest_path, history_path = save_outputs(output_dir, model_name, prompt, reply)

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
