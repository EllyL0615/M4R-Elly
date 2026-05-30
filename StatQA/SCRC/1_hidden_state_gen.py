#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1 for SCRC: extract hidden states and generate method-answer completions.

This script supports:
- Reproducible train split: train / calib (50/50 stratified by task)
- Fixed test split: test from mini-StatQA
- Hidden-state extraction at the final layer, final token position
- Greedy decoding (equivalent to vLLM temperature=0, top_p=1)
- Origin Answer-compatible CSV export
- Optional row-capped smoke validation via --smoke_rows_per_split
"""

import argparse
import ast
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hidden states and method completions for SCRC Step 1."
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="Data/Integrated Dataset/Dataset with Prompt/Training Set/D_train for methods-only.csv",
        help="Training CSV with prompt column.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="Data/Integrated Dataset/Dataset with Prompt/Test Set/mini-StatQA for methods-only.csv",
        help="Test CSV with prompt column.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/rds/general/user/yl9422/home/files/models/Meta-Llama-3-8B",
        help="Local model path for LLaMA-3 8B Base.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3_8b",
        help="Output file prefix.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for tokenizer/model inference (aligned with llama_evaluation).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per sample (aligned with non-CoT llama_evaluation).",
    )
    parser.add_argument(
        "--hidden_state_dir",
        type=str,
        default="SCRC/outputs/step1",
        help="Directory for hidden state outputs and split manifest.",
    )
    parser.add_argument(
        "--origin_answer_dir",
        type=str,
        default="Model Answer/Origin Answer",
        help="Directory for Origin Answer-compatible CSV outputs.",
    )
    parser.add_argument(
        "--smoke_rows_per_split",
        type=int,
        default=0,
        help="If >0, keep only first N rows per split (smoke mode). If <=0, use full splits.",
    )
    parser.add_argument(
        "--mock_inference",
        action="store_true",
        help=(
            "Skip model loading and create deterministic mock hidden states + answers. "
            "Useful for smoke test on login nodes."
        ),
    )
    return parser.parse_args()


def resolve_repo_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(main_folder_path, path_value)


def configure_determinism() -> None:
    # cuBLAS reproducibility hint for CUDA matmul kernels.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            torch.use_deterministic_algorithms(True, warn_only=True)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_df(train_df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "task" not in train_df.columns:
        raise ValueError("[!] Missing required column: task")

    temp = train_df.copy()
    temp["source_index"] = temp.index

    probe_idx, calib_idx = train_test_split(
        temp.index,
        test_size=0.5,
        random_state=seed,
        stratify=temp["task"],
    )

    probe_df = temp.loc[probe_idx].reset_index(drop=True)
    calib_df = temp.loc[calib_idx].reset_index(drop=True)
    return probe_df, calib_df


def parse_methods_from_results(results_value: str) -> List[str]:
    try:
        parsed = ast.literal_eval(
            str(results_value).replace("false", "False").replace("true", "True")
        )
    except Exception:
        return []

    methods: List[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            if item.get("conclusion") == "Not applicable":
                continue
            method_name = str(item.get("method", "")).strip()
            if method_name:
                methods.append(method_name)
    return methods


def parse_columns_from_relevant_column(relevant_value: str) -> List[str]:
    try:
        parsed = ast.literal_eval(
            str(relevant_value).replace("false", "False").replace("true", "True")
        )
    except Exception:
        return []

    columns: List[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            col_name = str(item.get("column_header", "")).strip()
            if col_name:
                columns.append(col_name)
    return columns


def build_ground_truth_json(row: pd.Series) -> str:
    columns = parse_columns_from_relevant_column(row.get("relevant_column", ""))
    methods = parse_methods_from_results(row.get("results", ""))
    return json.dumps({"columns": columns, "methods": methods}, ensure_ascii=False)


def load_model_and_tokenizer(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: Dict[str, object] = {"output_hidden_states": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not torch.cuda.is_available():
        model = model.to("cpu")

    if model.generation_config is not None:
        # Keep greedy mode deterministic and silence sampling-related warnings.
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0

    model.eval()
    return model, tokenizer


def extract_hidden_states_and_answers(
    model,
    tokenizer,
    df: pd.DataFrame,
    batch_size: int,
    max_new_tokens: int,
    mock_inference: bool,
    seed: int,
) -> Tuple[np.ndarray, List[str]]:
    if "prompt" not in df.columns:
        raise ValueError("[!] Missing required column: prompt")

    if mock_inference:
        rng = np.random.default_rng(seed + len(df))
        hidden_states = rng.standard_normal((len(df), 4096)).astype(np.float32)
        answers: List[str] = []
        for _, row in df.iterrows():
            methods = parse_methods_from_results(row.get("results", ""))
            if methods:
                answers.append('", "'.join(methods) + '"]}')
            else:
                answers.append('"]}')
        return hidden_states, answers

    hidden_state_batches: List[np.ndarray] = []
    answers: List[str] = []

    prompts = df["prompt"].astype(str).tolist()

    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]

        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}

        # current workflow: hidden state and generation are separate procedures
        # alternative: forward (hidden_states AND past_key_values) + greedy token-by-token generation using past_key values
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)
            batch_hidden_states = (
                outputs.hidden_states[-1][:, -1, :].detach().float().cpu().numpy()
            )
            hidden_state_batches.append(batch_hidden_states)

            generated = model.generate(
                **encoded,
                generation_config=model.generation_config,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = encoded["input_ids"].shape[1]
        completion_ids = generated[:, prompt_len:]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        answers.extend([text.strip() for text in completions])

        print(f"[+] Processed rows {start}..{end - 1}")

    hidden_states = np.concatenate(hidden_state_batches, axis=0)
    return hidden_states, answers


def build_origin_answer_df(df: pd.DataFrame, model_answers: List[str]) -> pd.DataFrame:
    required_columns = [
        "dataset",
        "refined_question",
        "relevant_column",
        "task",
        "difficulty",
        "results",
        "prompt",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"[!] Missing required columns: {missing_columns}")

    out_df = df[required_columns].copy()
    out_df["model_answer"] = model_answers
    out_df["ground_truth"] = df.apply(build_ground_truth_json, axis=1)
    return out_df


def save_split_outputs(
    split_name: str,
    split_df: pd.DataFrame,
    hidden_states: np.ndarray,
    answers: List[str],
    model_name: str,
    hidden_state_dir: str,
    origin_answer_dir: str,
) -> Dict[str, str]:
    hidden_state_file = os.path.join(
        hidden_state_dir, f"{model_name}_{split_name}_hidden-states.npy"
    )
    np.save(hidden_state_file, hidden_states)

    answer_file = os.path.join(origin_answer_dir, f"{model_name}_{split_name}.csv")
    out_df = build_origin_answer_df(split_df, answers)
    out_df.to_csv(answer_file, index=False, encoding="utf-8")

    return {
        "split": split_name,
        "hidden_states": hidden_state_file,
        "answers": answer_file,
        "num_rows": str(len(split_df)),
        "hidden_shape": str(tuple(hidden_states.shape)),
    }


def maybe_apply_smoke_subset(
    probe_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    smoke_rows_per_split: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if smoke_rows_per_split <= 0:
        return probe_df, calib_df, test_df

    probe_smoke = probe_df.head(smoke_rows_per_split).reset_index(drop=True)
    calib_smoke = calib_df.head(smoke_rows_per_split).reset_index(drop=True)
    test_smoke = test_df.head(smoke_rows_per_split).reset_index(drop=True)
    return probe_smoke, calib_smoke, test_smoke


def main() -> None:
    args = parse_args()
    configure_determinism()
    set_seed(args.seed)

    start_time = time.time()

    train_csv = resolve_repo_path(args.train_csv)
    test_csv = resolve_repo_path(args.test_csv)
    hidden_state_dir = resolve_repo_path(args.hidden_state_dir)
    origin_answer_dir = resolve_repo_path(args.origin_answer_dir)

    os.makedirs(hidden_state_dir, exist_ok=True)
    os.makedirs(origin_answer_dir, exist_ok=True)

    # Keep relative paths in the repository stable for any shared utilities.
    os.chdir(main_folder_path)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    probe_df, calib_df = split_train_df(train_df, seed=args.seed)

    test_df = test_df.copy()
    test_df["source_index"] = test_df.index

    probe_df, calib_df, test_df = maybe_apply_smoke_subset(
        probe_df,
        calib_df,
        test_df,
        smoke_rows_per_split=args.smoke_rows_per_split,
    )

    model = None
    tokenizer = None
    if not args.mock_inference:
        print("[i] Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        print("[+] Model loaded.")
    else:
        print("[i] Running in mock inference mode for smoke validation.")

    split_map = {
        "train": probe_df,
        "calib": calib_df,
        "test": test_df,
    }

    summary_rows: List[Dict[str, str]] = []
    manifest_rows: List[Dict[str, object]] = []

    for split_name, split_df in split_map.items():
        print("------------------------------------------------------------")
        print(f"[i] Running split: {split_name} (rows={len(split_df)})")

        hidden_states, answers = extract_hidden_states_and_answers(
            model=model,
            tokenizer=tokenizer,
            df=split_df,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            mock_inference=args.mock_inference,
            seed=args.seed,
        )

        if hidden_states.shape[0] != len(split_df):
            raise RuntimeError(
                f"[!] Hidden state row mismatch for {split_name}: "
                f"{hidden_states.shape[0]} vs {len(split_df)}"
            )

        if len(answers) != len(split_df):
            raise RuntimeError(
                f"[!] Answer row mismatch for {split_name}: "
                f"{len(answers)} vs {len(split_df)}"
            )

        summary = save_split_outputs(
            split_name=split_name,
            split_df=split_df,
            hidden_states=hidden_states,
            answers=answers,
            model_name=args.model_name,
            hidden_state_dir=hidden_state_dir,
            origin_answer_dir=origin_answer_dir,
        )
        summary_rows.append(summary)

        for _, row in split_df.iterrows():
            manifest_rows.append(
                {
                    "split": split_name,
                    "source_index": int(row.get("source_index", -1)),
                    "dataset": row.get("dataset", ""),
                    "task": row.get("task", ""),
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(hidden_state_dir, f"{args.model_name}_split_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(hidden_state_dir, f"{args.model_name}_step1_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    elapsed = time.time() - start_time
    print("============================================================")
    print("[+] Step 1 extraction finished.")
    print(f"[i] Manifest: {manifest_path}")
    print(f"[i] Summary : {summary_path}")
    print(f"[i] Elapsed : {elapsed:.2f}s")


if __name__ == "__main__":
    main()
