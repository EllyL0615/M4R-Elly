#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2 for SCRC: train a linear probe from hidden states and export
logits for probe-train, conformal-calib, and conformal-test splits.
"""

import argparse
import ast
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, main_folder_path)

import mappings  # noqa: E402
from prompt_wording import PROMPT_CLASSIFICATION  # noqa: E402


DEFAULT_TASK_ORDER = [
    "Correlation Analysis",
    "Distribution Compliance Test",
    "Contingency Table Test",
    "Descriptive Statistics",
    "Variance Test",
]


@dataclass
class SplitBundle:
    name: str
    hidden_states: np.ndarray
    labels: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SCRC linear probe and export split-level logits."
    )

    parser.add_argument(
        "--probe_hidden_states",
        type=str,
        default="SCRC/outputs/step1/llama3_8b_probe-train_hidden-states.npy",
        help="Path to probe-train hidden states .npy",
    )
    parser.add_argument(
        "--calib_hidden_states",
        type=str,
        default="SCRC/outputs/step1/llama3_8b_conformal-calib_hidden-states.npy",
        help="Path to conformal-calib hidden states .npy",
    )
    parser.add_argument(
        "--test_hidden_states",
        type=str,
        default="SCRC/outputs/step1/llama3_8b_conformal-test_hidden-states.npy",
        help="Path to conformal-test hidden states .npy",
    )

    parser.add_argument(
        "--probe_csv",
        type=str,
        default="Data/Integrated Dataset/Dataset with Prompt/Training Set/D_train for probe-train.csv",
        help="Path to probe-train CSV",
    )
    parser.add_argument(
        "--calib_csv",
        type=str,
        default="Data/Integrated Dataset/Dataset with Prompt/Training Set/D_train for conformal-calib.csv",
        help="Path to conformal-calib CSV",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="Data/Integrated Dataset/Dataset with Prompt/Test Set/mini-StatQA for methods-only.csv",
        help="Path to conformal-test CSV",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3_8b",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="SCRC/outputs/step2",
        help="Directory for split-level logits/labels outputs",
    )
    parser.add_argument(
        "--saved_probe_dir",
        type=str,
        default="SCRC/outputs/step2",
        help="Directory for probe.pt and probe_meta.json",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Compute device. cpu by default; choose auto/cuda when needed.",
    )

    parser.add_argument(
        "--use_pos_weight",
        action="store_true",
        help="Use class-wise positive weights in BCEWithLogitsLoss.",
    )
    parser.add_argument(
        "--save_labels",
        action="store_true",
        help="Save split-level label matrices as .npy files.",
    )

    return parser.parse_args()


def resolve_repo_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(main_folder_path, path_value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MethodProbe(nn.Module):
    def __init__(self, input_dim: int = 4096, num_methods: int = 27):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_methods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


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
            method_name = normalize_method_name(str(item.get("method", "")))
            if method_name:
                methods.append(method_name)
    return methods


def normalize_method_name(name: str) -> str:
    return name.strip().rstrip(".")


def parse_methods_from_prompt_classification(prompt_classification: str) -> List[str]:
    methods: List[str] = []
    for raw_line in prompt_classification.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(";"):
            line = line[:-1]
        if ":" not in line:
            continue
        _, rhs = line.split(":", 1)
        parts = [normalize_method_name(item) for item in rhs.split(",") if item.strip()]
        methods.extend(parts)
    return methods


def build_method_list_from_mapping() -> List[str]:
    methods: List[str] = []
    for task in DEFAULT_TASK_ORDER:
        if task not in mappings.tasks_to_methods:
            raise RuntimeError(f"Task missing in mappings.tasks_to_methods: {task}")
        methods.extend([normalize_method_name(x) for x in mappings.tasks_to_methods[task]])
    return methods


def validate_method_dictionary() -> List[str]:
    methods_mapping = build_method_list_from_mapping()
    methods_prompt = parse_methods_from_prompt_classification(PROMPT_CLASSIFICATION)

    if len(methods_mapping) != len(set(methods_mapping)):
        raise RuntimeError("Duplicate method names found in mappings.tasks_to_methods.")

    if len(methods_mapping) != 27:
        raise RuntimeError(
            f"Expected 27 methods from mappings.tasks_to_methods, got {len(methods_mapping)}"
        )

    if set(methods_mapping) != set(methods_prompt):
        only_mapping = sorted(set(methods_mapping) - set(methods_prompt))
        only_prompt = sorted(set(methods_prompt) - set(methods_mapping))
        raise RuntimeError(
            "Method set mismatch between mappings.tasks_to_methods and PROMPT_CLASSIFICATION. "
            f"only_mapping={only_mapping}, only_prompt={only_prompt}"
        )

    if methods_mapping != methods_prompt:
        raise RuntimeError(
            "Method order mismatch between mappings.tasks_to_methods and PROMPT_CLASSIFICATION."
        )

    return methods_mapping


def build_multi_hot_labels(df: pd.DataFrame, method_list: List[str]) -> np.ndarray:
    if "results" not in df.columns:
        raise RuntimeError("CSV is missing required column: results")

    method_to_idx = {name: idx for idx, name in enumerate(method_list)}
    labels = np.zeros((len(df), len(method_list)), dtype=np.float32)

    unknown_methods: Dict[str, int] = {}
    for row_idx, (_, row) in enumerate(df.iterrows()):
        methods = parse_methods_from_results(row.get("results", ""))
        for method_name in methods:
            if method_name not in method_to_idx:
                unknown_methods[method_name] = unknown_methods.get(method_name, 0) + 1
                continue
            labels[row_idx, method_to_idx[method_name]] = 1.0

    if unknown_methods:
        preview = sorted(unknown_methods.items(), key=lambda x: (-x[1], x[0]))[:10]
        raise RuntimeError(
            "Found unknown methods in results not present in method dictionary: "
            f"{preview}"
        )

    return labels


def load_hidden_states(path: str) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 2:
        raise RuntimeError(f"Hidden states must be rank-2 array, got shape {data.shape} from {path}")
    if data.shape[1] != 4096:
        raise RuntimeError(f"Hidden states second dimension must be 4096, got {data.shape[1]} from {path}")
    if not np.isfinite(data).all():
        raise RuntimeError(f"Hidden states contain NaN/Inf: {path}")
    return data.astype(np.float32)


def build_split_bundle(
    split_name: str,
    hidden_path: str,
    csv_path: str,
    method_list: List[str],
) -> SplitBundle:
    hidden_states = load_hidden_states(hidden_path)
    df = pd.read_csv(csv_path)
    labels = build_multi_hot_labels(df, method_list)

    if hidden_states.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"Row mismatch for {split_name}: hidden={hidden_states.shape[0]}, labels={labels.shape[0]}"
        )

    return SplitBundle(name=split_name, hidden_states=hidden_states, labels=labels)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_train_validation(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if val_ratio <= 0:
        return x, y, x.copy(), y.copy()

    n = x.shape[0]
    val_size = max(1, int(round(n * val_ratio)))
    val_size = min(val_size, n - 1)

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def make_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    weight = (neg + 1e-6) / (pos + 1e-6)
    weight = np.clip(weight, 1.0, 100.0)
    return torch.from_numpy(weight.astype(np.float32))


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: List[float] = []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else math.nan


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else math.nan


def train_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[MethodProbe, Dict[str, float]]:
    x_subtrain, y_subtrain, x_val, y_val = split_train_validation(
        x_train,
        y_train,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_loader = make_dataloader(x_subtrain, y_subtrain, args.batch_size, shuffle=True)
    val_loader = make_dataloader(x_val, y_val, args.batch_size, shuffle=False)

    model = MethodProbe(input_dim=x_train.shape[1], num_methods=y_train.shape[1]).to(device)

    if args.use_pos_weight:
        pos_weight = compute_pos_weight(y_subtrain).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device)

        print(
            f"[i] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(f"[i] Early stopping triggered at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best checkpoint captured.")

    model.load_state_dict(best_state)

    metrics = {
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "train_rows": int(x_train.shape[0]),
        "val_rows": int(x_val.shape[0]),
        "subtrain_rows": int(x_subtrain.shape[0]),
    }
    return model, metrics


def predict_logits(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = make_dataloader(x, np.zeros((x.shape[0], 1), dtype=np.float32), batch_size, shuffle=False)

    score_list: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            batch_logits = model(batch_x)
            score_list.append(batch_logits.detach().cpu().numpy())

    logits = np.concatenate(score_list, axis=0)
    return logits.astype(np.float32)


def save_array(path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def save_matrix_with_method_columns(
    path: str,
    matrix: np.ndarray,
    method_list: List[str],
) -> None:
    if matrix.ndim != 2:
        raise RuntimeError(
            f"Expected rank-2 matrix for named export, got shape={matrix.shape} at {path}"
        )

    if matrix.shape[1] != len(method_list):
        raise RuntimeError(
            "Column mismatch for named export: "
            f"matrix has {matrix.shape[1]} columns but method_list has {len(method_list)}"
        )

    if len(set(method_list)) != len(method_list):
        raise RuntimeError("Duplicate method names detected in method_list.")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    float_format = "%.9g" if np.issubdtype(matrix.dtype, np.floating) else None
    pd.DataFrame(matrix, columns=method_list).to_csv(
        path,
        index=False,
        encoding="utf-8",
        float_format=float_format,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.chdir(main_folder_path)

    method_list = validate_method_dictionary()

    probe_bundle = build_split_bundle(
        split_name="train",
        hidden_path=resolve_repo_path(args.probe_hidden_states),
        csv_path=resolve_repo_path(args.probe_csv),
        method_list=method_list,
    )
    calib_bundle = build_split_bundle(
        split_name="calib",
        hidden_path=resolve_repo_path(args.calib_hidden_states),
        csv_path=resolve_repo_path(args.calib_csv),
        method_list=method_list,
    )
    test_bundle = build_split_bundle(
        split_name="test",
        hidden_path=resolve_repo_path(args.test_hidden_states),
        csv_path=resolve_repo_path(args.test_csv),
        method_list=method_list,
    )

    device = choose_device(args.device)
    print(f"[i] Using device: {device}")

    model, train_metrics = train_probe(
        x_train=probe_bundle.hidden_states,
        y_train=probe_bundle.labels,
        args=args,
        device=device,
    )

    outputs_dir = resolve_repo_path(args.outputs_dir)
    saved_probe_dir = resolve_repo_path(args.saved_probe_dir)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(saved_probe_dir, exist_ok=True)

    method_columns_path = os.path.join(outputs_dir, f"{args.model_name}_method_columns.json")
    with open(method_columns_path, "w", encoding="utf-8") as f:
        json.dump(method_list, f, indent=2, ensure_ascii=False)

    all_bundles = [probe_bundle, calib_bundle, test_bundle]

    split_shapes = {}
    for bundle in all_bundles:
        logits = predict_logits(
            model=model,
            x=bundle.hidden_states,
            device=device,
            batch_size=args.batch_size,
        )

        score_path = os.path.join(outputs_dir, f"{args.model_name}_{bundle.name}_logits.npy")
        score_csv_path = os.path.join(outputs_dir, f"{args.model_name}_{bundle.name}_logits.csv")
        save_array(score_path, logits)
        save_matrix_with_method_columns(score_csv_path, logits, method_list)

        if args.save_labels:
            label_path = os.path.join(outputs_dir, f"{args.model_name}_{bundle.name}_labels.npy")
            label_csv_path = os.path.join(outputs_dir, f"{args.model_name}_{bundle.name}_labels.csv")
            save_array(label_path, bundle.labels)
            save_matrix_with_method_columns(label_csv_path, bundle.labels, method_list)

        split_shapes[bundle.name] = {
            "hidden_states": list(bundle.hidden_states.shape),
            "labels": list(bundle.labels.shape),
            "logits": list(logits.shape),
        }

        print(
            f"[+] Saved split={bundle.name} "
            f"logits_npy={score_path} logits_csv={score_csv_path}"
        )

    probe_path = os.path.join(saved_probe_dir, f"{args.model_name}_probe.pt")
    torch.save(model.state_dict(), probe_path)

    meta = {
        "model_name": args.model_name,
        "seed": args.seed,
        "device": str(device),
        "method_list": method_list,
        "num_methods": len(method_list),
        "train_hyperparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "val_ratio": args.val_ratio,
            "use_pos_weight": bool(args.use_pos_weight),
        },
        "input_paths": {
            "probe_hidden_states": resolve_repo_path(args.probe_hidden_states),
            "calib_hidden_states": resolve_repo_path(args.calib_hidden_states),
            "test_hidden_states": resolve_repo_path(args.test_hidden_states),
            "probe_csv": resolve_repo_path(args.probe_csv),
            "calib_csv": resolve_repo_path(args.calib_csv),
            "test_csv": resolve_repo_path(args.test_csv),
        },
        "output_paths": {
            "outputs_dir": outputs_dir,
            "saved_probe_dir": saved_probe_dir,
            "probe_weight": probe_path,
            "method_columns_json": method_columns_path,
        },
        "split_shapes": split_shapes,
        "training_metrics": train_metrics,
    }

    meta_path = os.path.join(saved_probe_dir, f"{args.model_name}_probe_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[+] Saved probe weights: {probe_path}")
    print(f"[+] Saved probe metadata: {meta_path}")


if __name__ == "__main__":
    main()
