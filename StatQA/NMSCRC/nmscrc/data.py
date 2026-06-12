"""Probe-label parser — reads the `results` column (GROUND TRUTH). CLEAN.

Builds multi-hot labels via parse_methods_from_results, normalize_method_name,
build_multi_hot_labels. It never touches `model_answer`, so the 3B
template-echo problem cannot affect it. Unknown method names -> HARD ERROR (these are ground
truth; silently dropping them would corrupt supervision + evaluation).

Three preserved behaviors (Gate-1):
  (1) ast.literal_eval on str(results).replace("false","False").replace("true","True")
  (2) skip items with conclusion == "Not applicable"
  (3) normalize_method_name strips trailing period
"""

import ast
import json
from typing import Dict, List

import numpy as np
import pandas as pd

from nmscrc import paths


def load_method_list() -> List[str]:
    """The ordered 27-method list (id <-> name). Asserts 27 unique names."""
    with open(paths.METHODS_JSON, "r", encoding="utf-8") as f:
        method_list = json.load(f)
    if len(method_list) != 27 or len(set(method_list)) != 27:
        raise RuntimeError(
            f"Method list must contain exactly 27 unique names; got {len(method_list)} "
            f"({len(set(method_list))} unique)."
        )
    return method_list


def normalize_method_name(name: str) -> str:
    return name.strip().rstrip(".")


def parse_methods_from_results(results_value: str) -> List[str]:
    """Extract ground-truth method names from the CSV 'results' column.

    Ported verbatim: literal_eval with false/true -> False/True, skip 'Not applicable',
    normalize_method_name strips trailing period.
    """
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


def build_multi_hot_labels(df: pd.DataFrame, method_list: List[str]) -> np.ndarray:
    """results-derived multi-hot Y in {0,1}^27 (supervision AND F1-evaluation truth).

    Unknown method names -> RuntimeError (ground truth must never be silently dropped).
    """
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
            f"Found unknown methods in results not present in method dictionary: {preview}"
        )

    return labels
