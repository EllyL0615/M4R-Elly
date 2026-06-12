"""Stratified pool split (split_dataset).

Ported verbatim: stratified train_test_split on the JOINT (task, difficulty) label, the
per-group pre-flight singleton check (every group needs >= 2 rows), ascending source-index
order, manifest CSV. Wired to config split_ratio (30/70) and nmscrc/paths.py.

DISCARDED: BASE_DIR absolutes, the setC plumbing (probe_train+calibtest = 100 => empty C),
global CONFIG cells. The split INDICES and label/row semantics are identical to legacy.

Output (one fixed, FROZEN split):
  {split_dir}/{model}_train_data.csv      + {model}_train_hs.npy        # probe_train fraction
  {split_dir}/{model}_calibtest_data.csv  + {model}_calibtest_hs.npy    # calibtest pool (the rep pool)
  {split_dir}/{model}_{SPLIT_NAME}_manifest.csv
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nmscrc import paths


def _preflight_strata(strat: pd.Series, n: int, frac_train: float, verbose: bool) -> None:
    """Per-(task x difficulty) group health; raise on singletons (cannot stratify)."""
    counts = strat.value_counts().sort_index()
    fracs = {"train": frac_train, "calibtest": 1.0 - frac_train}
    singletons = counts[counts < 2]
    thin = {g: c for g, c in counts.items()
            if any(int(round(c * f)) == 0 for f in fracs.values() if f > 0)}
    if verbose:
        print(f"[pre-flight] {len(counts)} (task x difficulty) groups; total={n}  "
              f"min={int(counts.min())}  max={int(counts.max())}  median={int(counts.median())}")
        print(f"             targets  train={frac_train*100:.0f}%  calibtest={(1-frac_train)*100:.0f}%")
        if thin:
            print(f"  [warn] {len(thin)} group(s) too small to appear in every split:")
            for g, c in sorted(thin.items(), key=lambda kv: kv[1]):
                alloc = "  ".join(f"{k}~{int(round(c*f))}" for k, f in fracs.items() if f > 0)
                print(f"           {g!r}: n={c}  ->  {alloc}")
        else:
            print("  [ok] every group is large enough to appear in all requested splits.")
    if len(singletons):
        raise ValueError(
            f"{len(singletons)} (task x difficulty) group(s) have only 1 row, so stratified "
            f"splitting is impossible: {list(singletons.index)}. "
            f"Merge/drop these difficulty levels or coarsen the strata before splitting."
        )


def split_pool(model: str, probe_train: int = 30, calibtest: int = 70,
               seed: int = 42, verbose: bool = True) -> Dict:
    """Stratified-on-(task x difficulty) split of {model}_pool into train/calibtest. FROZEN."""
    if not (0 < probe_train < 100) or not (0 < calibtest < 100):
        raise ValueError(f"ratios must be in (0,100): probe_train={probe_train}, calibtest={calibtest}")
    if probe_train + calibtest != 100:
        raise ValueError(f"probe_train + calibtest must equal 100, got {probe_train + calibtest}")

    df = pd.read_csv(paths.data_full(model))
    hs = np.load(paths.data_full_hs(model))
    n = len(df)
    if hs.shape[0] != n:
        raise ValueError(f"Row mismatch for {model}: CSV {n} vs hidden states {hs.shape[0]}")
    for col in ("task", "difficulty"):
        if col not in df.columns:
            raise AssertionError(f"CSV must contain a '{col}' column for stratified splitting.")

    strat = df["task"].astype(str) + " || " + df["difficulty"].astype(str)
    frac_train = probe_train / 100.0
    _preflight_strata(strat, n, frac_train, verbose)

    idx_train, idx_calibtest = train_test_split(
        np.arange(n),
        train_size=frac_train,
        test_size=calibtest / 100.0,
        random_state=seed,
        stratify=strat.values,
    )
    idx_train, idx_calibtest = np.sort(idx_train), np.sort(idx_calibtest)

    out_dir = paths.split_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # manifest
    has_dataset = "dataset" in df.columns
    rows = []
    for label, idxs in (("train", idx_train), ("calibtest", idx_calibtest)):
        for i in idxs:
            rows.append({"split": label, "source_index": int(i),
                         "dataset": (df.at[i, "dataset"] if has_dataset else ""),
                         "task": df.at[i, "task"], "difficulty": df.at[i, "difficulty"]})
    pd.DataFrame(rows).to_csv(paths.split_manifest(model), index=False, encoding="utf-8")

    # subsets (CSV + hidden states), ascending source-index order
    out = {}
    for label, idxs in (("train", idx_train), ("calibtest", idx_calibtest)):
        df.iloc[idxs].reset_index(drop=True).to_csv(paths.split_data(model, label), index=False, encoding="utf-8")
        np.save(paths.split_hs(model, label), hs[idxs])
        out[label] = {"csv": paths.split_data(model, label), "hs": paths.split_hs(model, label),
                      "n": int(len(idxs)), "index": idxs}

    if verbose:
        print(f"[{model}] total={n}  train={len(idx_train)} ({probe_train}%)  "
              f"calibtest={len(idx_calibtest)} ({calibtest}%)")
    return out
