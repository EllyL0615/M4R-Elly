"""Frozen Stage-0 artifacts container. Built once per model from the frozen-probe
logits + results-labels on the calibtest pool, then consumed read-only by every method/rep.
"""

from dataclasses import dataclass

import numpy as np

from nmscrc import losses, paths


@dataclass
class Artifacts:
    model: str
    p: np.ndarray         # (n,27)  sigmoid(logits)
    g: np.ndarray         # (n,)    selector max_k p_k
    Y: np.ndarray         # (n,27)  results-derived multi-hot (ground truth)
    L_f1: np.ndarray      # (n,m2)  1 - F1
    L_recall: np.ndarray  # (n,m2)  1 - recall (monotone)
    lambda1: np.ndarray   # (m1,)
    lambda2: np.ndarray   # (m2,)
    split_hash: str = None
    SS: np.ndarray = None  # (n,m2)  |C_{λ2}(x_i)| set sizes, precomputed once (speed)

    @property
    def n(self):
        return self.p.shape[0]


def grids_from_config(cfg):
    """Uniform fallback grids (mode=uniform / when p is unavailable)."""
    g = cfg["grids"]
    lam1 = losses.lambda_grid(g["lambda1"]["n_points"], *g["lambda1"]["range"])
    lam2 = losses.lambda_grid(g["lambda2"]["n_points"], *g["lambda2"]["range"])
    return lam1, lam2


def build_grids(p, cfg):
    """Per-model Λ1, Λ2 grids (same grid for all methods within a model).

    mode=quantile: λ1 from g=max_k p_k quantiles (coverage sweeps evenly in [cov_min,1]); λ2 from
    pooled-p quantiles (mean set size sweeps evenly). Built from the FROZEN pool only — no calib/test
    leak. mode=uniform: linspace [0,1]. Defends against probe saturation (g,p concentrate near 1).
    """
    gc, g1, g2 = cfg["grids"], cfg["grids"]["lambda1"], cfg["grids"]["lambda2"]
    g = p.max(axis=1)
    if g1.get("mode", "uniform") == "quantile":
        cov = np.linspace(g1.get("cov_min", 0.02), 1.0, g1["n_points"])
        lam1 = np.sort(np.clip(1.0 - np.quantile(g, 1.0 - cov), 0.0, 1.0))
    else:
        lam1 = losses.lambda_grid(g1["n_points"], *g1["range"])
    if g2.get("mode", "uniform") == "quantile":
        size = np.linspace(g2.get("size_min", 0.0) + 1e-3, 1.0, g2["n_points"])
        lam2 = np.sort(np.clip(1.0 - np.quantile(p.reshape(-1), 1.0 - size), 0.0, 1.0))
    else:
        lam2 = losses.lambda_grid(g2["n_points"], *g2["range"])
    return lam1, lam2


def build_artifacts(model, logits, labels, lambda1, lambda2, split_hash=None):
    p = losses.sigmoid(logits.astype(np.float64))
    Y = (labels > 0).astype(np.float64)
    g = losses.selector_g(p)
    L_f1 = losses.build_loss_tensor(p, Y, lambda2, kind="f1")
    L_recall = losses.build_loss_tensor(p, Y, lambda2, kind="recall")
    SS = (p[:, :, None] >= (1.0 - np.asarray(lambda2))[None, None, :]).sum(axis=1).astype(np.int16)
    return Artifacts(model, p, g, Y, L_f1, L_recall,
                     np.asarray(lambda1), np.asarray(lambda2), split_hash, SS=SS)


def load_artifacts(model, cfg, split_hash=None):
    """Load frozen Stage-0 logits+labels from disk and build the Artifacts container."""
    logits = np.load(paths.calibtest_logits(model))
    labels = np.load(paths.calibtest_labels(model))
    lam1, lam2 = grids_from_config(cfg)
    return build_artifacts(model, logits, labels, lam1, lam2, split_hash)
