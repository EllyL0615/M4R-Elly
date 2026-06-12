"""Shared loss / set / bound primitives. B = 1 throughout.

  p(x)        = sigmoid(probe logits) in [0,1]^27
  C_{λ2}(x)   = { k : p_k(x) >= 1 - λ2 }            (nested; larger λ2 => larger set)
  g(x)        = max_k p_k(x)                         (selector; DECIDED)
  U_i(λ1)     = 1{ g(x_i) >= 1 - λ1 }
  ℓ_F1        = 1 - F1(C, Y),  F1 = 2|C∩Y|/(|C|+|Y|),  ℓ_F1 = 1 when C = ∅
  ℓ_recall    = 1 - |C∩Y| / max(|Y|,1)              (monotone non-increasing in λ2)

Bounds (one-sided):
  ε  (DKW, selection)      = sqrt(log(1/δ_U)/(2n))                      uniform over Λ1
  t_V (Hoeffding, union)   = B sqrt(log(m1 m2/δ_V)/(2n))               union over Λ1×Λ2
  EB (Maurer-Pontil 2009, one-sided): mean error <= sqrt(2 V̂ log(M/δ)/n) + 7 log(M/δ)/(3(n-1)),
     V̂ = unbiased sample variance. t_V^EB unions Λ1×Λ2 (M=m1 m2); t_U^EB unions Λ1 (M=m1),
     applied to (1-U).  [exact constants documented here; flagged at Checkpoint 1]
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def lambda_grid(n_points, lo=0.0, hi=1.0):
    return np.linspace(lo, hi, n_points)


def selector_g(p):
    """g(x) = max_k p_k(x)  (top-method probability)."""
    return p.max(axis=1)


def set_sizes(p, lambda2_value):
    """|C_{λ2}(x)| for a single λ2 (threshold 1-λ2)."""
    return (p >= (1.0 - lambda2_value)).sum(axis=1)


def build_loss_tensor(p, Y, lambda2, kind="f1"):
    """ℓ(C_{λ2}(x_i), Y_i) over the pool x Λ2 -> (n, m2) float32. kind in {'f1','recall'}."""
    thr = 1.0 - np.asarray(lambda2)
    Ysum = Y.sum(axis=1)
    n, m2 = p.shape[0], len(thr)
    L = np.empty((n, m2), dtype=np.float32)
    for j in range(m2):
        C = (p >= thr[j])
        inter = (C * (Y > 0)).sum(axis=1).astype(np.float64)
        Csum = C.sum(axis=1).astype(np.float64)
        if kind == "f1":
            denom = Csum + Ysum
            f1 = np.where(denom > 0, 2.0 * inter / denom, 0.0)   # C=∅ (or both ∅) -> F1=0 -> ℓ=1
            L[:, j] = 1.0 - f1
        elif kind == "recall":
            L[:, j] = 1.0 - inter / np.maximum(Ysum, 1.0)        # |Y|=0 guard; non-increasing in λ2
        else:
            raise ValueError(f"unknown loss kind {kind!r}")
    return L


# ---- one-sided concentration bounds ----
def eps_dkw(n, delta_u):
    """One-sided DKW slack on the selection CDF (uniform over all thresholds)."""
    return float(np.sqrt(np.log(1.0 / delta_u) / (2.0 * n)))


def t_hoeffding(n, m1, m2, delta_v, B=1.0):
    """Hoeffding slack with a union over the Λ1×Λ2 grid."""
    return float(B * np.sqrt(np.log(m1 * m2 / delta_v) / (2.0 * n)))


def eb_bound(var, n, log_term):
    """Maurer-Pontil one-sided empirical-Bernstein slack (var = unbiased sample variance)."""
    var = np.maximum(var, 0.0)
    return np.sqrt(2.0 * var * log_term / n) + 7.0 * log_term / (3.0 * (n - 1))


def oracle_ell_star(g, Y, loss_tensor, lambda1, lambda2, xi):
    """Oracle best accepted-region conditional F1-risk at coverage >= ξ (full pool, no penalty).

    ℓ* = min over the SAME threshold space the method searches — { λ1 : φ(λ1) >= ξ } × Λ2 — of the
    accepted-region conditional risk. α = ℓ* + Δ. Minimising over λ1 too (not a fixed λ1)
    keeps ℓ* and the method on one oracle space, so the C1 'excess' axis cannot drift.
    """
    phi = (g[:, None] >= (1.0 - lambda1)[None, :]).mean(axis=0)   # (m1,)
    feas = np.where(phi >= xi)[0]
    if feas.size == 0:
        feas = np.array([len(lambda1) - 1])
    best, best_info = float("inf"), None
    for i1 in feas:
        acc = g >= (1.0 - lambda1[i1])
        if acc.sum() == 0:
            continue
        cond = loss_tensor[acc].mean(axis=0)                     # (m2,)
        j = int(np.argmin(cond))
        if cond[j] < best:
            best = float(cond[j])
            best_info = {"lambda1_idx": int(i1), "lambda2_idx": j, "coverage": float(acc.mean()),
                         "lambda1": float(lambda1[i1]), "lambda2": float(lambda2[j])}
    return best, best_info
