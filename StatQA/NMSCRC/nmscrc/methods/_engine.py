"""Shared per-rep statistics on a calib split, and test-set evaluation.

  φ̂(λ1) = (1/n) Σ U_i(λ1),   U_i(λ1) = 1{ g(x_i) >= 1-λ1 }
  R̂(λ1,λ2) = (1/n) Σ U_i(λ1) ℓ(C_{λ2}(x_i), Y_i),   r̂ = R̂/φ̂
  S2(λ1,λ2) = (1/n) Σ U_i(λ1) ℓ²   (for empirical-Bernstein variances)
"""

import numpy as np

from nmscrc import losses


def calib_stats(g_calib, L_calib, lambda1):
    """Return U (n,m1), phi (m1,), R (m1,m2), S2 (m1,m2) on a calib split for loss L_calib."""
    thr1 = 1.0 - np.asarray(lambda1)
    U = (g_calib[:, None] >= thr1[None, :]).astype(np.float64)   # (n, m1)
    n = U.shape[0]
    phi = U.mean(axis=0)                                         # (m1,)
    R = (U.T @ L_calib) / n                                      # (m1, m2)
    S2 = (U.T @ (L_calib ** 2)) / n                              # (m1, m2)
    return U, phi, R, S2


def first_stage_dkw(g_calib, lambda1, xi, delta_u):
    """Shared first stage (IDENTICAL to NM-SCRC-I): λ̂1' = inf{λ1 : φ̂(λ1) >= ξ + ε}, ε=DKW(δ_U).
    Returns (i1, phi, eps, xi_lcb) or None if no λ1 reaches ξ+ε."""
    thr1 = 1.0 - np.asarray(lambda1)
    phi = (g_calib[:, None] >= thr1[None, :]).mean(axis=0)
    eps = losses.eps_dkw(len(g_calib), delta_u)
    feas = phi >= (xi + eps)
    if not feas.any():
        return None
    i1 = int(np.argmax(feas))
    return i1, phi, eps, float(phi[i1] - eps)


def xu_engine(U, L_calib, target, B=1.0):
    """Xu's augmented-loss CRC counting: R̂v(λ2)=(1/n)Σ U_i ℓ•(C_{λ2},y_i);
    return inf{ λ2 : (n/(n+1)) R̂v + B/(n+1) <= target }  (real-valued RHS, NO ceiling, NO D).
    target = α·ξ̂_LCB. Returns λ2 index or None (ABSTAIN). REQUIRES ℓ• monotone non-increasing in λ2."""
    n = len(U)
    Rv = (U[:, None] * L_calib).mean(axis=0)             # (m2,)  augmented loss, 0 off accepted
    feas = np.where((n / (n + 1)) * Rv + B / (n + 1) <= target)[0]
    return int(feas[0]) if feas.size else None


def eval_on_test(A, test_idx, i1, j, L_test_attr="L_f1"):
    """Evaluate a chosen (λ1[i1], λ2[j]) pair on the test accepted region."""
    g_test = A.g[test_idx]
    L_test = getattr(A, L_test_attr)[test_idx]
    acc = g_test >= (1.0 - A.lambda1[i1])
    coverage = float(acc.mean())
    if acc.sum() == 0:
        return {"coverage": coverage, "risk_cond": float("nan"), "set_size": float("nan"),
                "n_accept": 0}
    risk_cond = float(L_test[acc, j].mean())
    ss = A.SS[test_idx][acc, j] if A.SS is not None else losses.set_sizes(A.p[test_idx][acc], A.lambda2[j])
    return {"coverage": coverage, "risk_cond": risk_cond, "set_size": float(ss.mean()),
            "n_accept": int(acc.sum())}
