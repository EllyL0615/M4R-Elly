"""naive composition (opponent; FAIL layer).

Meet coverage (same first stage as NM-SCRC), then over (λ1>=λ̂1', λ2) pick the smallest set among the
EMPIRICALLY feasible cells (bare r̂ <= α), WITHOUT the Λ1×Λ2-union correction (no t_V, no ε in the
rule). Differs from NM-SCRC-I ONLY by the missing union, so a FAIL is attributable to exactly that:
the Prop 4.4 winner's curse over the joint (λ1,λ2) search makes the empirical optimum optimistic, so
the test conditional risk exceeds α. Runs in search mode (the curse lives in the optimisation layer).
On benign real data the violation may be mild — the synthetic adversarial instance shows the clean FAIL.
"""

import numpy as np

from nmscrc import losses
from nmscrc.methods._engine import calib_stats, eval_on_test


def run(A, calib_idx, test_idx, xi, alpha, delta):
    n = len(calib_idx)
    m1, m2 = len(A.lambda1), len(A.lambda2)
    g_c, L_c = A.g[calib_idx], A.L_f1[calib_idx]
    U, phi, R, _ = calib_stats(g_c, L_c, A.lambda1)

    eps = losses.eps_dkw(n, delta["dU"])
    if not (phi >= (xi + eps)).any():
        return _abstain(A, xi, alpha, n, "first_stage_empty")
    i1_min = int(np.argmax(phi >= (xi + eps)))

    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.where(phi[:, None] > 0, R / phi[:, None], np.inf)
    feasible = rhat <= alpha                                   # BARE empirical rule (NO union, NO t_V)
    feasible[:i1_min, :] = False
    if not feasible.any():
        return _abstain(A, xi, alpha, n, "second_stage_empty")

    SS = A.SS[calib_idx]                                   # precomputed set sizes
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_size = (U.T @ SS) / np.maximum(U.sum(axis=0)[:, None], 1.0)
    masked = np.where(feasible, mean_size, np.inf)
    i1, j = divmod(int(np.argmin(masked)), m2)

    ev = eval_on_test(A, test_idx, i1, j, "L_f1")
    return {"abstained": False, "lambda1": float(A.lambda1[i1]), "lambda2": float(A.lambda2[j]),
            "xi": xi, "alpha": alpha, "n": n, "m1": m1, "m2": m2,
            "coverage": ev["coverage"], "risk_cond": ev["risk_cond"], "set_size": ev["set_size"],
            "excess": (ev["risk_cond"] - alpha) if np.isfinite(ev["risk_cond"]) else None,
            "phi_hat": float(phi[i1]), "lambda1_mode": "search"}


def _abstain(A, xi, alpha, n, why):
    return {"abstained": True, "why": why, "lambda1": None, "lambda2": None, "xi": xi, "alpha": alpha,
            "n": n, "m1": len(A.lambda1), "m2": len(A.lambda2), "coverage": None, "risk_cond": None,
            "set_size": None, "excess": None, "lambda1_mode": "search"}
