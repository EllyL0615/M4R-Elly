"""mono-counting CRC (opponent; FAIL layer).

Xu's augmented-loss CRC engine fed the RAW F1-loss (pretends F1 is monotone). F1 is U-shaped, so
R̂v is non-monotone and the inf-crossing does NOT certify the population risk -> expect FAIL on the
true conditional F1-risk. Same first stage + grid + frozen f,g as NM-SCRC (strongest, attributable
form): the ONLY difference from xu_proxy is the loss fed to the engine.
"""

import numpy as np

from nmscrc.methods._engine import first_stage_dkw, xu_engine, eval_on_test


def run(A, calib_idx, test_idx, xi, alpha, delta):
    n = len(calib_idx)
    fs = first_stage_dkw(A.g[calib_idx], A.lambda1, xi, delta["dU"])
    if fs is None:
        return _abstain(A, xi, alpha, n, "first_stage_empty")
    i1, phi, eps, xi_lcb = fs
    U = (A.g[calib_idx] >= (1.0 - A.lambda1[i1])).astype(np.float64)
    j = xu_engine(U, A.L_f1[calib_idx], alpha * xi_lcb)        # F1-loss (NON-monotone -> invalid)
    if j is None:
        return _abstain(A, xi, alpha, n, "engine_infeasible")
    ev = eval_on_test(A, test_idx, i1, j, "L_f1")
    return {"abstained": False, "lambda1": float(A.lambda1[i1]), "lambda2": float(A.lambda2[j]),
            "xi": xi, "alpha": alpha, "n": n, "m1": len(A.lambda1), "m2": len(A.lambda2),
            "coverage": ev["coverage"], "risk_cond": ev["risk_cond"], "set_size": ev["set_size"],
            "excess": (ev["risk_cond"] - alpha) if np.isfinite(ev["risk_cond"]) else None,
            "xi_lcb": xi_lcb, "phi_hat": float(phi[i1])}


def _abstain(A, xi, alpha, n, why):
    return {"abstained": True, "why": why, "lambda1": None, "lambda2": None, "xi": xi, "alpha": alpha,
            "n": n, "m1": len(A.lambda1), "m2": len(A.lambda2), "coverage": None, "risk_cond": None,
            "set_size": None, "excess": None}
