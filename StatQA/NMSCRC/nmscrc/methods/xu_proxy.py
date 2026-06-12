"""Xu-SCRC on a monotone proxy (opponent).

Xu's engine fed the MONOTONE recall-loss (the most charitable monotonization a Xu user would pick).
The engine is VALID and controls recall, but no monotone loss can encode F1's precision penalty, so
the TRUE F1-risk it incurs is uncontrolled (expected >> α) and its sets are typically larger than
NM-SCRC's. Reports TWO numbers on the test accepted region:
  (i)  recall-risk  E[ℓ_recall | accepted]  -> PASS on its own (proxy) objective
  (ii) true F1-risk E[1 − F1   | accepted]  -> expected >> α  (the real objective, uncontrolled)
Judged (PASS/FAIL) on the TRUE F1-risk (the real objective); recall-risk reported alongside.
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
    j = xu_engine(U, A.L_recall[calib_idx], alpha * xi_lcb)    # recall-loss (monotone, VALID)
    if j is None:
        return _abstain(A, xi, alpha, n, "engine_infeasible")
    ev_rec = eval_on_test(A, test_idx, i1, j, "L_recall")     # (i) recall-risk on accepted
    ev_f1 = eval_on_test(A, test_idx, i1, j, "L_f1")          # (ii) TRUE F1-risk on accepted
    return {"abstained": False, "lambda1": float(A.lambda1[i1]), "lambda2": float(A.lambda2[j]),
            "xi": xi, "alpha": alpha, "n": n, "m1": len(A.lambda1), "m2": len(A.lambda2),
            "coverage": ev_f1["coverage"], "risk_cond": ev_f1["risk_cond"],   # judged on TRUE F1
            "recall_risk": ev_rec["risk_cond"], "true_f1_risk": ev_f1["risk_cond"],
            "set_size": ev_f1["set_size"],
            "excess": (ev_f1["risk_cond"] - alpha) if np.isfinite(ev_f1["risk_cond"]) else None,
            "xi_lcb": xi_lcb, "phi_hat": float(phi[i1])}


def _abstain(A, xi, alpha, n, why):
    return {"abstained": True, "why": why, "lambda1": None, "lambda2": None, "xi": xi, "alpha": alpha,
            "n": n, "m1": len(A.lambda1), "m2": len(A.lambda2), "coverage": None, "risk_cond": None,
            "recall_risk": None, "true_f1_risk": None, "set_size": None, "excess": None}
