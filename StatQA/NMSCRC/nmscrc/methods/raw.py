"""raw-LLM baseline (floor; NOT judged PASS/FAIL).

Take the LLM's own method set from `model_answer` (the answer.csv multi-hot; echo/unknown/unparseable
already scored as the empty set during Stage 0c), no calibration, no selection. Report the marginal
F1-risk (>> α) AND the echo_rate. Lives in the motivation table, not the PASS/FAIL table.
"""

import numpy as np


def f1_loss_rows(pred, Y):
    inter = (pred * Y).sum(axis=1)
    denom = pred.sum(axis=1) + Y.sum(axis=1)
    f1 = np.where(denom > 0, 2.0 * inter / denom, 0.0)        # empty prediction -> F1=0 -> loss=1
    return 1.0 - f1


def run(A, calib_idx, test_idx, xi, alpha, answer_matrix, echo_rate):
    """answer_matrix: (n_pool, 27) raw-LLM multi-hot (echo -> all-zero). Risk on the rep's test rows."""
    pred = answer_matrix[test_idx]
    risk = float(f1_loss_rows(pred, A.Y[test_idx]).mean())
    return {"abstained": False, "lambda1": None, "lambda2": None, "xi": xi, "alpha": alpha,
            "n": len(calib_idx), "m1": len(A.lambda1), "m2": len(A.lambda2),
            "coverage": 1.0, "risk_cond": None, "risk_marginal": risk,
            "set_size": float(pred.sum(axis=1).mean()), "excess": risk - alpha,
            "echo_rate": float(echo_rate)}
