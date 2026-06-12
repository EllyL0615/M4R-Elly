"""CRC-NM-marginal (opponent; DIFFERENT caliber) — Aldirawi et al. 2026 Thm 1.

NO selection, NO ratio. Controls the MARGINAL risk E[ℓ] <= α over ALL points (a different quantity
from NM-SCRC's accepted-region conditional risk — never compare on the same caliber column). Genuine
marginal Aldirawi with the two-term correction D(m2, n) and the inf∅ = λ_max fallback (NOT abstain).
"""

import numpy as np

from nmscrc import losses


def run(A, calib_idx, test_idx, xi, alpha, delta, B=1.0):
    n = len(calib_idx)
    m2 = len(A.lambda2)
    R = A.L_f1[calib_idx].mean(axis=0)                         # over ALL calib points; no selection
    D = B * np.sqrt(np.log(2 * m2) / (2 * n)) + B / (2 * np.sqrt(2 * n * np.log(2 * m2)))
    feas = np.where((n / (n + 1)) * R + B / (n + 1) <= alpha - D)[0]
    j = int(feas[0]) if feas.size else (m2 - 1)               # inf∅ = λ_max  (FALLBACK, not abstain)

    risk_marg = float(A.L_f1[test_idx][:, j].mean())          # MARGINAL: ALL test points
    set_size = float(A.SS[test_idx][:, j].mean())
    return {"abstained": False, "lambda1": None, "lambda2": float(A.lambda2[j]),
            "xi": xi, "alpha": alpha, "n": n, "m1": len(A.lambda1), "m2": m2,
            "coverage": 1.0, "risk_cond": None, "risk_marginal": risk_marg, "set_size": set_size,
            "excess": risk_marg - alpha, "D": float(D), "fallback_lambda_max": bool(feas.size == 0)}
