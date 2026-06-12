"""NM-SCRC-T (transductive, symmetric) — in-expectation, exact coverage (thesis Thm 4.10,
Eq 4.3, framework.tex). Symmetric first-stage threshold (top-m_star by g) gives exact coverage >= ξ;
on the accepted bag run calibrated CRC-NM with the two-term non-monotone correction D(m2, m_star);
report the selection-instability count K (closed form 4.3) and K/M.

  M       = ceil(ξ (n+1)),  m_star = M - 1  (accepted calib points; bag incl. test has M)
  D(m2,m_star) = B*sqrt(log(2 m2)/(2 m_star)) + B/(2 sqrt(2 m_star log(2 m2)))   (Aldirawi 2-term)
  deployed λ̂2 = inf{ λ2 : (m_star/(m_star+1)) R̂(λ2) + B/(m_star+1) <= α − D }     (calibrated)
  λ⋆      = inf{ λ2 : L̄_M(λ2) <= α }   (full-bag threshold for the instability count)
  K       = #{ j in bag : L_j(λ⋆) < c },  c = α + B − M δ,  δ = α − L̄_M(λ⋆)        (Eq 4.3)
  risk(sel) <= α + 2 K B / M  (exact, assumption-free, Thm 4.10)
"""

import numpy as np


def _D(m2, m_star, B=1.0):
    return B * np.sqrt(np.log(2 * m2) / (2 * m_star)) + B / (2 * np.sqrt(2 * m_star * np.log(2 * m2)))


def instability_count(L_bag, alpha, M, B=1.0):
    """Eq 4.3 on a bag loss matrix L_bag (m_star, m2). Returns dict(lam2_star_idx, mu, delta, c, K, K_over_M)."""
    Lbar = L_bag.mean(axis=0)                              # (m2,) full-bag mean loss per λ2
    feas = np.where(Lbar <= alpha)[0]
    js = int(feas[0]) if feas.size else int(np.argmin(Lbar))
    mu = float(Lbar[js])
    delta = alpha - mu
    c = alpha + B - M * delta
    Lstar = L_bag[:, js]                                  # per-point loss at λ⋆
    K = int((Lstar < c).sum())
    return {"lam2_star_idx": js, "mu": mu, "delta": float(delta), "c": float(c),
            "K": K, "K_over_M": K / M, "M": int(M), "feasible": bool(feas.size)}


def run(A, calib_idx, test_idx, xi, alpha, delta=None, B=1.0):
    n = len(calib_idx)
    m1, m2 = len(A.lambda1), len(A.lambda2)
    M = int(np.ceil(xi * (n + 1)))
    m_star = M - 1
    if m_star < 2:
        return _abstain(A, xi, alpha, n, "m_star<2")

    # symmetric first-stage: accept the top-m_star calib points by g (bag); threshold tau
    g_c = A.g[calib_idx]
    order = np.argsort(-g_c)
    bag_local = order[:m_star]
    tau = float(g_c[bag_local[-1]])                      # acceptance threshold g >= tau
    L_bag = A.L_f1[calib_idx][bag_local]                 # (m_star, m2)

    # instability count (Eq 4.3) on the bag
    ic = instability_count(L_bag, alpha, M, B)

    # deployed calibrated CRC-NM threshold on the bag
    Rhat = L_bag.mean(axis=0)                            # (m2,)
    Dcorr = _D(m2, m_star, B)
    lhs = (m_star / (m_star + 1)) * Rhat + B / (m_star + 1)
    feas = np.where(lhs <= alpha - Dcorr)[0]
    if feas.size == 0:
        out = _abstain(A, xi, alpha, n, "crcnm_infeasible")
        out.update({"K": ic["K"], "K_over_M": ic["K_over_M"]})
        return out
    j = int(feas[0])

    # evaluate on test accepted region (g >= tau)
    g_t = A.g[test_idx]
    acc = g_t >= tau
    coverage = float(acc.mean())
    risk = float(A.L_f1[test_idx][acc, j].mean()) if acc.sum() else float("nan")
    set_size = float(A.SS[test_idx][acc, j].mean()) if acc.sum() else float("nan")

    return {
        "abstained": False, "lambda1": float(1.0 - tau), "lambda2": float(A.lambda2[j]),
        "xi": xi, "alpha": alpha, "n": n, "m1": m1, "m2": m2,
        "coverage": coverage, "risk_cond": risk, "set_size": set_size,
        "excess": (risk - alpha) if np.isfinite(risk) else None,
        "K": ic["K"], "K_over_M": ic["K_over_M"], "M": ic["M"], "m_star": m_star,
        "D": float(Dcorr), "mu": ic["mu"], "delta_slack": ic["delta"],
    }


def _abstain(A, xi, alpha, n, why):
    return {"abstained": True, "why": why, "lambda1": None, "lambda2": None,
            "xi": xi, "alpha": alpha, "n": n, "m1": len(A.lambda1), "m2": len(A.lambda2),
            "coverage": None, "risk_cond": None, "set_size": None, "excess": None,
            "K": None, "K_over_M": None}
