"""C2 / Theorem 4.9 — transductive leave-one-out (LOO) certificate (PURE functions).

DISTINCT from exp6's INDUCTIVE union-tax certificate. This is the transductive LOO *vacuity*: as the
Λ2 grid refines (m2 up), the operational LOO certificate slides from ~α toward α+B, while the true
held-out risk stays ~α. Same m2 axis as exp6, different object.

Deterministic LOO identity (thesis Thm 4.9):  R̂^(−j)(λ) = (M·L̄(λ) − L_j(λ)) / (M−1).
Operational instability:  λ̂₂^(−j) = first t with R̂^(−j)[t] ≤ α − B/(M−1);  K_direct = #{j: λ̂₂^(−j) ≠ t⋆}.
Closed form (Eq 4.3):     c = α + B − M·Δ ;  K_closed = #{j: L_j(λ⋆) < c}.   (the two must agree)
Certificate:              cert = α + K_direct·B/M.
"""

import numpy as np


def loo_risk_matrix(L):
    """R̂^(−j)[t] = mean_{k≠j} L[k,t] via the deterministic identity (M·L̄ − L)/(M−1). Shape (M, m2)."""
    M = L.shape[0]
    Lbar = L.mean(axis=0)
    return (M * Lbar[None, :] - L) / (M - 1)


def loo_instability_count(L, alpha, B=1.0):
    """K_direct (operational LOO) + K_closed (Thm 4.9 Eq 4.3) for an accepted-bag loss matrix L (M,m2).
    Returns None if the bag is infeasible (no t with L̄(t) ≤ α)."""
    M = L.shape[0]
    Lbar = L.mean(axis=0)
    feas = np.where(Lbar <= alpha)[0]
    if feas.size == 0:
        return None
    tstar = int(feas[0])
    Delta = float(alpha - Lbar[tstar])
    Rloo = (M * Lbar[None, :] - L) / (M - 1)                      # LOO identity (no literal recompute)
    fl = Rloo <= (alpha - B / (M - 1))                           # calibrated LOO target, m⋆ = M−1
    has = fl.any(axis=1)
    lhat = np.where(has, fl.argmax(axis=1), -1)                  # first feasible t per j; -1 if none
    K_direct = int(np.sum(lhat != tstar))                       # incl. LOO-infeasible (definitely shifted)
    c = alpha + B - M * Delta
    K_closed = int(np.sum(L[:, tstar] < c))
    return {"M": M, "t_star": tstar, "Delta": Delta, "c": float(c),
            "K_direct": K_direct, "K_closed": K_closed}


def transductive_loo_certificate(L, alpha, lambda2, B=1.0):
    """Full per-bag record: t⋆, λ2⋆, Δ, local slope ŝ, K_direct, K_closed, cert=α+K·B/M."""
    ic = loo_instability_count(L, alpha, B)
    if ic is None:
        return {"infeasible": True}
    t, M = ic["t_star"], ic["M"]
    Lbar = L.mean(axis=0)
    if t == 0:
        s_hat = abs(Lbar[1] - Lbar[0]) / abs(lambda2[1] - lambda2[0]) if len(lambda2) > 1 else 0.0
    else:
        s_hat = abs(Lbar[t - 1] - Lbar[t]) / abs(lambda2[t] - lambda2[t - 1])
    return {"infeasible": False, "M": M, "t_star": t, "lambda2_star": float(lambda2[t]),
            "Delta": ic["Delta"], "s_hat": float(s_hat), "K_direct": ic["K_direct"],
            "K_closed": ic["K_closed"], "cert": float(alpha + ic["K_direct"] * B / M)}


def assert_loo_identity():
    """Unit test: R̂^(−j)[t] from the identity == the literal leave-one-out mean. Fail loud."""
    rng = np.random.default_rng(0)
    L = rng.random((6, 5))
    Rloo = loo_risk_matrix(L)
    for j in range(L.shape[0]):
        ref = L[[k for k in range(L.shape[0]) if k != j]].mean(axis=0)
        if not np.allclose(Rloo[j], ref):
            raise AssertionError(f"LOO identity FAILED at j={j}")
    return True
