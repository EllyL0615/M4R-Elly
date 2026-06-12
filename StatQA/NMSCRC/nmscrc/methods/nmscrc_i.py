"""NM-SCRC-I (inductive) — PAC, accepted-region conditional caliber.

First stage:  λ̂1' = inf{ λ1 ∈ Λ1 : φ̂(λ1) >= ξ + t_U }
Second stage over (λ1>=λ̂1', λ2), feasible iff rule (4.1):
              r̂(λ1,λ2) <= α − (t_V + α·t_U)/φ̂(λ1)
Feasible set nonempty -> pick the pair with the SMALLEST measured set (min mean |C| on the calib
accepted region; tie -> max coverage); else ABSTAIN.

variant="hoeffding":  t_U = DKW ε (uniform over Λ1),  t_V = Hoeffding union over Λ1×Λ2.
variant="eb":         t_U = t_U^EB (Maurer-Pontil on 1-U, union Λ1),
                      t_V = t_V^EB (Maurer-Pontil on U·ℓ, union Λ1×Λ2).

The reported half-widths are variant-specific (t_V xor t_V_eb; t_U xor t_U_eb) so the C1 finding
(EB cert/φ̂ ∝ 1/√φ̂ slope −0.5 vs Hoeffding ∝ 1/φ̂ slope −1.0) is rebuildable from the json.
"""

import numpy as np

from nmscrc import losses
from nmscrc.methods._engine import calib_stats, eval_on_test


def run(A, calib_idx, test_idx, xi, alpha, delta, variant="eb", lam1_mode="search"):
    """lam1_mode: 'search' = joint (λ1>=λ̂1', λ2) optimisation (full NM-SCRC, set-size layer, Prop 4.4
    winner's curse lives here); 'pinned' = λ1 fixed at λ̂1' (pure Def 3.1 validity, coverage ~ ξ)."""
    n = len(calib_idx)
    m1, m2 = len(A.lambda1), len(A.lambda2)
    g_c = A.g[calib_idx]
    L_c = A.L_f1[calib_idx]
    U, phi, R, S2 = calib_stats(g_c, L_c, A.lambda1)

    # ---- selection slack t_U (DKW scalar broadcast, or EB per-λ1 on 1-U) ----
    if variant == "hoeffding":
        tU = np.full(m1, losses.eps_dkw(n, delta["dU"]))
    elif variant == "eb":
        logU = np.log(m1 / delta["dU"])
        tU = losses.eb_bound((1.0 - U).var(axis=0, ddof=1), n, logU)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    # second-stage slack t_V (Hoeffding constant, or EB per-cell on U·ℓ) — computed up front so the
    # well-defined half-widths are reported even on ABSTAIN.
    if variant == "hoeffding":
        tV = np.full((m1, m2), losses.t_hoeffding(n, m1, m2, delta["dV"]))
        tV_rep, tU_rep, tV_eb_rep, tU_eb_rep = float(tV[0, 0]), float(tU[0]), None, None
    else:
        var_Z = (S2 - R ** 2) * (n / (n - 1))
        tV = losses.eb_bound(var_Z, n, np.log(m1 * m2 / delta["dV"]))
        tV_rep, tU_rep, tV_eb_rep, tU_eb_rep = None, None, None, None

    if not (phi >= (xi + tU)).any():
        return _abstain(A, xi, alpha, n, variant, "first_stage_empty",
                        t_V=tV_rep, t_U=tU_rep, t_V_eb=tV_eb_rep, t_U_eb=tU_eb_rep, lam1_mode=lam1_mode)
    i1_min = int(np.argmax(phi >= (xi + tU)))
    if variant == "eb":
        tU_eb_rep = float(tU[i1_min])

    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.where(phi[:, None] > 0, R / phi[:, None], np.inf)

    with np.errstate(divide="ignore", invalid="ignore"):
        rhs = alpha - (tV + alpha * tU[:, None]) / phi[:, None]
    feasible = rhat <= rhs
    feasible[:i1_min, :] = False
    if lam1_mode == "pinned":
        feasible[i1_min + 1:, :] = False          # λ1 fixed at λ̂1' (no set-size search over λ1)
    if not feasible.any():
        return _abstain(A, xi, alpha, n, variant, "second_stage_empty",
                        t_V=tV_rep, t_U=tU_rep, t_V_eb=tV_eb_rep, t_U_eb=tU_eb_rep, lam1_mode=lam1_mode)

    # ---- smallest set: minimise measured mean |C| on the calib accepted region; tie -> max φ ----
    SS = A.SS[calib_idx]                                   # (n,m2) precomputed set sizes
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_size = (U.T @ SS) / np.maximum(U.sum(axis=0)[:, None], 1.0)               # (m1,m2)
    masked = np.where(feasible, mean_size, np.inf)
    flat = int(np.argmin(masked))
    i1, j = divmod(flat, m2)
    ties = np.where(masked == masked[i1, j])
    if len(ties[0]) > 1:
        k = int(np.argmax(phi[ties[0]]))
        i1, j = int(ties[0][k]), int(ties[1][k])

    ev = eval_on_test(A, test_idx, i1, j, "L_f1")
    cert = float((tV[i1, j] + alpha * tU[i1]) / phi[i1])
    out = {
        "abstained": False, "variant": variant,
        "lambda1": float(A.lambda1[i1]), "lambda2": float(A.lambda2[j]),
        "xi": xi, "alpha": alpha, "n": n, "m1": m1, "m2": m2,
        "coverage": ev["coverage"], "risk_cond": ev["risk_cond"], "set_size": ev["set_size"],
        "excess": (ev["risk_cond"] - alpha) if np.isfinite(ev["risk_cond"]) else None,
        "cert_half_width": cert,
        "R_hat": float(R[i1, j]), "phi_hat": float(phi[i1]), "phi_lcb": float(phi[i1] - tU[i1]),
        "t_V": float(tV[i1, j]) if variant == "hoeffding" else None,
        "t_V_eb": float(tV[i1, j]) if variant == "eb" else None,
        "t_U": float(tU[i1]) if variant == "hoeffding" else None,
        "t_U_eb": float(tU[i1]) if variant == "eb" else None,
        "lambda1_mode": lam1_mode,
    }
    return out


def _abstain(A, xi, alpha, n, variant, why, t_V=None, t_U=None, t_V_eb=None, t_U_eb=None, lam1_mode=None):
    return {
        "abstained": True, "variant": variant, "why": why,
        "lambda1": None, "lambda2": None, "xi": xi, "alpha": alpha, "n": n,
        "m1": len(A.lambda1), "m2": len(A.lambda2),
        "coverage": None, "risk_cond": None, "set_size": None, "excess": None,
        "cert_half_width": None, "R_hat": None, "phi_hat": None, "phi_lcb": None,
        "t_V": t_V, "t_V_eb": t_V_eb, "t_U": t_U, "t_U_eb": t_U_eb, "lambda1_mode": lam1_mode,
    }
