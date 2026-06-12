"""Synthetic hard instance (Stage 1) — confirm the C1/C2 predictions (EB cert slope −0.5 vs
Hoeffding −1.0; feasibility floor ∝ √(log/n); union-tax ∝ √(log m)) BEFORE touching real data.

hard_family = aldirawi_prop1 : the mean-loss curve L̄(λ2) crosses α with a SHALLOW ramp (slope δ), so
many grid points sit near α -> the Λ2-union is stressed (worst case for the union bound). At the
boundary λ2* the per-point loss is Bernoulli(α): conditional variance α(1−α), which is exactly what
makes the EB half-width scale as √(α(1−α) φ log/n) -> cert/φ ∝ 1/√φ (slope −0.5), while Hoeffding's
φ-blind t_V gives cert/φ ∝ 1/φ (slope −1.0).
coupling = rejected_constant_B : rejected points carry loss ≡ B (they do not enter the accepted-region
risk; they only affect the marginal view).
"""

import numpy as np

from nmscrc import losses
from nmscrc.artifacts import Artifacts


def build_synthetic_artifacts(n, m1, m2, alpha, delta, seed,
                              family="aldirawi_prop1", coupling="rejected_constant_B"):
    rng = np.random.default_rng(seed)
    g = rng.random(n)                                          # random selector -> coverage = ξ exactly
    lam2 = losses.lambda_grid(m2, 0.0, 1.0)
    lam1 = losses.lambda_grid(m1, 0.0, 1.0)

    # shallow ramp: mean loss crosses α at λ2* (middle), slope δ -> many grid points near α (hard).
    j_star = m2 // 2
    mean_j = np.clip(alpha + (j_star - np.arange(m2)) * delta, 0.0, 1.0)   # decreasing in λ2
    q = 1.0 - mean_j                                           # per-grid acceptance quantile
    u = rng.random(n)
    L = (u[:, None] > q[None, :]).astype(np.float32)           # ℓ_i(λ2)=1{u_i>q_j}; mean_j at col j

    p = rng.random((n, 27))                                    # dummy scores (set sizes only)
    Y = (rng.random((n, 27)) < 0.06).astype(np.float64)
    SS = (p[:, :, None] >= (1.0 - lam2)[None, None, :]).sum(axis=1).astype(np.int16)
    A = Artifacts(model=f"synthetic_{family}", p=p, g=g, Y=Y, L_f1=L, L_recall=L,
                  lambda1=lam1, lambda2=lam2, split_hash=f"synthetic:{family}:{coupling}", SS=SS)
    A.synthetic = {"alpha": alpha, "delta": delta, "j_star": int(j_star), "mean_at_star": float(mean_j[j_star]),
                   "family": family, "coupling": coupling, "n": n, "m1": m1, "m2": m2}
    return A


def build_synthetic_f1(n, m1, m2, seed, K=27, n_easy=2, n_hard=3, n_conf=4,
                       sep_easy=2.5, sep_hard=0.0, sep_neg=-2.5, sd=0.8):
    """FIXED, principled weak-probe multi-label instance (negative controls). NOT tuned to a
    target outcome. Each row has `n_easy` clearly-ranked + `n_hard` low-logit true methods,
    plus `n_conf` confusable negatives sharing the hard-positive logit level (so high recall drags in
    false positives -> a genuine precision penalty). build_artifacts yields a REAL F1 U-shape +
    monotone recall-loss, so the predicted failure modes can be measured honestly:
      naive   -> winner's curse over the (λ1,λ2) search;
      mono    -> non-monotone F1 -> the counting inf-crossing is not a valid certificate;
      xu_proxy-> recall-loss PASSes, but the recall-sized set carries the precision penalty (true-F1 gap).
    Whether each bites, and how hard, is reported as-measured.
    """
    from nmscrc.artifacts import build_artifacts
    rng = np.random.default_rng(seed)
    Y = np.zeros((n, K), dtype=np.float64)
    logit = rng.normal(sep_neg, sd, (n, K))
    for i in range(n):
        perm = rng.permutation(K)
        pos, conf = perm[:n_easy + n_hard], perm[n_easy + n_hard:n_easy + n_hard + n_conf]
        Y[i, pos] = 1.0
        logit[i, pos[:n_easy]] = rng.normal(sep_easy, sd, n_easy)
        logit[i, pos[n_easy:]] = rng.normal(sep_hard, sd, n_hard)
        logit[i, conf] = rng.normal(sep_hard, sd, n_conf)        # confusable negatives
    lam1 = losses.lambda_grid(m1, 0.0, 1.0)
    lam2 = losses.lambda_grid(m2, 0.0, 1.0)
    return build_artifacts("synthetic_f1", logit, Y, lam1, lam2, split_hash="synthetic_f1:fixed")

