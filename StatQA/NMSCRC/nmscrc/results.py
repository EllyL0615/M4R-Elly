"""Output contract: every experiment writes only structured json; plotting reads these
json and NEVER recomputes. This module defines the schema + a writer that fills unused -> null.
"""

import json
from pathlib import Path

# Superset of fields; any unused field is null.
FIELDS = [
    "exp", "method", "rep", "seed", "split_hash",
    "xi", "alpha", "ell_star", "delta", "m1", "m2", "n",
    "coverage", "risk_cond", "risk_marginal", "set_size",
    "abstained", "state",
    "K", "K_over_M",
    "echo_rate",                       # raw-LLM only; null for probe-based methods
    "excess",                          # DIAGNOSTIC: risk_cond - alpha (test gap)
    "cert_half_width",                 # C1 axis: (t_V + alpha*t_U)/phi_hat  (Thm 4.15 certificate width)
    "t_V", "t_V_eb",                   # Hoeffding / EB second-stage half-width (one is null per variant)
    "t_U", "t_U_eb",                   # DKW eps / EB selection half-width  (one is null per variant)
    "R_hat", "phi_hat", "phi_lcb",     # R̂(λ1,λ2), φ̂(λ1), ξ̂_LCB = φ̂ - t_U  (rebuild C1 from json)
    "lambda1", "lambda2", "lambda1_mode",   # 'search' (optimisation layer, Prop 4.4) | 'pinned' (Def 3.1)
    "recall_risk", "true_f1_risk",     # xu_proxy two-number report
    "rung", "extra",                   # model rung label + method-specific extras
]


def make_result(**kw):
    r = {k: None for k in FIELDS}
    for k, v in kw.items():
        if k not in r:
            r.setdefault("extra", {})
            if r["extra"] is None:
                r["extra"] = {}
            r["extra"][k] = v
        else:
            r[k] = v
    return r


def write_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=float)
    return path


def write_jsonl(path, rows):
    """One JSON object per rep-line (per-rep records, batched for fast IO)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    return path
