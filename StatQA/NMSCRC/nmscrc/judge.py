"""One judging function, used everywhere. Three states stay THREE.

Per rep:
  ABSTAIN  — the method declined to emit (feasibility failed).
  PASS     — emitted and test risk <= α.
  FAIL     — emitted and test risk > α.

Method verdict over the 100 reps (a rep is "non-violating" if it abstained OR risk<=α; abstaining
is honest safety, not a violation):
  PAC methods (I, opponents):  PASS iff mean(non-violating) >= 1-δ AND abstain_rate < 1
                               (AND coverage target met to the same standard when checked).
  in-expectation (T):          PASS iff mean(test risk over emitted reps) <= α AND abstain_rate < 1.
  FAIL otherwise (controlled fraction below nominal). A method that ALWAYS abstains -> ABSTAIN.
"""

import numpy as np

PASS, FAIL, ABSTAIN = "PASS", "FAIL", "ABSTAIN"


def rep_state(abstained, risk_cond, alpha):
    if abstained:
        return ABSTAIN
    return PASS if risk_cond <= alpha else FAIL


def method_verdict(states, risks, alpha, delta, *, kind="pac",
                   coverages=None, xi=None, cover_slack=0.0):
    """Aggregate per-rep states/risks into one PASS/FAIL/ABSTAIN verdict."""
    states = list(states)
    risks = np.asarray(risks, dtype=float)
    n = len(states)
    abstain_mask = np.array([s == ABSTAIN for s in states])
    abstain_rate = float(abstain_mask.mean()) if n else 1.0
    emitted = ~abstain_mask

    out = {"abstain_rate": abstain_rate, "n_reps": n,
           "n_pass": int(sum(s == PASS for s in states)),
           "n_fail": int(sum(s == FAIL for s in states)),
           "n_abstain": int(abstain_mask.sum())}

    if abstain_rate >= 1.0:                       # cannot pass by always abstaining
        out["verdict"] = ABSTAIN
        out["controlled_fraction"] = float("nan")
        return out

    cover_ok = True
    if coverages is not None and xi is not None and emitted.any():
        cov = np.asarray(coverages, dtype=float)[emitted]
        cover_ok = bool((cov >= xi - cover_slack).mean() >= 1.0 - delta)

    if kind == "pac":
        non_violating = abstain_mask | (risks <= alpha)   # abstain = safe
        frac = float(non_violating.mean())
        out["controlled_fraction"] = frac
        out["verdict"] = PASS if (frac >= 1.0 - delta and abstain_rate < 1.0 and cover_ok) else FAIL
    elif kind == "expectation":
        mean_risk = float(np.nanmean(risks[emitted])) if emitted.any() else float("nan")
        out["controlled_fraction"] = mean_risk
        out["verdict"] = PASS if (mean_risk <= alpha and abstain_rate < 1.0 and cover_ok) else FAIL
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return out
