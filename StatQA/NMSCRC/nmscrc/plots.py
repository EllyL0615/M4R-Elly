"""Plotting + tables (read results/*.json ONLY; never recompute). Each function loads the
relevant experiment's json and returns a matplotlib Figure (rendered inline in the notebook) and/or
a DataFrame. Style is whatever rcParams the caller set.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nmscrc import experiments as E, judge, paths


def _group(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in keys)].append(r)
    return out


def _risk(r):
    return r["risk_cond"] if r["risk_cond"] is not None else r["risk_marginal"]


# ----------------------------------------------------------------- exp1 PAC histogram
def plot_exp1(delta_gap=0.05):
    rows = E.load_results("exp1")
    tag = f"d{int(delta_gap*100):02d}"
    rungs = ["llama3_2_1b", "llama3_2_3b", "llama3_1_8b"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, rung in zip(axes, rungs):
        eb = [r for r in rows if r["rung"] == rung and r["method"] == "nmscrc_i_eb"
              and abs(r["alpha"] - (r["ell_star"] + delta_gap)) < 1e-9]
        if not eb:
            continue
        alpha = eb[0]["alpha"]
        risks = np.array([_risk(r) for r in eb if not r["abstained"]], dtype=float)
        frac = np.mean([(r["abstained"] or _risk(r) <= alpha) for r in eb])
        ax.hist(risks, bins=25, color="#4C72B0", alpha=0.85)
        ax.axvline(alpha, color="crimson", lw=2, label=f"α={alpha:.3f}")
        ax.set_title(f"{rung}\nP(risk≤α or abstain)={frac:.2f} (≥1−δ=0.9)")
        ax.set_xlabel("test conditional F1-risk"); ax.legend(fontsize=8)
    axes[0].set_ylabel("reps")
    fig.suptitle(f"① NM-SCRC-I-EB validity (PAC histogram over 100 reps, Δ={delta_gap}, search)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- exp3 U-shape
def plot_exp3():
    import json
    rungs = ["llama3_2_1b", "llama3_2_3b", "llama3_1_8b"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for rung in rungs:
        p = paths.ushape_json(rung)
        if not p.exists():
            continue
        d = json.loads(p.read_text())["extra"]
        ax.plot(d["set_size"], d["cond_risk"], marker=".", ms=4, label=rung)
    ax.set_xlabel("mean set size |C| (accepted region)"); ax.set_ylabel("conditional F1-risk")
    ax.set_title("③ F1-risk is U-shaped in set size (non-monotone)"); ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- exp5 ξ-slope (C1)
def plot_exp5():
    rows = E.load_results("exp5")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    summary = {}
    for ax, rung in zip(axes, ["llama3_2_1b", "synthetic"]):
        for variant, col in [("eb", "#2E7D32"), ("hoeffding", "#C62828")]:
            g = _group([r for r in rows if r["rung"] == rung and r["method"] == f"nmscrc_i_{variant}"], "xi")
            xs, ys = [], []
            for xi, rs in sorted(g.items()):
                cov = [r["phi_hat"] for r in rs if r["phi_hat"] is not None]
                cert = [r["cert_half_width"] for r in rs if r["cert_half_width"] is not None]
                if cov and cert:
                    xs.append(np.mean(cov)); ys.append(np.mean(cert))
            if len(xs) >= 2:
                slope = np.polyfit(np.log(xs), np.log(ys), 1)[0]
                summary[(rung, variant)] = slope
                ax.plot(xs, ys, "o-", color=col, label=f"{variant} (slope {slope:.2f})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("measured coverage φ̂"); ax.set_ylabel("certificate half-width")
        ax.set_title(f"⑤ C1: {rung}\n(EB≈−0.5, Hoeffding≈−1.0)"); ax.legend(fontsize=8)
    fig.suptitle("⑤ ξ-slope (C1): EB recovers the optimal √-rate; Hoeffding pays 1/φ̂")
    fig.tight_layout()
    return fig, summary


# ----------------------------------------------------------------- exp6 phase transition
def plot_exp6():
    rows = E.load_results("exp6")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    for variant, col in [("eb", "#2E7D32"), ("hoeffding", "#C62828")]:
        g = _group([r for r in rows if r["method"] == f"nmscrc_i_{variant}"], "m2")
        ms, certs, absts = [], [], []
        for tagkey, rs in g.items():
            m2 = rs[0]["m2"]
            cert = [r["cert_half_width"] for r in rs if r["cert_half_width"] is not None]
            ms.append(m2); certs.append(np.mean(cert) if cert else np.nan)
            absts.append(np.mean([r["abstained"] for r in rs]))
        order = np.argsort(ms); ms = np.array(ms)[order]
        ax1.plot(ms, np.array(certs)[order], "o-", color=col, label=f"{variant} cert")
        ax2.plot(ms, np.array(absts)[order], "s--", color=col, alpha=0.5, label=f"{variant} abstain")
    ax1.set_xscale("log"); ax1.set_xlabel("m₂ (Λ₂ grid size)"); ax1.set_ylabel("certificate half-width")
    ax2.set_ylabel("abstain rate"); ax1.set_title("⑥ phase transition vs m₂ (8b)")
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- exp8 I vs T
def plot_exp8():
    rows = E.load_results("exp8")
    recs = []
    for (rung, method), rs in _group(rows, "rung", "method").items():
        risks = [_risk(r) for r in rs if not r["abstained"]]
        recs.append({"rung": rung, "method": method, "n": len(rs),
                     "abstain_rate": np.mean([r["abstained"] for r in rs]),
                     "mean_risk": np.nanmean(risks) if risks else np.nan,
                     "mean_cov": np.nanmean([r["coverage"] for r in rs if r["coverage"] is not None]),
                     "mean_K_over_M": np.nanmean([r["K_over_M"] for r in rs if r["K_over_M"] is not None])})
    return pd.DataFrame(recs).sort_values(["rung", "method"]).reset_index(drop=True)


# ----------------------------------------------------------------- Stage-1 synth: floor + union-tax
def plot_synth():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    floor = E.load_results("exp9_floor")
    for variant, col in [("eb", "#2E7D32"), ("hoeffding", "#C62828")]:
        g = _group([r for r in floor if r["method"] == f"nmscrc_i_{variant}"], "n")
        pts = sorted((int(rs[0]["n"]), float(np.mean([bool(r["abstained"]) for r in rs]))) for rs in g.values())
        if pts:
            xs, ys = zip(*pts)
            axes[0].plot(list(xs), list(ys), "o-", color=col, label=variant)
    axes[0].set_xscale("log"); axes[0].set_xlabel("n (calib)"); axes[0].set_ylabel("abstain rate")
    axes[0].set_title("⑨ feasibility floor (EB lower than Hoeffding)"); axes[0].legend()

    union = E.load_results("union_tax")
    for variant, col in [("eb", "#2E7D32"), ("hoeffding", "#C62828")]:
        g = _group([r for r in union if r["method"] == f"nmscrc_i_{variant}"], "m1")
        pts = []
        for rs in g.values():
            cert = [r["cert_half_width"] for r in rs if r["cert_half_width"] is not None]
            if cert:
                pts.append((int(rs[0]["m1"]) * int(rs[0]["m2"]), float(np.mean(cert))))
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            axes[1].plot(list(xs), list(ys), "o-", color=col, label=variant)
    axes[1].set_xscale("log"); axes[1].set_xlabel("m₁·m₂ (grid size)"); axes[1].set_ylabel("certificate half-width")
    axes[1].set_title("union-tax: cert grows ∝ √log(m₁m₂)"); axes[1].legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- mono grid-refinement (Prop 4.4)
def plot_mono_gridrefine():
    """mono (no cushion) FAILs as m₂ grows on BOTH rungs; NM-SCRC-EB (t_V cushion) stays at ~0."""
    rows = E.load_results("mono_gridrefine")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    styles = {("mono", "llama3_2_1b"): ("#C62828", "o-", "mono — 1b (no cushion)"),
              ("mono", "llama3_1_8b"): ("#8E0000", "s-", "mono — 8b (no cushion)"),
              ("nmscrc_i_eb", "llama3_2_1b"): ("#2E7D32", "o--", "NM-SCRC-EB — 1b (t_V cushion)"),
              ("nmscrc_i_eb", "llama3_1_8b"): ("#1B5E20", "s--", "NM-SCRC-EB — 8b (t_V cushion)")}
    for (method, rung), (col, ls, lab) in styles.items():
        g = _group([r for r in rows if r["method"] == method and r["rung"] == rung], "m2")
        pts = []
        for rs in g.values():
            viol = np.mean([(r["risk_cond"] is not None and r["risk_cond"] > r["alpha"]) for r in rs])
            pts.append((int(rs[0]["m2"]), float(viol)))
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            ax.plot(list(xs), list(ys), ls, color=col, lw=2, label=lab)
    ax.axhline(0.1, color="gray", ls=":", lw=1.5, label="FAIL threshold (δ = 0.1)")
    ax.annotate("NM-SCRC-EB: viol≈0, flat in m₂\n(t_V cushion immunises)", xy=(160, 0.02),
                xytext=(160, 0.30), fontsize=9, color="#1B5E20",
                arrowprops=dict(arrowstyle="->", color="#1B5E20"))
    ax.set_xscale("log"); ax.set_xlabel("m₂  (Λ₂ grid size — search density)")
    ax.set_ylabel("violation fraction  (test risk > α over 100 reps)")
    ax.set_title("mono's benign-PASS is grid luck (FAILs as the grid densifies, both rungs);\n"
                 "NM-SCRC's t_V cushion makes it immune to grid refinement")
    ax.legend(fontsize=8, loc="center left"); ax.set_ylim(-0.03, None)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- C2 transductive LOO cert (Thm 4.9)
def _ex(r, k):
    return (r.get("extra") or {}).get(k)


def plot_c2_transductive_cert():
    """Panel A: full-M cert slides α→α+B while true held-out risk stays ~α. Panel B: onset shifts
    RIGHT as M grows (Δ<(α+B)/M ⇒ onset ∝ M). Quantile Λ2 grid (option A)."""
    rows = E.load_results("c2_transductive_cert")
    alpha = rows[0]["alpha"]; B = 1.0
    grp = {}
    for r in rows:
        grp.setdefault((_ex(r, "M"), r["m2"]), []).append(r)
    Ms = sorted({_ex(r, "M") for r in rows})
    fullM = max(Ms)

    def curve(Msel):
        pts = []
        for (M, m2), rs in grp.items():
            if M != Msel:
                continue
            feas = [x for x in rs if not _ex(x, "infeasible")]
            if feas:
                pts.append((m2, float(np.mean([_ex(x, "cert") for x in feas])),
                            float(np.nanmean([_ex(x, "true_held_risk") for x in feas]))))
        return sorted(pts)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.5))
    pa = curve(fullM)
    m2s, certs, ths = zip(*pa)
    axA.plot(m2s, certs, "o-", color="#C62828", lw=2, label="LOO certificate  α+K·B/M")
    axA.plot(m2s, ths, "s-", color="#2E7D32", lw=2, label="true held-out risk")
    axA.axhline(alpha, ls=":", color="gray", label=f"α={alpha:.3f}")
    axA.axhline(alpha + B, ls="--", color="black", label=f"α+B={alpha + B:.3f}")
    onset = next((m for m, ce, _ in pa if ce > alpha + 0.05), None)
    if onset:
        axA.axvline(onset, ls="-.", color="purple", alpha=0.6, label=f"onset m₂≈{onset} (Δ<(α+B)/M)")
    axA.set_xscale("log"); axA.set_xlabel("m₂ (Λ₂ grid size)"); axA.set_ylabel("risk / certificate")
    axA.set_title(f"C2 (Thm 4.9) transductive LOO certificate — 8b, M={fullM}\n"
                  "cert slides α→α+B (vacuity); true risk stays ~α"); axA.legend(fontsize=7, loc="center left")

    for M, col in zip(Ms, ["#1f77b4", "#ff7f0e", "#d62728"]):
        pb = curve(M)
        if pb:
            xs, ys, _ = zip(*pb)
            axB.plot(xs, ys, "o-", color=col, lw=2, label=f"M={M}")
    axB.axhline(alpha, ls=":", color="gray"); axB.axhline(alpha + B, ls="--", color="black")
    axB.set_xscale("log"); axB.set_xlabel("m₂"); axB.set_ylabel("LOO certificate")
    axB.set_title("onset shifts RIGHT as M grows\n(Δ<(α+B)/M  ⇒  onset ∝ M)"); axB.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- head-to-head table
def head2head_table(cfg, delta_gap=0.05):
    rows = E.load_results("head2head")
    delta = cfg["delta"]["total"]
    recs = []
    for (rung, method), rs in _group(rows, "rung", "method").items():
        alpha = rs[0]["alpha"]
        risks = [_risk(r) for r in rs]
        risks_f = [x for x in risks if x is not None]
        cov = [r["coverage"] for r in rs if r["coverage"] is not None]
        ss = [r["set_size"] for r in rs if r["set_size"] is not None]
        rec = {"rung": rung, "method": method, "n_reps": len(rs), "alpha": round(alpha, 4),
               "abstain": round(np.mean([r["abstained"] for r in rs]), 2),
               "mean_risk": round(np.nanmean(risks_f), 4) if risks_f else np.nan,
               "mean_cov": round(np.mean(cov), 3) if cov else np.nan,
               "mean_set": round(np.mean(ss), 3) if ss else np.nan}
        if method in ("raw_llm", "rand"):
            rec["verdict"] = "floor"
            rec["echo_rate"] = rs[0].get("echo_rate")
        else:
            v = E._verdict_for(method, rs, alpha, delta)
            rec["verdict"] = v["verdict"]
            rec["ctrl_frac"] = round(v["controlled_fraction"], 3) if v["controlled_fraction"] == v["controlled_fraction"] else None
        if method == "xu_proxy":
            rec["recall_risk"] = round(np.nanmean([r["recall_risk"] for r in rs if r["recall_risk"] is not None]), 4)
            rec["true_f1_risk"] = round(np.nanmean([r["true_f1_risk"] for r in rs if r["true_f1_risk"] is not None]), 4)
        if method == "nmscrc_t":
            rec["K_over_M"] = round(np.nanmean([r["K_over_M"] for r in rs if r["K_over_M"] is not None]), 3)
        recs.append(rec)
    order = ["nmscrc_i_eb", "nmscrc_i_hoeff", "nmscrc_t", "mono", "naive", "crcnm_marginal",
             "xu_proxy", "raw_llm", "rand"]
    df = pd.DataFrame(recs)
    df["__o"] = df["method"].map({m: i for i, m in enumerate(order)}).fillna(99)
    return df.sort_values(["rung", "__o"]).drop(columns="__o").reset_index(drop=True)
