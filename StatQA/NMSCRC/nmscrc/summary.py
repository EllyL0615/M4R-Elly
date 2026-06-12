"""Objective results record (NUMBERS ONLY, no interpretation). Each function reads
results/*.jsonl and returns a tidy DataFrame. No prose, no conclusions.
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from nmscrc import paths, hashing
from nmscrc import experiments as E


def _grp(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in keys)].append(r)
    return out


def _risk(r):
    return r["risk_cond"] if r["risk_cond"] is not None else r["risk_marginal"]


def _agg(rs):
    risks = np.array([_risk(r) for r in rs if not r["abstained"] and _risk(r) is not None], float)
    cov = np.array([r["coverage"] for r in rs if r["coverage"] is not None], float)
    ss = np.array([r["set_size"] for r in rs if r["set_size"] is not None], float)
    cert = np.array([r["cert_half_width"] for r in rs if r["cert_half_width"] is not None], float)
    ab = np.mean([bool(r["abstained"]) for r in rs])
    alpha = rs[0]["alpha"]
    return {
        "n_reps": len(rs), "abstain_rate": round(ab, 3),
        "frac_risk_le_alpha": round(float(np.mean(risks <= alpha)), 3) if risks.size else None,
        "frac_safe": round(float(np.mean([(r["abstained"] or (_risk(r) is not None and _risk(r) <= alpha)) for r in rs])), 3),
        "mean_risk": round(float(risks.mean()), 4) if risks.size else None,
        "p05_risk": round(float(np.percentile(risks, 5)), 4) if risks.size else None,
        "p95_risk": round(float(np.percentile(risks, 95)), 4) if risks.size else None,
        "mean_cov": round(float(cov.mean()), 3) if cov.size else None,
        "mean_set": round(float(ss.mean()), 3) if ss.size else None,
        "mean_cert": round(float(cert.mean()), 4) if cert.size else None,
    }


def exp1(cfg):
    rows = E.load_results("exp1")
    rec = []
    for (rung, method, tag), rs in _grp(rows, "rung", "method", "_tag").items():
        alpha = rs[0]["alpha"]
        v = E._verdict_for(method, rs, alpha, cfg["delta"]["total"])
        rec.append({"rung": rung, "method": method, "Delta": round(alpha - rs[0]["ell_star"], 2),
                    "alpha": round(alpha, 4), **_agg(rs), "verdict": v["verdict"]})
    return pd.DataFrame(rec).sort_values(["rung", "method", "Delta"]).reset_index(drop=True)


def exp3():
    rec = []
    for rung in ["llama3_2_1b", "llama3_2_3b", "llama3_1_8b"]:
        p = paths.ushape_json(rung)
        if p.exists():
            d = json.loads(p.read_text())
            e = d["extra"]
            rec.append({"rung": rung, "xi": d["xi"], "coverage": round(d["coverage"], 3),
                        "min_cond_risk(ell*)": round(e["min_cond_risk"], 4),
                        "argmin_lambda2": round(e["argmin_lambda2"], 4)})
    return pd.DataFrame(rec)


def exp5():
    rows = E.load_results("exp5")
    rec, slopes = [], []
    for (rung, method), rs0 in _grp(rows, "rung", "method").items():
        xs, ys = [], []
        for xi, rs in sorted(_grp(rs0, "xi").items()):
            cov = [r["phi_hat"] for r in rs if r["phi_hat"] is not None]
            cert = [r["cert_half_width"] for r in rs if r["cert_half_width"] is not None]
            ab = np.mean([bool(r["abstained"]) for r in rs])
            mc = float(np.mean(cov)) if cov else None
            ce = float(np.mean(cert)) if cert else None
            rec.append({"rung": rung, "method": method, "xi": xi[0],
                        "mean_cov": round(mc, 3) if mc else None,
                        "mean_cert": round(ce, 5) if ce else None, "abstain_rate": round(float(ab), 2)})
            if mc and ce:
                xs.append(mc); ys.append(ce)
        if len(xs) >= 2:
            slopes.append({"rung": rung, "method": method,
                           "slope_loglog(cert~cov)": round(float(np.polyfit(np.log(xs), np.log(ys), 1)[0]), 3)})
    return pd.DataFrame(rec).sort_values(["rung", "method", "xi"]).reset_index(drop=True), pd.DataFrame(slopes)


def exp6():
    rows = E.load_results("exp6")
    rec = []
    for (method, m2), rs in _grp(rows, "method", "m2").items():
        cert = [r["cert_half_width"] for r in rs if r["cert_half_width"] is not None]
        rec.append({"rung": "llama3_1_8b", "method": method, "m2": m2,
                    "abstain_rate": round(float(np.mean([bool(r["abstained"]) for r in rs])), 2),
                    "mean_cert": round(float(np.mean(cert)), 4) if cert else None,
                    "mean_cov": round(float(np.mean([r["coverage"] for r in rs if r["coverage"] is not None])), 3) if any(r["coverage"] is not None for r in rs) else None})
    return pd.DataFrame(rec).sort_values(["method", "m2"]).reset_index(drop=True)


def exp8():
    rows = E.load_results("exp8")
    rec = []
    for (rung, method), rs in _grp(rows, "rung", "method").items():
        a = _agg(rs)
        rec.append({"rung": rung, "method": method, "alpha": round(rs[0]["alpha"], 4),
                    "abstain_rate": a["abstain_rate"], "mean_risk": a["mean_risk"], "mean_cov": a["mean_cov"],
                    "mean_set": a["mean_set"],
                    "mean_K_over_M": round(float(np.nanmean([r["K_over_M"] for r in rs if r["K_over_M"] is not None])), 4) if any(r["K_over_M"] is not None for r in rs) else None})
    return pd.DataFrame(rec).sort_values(["rung", "method"]).reset_index(drop=True)


def floor():
    rows = E.load_results("exp9_floor")
    rec = []
    for (method, n), rs in _grp(rows, "method", "n").items():
        cert = [r["cert_half_width"] for r in rs if r["cert_half_width"] is not None]
        rec.append({"method": method, "n_calib": n,
                    "abstain_rate": round(float(np.mean([bool(r["abstained"]) for r in rs])), 2),
                    "mean_cert": round(float(np.mean(cert)), 4) if cert else None})
    return pd.DataFrame(rec).sort_values(["method", "n_calib"]).reset_index(drop=True)


def union_tax():
    rows = E.load_results("union_tax")
    rec = []
    for (method, m1), rs in _grp(rows, "method", "m1").items():
        cert = [r["cert_half_width"] for r in rs if r["cert_half_width"] is not None]
        rec.append({"method": method, "m1*m2": rs[0]["m1"] * rs[0]["m2"],
                    "mean_cert": round(float(np.mean(cert)), 4) if cert else None,
                    "abstain_rate": round(float(np.mean([bool(r["abstained"]) for r in rs])), 2)})
    return pd.DataFrame(rec).sort_values(["method", "m1*m2"]).reset_index(drop=True)


def mono_gridrefine():
    rows = E.load_results("mono_gridrefine")
    rec = []
    for (rung, method, m2), rs in _grp(rows, "rung", "method", "m2").items():
        viol = np.mean([(r["risk_cond"] is not None and r["risk_cond"] > r["alpha"]) for r in rs])
        rec.append({"rung": rung, "method": method, "m2": m2, "viol_frac": round(float(viol), 3),
                    "abstain_rate": round(float(np.mean([bool(r["abstained"]) for r in rs])), 2)})
    return pd.DataFrame(rec).sort_values(["rung", "method", "m2"]).reset_index(drop=True)


def c2_transductive_cert():
    rows = E.load_results("c2_transductive_cert")
    ex = lambda r, k: (r.get("extra") or {}).get(k)
    rec = []
    for tag, rs in _grp(rows, "_tag").items():
        feas = [r for r in rs if not ex(r, "infeasible")]
        rec.append({"M": ex(rs[0], "M"), "m2": rs[0]["m2"], "n_reps": len(rs),
                    "infeasible_rate": round(1 - len(feas) / len(rs), 2),
                    "mean_Delta": round(float(np.mean([ex(r, "c2_delta") for r in feas])), 5) if feas else None,
                    "mean_cert": round(float(np.mean([ex(r, "cert") for r in feas])), 4) if feas else None,
                    "mean_true_held": round(float(np.nanmean([ex(r, "true_held_risk") for r in feas])), 4) if feas else None,
                    "K_direct_eq_K_closed": round(float(np.mean([ex(r, "K_direct") == ex(r, "K_closed") for r in feas])), 3) if feas else None})
    return pd.DataFrame(rec).sort_values(["M", "m2"]).reset_index(drop=True)


def _md_table(df):
    """DataFrame -> GitHub-markdown table (no tabulate dependency). NaN/None -> empty cell."""
    esc = lambda x: str(x).replace("|", "\\|")
    cols = [esc(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else esc(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def export_report(cfg, out_path=None):
    """Assemble all objective result tables + figures into one markdown report (titles + numbers +
    embedded plots, NO interpretation). Reads results/*.jsonl; saves figures to
    results/figures/*.png and embeds them under their section. Re-runnable after any experiment re-run."""
    from nmscrc import plots                                    # local import (avoid cycle)
    import matplotlib.pyplot as plt
    from pathlib import Path
    out_path = paths.report_md() if out_path is None else Path(out_path)
    figdir = paths.figures_dir()
    figdir.mkdir(parents=True, exist_ok=True)
    figrel = figdir.name                                        # e.g. v1_figures (relative to report dir)

    def fig_md(name, figobj):
        figobj.savefig(figdir / f"{name}.png", bbox_inches="tight", dpi=120)
        plt.close(figobj)                                       # don't also display in the export cell
        return f"![{name}]({figrel}/{name}.png)"

    img1 = fig_md("exp1_validity", plots.plot_exp1())
    img3 = fig_md("exp3_ushape", plots.plot_exp3())
    fig5, _ = plots.plot_exp5(); img5 = fig_md("exp5_xislope", fig5)
    img6 = fig_md("exp6_phase", plots.plot_exp6())
    img9 = fig_md("synth_floor_union", plots.plot_synth())
    img63 = fig_md("mono_gridrefine", plots.plot_mono_gridrefine())
    img67 = fig_md("c2_transductive_cert", plots.plot_c2_transductive_cert())

    d0 = stage0(cfg)
    d5d, d5s = exp5()
    mg = mono_gridrefine().pivot_table(index=["rung", "method"], columns="m2", values="viol_frac").reset_index()
    au_states = {}
    for exp in ["exp1", "exp5", "exp6", "exp8", "head2head", "mono_gridrefine", "exp9_floor", "union_tax"]:
        for r in E.load_results(exp):
            if r.get("state"):
                au_states[r["state"]] = au_states.get(r["state"], 0) + 1

    parts = [
        "# NM-SCRC — Results report (objective numbers only)",
        "",
        f"- combined artifact sha256: `{d0.attrs['combined_hash']}`",
        f"- models: {cfg['models']}  ·  split: {cfg['split_name']}  ·  reps: {cfg['reps']['n']}  ·  "
        f"selector: {cfg['selector']['type']}  ·  grids: {cfg['grids']['lambda1']['mode']}",
        f"- use_3b_for_transition: {d0.attrs['use_3b_for_transition']}",
        f"- rep-state totals (all judged experiments): {au_states}",
        "",
        "## Stage 0 — frozen artifacts (probe AUC, oracle ell*, raw-LLM rates, K/M vs M)", _md_table(d0),
        "## (1) Validity + PAC histogram — per rung x variant x Delta", img1, _md_table(exp1(cfg)),
        "## (3) F1-risk U-shape (oracle ell*, argmin lambda2)", img3, _md_table(exp3()),
        "## (5) xi-slope (C1) — per (rung, variant, xi)", img5, _md_table(d5d),
        "## (5) xi-slope — fitted log-log slopes (cert ~ coverage)", _md_table(d5s),
        "## (6) Phase transition vs m2 (llama3_1_8b)", img6, _md_table(exp6()),
        "## (8) NM-SCRC-I vs NM-SCRC-T", _md_table(exp8()),
        "## Head-to-head (6 judged + raw-LLM/RAND floors; CRC-NM-marginal = MARGINAL caliber)",
        _md_table(plots.head2head_table(cfg)),
        "## (9) Feasibility floor + union-tax", img9, _md_table(floor()), _md_table(union_tax()),
        "## (6.3) mono grid-refinement — violation fraction vs m2", img63, _md_table(mg),
        "## (6.7) C2 transductive LOO certificate (Thm 4.9) — cert slides α->α+B, true risk ~α",
        img67, _md_table(c2_transductive_cert()),
    ]
    md = "\n\n".join(parts) + "\n"
    paths.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    return out_path


def stage0(cfg):
    qc = json.loads(paths.qc_3b().read_text())
    flags = json.loads(paths.answer_flags().read_text())
    rec = []
    for m in cfg["models"]:
        meta = json.loads(paths.probe_meta(m).read_text())
        km = qc["K_over_M_vs_M"][m]
        a = flags[m]
        rec.append({"model": m, "probe_val_auc": round(meta["val_macro_auc"], 4),
                    "input_dim": meta["input_dim"], "calibtest_n": meta["calibtest_rows"],
                    "ell*(xi=0.3)": round(qc["ell_star"][m], 4),
                    "echo_rate": round(a["echo_rate"], 4), "unparseable": round(a["unparseable_rate"], 4),
                    "ok_rate": round(a["ok_rate"], 4),
                    "K/M@M20": km.get("20"), "K/M@M40": km.get("40"), "K/M@M80": km.get("80")})
    df = pd.DataFrame(rec)
    df.attrs["combined_hash"] = hashing.read_combined_hash(paths.artifact_hash())
    df.attrs["use_3b_for_transition"] = qc["use_3b_for_transition"]
    return df
