"""Chapter-6 publication figures -> results/figures_beautify_v2/.

A calm, editorial palette: muted steel-blue (ours / EB / NM-SCRC), warm terracotta (certificate /
baseline), muted plum (third category); neutral grey for reference lines and grid. Within a method
family, light/dark shades separate the rungs. Read-only: consumes the frozen results
(results/*.jsonl, exp3 json) and renders figures; it never recomputes or mutates results.

Run from the repo root:  python scripts/make_figs_beautify_v2.py
"""
import os
os.environ.setdefault("NMSCRC_VERSION", "v3")
os.environ.setdefault("NMSCRC_RAW_DIR", "data-full_v3")
import json
import sys
from collections import defaultdict
from pathlib import Path

# Run from anywhere: put the repo root on sys.path so `import nmscrc` resolves without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cycler
import numpy as np

from nmscrc import paths, experiments as E   # READ-ONLY

# ----------------------------------------------------------------- palette (user-chosen 4-colour set)
BLUE, BLUE_L, BLUE_D = "#205D89", "#5E8FB5", "#16415F"      # navy   — ours / EB / NM-SCRC
RUST, RUST_L, RUST_D = "#CF784B", "#E0A37D", "#A1542C"      # orange — certificate / baseline / mono
PLUM = "#73A87C"                                            # gray-green — third category (rung/version)
GOLD = "#C1BC78"                                            # dried-grass yellow — accent
SLATE = "#5C6B79"
GREY, RED = "#7a7a7a", "#414C57"                            # neutral reference / threshold
PAL = [BLUE, RUST, PLUM, GOLD, SLATE]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "mathtext.fontset": "cm",
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
    "axes.titlecolor": "#272727", "text.color": "#272727",
    "axes.titlepad": 9, "axes.labelpad": 5,
    "axes.linewidth": 0.8, "axes.edgecolor": "#9a9a9a",
    "axes.labelcolor": "#272727",
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#d7d7d7", "grid.linewidth": 0.6, "grid.alpha": 0.55,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.fontsize": 9.5, "legend.frameon": True, "legend.framealpha": 0.96,
    "legend.edgecolor": "#d8d8d8", "legend.borderpad": 0.5, "legend.handlelength": 1.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 6, "lines.markeredgewidth": 0.7,
    "lines.markeredgecolor": "white",
    "axes.prop_cycle": cycler(color=PAL),
})
RUNGS = {"llama3_2_1b": ("1B", BLUE, "o"), "llama3_2_3b": ("3B", RUST, "s"),
         "llama3_1_8b": ("8B", PLUM, "^")}
OUT = Path("results/figures_beautify_v2"); OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png"); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)
    print(f"  wrote {name}.png + .pdf")


def group(rows, *keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r.get(k) for k in keys)].append(r)
    return g


def _ex(r, k):
    return (r.get("extra") or {}).get(k)


# ----------------------------------------------------------------- Fig 6.1
def fig61():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for rung, (lab, col, mk) in RUNGS.items():
        d = json.loads(paths.ushape_json(rung).read_text())["extra"]
        ss, cr = np.array(d["set_size"]), np.array(d["cond_risk"])
        ax.plot(ss, cr, color=col, marker=mk, ms=4.5, lw=1.9, label=f"{lab} probe", markevery=6)
        jmin = int(np.argmin(cr))
        ax.scatter([ss[jmin]], [cr[jmin]], s=68, facecolor=col, edgecolor="white", linewidth=1.1, zorder=5)
    ax.annotate("interior minimum\n(precision–recall optimum)", xy=(2.0, 0.02), xytext=(6.5, 0.16),
                fontsize=9.5, color="#3a3a3a", ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0, connectionstyle="arc3,rad=-0.2"))
    ax.set_xlabel("mean prediction-set size  $|\\,C_{\\lambda_2}(x)\\,|$")
    ax.set_ylabel("conditional $F_1$-risk  $\\;1-F_1$")
    ax.set_title("Non-monotone loss: $F_1$-risk is U-shaped in set size")
    ax.set_xlim(0, None); ax.set_ylim(0, None)
    ax.legend(loc="upper center", ncol=3, columnspacing=1.3)
    save(fig, "fig61_ushape")


# ----------------------------------------------------------------- Fig 6.2
def fig62(delta_gap=0.05):
    rows = E.load_results("exp1")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)
    for ax, (rung, (lab, col, mk)) in zip(axes, RUNGS.items()):
        eb = [r for r in rows if r["rung"] == rung and r["method"] == "nmscrc_i_eb"
              and abs(r["alpha"] - (r["ell_star"] + delta_gap)) < 1e-9]
        alpha = eb[0]["alpha"]
        risks = np.array([r["risk_cond"] for r in eb if not r["abstained"]], float)
        frac = np.mean([(r["abstained"] or r["risk_cond"] <= alpha) for r in eb])
        ax.hist(risks, bins=22, color=col, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvline(alpha, color=RED, lw=1.8, zorder=5)
        ax.text(alpha, ax.get_ylim()[1] * 0.96, f"  $\\alpha={alpha:.3f}$", color=RED,
                fontsize=9.5, va="top", ha="left")
        ax.set_title(f"{lab} probe", color=col)
        ax.set_xlabel("test conditional $F_1$-risk")
        ax.text(0.97, 0.85, f"P(risk ≤ α or abstain)\n= {frac:.2f}   (≥ 1−δ)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.2,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#dcdcdc", alpha=0.92))
    axes[0].set_ylabel("number of calibration draws (of 100)")
    fig.suptitle("NM-SCRC-I (EB) validity — PAC histogram over 100 calibration draws  ($\\Delta=0.05$)",
                 fontsize=12.5)
    save(fig, "fig62_pac")


# ----------------------------------------------------------------- Fig 6.3
def fig63():
    rows = E.load_results("mono_gridrefine")
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    styles = {("mono", "llama3_2_1b"): (RUST_L, "o", "-", "monotone CRC — 1B"),
              ("mono", "llama3_1_8b"): (RUST_D, "s", "-", "monotone CRC — 8B"),
              ("nmscrc_i_eb", "llama3_2_1b"): (BLUE_L, "o", "--", "NM-SCRC-I — 1B"),
              ("nmscrc_i_eb", "llama3_1_8b"): (BLUE_D, "s", "--", "NM-SCRC-I — 8B")}
    for (method, rung), (col, mk, ls, lab) in styles.items():
        g = group([r for r in rows if r["method"] == method and r["rung"] == rung], "m2")
        pts = sorted((int(rs[0]["m2"]),
                      float(np.mean([(r["risk_cond"] is not None and r["risk_cond"] > r["alpha"]) for r in rs])))
                     for rs in g.values())
        xs, ys = zip(*pts)
        ax.plot(xs, ys, ls, color=col, marker=mk, lw=2.0, label=lab)
    ax.axhline(0.10, color=GREY, ls=":", lw=1.3)
    ax.text(70, 0.108, "FAIL threshold  $\\delta=0.1$", color=GREY, fontsize=9.5, va="bottom")
    ax.annotate("NM-SCRC-I: $t_V$ cushion\n$\\Rightarrow$ violation $\\approx 0$ (flat)", xy=(620, 0.004),
                xytext=(95, 0.052), fontsize=9.2, color=BLUE_D, ha="left",
                arrowprops=dict(arrowstyle="->", color=BLUE_D, lw=1.0, connectionstyle="arc3,rad=-0.25"))
    ax.set_xscale("log", base=2); ax.set_xticks([40, 80, 160, 320, 640, 1280, 2560])
    ax.set_xticklabels([40, 80, 160, 320, 640, 1280, 2560]); ax.minorticks_off()
    ax.set_xlabel("second-stage grid size  $m_2$")
    ax.set_ylabel("violation fraction over 100 draws")
    ax.set_title("Violation fraction vs grid size $m_2$: monotone CRC (1B, 8B) vs NM-SCRC-I")
    ax.set_ylim(-0.015, 0.205)
    ax.legend(loc="upper left", ncol=2, columnspacing=1.2)
    save(fig, "fig63_monogrid")


# ----------------------------------------------------------------- Fig 6.4
def fig64():
    rows = E.load_results("exp5")
    panels = [("llama3_2_1b", "1B probe (real)"), ("synthetic", "monotone-slope construction")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    eb_slopes = {}
    for ax, (rung, title) in zip(axes, panels):
        for variant, col, mk in [("eb", BLUE, "o"), ("hoeffding", RUST, "s")]:
            g = group([r for r in rows if r["rung"] == rung and r["method"] == f"nmscrc_i_{variant}"], "xi")
            xs, ys = [], []
            for xi, rs in sorted(g.items()):
                cov = [r["phi_hat"] for r in rs if r["phi_hat"] is not None]
                ce = [r["cert_half_width"] for r in rs if r["cert_half_width"] is not None]
                if cov and ce:
                    xs.append(np.mean(cov)); ys.append(np.mean(ce))
            sl = np.polyfit(np.log(xs), np.log(ys), 1)[0]
            if variant == "eb":
                eb_slopes[rung] = sl
            lab = ("empirical Bernstein" if variant == "eb" else "Hoeffding") + f"  (slope ${sl:.2f}$)"
            ax.plot(xs, ys, color=col, marker=mk, lw=2.0, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("measured selection rate  $\\hat\\varphi\\;(\\approx\\xi)$"); ax.set_title(title)
        ax.legend(loc="lower left")
    axes[0].set_ylabel("certificate half-width")
    fig.suptitle(f"C1 rate: EB approaches the √-rate (finite-$n$ ${eb_slopes['llama3_2_1b']:.2f}$/${eb_slopes['synthetic']:.2f}$); "
                 f"Hoeffding pays $1/\\hat\\varphi$ ($-1.0$)", fontsize=12.5)
    save(fig, "fig64_xislope")


# ----------------------------------------------------------------- Fig 6.5
def fig65():
    floor = E.load_results("exp9_floor"); union = E.load_results("union_tax")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.9))
    for variant, col, mk, lab in [("eb", BLUE, "o", "empirical Bernstein"), ("hoeffding", RUST, "s", "Hoeffding")]:
        g = group([r for r in floor if r["method"] == f"nmscrc_i_{variant}"], "n")
        pts = sorted((int(rs[0]["n"]), float(np.mean([bool(r["abstained"]) for r in rs]))) for rs in g.values())
        axL.plot(*zip(*pts), color=col, marker=mk, lw=2.0, label=lab)
    axL.set_xscale("log"); axL.set_xlabel("calibration size  $n$"); axL.set_ylabel("abstention rate")
    axL.set_title("Feasibility floor")
    axL.annotate("EB certifies\nfrom $n=4{,}000$", xy=(4000, 0.02), xytext=(4600, 0.45), fontsize=9.0,
                 color=BLUE_D, arrowprops=dict(arrowstyle="->", color=BLUE_D, lw=1.0))
    for variant, col, mk, lab in [("eb", BLUE, "o", "empirical Bernstein"), ("hoeffding", RUST, "s", "Hoeffding")]:
        g = group([r for r in union if r["method"] == f"nmscrc_i_{variant}"], "m1")
        pts = []
        for rs in g.values():
            ce = [r["cert_half_width"] for r in rs if r["cert_half_width"] is not None]
            if ce:
                pts.append((int(rs[0]["m1"]) * int(rs[0]["m2"]), float(np.mean(ce))))
        if pts:
            axR.plot(*zip(*sorted(pts)), color=col, marker=mk, lw=2.0, label=lab)
    axR.set_xscale("log"); axR.set_xlabel("grid size  $m_1 m_2$"); axR.set_ylabel("certificate half-width")
    axR.set_title("Union tax  ($\\propto\\sqrt{\\log m_1 m_2}$)")
    fig.suptitle("Stage-1 synthetic checks: feasibility floor and union tax", fontsize=12.5)
    fig.legend(*axL.get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=True)
    save(fig, "fig65_floorunion")


# ----------------------------------------------------------------- Fig 6.6
def fig66():
    rows = E.load_results("c2_transductive_cert")
    alpha, B = rows[0]["alpha"], 1.0
    grp = defaultdict(list)
    for r in rows:
        grp[(_ex(r, "M"), r["m2"])].append(r)
    Ms = sorted({_ex(r, "M") for r in rows}); full = max(Ms)

    def curve(Msel):
        out = []
        for (M, m2), rs in grp.items():
            if M != Msel:
                continue
            feas = [x for x in rs if not _ex(x, "infeasible")]
            if feas:
                out.append((int(m2), float(np.mean([_ex(x, "cert") for x in feas])),
                            float(np.nanmean([_ex(x, "true_held_risk") for x in feas]))))
        return sorted(out)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.6))
    m2s, cert, held = zip(*curve(full))
    axA.plot(m2s, cert, color=RUST, marker="o", lw=2.1, label="LOO certificate  $\\alpha+\\bar K B/M$")
    axA.plot(m2s, held, color=BLUE, marker="s", lw=1.7, ls="--", ms=5, label="held-out risk")
    axA.axhline(alpha + B, color=GREY, ls=":", lw=1.3); axA.axhline(alpha, color=GREY, ls=":", lw=1.3)
    axA.text(m2s[0], alpha + B + 0.012, "trivial ceiling  $\\alpha+B$", color=GREY, fontsize=9, va="bottom")
    axA.text(m2s[0], alpha - 0.055, "$\\alpha$", color=GREY, fontsize=9.5, va="bottom")
    axA.set_xscale("log", base=2); axA.set_xticks(m2s); axA.set_xticklabels([str(m) for m in m2s])
    axA.minorticks_off(); axA.set_ylim(0, 1.18)
    axA.set_xlabel("second-stage grid size  $m_2$"); axA.set_ylabel("certificate / held-out risk")
    axA.set_title(f"8B operating bag  $M={full:,}$"); axA.legend(loc="center left")

    for M, col, mk in zip(Ms, [RUST_L, RUST, RUST_D], ["D", "v", "o"]):
        xs, ys, _ = zip(*curve(M))
        axB.plot(xs, ys, color=col, marker=mk, lw=2.0, label=f"$M={M}$")
    axB.axhline(alpha + B, color=GREY, ls=":", lw=1.3); axB.axhline(alpha, color=GREY, ls=":", lw=1.3)
    axB.set_xscale("log", base=2); axB.set_xticks(xs); axB.set_xticklabels([str(m) for m in xs])
    axB.minorticks_off(); axB.set_ylim(0, 1.18)
    axB.set_xlabel("second-stage grid size  $m_2$"); axB.set_ylabel("LOO certificate")
    axB.set_title("Onset shifts right as $M$ grows"); axB.legend(loc="center left")
    fig.suptitle("C2 transductive leave-one-out certificate (Thm 4.9): the vacuity slide toward $\\alpha+B$",
                 fontsize=12.5)
    save(fig, "fig66_c2slide")


# ----------------------------------------------------------------- Fig 6.7
def fig67():
    M2 = [80, 160, 320, 640, 1280, 2560, 5120, 10240]
    style = {"v1": (BLUE, "o"), "v2": (RUST, "s"), "v3": (PLUM, "^")}
    alpha, B = 0.104, 1.0

    def read(v):
        cert, held = [], []
        for m2 in M2:
            f = Path(f"results/c2_transductive_cert/{v}_loo_cert__llama3_1_8b__Mfull_m2_{m2:05d}.jsonl")
            rs = [json.loads(l)["extra"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            feas = [e for e in rs if not e.get("infeasible")]
            cert.append(float(np.mean([e["cert"] for e in feas])))
            held.append(float(np.nanmean([e["true_held_risk"] for e in feas])))
        return cert, held

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for v, (col, mk) in style.items():
        cert, held = read(v)
        ax.plot(M2, cert, color=col, marker=mk, lw=2.1, label=v)
        ax.plot(M2, held, color=col, ls="--", lw=1.1, alpha=0.85)
    ax.axhline(alpha + B, color=GREY, ls=":", lw=1.3); ax.axhline(alpha, color=GREY, ls=":", lw=1.3)
    ax.text(M2[0], alpha + B + 0.012, "trivial ceiling  $\\alpha+B=1.104$", color=GREY, fontsize=9, va="bottom")
    ax.text(M2[0], alpha - 0.055, "$\\alpha=0.104$", color=GREY, fontsize=9.5, va="bottom")
    ax.text(2560, 0.66, "certificate", fontsize=10.5, rotation=34, color="#333")
    ax.text(300, 0.135, "held-out risk", fontsize=10.5, color="#333")
    ax.set_xscale("log", base=2); ax.set_xticks(M2); ax.set_xticklabels([str(m) for m in M2])
    ax.minorticks_off(); ax.set_ylim(0, 1.18)
    ax.set_xlabel("second-stage grid size  $m_2$"); ax.set_ylabel("certificate / held-out risk")
    ax.set_title("C2 certificate across prompts  (8B, $M=1{,}220$)")
    ax.legend(title="prompt", loc="center right")
    save(fig, "fig67_crossprompt")


if __name__ == "__main__":
    print("rendering -> results/figures_beautify_v2/")
    for fn in (fig61, fig62, fig63, fig64, fig65, fig66, fig67):
        try:
            fn()
        except Exception as e:
            import traceback; print(f"  [ERR] {fn.__name__}: {e}"); traceback.print_exc()
    print("done.")
