"""Stage 1/2 experiment orchestration. Each experiment reads the FROZEN Stage-0 artifacts
(per-rep re-split of the calibtest pool; probe never touched) and writes per-rep structured records,
batched as one JSONL per config: results/{exp}/{method}__{rung}__{tag}.jsonl (one JSON object per
rep-line; per-rep contract, batched for fast IO). Plotting/audit read these only.

Modes (user decision C): 'search' = full NM-SCRC optimisation layer (Prop 4.4 winner's curse -> naive
FAILs) for head-to-head; 'pinned' = λ1 fixed at ξ-floor (pure Def 3.1 validity) for Exp5 + vs-RAND.
Grids are the global frozen QUANTILE grids (decision B) so set sizes are comparable across methods.
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nmscrc import paths, losses, repeng, judge, results, data, stage0, hashing, c2
from nmscrc.artifacts import build_artifacts, build_grids
from nmscrc.synthetic import build_synthetic_artifacts, build_synthetic_f1
from nmscrc.methods import nmscrc_i, nmscrc_t, mono, naive, crcnm_marginal, xu_proxy, raw, rand


def _seeds(cfg):
    return repeng.rep_seeds(cfg["reps"]["n"], cfg["reps"]["seed_start"])


def _frac(cfg):
    return cfg["rep_split"]["calib_frac_of_pool"]


def ell_star(A, cfg, xi):
    return losses.oracle_ell_star(A.g, A.Y, A.L_f1, A.lambda1, A.lambda2, xi)[0]


def _emit(buf, exp, method, rung, tag, rep, seed, split_hash, r, cfg, *,
          alpha=None, ell=None, xi=None, judged=True):
    risk = r.get("risk_cond") if r.get("risk_cond") is not None else r.get("risk_marginal")
    state = judge.rep_state(r.get("abstained", False), risk if risk is not None else 1.0, alpha) \
        if (judged and alpha is not None) else None
    res = results.make_result(
        exp=exp, method=method, rung=rung, rep=rep, seed=seed, split_hash=split_hash,
        xi=xi if xi is not None else r.get("xi"), alpha=alpha, ell_star=ell, delta=cfg["delta"]["total"],
        m1=r.get("m1"), m2=r.get("m2"), n=r.get("n"),
        coverage=r.get("coverage"), risk_cond=r.get("risk_cond"), risk_marginal=r.get("risk_marginal"),
        set_size=r.get("set_size"), abstained=r.get("abstained"), state=state,
        K=r.get("K"), K_over_M=r.get("K_over_M"), echo_rate=r.get("echo_rate"),
        excess=r.get("excess"), cert_half_width=r.get("cert_half_width"),
        t_V=r.get("t_V"), t_V_eb=r.get("t_V_eb"), t_U=r.get("t_U"), t_U_eb=r.get("t_U_eb"),
        R_hat=r.get("R_hat"), phi_hat=r.get("phi_hat"), phi_lcb=r.get("phi_lcb"),
        lambda1=r.get("lambda1"), lambda2=r.get("lambda2"), lambda1_mode=r.get("lambda1_mode"),
        recall_risk=r.get("recall_risk"), true_f1_risk=r.get("true_f1_risk"))
    buf.setdefault((method, rung, tag), []).append(res)


def _flush(exp, buf):
    for (method, rung, tag), rows in buf.items():
        results.write_jsonl(paths.result_jsonl(exp, method, rung, tag), rows)


# ===================================================================== Stage 2
def run_exp1(cfg, split_hash, verbose=True):
    """① validity + PAC histogram — 3 rungs, NM-SCRC-I {eb, hoeffding}, SEARCH, all Δ."""
    buf = {}
    for rung in cfg["models"]:
        A = stage0.build_or_load_artifacts(rung, cfg, split_hash)
        xi = cfg["targets"]["xi_default"]
        ell = ell_star(A, cfg, xi)
        for Dg in cfg["targets"]["delta_gap"]:
            alpha = ell + Dg
            for variant in ("eb", "hoeffding"):
                for s in _seeds(cfg):
                    c, t = repeng.rep_split(A.n, s, _frac(cfg))
                    r = nmscrc_i.run(A, c, t, xi, alpha, cfg["delta"], variant=variant, lam1_mode="search")
                    _emit(buf, "exp1", f"nmscrc_i_{variant}", rung, f"d{int(Dg*100):02d}", s, s, split_hash,
                          r, cfg, alpha=alpha, ell=ell, xi=xi)
        if verbose:
            print(f"[exp1] {rung} done (ell*={ell:.4f})")
    _flush("exp1", buf)


def run_exp3(cfg, split_hash, verbose=True):
    """③ U-shape — conditional F1-risk vs λ2 (set size) at fixed coverage ξ, full pool (descriptive)."""
    xi = cfg["targets"]["xi_default"]
    for rung in cfg["models"]:
        A = stage0.build_or_load_artifacts(rung, cfg, split_hash)
        phi = (A.g[:, None] >= (1.0 - A.lambda1)[None, :]).mean(0)
        i1 = int(np.argmax(phi >= xi)) if (phi >= xi).any() else len(A.lambda1) - 1
        acc = A.g >= (1.0 - A.lambda1[i1])
        cond_risk = A.L_f1[acc].mean(0)
        set_size = A.SS[acc].mean(0)
        res = results.make_result(exp="exp3", method="ushape", rung=rung, split_hash=split_hash, xi=xi,
                                  coverage=float(acc.mean()), m2=len(A.lambda2))
        res["extra"] = {"lambda2": A.lambda2.tolist(), "set_size": set_size.tolist(),
                        "cond_risk": cond_risk.tolist(),
                        "argmin_lambda2": float(A.lambda2[int(np.argmin(cond_risk))]),
                        "min_cond_risk": float(cond_risk.min())}
        results.write_result(res, paths.ushape_json(rung))
        if verbose:
            print(f"[exp3] {rung}: min cond-risk={cond_risk.min():.4f} at set~{set_size[int(np.argmin(cond_risk))]:.2f}")


def run_exp5(cfg, split_hash, verbose=True):
    """⑤ ξ-slope (C1) — PINNED, cert_half_width vs measured φ̂; real 1b + synthetic. Fixed α."""
    buf = {}
    alpha = 0.10
    for rung, A in [("llama3_2_1b", stage0.build_or_load_artifacts("llama3_2_1b", cfg, split_hash)),
                    ("synthetic", build_synthetic_artifacts(n=2 * 12000, m1=cfg["grids"]["m1"],
                                                            m2=cfg["grids"]["m2_default"], alpha=alpha,
                                                            delta=0.002, seed=12345))]:
        sh = split_hash if rung != "synthetic" else A.split_hash
        for xi in cfg["targets"]["xi_sweep_coverage"]:
            for variant in ("eb", "hoeffding"):
                for s in _seeds(cfg):
                    c, t = repeng.rep_split(A.n, s, _frac(cfg))
                    r = nmscrc_i.run(A, c, t, xi, alpha, cfg["delta"], variant=variant, lam1_mode="pinned")
                    _emit(buf, "exp5", f"nmscrc_i_{variant}", rung, f"xi{int(xi*100):02d}", s, s, sh,
                          r, cfg, alpha=alpha, ell=None, xi=xi)
        if verbose:
            print(f"[exp5] {rung} done")
    _flush("exp5", buf)


def run_exp6(cfg, split_hash, verbose=True):
    """⑥ phase transition — 8b, sweep m₂ (m₂-driven). PINNED, NM-SCRC-I eb+hoeffding."""
    buf = {}
    rung = "llama3_1_8b"
    xi = cfg["targets"]["xi_default"]
    alpha = ell_star(stage0.build_or_load_artifacts(rung, cfg, split_hash), cfg, xi) + 0.05
    logits = np.load(paths.calibtest_logits(rung))
    labels = np.load(paths.calibtest_labels(rung))
    p = losses.sigmoid(logits.astype(np.float64))
    for m2 in cfg["grids"]["m2_sweep"]:
        cfg_m2 = json.loads(json.dumps(cfg))
        cfg_m2["grids"]["lambda2"]["n_points"] = m2
        lam1, lam2 = build_grids(p, cfg_m2)
        A = build_artifacts(rung, logits, labels, lam1, lam2, split_hash)
        for variant in ("eb", "hoeffding"):
            for s in _seeds(cfg):
                c, t = repeng.rep_split(A.n, s, _frac(cfg))
                r = nmscrc_i.run(A, c, t, xi, alpha, cfg["delta"], variant=variant, lam1_mode="pinned")
                _emit(buf, "exp6", f"nmscrc_i_{variant}", rung, f"m2_{m2:04d}", s, s, split_hash,
                      r, cfg, alpha=alpha, ell=alpha - 0.05, xi=xi)
        if verbose:
            print(f"[exp6] m2={m2} done")
    _flush("exp6", buf)


def run_exp8(cfg, split_hash, verbose=True):
    """⑧ I vs T — 1b + 8b, NM-SCRC-I (eb, pinned) vs NM-SCRC-T (K/M). Δ=0.10 (so T can emit)."""
    buf = {}
    Dg = cfg["targets"]["delta_gap"][2]
    xi = cfg["targets"]["xi_default"]
    for rung in ("llama3_2_1b", "llama3_1_8b"):
        A = stage0.build_or_load_artifacts(rung, cfg, split_hash)
        ell = ell_star(A, cfg, xi)
        alpha = ell + Dg
        for s in _seeds(cfg):
            c, t = repeng.rep_split(A.n, s, _frac(cfg))
            _emit(buf, "exp8", "nmscrc_i_eb", rung, f"d{int(Dg*100):02d}", s, s, split_hash,
                  nmscrc_i.run(A, c, t, xi, alpha, cfg["delta"], "eb", "pinned"), cfg, alpha=alpha, ell=ell, xi=xi)
            _emit(buf, "exp8", "nmscrc_t", rung, f"d{int(Dg*100):02d}", s, s, split_hash,
                  nmscrc_t.run(A, c, t, xi, alpha, cfg["delta"]), cfg, alpha=alpha, ell=ell, xi=xi)
        if verbose:
            print(f"[exp8] {rung} done")
    _flush("exp8", buf)


def run_head2head(cfg, split_hash, verbose=True):
    """6 judged methods (SEARCH so naive can FAIL) + raw/rand floors. Real rungs (mild, near-oracle)
    + a FIXED adversarial synthetic_f1 arm (where predicted opponent failures can bite). Δ=0.05."""
    buf = {}
    Dg = cfg["targets"]["delta_gap"][1]
    xi = cfg["targets"]["xi_default"]
    tag = f"d{int(Dg*100):02d}"
    methods_list = data.load_method_list()
    flags = json.loads(paths.answer_flags().read_text())

    arms = []
    for rung in cfg["models"]:
        A = stage0.build_or_load_artifacts(rung, cfg, split_hash)
        ans = pd.read_csv(paths.answer_csv(rung))[methods_list].to_numpy().astype(np.float64)
        arms.append((rung, A, split_hash, ans, flags[rung]["echo_rate"]))
    A_syn = build_synthetic_f1(n=2 * 12000, m1=cfg["grids"]["m1"], m2=cfg["grids"]["m2_default"], seed=20260607)
    arms.append(("synthetic_f1", A_syn, A_syn.split_hash, None, None))

    for rung, A, sh, ans, echo in arms:
        ell = ell_star(A, cfg, xi)
        alpha = ell + Dg
        for s in _seeds(cfg):
            c, t = repeng.rep_split(A.n, s, _frac(cfg))
            for name, r in [
                ("nmscrc_i_eb", nmscrc_i.run(A, c, t, xi, alpha, cfg["delta"], "eb", "search")),
                ("nmscrc_i_hoeff", nmscrc_i.run(A, c, t, xi, alpha, cfg["delta"], "hoeffding", "search")),
                ("nmscrc_t", nmscrc_t.run(A, c, t, xi, alpha, cfg["delta"])),
                ("mono", mono.run(A, c, t, xi, alpha, cfg["delta"])),
                ("naive", naive.run(A, c, t, xi, alpha, cfg["delta"])),
                ("crcnm_marginal", crcnm_marginal.run(A, c, t, xi, alpha, cfg["delta"])),
                ("xu_proxy", xu_proxy.run(A, c, t, xi, alpha, cfg["delta"])),
            ]:
                _emit(buf, "head2head", name, rung, tag, s, s, sh, r, cfg, alpha=alpha, ell=ell, xi=xi)
            if ans is not None:
                _emit(buf, "head2head", "raw_llm", rung, tag, s, s, sh,
                      raw.run(A, c, t, xi, alpha, ans, echo), cfg, alpha=alpha, ell=ell, xi=xi, judged=False)
            _emit(buf, "head2head", "rand", rung, tag, s, s, sh,
                  rand.run(A, c, t, xi, alpha, cfg["delta"], s), cfg, alpha=alpha, ell=ell, xi=xi, judged=False)
        if verbose:
            print(f"[head2head] {rung} done (alpha={alpha:.4f})")
    _flush("head2head", buf)


def run_mono_gridrefine(cfg, split_hash, verbose=True):
    """single-figure evidence: as the Λ₂ grid densifies, mono (NO t_V cushion) FAILs
    (viol_frac -> 1) on BOTH rungs — its benign-PASS at coarse grids is grid luck, NOT validity, and
    NOT a probe-strength effect. NM-SCRC-I-EB (t_V cushion, same search) stays at viol_frac ≈ 0
    regardless of m₂ — the cushion immunises against grid refinement. Δ=0.05, search."""
    buf = {}
    xi = cfg["targets"]["xi_default"]
    Dg = cfg["targets"]["delta_gap"][1]
    m2_grid = [40, 80, 160, 320, 640, 1280, 2560]
    for rung in ("llama3_2_1b", "llama3_1_8b"):
        logits = np.load(paths.calibtest_logits(rung))
        labels = np.load(paths.calibtest_labels(rung))
        p = losses.sigmoid(logits.astype(np.float64))
        for m2 in m2_grid:
            cfg_m2 = json.loads(json.dumps(cfg))
            cfg_m2["grids"]["lambda2"]["n_points"] = m2
            lam1, lam2 = build_grids(p, cfg_m2)
            A = build_artifacts(rung, logits, labels, lam1, lam2, split_hash)
            ell = ell_star(A, cfg, xi)
            alpha = ell + Dg
            for s in _seeds(cfg):
                c, t = repeng.rep_split(A.n, s, _frac(cfg))
                _emit(buf, "mono_gridrefine", "mono", rung, f"m2_{m2:04d}", s, s, split_hash,
                      mono.run(A, c, t, xi, alpha, cfg["delta"]), cfg, alpha=alpha, ell=ell, xi=xi)
                _emit(buf, "mono_gridrefine", "nmscrc_i_eb", rung, f"m2_{m2:04d}", s, s, split_hash,
                      nmscrc_i.run(A, c, t, xi, alpha, cfg["delta"], "eb", "search"), cfg, alpha=alpha, ell=ell, xi=xi)
        if verbose:
            print(f"[mono_gridrefine] {rung} done")
    _flush("mono_gridrefine", buf)


def run_synth(cfg, verbose=True):
    """Stage 1 (no model): ⑨ feasibility floor (sweep n) + union-tax (sweep m₁m₂) on the hard family."""
    alpha = 0.10
    xi = cfg["targets"]["xi_default"]
    buf = {}
    for n_calib in cfg["n_sweep"] + [8000, 16000]:
        A = build_synthetic_artifacts(n=2 * n_calib, m1=cfg["grids"]["m1"], m2=cfg["grids"]["m2_default"],
                                      alpha=alpha, delta=0.002, seed=777)
        for variant in ("eb", "hoeffding"):
            for s in _seeds(cfg):
                c, t = repeng.rep_split(A.n, s, _frac(cfg))
                r = nmscrc_i.run(A, c, t, xi, alpha, cfg["delta"], variant=variant, lam1_mode="pinned")
                _emit(buf, "exp9_floor", f"nmscrc_i_{variant}", "synthetic", f"n{n_calib:05d}", s, s,
                      A.split_hash, r, cfg, alpha=alpha, ell=None, xi=xi)
    _flush("exp9_floor", buf)
    buf = {}
    for m in cfg["grids"]["m2_sweep"]:
        A = build_synthetic_artifacts(n=2 * 12000, m1=m, m2=m, alpha=alpha, delta=0.002, seed=888)
        for variant in ("eb", "hoeffding"):
            for s in _seeds(cfg):
                c, t = repeng.rep_split(A.n, s, _frac(cfg))
                r = nmscrc_i.run(A, c, t, xi, alpha, cfg["delta"], variant=variant, lam1_mode="pinned")
                _emit(buf, "union_tax", f"nmscrc_i_{variant}", "synthetic", f"m{m:04d}", s, s,
                      A.split_hash, r, cfg, alpha=alpha, ell=None, xi=xi)
    _flush("union_tax", buf)
    if verbose:
        print("[stage1] synth floor + union-tax done")


def run_c2(cfg, split_hash, verbose=True):
    """C2 / Thm 4.9 transductive LOO certificate phase transition (ADDED experiment; distinct from
    exp6's inductive union-tax). rung=8b, α = exp8's (ℓ*+0.10), B=1. Per (M,m2,rep) on the FIXED
    NM-SCRC-T accepted bag: t⋆, Δ, ŝ, K_direct (LOO identity), K_closed (Eq 4.3), cert=α+K·B/M,
    and out-of-sample true_held_risk at λ⋆. Quantile Λ2 grid (option A); onset read via Δ<(α+B)/M."""
    buf = {}
    rung = "llama3_1_8b"
    xi = cfg["targets"]["xi_default"]
    B = 1.0
    A0 = stage0.build_or_load_artifacts(rung, cfg, split_hash)
    ell = ell_star(A0, cfg, xi)
    alpha = ell + cfg["targets"]["delta_gap"][2]                  # SAME α as exp8 on 8b
    logits = np.load(paths.calibtest_logits(rung))
    labels = np.load(paths.calibtest_labels(rung))
    p = losses.sigmoid(logits.astype(np.float64))
    g = losses.selector_g(p)
    Y = (labels > 0).astype(np.float64)
    m2_grid = [80, 160, 320, 640, 1280, 2560, 5120, 10240]       # to saturation near α+B
    M_sizes = ["full", 500, 250]
    for m2 in m2_grid:
        cfg_m2 = json.loads(json.dumps(cfg))
        cfg_m2["grids"]["lambda2"]["n_points"] = m2
        lam2 = build_grids(p, cfg_m2)[1]
        L_f1 = losses.build_loss_tensor(p, Y, lam2, "f1")        # only the loss tensor (no SS) — cheap at large m2
        for s in _seeds(cfg):
            c, t = repeng.rep_split(len(g), s, _frac(cfg))
            g_c = g[c]
            order = np.argsort(-g_c)
            K_acc = int(np.ceil(xi * (len(c) + 1)) - 1)          # exp8 NM-SCRC-T accepted count
            tau = float(g_c[order[K_acc - 1]])
            acc_t = g[t] >= tau
            Lc, Lt = L_f1[c], L_f1[t]
            for Msz in M_sizes:
                Mbag = K_acc if Msz == "full" else min(int(Msz), K_acc)
                r = c2.transductive_loo_certificate(Lc[order[:Mbag]], alpha, lam2, B)
                tag = f"M{'full' if Msz == 'full' else f'{Mbag:04d}'}_m2_{m2:05d}"
                base = dict(exp="c2_transductive_cert", method="loo_cert", rung=rung, rep=s, seed=s,
                            split_hash=split_hash, xi=xi, alpha=alpha, ell_star=ell, m2=m2, n=len(c))
                if r["infeasible"]:
                    res = results.make_result(**base, abstained=True, M=Mbag, infeasible=True)
                else:
                    th = float(Lt[acc_t, r["t_star"]].mean()) if acc_t.sum() else float("nan")
                    res = results.make_result(**base, abstained=False, M=r["M"], t_star=r["t_star"],
                        lambda2_star=r["lambda2_star"], c2_delta=r["Delta"], s_hat=r["s_hat"],
                        K_direct=r["K_direct"], K_closed=r["K_closed"], cert=r["cert"],
                        true_held_risk=th, infeasible=False)
                buf.setdefault(("loo_cert", rung, tag), []).append(res)
        if verbose:
            print(f"[c2] m2={m2} done")
    _flush("c2_transductive_cert", buf)


# ===================================================================== load + audit
def load_results(exp):
    rows = []
    # version-filtered (read side of the _v() write prefix): only the CURRENT version's jsonl, so v1_
    # and v2_ runs coexisting in the same results/{exp}/ dir never merge.
    for p in glob.glob(str(paths.results_dir(exp) / f"{paths.out_version()}_*.jsonl")):
        tag = Path(p).stem.split("__")[-1]
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                d["_tag"] = tag
                rows.append(d)
    return rows


def _verdict_for(method, rows, alpha, delta):
    states = [r["state"] for r in rows]
    risks = [(r["risk_cond"] if r["risk_cond"] is not None else r["risk_marginal"]) for r in rows]
    risks = [(x if x is not None else np.nan) for x in risks]
    kind = "expectation" if method == "nmscrc_t" else "pac"
    return judge.method_verdict(states, risks, alpha, delta, kind=kind)


def build_audit(cfg, expected_reps=None):
    """results/AUDIT.md — compliance, NUMBERS ONLY (no interpretation). Per
    (exp, method, rung, tag): rep count (assert == n_reps), PASS/FAIL/ABSTAIN counts, split_hash
    match; per-model raw-LLM echo_rate; qc_3b verdict + rung ⑥; FLAG anomalies."""
    expected_reps = expected_reps or cfg["reps"]["n"]
    combined = hashing.read_combined_hash(paths.artifact_hash())
    qc = json.loads(paths.qc_3b().read_text())
    flags = json.loads(paths.answer_flags().read_text())

    PER_REP = ["exp1", "exp5", "exp6", "exp8", "head2head", "mono_gridrefine", "c2_transductive_cert",
               "exp9_floor", "union_tax"]
    L = ["# AUDIT — compliance (numbers only)", "",
         f"ARTIFACT_HASH combined sha256: `{combined}`",
         f"qc_3b: use_3b_for_transition={qc['use_3b_for_transition']}  exp6_primary_rung={qc['exp6_primary_rung']}  "
         f"transition_detected_small_M={qc.get('transition_detected_small_M')} ({qc.get('transition_axis')})",
         "", "## per (exp, method, rung, tag): reps / states / hash",
         "| exp | method | rung | tag | reps | PASS | FAIL | ABSTAIN | hash_ok | FLAG |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    flagged, totals = [], {"PASS": 0, "FAIL": 0, "ABSTAIN": 0}
    for exp in PER_REP:
        groups = {}
        for r in load_results(exp):
            groups.setdefault((r["method"], r["rung"], r["_tag"]), []).append(r)
        for (method, rung, tag), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
            sc = {s: sum(1 for r in rs if r["state"] == s) for s in ("PASS", "FAIL", "ABSTAIN")}
            for s in totals:
                totals[s] += sc[s]
            hashes = {r["split_hash"] for r in rs}
            hash_ok = all((h == combined) or (h or "").startswith("synthetic") for h in hashes)
            flag = []
            if len(rs) != expected_reps:
                flag.append(f"reps={len(rs)}!={expected_reps}")
            if not hash_ok:
                flag.append("HASH_MISMATCH")
            if flag:
                flagged.append(f"{exp}/{method}/{rung}/{tag}: {','.join(flag)}")
            L.append(f"| {exp} | {method} | {rung} | {tag} | {len(rs)} | {sc['PASS']} | {sc['FAIL']} | "
                     f"{sc['ABSTAIN']} | {hash_ok} | {','.join(flag) if flag else '-'} |")

    coexist = all(v > 0 for v in totals.values())
    L += ["", "## three-state coexistence (across all judged reps)",
          f"PASS={totals['PASS']}  FAIL={totals['FAIL']}  ABSTAIN={totals['ABSTAIN']}  all_three_present={coexist}",
          "", "## raw-LLM echo_rate (separate from F1; raw-LLM is a floor, not judged)"]
    for m in cfg["models"]:
        a = flags[m]
        L.append(f"- {m}: echo_rate={a['echo_rate']:.4f}  unparseable={a['unparseable_rate']:.4f}  "
                 f"genuine_empty={a['genuine_empty_rate']:.4f}  all_unknown={a['all_unknown_rate']:.4f}  ok={a['ok_rate']:.4f}")

    if not coexist:
        flagged.append("THREE_STATE_COEXISTENCE_FAILED")
    L += ["", "## FLAGS"] + ([("- " + f) for f in flagged] if flagged else ["- none"])

    paths.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    paths.audit_md().write_text("\n".join(L), encoding="utf-8")
    return {"flagged": flagged, "coexist": coexist, "totals": totals}
