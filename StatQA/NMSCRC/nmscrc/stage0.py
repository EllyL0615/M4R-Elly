"""Stage 0 orchestration: the SERIAL upstream bottleneck. Produces the frozen artifacts
every experiment consumes. Steps 0a split -> 0b probe(freeze) -> 0c scores+answer+loss_tensor ->
0d qc_3b -> 0e freeze+hash. The notebook calls these; logic lives here so it is importable/testable.

All steps are cache-aware: if outputs exist and force=False they are loaded, so re-running the
notebook (e.g. at the two checkpoints) does not recompute the probe/loss tensors.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nmscrc import data, parser, paths, probe, losses, split, hashing
from nmscrc.artifacts import build_artifacts, build_grids


# ---------------------------------------------------------------- 0a split
def stage0a_split(cfg, force=False, verbose=True):
    out = {}
    for m in cfg["models"]:
        if (not force) and paths.split_hs(m, "train").exists() and paths.split_hs(m, "calibtest").exists():
            if verbose:
                print(f"[0a] {m}: split exists -> skip")
            out[m] = "cached"
            continue
        out[m] = split.split_pool(m, probe_train=cfg["split_ratio"]["probe_train"],
                                  calibtest=cfg["split_ratio"]["calibtest"],
                                  seed=cfg["split_seed"], verbose=verbose)
    return out


# ---------------------------------------------------------------- 0b probe (freeze)
def stage0b_probe(cfg, force=False, verbose=True):
    methods = data.load_method_list()
    p = cfg["probe"]
    meta_all = {}
    paths.probe_dir().mkdir(parents=True, exist_ok=True)
    for m in cfg["models"]:
        if (not force) and paths.calibtest_logits(m).exists() and paths.calibtest_labels(m).exists():
            if verbose:
                print(f"[0b] {m}: frozen-probe logits exist -> skip")
            meta_all[m] = json.loads(paths.probe_meta(m).read_text(encoding="utf-8")) \
                if paths.probe_meta(m).exists() else "cached"
            continue
        x_tr = probe.load_hidden_states(paths.split_hs(m, "train"))
        y_tr = data.build_multi_hot_labels(pd.read_csv(paths.split_data(m, "train")), methods)
        x_ct = probe.load_hidden_states(paths.split_hs(m, "calibtest"))
        y_ct = data.build_multi_hot_labels(pd.read_csv(paths.split_data(m, "calibtest")), methods)

        probe.set_seed(cfg["split_seed"])
        model, metrics = probe.train_probe(
            x_tr, y_tr, seed=cfg["split_seed"], epochs=p["epochs"], batch_size=p["batch_size"],
            lr=float(p["lr"]), weight_decay=float(p["weight_decay"]), patience=p["patience"],
            val_ratio=p["val_ratio"], use_pos_weight=p["use_pos_weight"],
            device=torch.device(p["device"]), verbose=False,
        )
        logits = probe.predict_logits(model, x_ct, torch.device(p["device"]), p["batch_size"])

        torch.save(model.state_dict(), paths.probe_pt(m))
        np.save(paths.calibtest_logits(m), logits)
        np.save(paths.calibtest_labels(m), y_ct)
        pd.DataFrame(logits, columns=methods).to_csv(paths.calibtest_logits_csv(m), index=False)
        pd.DataFrame(y_ct.astype(int), columns=methods).to_csv(paths.calibtest_labels_csv(m), index=False)
        meta = {"model": m, "method_list_len": len(methods), "calibtest_rows": int(logits.shape[0]),
                "logit_width_methods": int(logits.shape[1]), **metrics}
        paths.probe_meta(m).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        meta_all[m] = meta
        if verbose:
            print(f"[0b] {m}: input_dim={metrics['input_dim']} val_auc={metrics['val_macro_auc']:.4f} "
                  f"calibtest_logits={logits.shape}")
    return meta_all


# ---------------------------------------------------------------- 0c scores + answer + loss tensor
def stage0c_scores(cfg, force=False, verbose=True):
    methods = data.load_method_list()
    paths.artifacts_dir().mkdir(parents=True, exist_ok=True)
    all_flags = {}
    for m in cfg["models"]:
        logits = np.load(paths.calibtest_logits(m))
        labels = np.load(paths.calibtest_labels(m))
        p = losses.sigmoid(logits.astype(np.float64))
        g = losses.selector_g(p)
        Y = (labels > 0).astype(np.float64)

        # per-model FROZEN grids (quantile by default; derived from the pool only)
        lam1, lam2 = build_grids(p, cfg)
        np.save(paths.lambda1_grid(m), lam1)
        np.save(paths.lambda2_grid(m), lam2)

        # raw-LLM answer matrix + per-row flag + rates (on the calibtest pool's model_answer)
        df_ct = pd.read_csv(paths.split_data(m, "calibtest"))
        ans, flags, rates = parser.build_answer_matrix(df_ct, methods)
        ans.insert(0, "_flag", flags.values)
        ans.to_csv(paths.answer_csv(m), index=False)
        all_flags[m] = rates

        # derived tensors (loss tensor uses the frozen λ2 grid)
        if force or not paths.loss_tensor(m).exists():
            np.save(paths.loss_tensor(m), losses.build_loss_tensor(p, Y, lam2, kind="f1"))
        np.save(paths.scores_p(m), p.astype(np.float32))
        np.save(paths.selection(m), g.astype(np.float32))
        if verbose:
            print(f"[0c] {m}: echo_rate={rates['echo_rate']:.4f} ok_rate={rates['ok_rate']:.4f} "
                  f"allzero={rates['all_zero_rows']}/{rates['n']} | lam1[{lam1[0]:.3f},{lam1[-1]:.3f}] "
                  f"lam2[{lam2[0]:.3f},{lam2[-1]:.3f}] | loss_tensor saved")

    paths.answer_flags().write_text(json.dumps(all_flags, indent=2), encoding="utf-8")
    return all_flags


# ---------------------------------------------------------------- 0d QC: 3B K/M transition vs degenerate
def stage0d_qc(cfg, verbose=True):
    """Is 3B's K/M a genuine Δ<α/M transition (Eq 4.3: K>0 iff δ<(α+B)/M) or a degenerate
    weak-probe loss distribution? Sweeps bag size M per rung; emits use_3b_for_transition + the α,
    the slack Δ vs α/M, and which rung exp⑥ uses. NUMBERS + the control boolean only."""
    from nmscrc.methods import nmscrc_t
    xi = cfg["targets"]["xi_default"]
    Dg = cfg["targets"]["delta_gap"][1]                  # default Δ = 0.05
    M_grid = [20, 40, 80, 160, 320, 640, 1280]

    km_vs_M, ell_star, alpha_by = {}, {}, {}
    for m in cfg["models"]:
        A = build_or_load_artifacts(m, cfg)
        ell, _ = losses.oracle_ell_star(A.g, A.Y, A.L_f1, A.lambda1, A.lambda2, xi)
        alpha = ell + Dg
        ell_star[m], alpha_by[m] = ell, alpha
        order = np.argsort(-A.g)
        km = {}
        for M in M_grid + [A.n]:
            if M - 1 < 2:
                continue
            ic = nmscrc_t.instability_count(A.L_f1[order[:M - 1]], alpha, M)
            km[str(M)] = ic["K_over_M"]
        km_vs_M[m] = km

    # 3B transition detail
    A = build_or_load_artifacts("llama3_2_3b", cfg)
    alpha = alpha_by["llama3_2_3b"]
    order = np.argsort(-A.g)
    trans = []
    for M in M_grid:
        ic = nmscrc_t.instability_count(A.L_f1[order[:M - 1]], alpha, M)
        Lstar = A.L_f1[order[:M - 1]][:, ic["lam2_star_idx"]]
        trans.append({"M": M, "K_over_M": ic["K_over_M"], "delta_slack": ic["delta"],
                      "alpha_plus_B_over_M": (alpha + 1.0) / M, "alpha_over_M": alpha / M,
                      "c": ic["c"], "frac_loss_zero_at_lamstar": float(np.mean(Lstar == 0.0))})
    high_M = [t["M"] for t in trans if t["K_over_M"] >= 0.5]
    transition_detected = len(high_M) > 0
    # DECISION A (user-confirmed): 3B's K/M transition is along the BAG-SIZE (M) axis and entangled
    # with a degenerate near-oracle loss (mass at L=0); exp⑥'s transition is m₂-driven on 8b — a
    # DIFFERENT axis. So 3B is NOT used as exp⑥'s main evidence; it is an appendix one-liner + small fig.
    use_3b = False

    qc = {
        "xi": xi, "delta_gap": Dg,
        "transition_condition": "K>0 iff delta < (alpha+B)/M  (Eq 4.3, B=1)",
        "ell_star": ell_star, "alpha": alpha_by,
        "K_over_M_vs_M": km_vs_M,
        "transition_3b": trans,
        "transition_detected_small_M": bool(transition_detected),
        "transition_axis": "bag-size M (NOT the m2 axis of exp6)",
        "K_over_M_high_at_M": high_M,
        "use_3b_for_transition": bool(use_3b),            # DECISION A = False (appendix only)
        "exp6_primary_rung": "llama3_1_8b",
        "exp6_supporting_3b_panel": bool(use_3b),
    }
    paths.qc_3b().write_text(json.dumps(qc, indent=2, default=float), encoding="utf-8")
    if verbose:
        print(f"[0d] qc_3b: K/M(M=20)= " +
              ", ".join(f"{m.split('_')[-1]}={km_vs_M[m].get('20', float('nan')):.3f}" for m in cfg["models"]) +
              f"; use_3b_for_transition(tentative)={use_3b}; exp6 rung=llama3_1_8b")
    return qc


# ---------------------------------------------------------------- 0e freeze + sha256
def stage0e_freeze(cfg, verbose=True):
    """Hash every Stage-0 artifact -> ARTIFACT_HASH.txt (combined digest = each json's split_hash)."""
    files = []
    for m in cfg["models"]:
        files += [paths.split_data(m, "train"), paths.split_hs(m, "train"),
                  paths.split_data(m, "calibtest"), paths.split_hs(m, "calibtest"),
                  paths.split_manifest(m), paths.answer_csv(m),
                  paths.probe_pt(m), paths.calibtest_logits(m), paths.calibtest_logits_csv(m),
                  paths.calibtest_labels(m), paths.calibtest_labels_csv(m), paths.probe_meta(m),
                  paths.loss_tensor(m), paths.scores_p(m), paths.selection(m),
                  paths.lambda1_grid(m), paths.lambda2_grid(m)]
    files += [paths.answer_flags(), paths.qc_3b(), paths.METHODS_JSON]
    files = [f for f in files if Path(f).exists()]
    combined = hashing.write_artifact_hash(files, paths.artifact_hash(), paths.DATA_ROOT)
    if verbose:
        print(f"[0e] froze {len(files)} files; combined sha256={combined[:16]}... -> {paths.artifact_hash().name}")
    return combined


def build_or_load_artifacts(model, cfg, split_hash=None):
    """In-memory Artifacts for experiments. Loads the FROZEN per-model grids (falls back to building
    them from p if absent), recomputes loss tensors from frozen logits (fast)."""
    logits = np.load(paths.calibtest_logits(model))
    labels = np.load(paths.calibtest_labels(model))
    if paths.lambda1_grid(model).exists() and paths.lambda2_grid(model).exists():
        lam1 = np.load(paths.lambda1_grid(model))
        lam2 = np.load(paths.lambda2_grid(model))
    else:
        lam1, lam2 = build_grids(losses.sigmoid(logits.astype(np.float64)), cfg)
    return build_artifacts(model, logits, labels, lam1, lam2, split_hash)
