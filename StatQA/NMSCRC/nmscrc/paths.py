"""The single place all filesystem paths are resolved.

Three env-overridable roots; everything else is derived via pathlib (Windows '\\' vs
Linux '/' handled automatically). NO absolute path literal appears anywhere in the code.

  Local : set nothing -> PROJECT_ROOT auto-infers, DATA_ROOT = repo/data.
  CX3   : `export NMSCRC_DATA=$EPHEMERAL/nmscrc/data` (or RDS). No code change.

Never rely on the current working directory: Path(__file__).resolve() makes resolution
CWD-independent (PBS/Slurm jobs start elsewhere).
"""

import os
from functools import lru_cache
from pathlib import Path

# ---- three env-overridable roots (everything else derived) ----
PROJECT_ROOT = Path(os.environ.get("NMSCRC_ROOT", Path(__file__).resolve().parents[1]))   # repo root (auto)
DATA_ROOT    = Path(os.environ.get("NMSCRC_DATA", PROJECT_ROOT / "data"))                  # CX3: scratch/RDS
RESULTS_ROOT = Path(os.environ.get("NMSCRC_RESULTS", PROJECT_ROOT / "results"))
SPLIT_NAME   = os.environ.get("NMSCRC_SPLIT", "pool_probe30calibtest70")
RAW_DIR      = os.environ.get("NMSCRC_RAW_DIR")   # raw-data folder override; None -> config raw_dir / "data-full"

# 3 rungs: weak / mid / strong — EXACT on-disk strings
MODELS = ["llama3_2_1b", "llama3_2_3b", "llama3_1_8b"]

CONFIG_PATH  = PROJECT_ROOT / "config.yaml"

# ---- INPUTS (already on disk; preflight asserts present for all 3 models) ----
# raw-data folder is a config/env knob (data swap): NMSCRC_RAW_DIR env > config raw_dir > "data-full".
def _raw_dir():       return RAW_DIR or load_config().get("raw_dir", "data-full")
def data_full(m):     return DATA_ROOT / _raw_dir() / f"{m}_pool_data.csv"
def data_full_hs(m):  return DATA_ROOT / _raw_dir() / f"{m}_pool_hs.npy"
METHODS_JSON = DATA_ROOT / "method_columns.json"   # shared 27-method list (same for all data versions)

# ---- output version prefix (every GENERATED file is prefixed so v1/v2 runs coexist) ----
# resolution: NMSCRC_VERSION env > config `version` > "v1". Inputs above are NEVER prefixed.
def out_version():    return os.environ.get("NMSCRC_VERSION") or load_config().get("version", "v1")
def _v(name):         return f"{out_version()}_{name}"

# ---- derived dirs ----
def split_dir():      return DATA_ROOT / "data-split"  / SPLIT_NAME
def probe_dir():      return DATA_ROOT / "probe-train" / SPLIT_NAME
def artifacts_dir():  return DATA_ROOT / "artifacts"   / SPLIT_NAME

# ---- Stage 0a split outputs (part in {"train","calibtest"}) ----
def split_data(m, part):  return split_dir() / _v(f"{m}_{part}_data.csv")
def split_hs(m, part):    return split_dir() / _v(f"{m}_{part}_hs.npy")
def split_manifest(m):    return split_dir() / _v(f"{m}_{SPLIT_NAME}_manifest.csv")

# ---- Stage 0b probe outputs (frozen; inference on calibtest pool) ----
def probe_pt(m):              return probe_dir() / _v(f"{m}_probe.pt")
def calibtest_logits(m):      return probe_dir() / _v(f"{m}_calibtest_logits.npy")
def calibtest_logits_csv(m):  return probe_dir() / _v(f"{m}_calibtest_logits.csv")
def calibtest_labels(m):      return probe_dir() / _v(f"{m}_calibtest_labels.npy")
def calibtest_labels_csv(m):  return probe_dir() / _v(f"{m}_calibtest_labels.csv")
def probe_meta(m):            return probe_dir() / _v(f"{m}_probe_meta.json")

# ---- Stage 0c raw-LLM answer + derived tensors ----
def answer_csv(m):    return split_dir() / _v(f"{m}_calibtest_answer.csv")   # raw-LLM multi-hot + per-row flag
def loss_tensor(m):   return artifacts_dir() / _v(f"{m}_loss_tensor.npy")    # ell(C_{l2}(x_i), y_i) on pool x Λ2
def scores_p(m):      return artifacts_dir() / _v(f"{m}_scores_p.npy")       # full p(x) in [0,1]^27 (selector source)
def selection(m):     return artifacts_dir() / _v(f"{m}_selection.npy")      # g(x) scalar on pool (max_k p_k)
def lambda1_grid(m):  return artifacts_dir() / _v(f"{m}_lambda1.npy")        # frozen per-model Λ1 grid
def lambda2_grid(m):  return artifacts_dir() / _v(f"{m}_lambda2.npy")        # frozen per-model Λ2 grid
def answer_flags():   return artifacts_dir() / _v("answer_flags.json")       # per-model echo/genuine_empty/... rates

# ---- Stage 0d-0e QC + freeze ----
def qc_3b():          return artifacts_dir() / _v("qc_3b.json")
def artifact_hash():  return artifacts_dir() / _v("ARTIFACT_HASH.txt")

# ---- results (downstream; never recomputed by plots) — all prefixed ----
def results_dir(exp):              return RESULTS_ROOT / exp
def result_jsonl(exp, method, rung, tag):  return RESULTS_ROOT / exp / _v(f"{method}__{rung}__{tag}.jsonl")
def ushape_json(rung):             return RESULTS_ROOT / "exp3" / _v(f"ushape__{rung}.json")
def audit_md():       return RESULTS_ROOT / _v("AUDIT.md")
def report_md():      return RESULTS_ROOT / _v("RESULTS_REPORT.md")
def figures_dir():    return RESULTS_ROOT / _v("figures")
def notebook_name():  return _v("experiments.ipynb")


@lru_cache(maxsize=1)
def load_config():
    """Numeric knobs from config.yaml. Paths come from this module, not config."""
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolved_roots():
    return {"PROJECT_ROOT": str(PROJECT_ROOT), "DATA_ROOT": str(DATA_ROOT),
            "RESULTS_ROOT": str(RESULTS_ROOT), "SPLIT_NAME": SPLIT_NAME, "raw_dir": _raw_dir()}
