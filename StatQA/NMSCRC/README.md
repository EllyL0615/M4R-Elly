# NM-SCRC: Non-Monotone Selective Conformal Risk Control

Distribution-free risk control for a non-monotone loss (the F1 loss), combined with a selective
(abstaining) prediction rule. The method is applied to statistical-method selection: given a
data-analysis question, decide which statistical methods are applicable, with a finite-sample,
high-probability guarantee on the F1 risk over the questions the system chooses to answer.

## Overview

For each question we read a language model's hidden state, pass it through a frozen linear probe to
obtain per-method probabilities `p(x)` in `[0,1]^27`, and form a nested prediction set

```
C_{l2}(x) = { method k : p_k(x) >= 1 - l2 }.
```

A scalar selector `g(x) = max_k p_k(x)` decides whether to answer at all: the system accepts the
question when `g(x) >= 1 - l1` and abstains otherwise. The loss on an accepted question is

```
loss = 1 - F1( C_{l2}(x), Y ),     loss = 1 when C is empty,
```

where `Y` is the ground-truth set of applicable methods. F1 is not monotone in the set size `l2`:
enlarging the set raises recall but eventually hurts precision, so the risk is U-shaped. Standard
monotone conformal-risk-control machinery cannot certify this, which is what NM-SCRC is built to
handle. NM-SCRC calibrates `(l1, l2)` so that the accepted-region conditional risk
`E[ loss | g(X) >= 1 - l1 ]` is controlled at a target `alpha` with confidence `1 - delta`.

The study uses three model rungs, from weakest to strongest: `llama3_2_1b`, `llama3_2_3b`,
`llama3_1_8b`. It compares NM-SCRC against baselines and against two published alternatives (Xu et
al. and Aldirawi et al.). See [METHODS.md](METHODS.md) for the full protocol, method definitions, and
certificates.

## Repository structure

```
.
├── README.md              this file
├── METHODS.md             methodology: protocol, method definitions, losses, certificates
├── config.yaml            all numeric knobs (splits, grids, targets, confidence budget)
├── pyproject.toml         package metadata (installable as `nmscrc`)
├── requirements.txt       exact dependency versions used for the results
│
├── nmscrc/                the experiment package (importable library)
│   ├── paths.py           single point of path resolution (env-overridable roots)
│   ├── data.py, parser.py, split.py, probe.py     Stage-0 data pipeline and frozen probe
│   ├── losses.py, judge.py, repeng.py             risk losses, PASS/FAIL/ABSTAIN judging, rep splits
│   ├── stage0.py, artifacts.py, hashing.py        frozen-artifact build, caching, hash checks
│   ├── experiments.py, summary.py, results.py     experiment orchestration and result tables
│   ├── plots.py, synthetic.py, c2.py              figures, synthetic instances, transductive cert
│   └── methods/           one module per method (see METHODS.md)
│       ├── nmscrc_i.py, nmscrc_t.py               NM-SCRC inductive / transductive
│       ├── naive.py, raw.py, rand.py, mono.py     baselines
│       ├── xu_proxy.py, crcnm_marginal.py         published opponents
│       └── _engine.py                             shared calibration engine
│
├── llm_gen/               upstream data generation (steps 0-1, GPU): raw StatQA to hidden states
│   ├── step0_make_prompts.py        methods-only prompts   -> data/prompts/
│   ├── step1_hidden_states.py       hidden states + answers -> data/data-full/
│   └── jobs/                         PBS launchers for step1
│
├── prompt_console/        optional: interactive vLLM console to test a prompt on all 3 models
│   ├── run_console.sh, run_prompt_loop_vllm.py
│   └── prompts/linear_probe.txt, outputs/
│
├── notebooks/             experiment drivers (run top-to-bottom; figures render inline)
│   ├── v1_experiments.ipynb, v2_experiments.ipynb, v3_experiments.ipynb
│   └── pool.ipynb         build the pool split (concatenate medium + hard)
│
├── scripts/
│   └── make_figs_beautify_v2.py     re-render the publication figures from frozen results
│
├── data/                  inputs and frozen Stage-0 artifacts (see "Data layout" below)
└── results/               per-experiment outputs (*.jsonl), figures, and results reports
```

### Versions v1 / v2 / v3

The experiment was run three times on three independent data draws (`data-full`, `data-full_v2`,
`data-full_v3`). Every generated file is prefixed with its version (`v3_...`), so the three runs
coexist. v3 is the most recent run. Each version has its own driver notebook, figure folder
(`results/v{1,2,3}_figures/`), and results report (`results/v{1,2,3}_RESULTS_REPORT.md`).

## Installation

Python 3.12. From the repository root:

```bash
pip install -r requirements.txt
pip install -e .          # makes `import nmscrc` work from anywhere (notebooks/, scripts/)
```

The editable install is optional if you always launch Jupyter from the repository root (the
notebooks add the repo root to `sys.path` themselves), but it is the most robust option.

## Running the experiments

The pipeline is six steps. Steps 0 to 2 (GPU) regenerate the raw pool data from raw StatQA
questions and are optional: the frozen `data-full*/` outputs are shipped, so you normally start at
step 3. All commands run from the repository root. Every path is resolved by `nmscrc/paths.py` (see
[Paths and environment](#paths-and-environment)); no absolute path is hard-coded.

**Step 0, make the methods-only prompts** (CPU, about 1 min). Self-contained (no dependency on the
StatQA package code; the SCRC prompt template is inlined).

```bash
python llm_gen/step0_make_prompts.py --set both      # or --set train / --set test
```
- input: StatQA repo files, resolved from `$STATQA_ROOT` (default: the sibling `StatQA/` dir):
  `Data/Integrated Dataset/Dataset with Prompt/Training Set/D_train for zero-shot.csv`,
  `Data/Integrated Dataset/Balanced Benchmark/mini-StatQA.csv`, `Data/Metadata/Column Metadata/*`.
- output: `data/prompts/{D_train, mini-StatQA} for methods-only.csv` (plus `D_train ...report.csv`).

**Step 1, extract hidden states** (GPU node, conda env `M4R_llama3x`, about 2h20 for 8B). For each
prompt it stores the final-layer, final-token hidden state and greedily decodes the method answer.

```bash
bash llm_gen/jobs/step1_submit_all.sh                # one PBS job per model; use bash, not qsub
qstat -u yl9422                                       # monitor
# no-GPU smoke (mock hidden states, 3 rows per split):
python llm_gen/step1_hidden_states.py --model_type 3_2_1b --mock_inference \
    --smoke_rows_per_split 3 --hidden_state_dir /tmp/smoke --origin_answer_dir /tmp/smoke
```
- input: `data/prompts/*` plus LLaMA weights at `/rds/.../files/models/Llama-3.{1-8B,2-1B,2-3B}`.
- output: `data/data-full/{model}_{medium,hard}_{data.csv,hs.npy}` for the three rungs
  `llama3_1_8b`, `llama3_2_1b`, `llama3_2_3b` (split `medium` = D_train block, `hard` = mini-StatQA).

**Step 2, build the pool split** (CPU). Run `notebooks/pool.ipynb` top-to-bottom.
- input: `data/data-full/{model}_{medium,hard}_*`.
- output: `data/data-full/{model}_pool_{data.csv,hs.npy}` (medium then hard, CSV and `.npy` kept
  row-aligned). These are the inputs the steps below require.

**Step 3, check the inputs are present** (resolves and asserts every input file):

```bash
python -m nmscrc.check_paths
```
- checks: `{model}_pool_data.csv` + `{model}_pool_hs.npy` (three rungs) + `method_columns.json`.

**Step 4, run a driver notebook** top-to-bottom, for example `notebooks/v3_experiments.ipynb`. Stage
0 (the frozen, hash-checked probe build) and the 100-rep experiments are cache-aware: with the
shipped `results/` populated, the notebook re-uses cached outputs and just renders the figures and
tables. Set `FORCE` / `FORCE_EXP = True` in the setup cell to recompute from scratch.
- output: `results/<exp>/*.jsonl`, `results/v3_figures/`, `results/v3_RESULTS_REPORT.md`.

**Step 5, re-render the publication figures** (reads frozen results):

```bash
python scripts/make_figs_beautify_v2.py
```
- output: `results/figures_beautify_v2/`.

## Interactive prompt testing (optional)

Before committing to a full step-0/step-1 run, you can test a prompt on all three models
interactively. `prompt_console/` keeps the three rungs resident on one GPU (vLLM); you edit a prompt
file, press Enter, and read each model's reply, with no reloading between models.

```bash
# in an interactive GPU session, env M4R_llama3x_vllm (loading the 3 models takes about 5 min):
qsub -I -l select=1:ncpus=8:mem=64gb:ngpus=1 -l walltime=02:00:00
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R_llama3x_vllm
bash prompt_console/run_console.sh
```

- input: `prompt_console/prompts/linear_probe.txt`. Line 1 must be `#model: <name>` (`3_1_8b`,
  `3_2_3b`, or `3_2_1b`); the prompt body follows. Switch model by editing line 1, then press Enter.
- output: `prompt_console/outputs/latest_reply.txt` plus a timestamped `reply_<model>_<job>_<time>.txt`.
- Greedy decoding (`temperature=0`). The GPU-memory split defaults to `8B=0.45 / 3B=0.25 / 1B=0.15`
  (about 0.85, fits a 46GB L40S); lower `--gpu-frac` or `--max-model-len` on a smaller card.

### Paths and environment

All filesystem paths are resolved in `nmscrc/paths.py` from a few environment-overridable roots; no
absolute path is hard-coded. Run locally with nothing set and the roots auto-infer to the
repository:

| Variable | Default | Purpose |
|---|---|---|
| `NMSCRC_ROOT`    | repo root (auto) | project root |
| `NMSCRC_DATA`    | `<root>/data`    | point at scratch/RDS on a cluster |
| `NMSCRC_RESULTS` | `<root>/results` | results location |
| `NMSCRC_SPLIT`   | `pool_probe30calibtest70` | split name |
| `NMSCRC_RAW_DIR` | `data-full` (config) | raw-data folder (`data-full_v3` for the v3 run) |
| `NMSCRC_VERSION` | `v1` (config) | output-file prefix |
| `STATQA_ROOT`    | sibling `StatQA/` (auto) | StatQA repo holding the step-0 raw inputs |

## Configuration

`config.yaml` is the single place every numeric knob is defined: the 30/70 probe/calibtest split,
the 100 reps, the confidence budget `delta = d1 + dV + dU`, the `L1 x L2` grids, the targets
`alpha = loss* + Delta` with `Delta` in `{0.02, 0.05, 0.10}`, and the selector. No numbers are
hard-coded in the package.

## Data layout

```
data/
├── method_columns.json                       ordered 27-method list (id, name)
├── prompts/          methods-only prompt CSVs (step0 output; input to step1)
├── data-full/        {model}_pool_data.csv + {model}_pool_hs.npy     (v1 raw pool)
│                     (plus the {model}_{medium,hard}_* files step1 emits, pooled by pool.ipynb)
├── data-full_v2/     ...                                             (v2 raw pool)
├── data-full_v3/     ...                                             (v3 raw pool)
├── data-split/pool_probe30calibtest70/        stratified 30/70 split per model
├── probe-train/pool_probe30calibtest70/       frozen-probe logits, labels, weights (.pt)
└── artifacts/pool_probe30calibtest70/         loss tensors, grids, hashes (Stage-0 freeze)
```

Each `_pool_data.csv` carries `task`, `difficulty` (stratification keys), `results` (ground-truth
applicable methods, used for both probe labels and F1 evaluation), and `model_answer` (the raw LLM
text, used only by the raw-LLM baseline). The `_pool_hs.npy` files are the row-aligned hidden states
(width 2048 / 3072 / 4096 for the three rungs).

Note: the large `.npy` arrays are kept on the filesystem and are not versioned in git (they exceed
GitHub's 100MB file limit); regenerate them with `llm_gen/` if needed.

## Results

- `results/<exp>/*.jsonl`: per-repetition outputs for each experiment (validity, U-shape, xi-slope,
  phase transition, inductive vs transductive, head-to-head, grid refinement, transductive
  certificate, synthetic floor / union-tax).
- `results/v{1,2,3}_figures/` and `results/figures_beautify_v2/`: figures.
- `results/v{1,2,3}_RESULTS_REPORT.md`: the objective numerical tables for each run.
