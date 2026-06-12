# Methodology

This document describes the experimental protocol, the method definitions, the losses, and the
certificates. `config.yaml` holds the numeric values; this document explains what they mean.

## 1. Setup and notation

Each example `x` is a data-analysis question. A language model produces a hidden state, which a
frozen linear probe `f` maps to per-method probabilities `p(x) = sigmoid(f(x)) ∈ [0,1]^27` over the
27 candidate statistical methods. The ground truth `Y ∈ {0,1}^27` is the set of applicable methods
for the question.

- Prediction set (nested in `λ₂`): `C_{λ₂}(x) = { k : p_k(x) ≥ 1 − λ₂ }`. A larger `λ₂` gives a larger set.
- Selector: `g(x) = maxₖ p_k(x)`. The system accepts `x` iff `g(x) ≥ 1 − λ₁`, otherwise it abstains.
  Write `Uᵢ(λ₁) = 1{ g(xᵢ) ≥ 1 − λ₁ }`.
- Loss: `ℓ(C_{λ₂}(x), Y) = 1 − F1(C_{λ₂}(x), Y)`, bounded by `B = 1`; `ℓ = 1` when `C = ∅`.
- Target risk: the accepted-region conditional risk `R_cond(λ₁,λ₂) = E[ ℓ | g(X) ≥ 1 − λ₁ ]`.
- Selection ratio: `φ̂_n(λ₁) = (1/n) Σᵢ Uᵢ(λ₁)`, the empirical fraction of accepted calibration points.

The goal is to choose `(λ₁, λ₂)` from calibration data so that, with probability `≥ 1 − δ`,
`R_cond ≤ α`, while keeping the prediction sets small and abstaining as rarely as possible.

## 2. Why the F1 loss is hard (non-monotonicity)

A monotone loss can only ever decrease as the set grows, so monotone conformal-risk-control counts
the loss at the smallest set that crosses the target. The F1 loss is U-shaped in `λ₂`: enlarging the
set raises recall but adds false positives, lowering precision, so the loss rises again at large
`λ₂`. The non-monotonicity is the precision penalty. No monotone surrogate can represent that rising
cost, so the standard machinery does not certify F1 directly. This is the gap NM-SCRC closes.

## 3. Frozen protocol

| Item | Setting |
|---|---|
| Split | A fixed, stratified-on-(task × difficulty) split of each model's pool into a 30% probe-training set and a 70% calib/test pool (`pool_probe30calibtest70`). The probe is trained on the 30% and then frozen. |
| Reps | 100 repetitions, seeds `0…99`. Each rep re-splits the 70% pool into calib / test (50/50). The probe is never re-trained; only the calib/test boundary moves. This is what makes the validity histogram samplable: the randomness sampled is the draw of the calibration set. |
| `f, g` | The probe `f` and selector `g` are a fixed function of the frozen probe output, shared by all methods and all reps. Re-training the probe per rep would inject probe randomness and break the conditional-on-`(f,g)` guarantee. |
| Targets | Sweep `(ξ, α)`. `α = ℓ* + Δ` with `Δ ∈ {0.02, 0.05, 0.10}`, where `ℓ*` is the oracle minimum conditional risk. Default `ξ = 0.3`, with sweeps for the slope/coverage studies. |
| Grids | The same `Λ₁ × Λ₂` grid for every method, sizes `m₁ = m₂ = 80` by default. Grid points are quantiles of `g` (for `Λ₁`) and of the pooled probabilities (for `Λ₂`), derived from the frozen pool only, so coverage and set size sweep evenly despite probe saturation near 1. |
| Confidence | `δ = 0.1`, split into a budget `δ = δ₁ + δ_V + δ_U` (`config.yaml: delta`). |
| Risk caliber | The accepted-region conditional risk, for every method except CRC-NM-marginal, which is marginal by construction (see the Methods section). The two calibers are never compared in the same column. |
| Reporting | Every validity quantity is reported as a distribution over the 100 reps (mean plus 5/95th percentile), never a single run. Reps where the risk is violated are kept and reported, never dropped or re-seeded. |

## 4. Data contract

Two parsers read different columns and must not be confused:

- Probe-label parser: reads the `results` column (the dataset's ground-truth applicable methods)
  into the multi-hot `Y`. This supplies both probe supervision and F1 evaluation. It never touches
  `model_answer`. Unknown method names are a hard error (they are ground truth).
- Raw-LLM-answer parser: reads `model_answer` (the LLM's free-text answer). This is the data source
  for the raw-LLM baseline only. It classifies each row before scoring: `ok` (at least one valid
  method, mapped to multi-hot), or one of `echo` / `all_unknown` / `genuine_empty` / `unparseable`
  (each scores as the empty set, the fair penalty for a failed answer). The failure rows are kept
  and counted (per-model `echo_rate`, `unparseable_rate`, and so on), never dropped, so the raw-LLM
  row can be interpreted honestly (for example the weak and mid rungs' template-echo behaviour).

Template echo affects only the raw-LLM baseline. It cannot touch the probe (whose labels come from
`results` and whose input is hidden states), hence cannot touch NM-SCRC or any probe-based result.

## 5. Methods

All methods share the same frozen `(f, g)`, the same grids `(Λ₁, Λ₂)`, and the same targets
`(ξ, α)`. The opponents share a single calibration engine and differ only in the loss fed in and the
second-stage rule, which isolates exactly the component under test (non-monotone handling).

Common first stage (selection level): choose `λ₁` as the smallest grid value whose selection ratio
clears `ξ` plus a DKW slack, and take a lower confidence bound `ξ̂_LCB = φ̂(λ₁) − ε(n, δ₁)` on the
selection probability, with `ε_{n,δ} = √(log(1/δ) / 2n)`.

### NM-SCRC inductive (`nmscrc_i`)
The proposed method. After the first stage it calibrates `λ₂` against the conditional F1 risk using
a `t_V` cushion / union correction over the grid, which immunises the certificate against grid
refinement (a denser `Λ₂` does not break validity). Two concentration variants:
- EB (empirical Bernstein): variance-adaptive; the certificate decays roughly as `√` in coverage.
- Hoeffding: variance-free; pays a `1/φ̂` selection factor, so it abstains more at low coverage.

### NM-SCRC transductive (`nmscrc_t`)
A leave-one-out / transductive variant giving exact coverage. It also reports the instability count
`K/M`, which is about 0 on the real data (the favourable regime); the small-`M` / degenerate
transition is an appendix-only phenomenon.

### Baselines
- `naive`: calibrates with no correction; expected to violate the target under a stronger search.
- `raw` (raw-LLM): the LLM's own parsed answer; a capability floor, not a calibrated method.
- `rand`: random selection; a floor.
- `mono` (mono-counting): the opponents' shared engine fed the raw F1-loss, that is, pretending F1
  is monotone. Because F1 is U-shaped the crossing does not certify, so it is expected to FAIL once
  the grid is dense enough; its PASS at coarse grids is grid luck, not validity.

### Published opponents
- Xu-proxy (`xu_proxy`): Xu et al. (2025). The honest monotonization: feed the engine the monotone
  recall-loss `1 − |C ∩ Y| / max(|Y|, 1)`. It controls its own recall objective (PASS) but its true
  F1 risk is uncontrolled (typically `> α`) and its sets are larger, because a recall-sized set is
  blind to precision. The F1-risk gap versus NM-SCRC at the same `(ξ, α)` is the headline comparison.
- CRC-NM-marginal (`crcnm_marginal`): Aldirawi et al. (2026). No selection and no ratio: it controls
  the marginal F1 risk over all test points, with its own two-term correction `D`. It is judged on
  the marginal caliber and is never compared against the conditional-risk methods in the same column.

## 6. Certificates and scaling

- EB vs Hoeffding. On the ξ-slope study the EB certificate scales as `cert ∝ coverage^s` with
  `s ≈ −0.5 … −0.7`; Hoeffding pays the full selection factor, `s ≈ −1.0`. EB is therefore tighter
  and feasible at lower coverage.
- Feasibility floor. EB becomes feasible (stops abstaining) at a smaller calibration size `n` than
  Hoeffding.
- Union-tax. The certificate grows like `√log(m₁ · m₂)` as the grid is enlarged, the price of
  searching a `Λ₁ × Λ₂` union, immunised by the `t_V` cushion.
- Phase transition (vs `m₂`). As the `Λ₂` union widens, the inductive certificate grows and the
  method eventually abstains; the transductive LOO certificate (`c2.py`) slides toward vacuity while
  the true held-out risk stays near `α`.

## 7. Experiment map

| Experiment | Question | Output folder |
|---|---|---|
| Validity + PAC histogram | Is `R_cond ≤ α` for `≥ 1 − δ` of reps? | `results/exp1/` |
| F1-risk U-shape | The non-monotonicity every opponent must respect | `results/exp3/` |
| ξ-slope | EB vs Hoeffding certificate decay in coverage | `results/exp5/` |
| Phase transition vs `m₂` | Certificate growth / abstention as `Λ₂` widens | `results/exp6/` |
| Inductive vs transductive | Coverage exactness and `K/M` instability | `results/exp8/` |
| Head-to-head | All methods at matched `(ξ, α)`; the F1-gap of the opponents | `results/head2head/` |
| Grid refinement | `mono` fails as the grid densifies; NM-SCRC stays valid | `results/mono_gridrefine/` |
| Transductive certificate | C2 LOO certificate vacuity vs true risk | `results/c2_transductive_cert/` |
| Synthetic floor / union-tax | Feasibility floor and `√log(m₁m₂)` union-tax | `results/exp9_floor/`, `results/union_tax/` |

The per-run numerical tables for all of the above are assembled in
`results/v{1,2,3}_RESULTS_REPORT.md`.
