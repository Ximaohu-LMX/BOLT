# Protocols and naming

## Method names

| Public name / CLI filter | Internal implementation | Interpretation |
|---|---|---|
| BOLT | `CLIQUE_bound_mono`, old name `BIMO` | Lower/upper bounded output and chain order loss |
| CLIQUE / UB | `CLIQUE` | Original upper-bound estimator |
| UB_LB | `CLIQUE_bound`, old `BOUND` | Bounds without order loss |
| UB_ORDER | `CLIQUE_mono`, old `MONO` | Upper bound and order loss |
| LB_ORDER | `lb_order` | Same lower-bound/chain rules, database size as upper endpoint, no UB summary or UB feature |
| NO_BOUND | `no_bound` | Neural estimator without bound/order mechanisms |
| BOLT_L0 | strict λ=0 path in `CLIQUE_bound_mono` | Same chain-aware path as BOLT, zero order-loss weight |
| ZERO_LB / ZERO_MID | `main_zero_label` | Static lower bound / shifted midpoint |
| ZERO_MID_MONO | `main_zero_label` | Order-only optimization without exact training labels |
| ANCHORED_MID_MONO | `main_zero_label` | Order-only initialization plus endpoint supervision |
| DREAM / LPLM / LBS | original model-specific entry points | External baselines |

Legacy names remain in saved filenames and manifests to preserve their provenance. Bash entry points use the public names above. `main_BOLT.py` is an alias for the BOLT implementation, but the task runner is preferred because it selects every paper parameter explicitly.

## Supervision and splits

A0 reveals the union of training-chain endpoint identities. It is **not zero labels**: about 21% of the full unique training pool is labeled. A1/A10 retain 1%/10% of the interior identities in addition to all endpoints. A25/A50/A75 correspond to the paper's interior-retention settings; the realized fractions of the full pool are approximately 40%/60%/80%. Their historical identifiers include L40/L60/L80. `budget_ledger.csv` gives exact denominators and the fixed validation-label counts.

The seven masks are nested. CLIQUE and BOLT receive identical revealed labels at a given setting. BOLT retains unlabeled nodes with target `-1` for chain edges; supervised losses must never consume those targets. The full training pool, validation, and test identities are disjoint. Frozen historical chain generation is recorded as `UNVERIFIED_LEGACY`; the released chains/masks are the reproducible starting point, not a promise that a new seed-only run recreates them.

The final WIKI test population contains 70,043 unique queries and 62,476 adjacent chain occurrences. A malformed backslash predicate is excluded uniformly. Held-out pair and whole-chain denominators differ, and singleton chains do not enter whole-chain order scores.

## Configuration and checkpoint selection

The main experiments use model seed 0, λ=5, τ=0.1, and γ=1 unless the task explicitly names a control or sensitivity/construction configuration. The main LB library uses 500 strings, 200 substrings, lengths 5–15, minimum frequency 0.0001. Complete parameters are in `configs/runs/`.

The JOB configuration is separate: λ=1, column-specific dimensions, and the original 5–10 / 500-substring / 0.0005 library settings. The JOB movie-company note batch size is 63. Do not substitute the four main dataset configurations for these tasks.

Standard model checkpoints use validation Q-error. The inherited main loops may report test diagnostics during epochs; they do not select the checkpoint by test error. The published JOB training protocol removes epoch test access while preserving the historical DataLoader RNG transition. Both zero-label order-only and endpoint-anchored midpoint controls use the declared training-order checkpoint rule, without exact validation targets; endpoints affect the anchored control's gradients. These are separate calibration diagnostics, not a claim that A0 has no labels.

## Metrics

Q-error uses continuous served predictions, with both truth and prediction floored at one for the ratio. Quantiles use NumPy's linear interpolation. Every prediction file must match the frozen test query/truth/order exactly.

`R_int` is the fraction of adjacent tight→loose edges satisfying `round(pred_tight) <= round(pred_loose)`, using ties-to-even rounding. `R_tie` is the equality fraction after that rounding. These post-hoc statistics do not change the served predictions. Historical continuous diagnostics use `log1p(pred_tight)-log1p(pred_loose) <= 1e-6` or `<= log(1.1)+1e-6`, with whole-chain success requiring every edge to pass.

LB-stratified tables group queries by the shared frozen raw lower bound (`LB=0` versus `LB>0`). These are descriptive applicability analyses. Original bound-export limitations and source hashes remain in `bound_join_audit.csv`; no new randomized causal claim is inferred from the stratification.

## Timing

Training task receipts contain elapsed time for diagnostics only. Formal construction time measures process wall time and subtracts instrumented post-training evaluation/persistence phases. Cold and warm refer to **summary caches**, not an operating-system cache flush. The controlled construction configurations use γ=0.25 for BOLT A1, and γ=0.5 for DBLP A0 / 0.25 for the other A0 datasets. They have their own accuracy and cannot borrow the γ=1 accuracy cells.

The sensitivity sweep is AUTHOR/DBLP/WIKI A1 at γ∈{0,0.25,0.5,1}. γ=0 is ordinary UB+LB with the same labels and no monotonic edges; it is distinct from the matched chain-preserving BOLT_L0 control. There is no published IMDB ratio-sweep claim.

JOB reports planning + execution time from PostgreSQL, excluding offline neural inference. One warmup is followed by ten measured trials per query/arm. Tukey 1.5-IQR filtering is applied within each query/arm to total time; the summary sums the retained query means. All raw samples and failures remain available. Hardware and system load affect all new timing measurements.
