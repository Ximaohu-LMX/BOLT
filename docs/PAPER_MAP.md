# Figure and table reproduction index

Every empirical display has a command and an input trail below. A separate training job for each display is unnecessary: figures and tables that share experimental cells are generated from the same verified predictions. Tables 1–2 and Figure 2 are definitions/design illustrations. Figure 1 is an explicitly selected empirical illustration, not a representative aggregate.

Run `bash scripts/render_paper.sh --source reference` to recompute saved-prediction metrics and render all figures. Run `bash scripts/audit_data.sh` for raw-column/workload statistics. The exact final-paper table TeX is preserved in `paper_results/table_sources/` and copied to `paper_results/generated/reference/paper_display_tex/`. Fresh model outputs go to `paper_results/generated/new/`; reference timing tables remain historical evidence.

## Main paper

| Item | Content | Fresh experiment / derivation | Output or original display |
|---|---|---|---|
| Figure 1 | Eight-pattern AUTHOR illustrative case | `bash scripts/audit_figure1.sh`; `bash scripts/render_paper.sh` | `reference/figure1_filled_case/nodes.csv`; generated `figs/F1_motivation.pdf` |
| Figure 2 | BOLT architecture | Static source; copied by `render_paper.sh` | `figure_sources/F2_framework.{tex,pdf}` |
| Figure 3 | Seven-budget CLIQUE/BOLT accuracy | `bash scripts/run_accuracy.sh --method CLIQUE BOLT` | `figs/F3_accuracy.pdf`; `all_metrics.csv` |
| Figure 4 | Bound/order ablations | `bash scripts/run_ablations.sh` and `bash scripts/run_accuracy.sh --method CLIQUE BOLT` | `figs/F4_ablations.pdf` |
| Figure 5 | Matched λ=0 control | `bash scripts/run_ablations.sh --method BOLT_L0`; `bash scripts/run_accuracy.sh --budget A0 --method BOLT` | `figs/F5_loss.pdf` |
| Table 1 | Notation | No experiment | `T0_notation.tex` |
| Table 2 | LIKE pattern forms | No experiment | `T0_pattern_types.tex` |
| Table 3 | Datasets/workloads | `bash scripts/audit_data.sh` | `generated/data/Table03_columns.csv`, `S02_distributions.csv`; `T1_workloads.tex` |
| Table 4 | Realized label budgets | `bash scripts/prepare_data.sh`; `bash scripts/audit_data.sh` | `generated/data/Table04_S01_budgets.csv`; `T2_budget.tex` |
| Table 5 | All estimators at A100/A0 | `bash scripts/run_accuracy.sh --budget A100 A0` | `tables/Table05_accuracy.csv`; `T3_external.tex` |
| Table 6 | A0 integer order/ties | A0 accuracy + ablation cells, then `collect_results.sh` | `tables/Table06_order.csv`; `T_order_consistency_view.tex` |
| Table 7 | Calibration controls | `bash scripts/run_calibration.sh` | `tables/Table07_calibration.csv`; `T_calibration_controls.tex` |
| Table 8 | LB-conditioned mean Q-error | A0 accuracy + bound/order ablations, then collect | `tables/Table08_lower_bound.csv`; `T_lb_conditional.tex` |
| Table 9 | Upper-bound diagnostics | `bash scripts/collect_results.sh --source reference` | `tables/Table09_S21_upper_bounds.csv`; `T_upper_bound_diagnostics.tex` |
| Table 10 | Exact-label acquisition | `bash scripts/run_costs.sh --phase labels` | fresh `tables/Table10_labels.csv`; `T_label_acquisition.tex` |
| Table 11 | Cold/warm construction with retained chains | `bash scripts/run_costs.sh --phase construction` | fresh `tables/Table11_construction.csv`, `construction_repetitions.csv`; `T4_construction.tex` |
| Table 12 | CPU singleton latency and storage | `bash scripts/run_costs.sh --phase inference` | fresh `tables/Table12_inference.csv`; `T5_inference.tex` |
| Table 13 | 59-query JOB execution | `bash scripts/run_job.sh`; fast saved-evidence check: `bash scripts/run_job.sh --verify` | fresh `tables/Table13_JOB.csv`, raw `res/job/*/attempt_*/timings.jsonl`; `T_JOB_E2E.tex` |

Paths such as `tables/` and `figs/` in the output column are relative to `paper_results/generated/<reference|new>/`, unless another location is stated. Static source filenames are relative to `paper_results/` or its `table_sources/` directory.

## Supplement

| Item | Content | Command / contributing cells | Output / source |
|---|---|---|---|
| S1 | Exact supervision counts | `bash scripts/audit_data.sh` | `generated/data/Table04_S01_budgets.csv`; `A_budget_counts.tex` |
| S2 | Workload distributions | `bash scripts/audit_data.sh` | `generated/data/S02_distributions.csv`; `A_distributions.tex` |
| S3 | Cardinality of revealed labels | `bash scripts/audit_data.sh` | `generated/data/S03_label_cardinality.csv`; `A_sampling_shift.tex` |
| S4 | Configuration | `configs/runs/*.json`; frozen parameter ledger | `reference/configuration.csv`; `A_configuration.tex` |
| S5 | External method supervision resources | `bash scripts/audit_data.sh` | `generated/data/S05_external_targets.csv`; `A_external_ledger.tex` |
| S6 | AUTHOR core accuracy | Accuracy + ablation cells, `--dataset AUTHOR` | `tables/S06_accuracy.csv`; `A_accuracy_AUTHOR.tex` |
| S7 | DBLP core accuracy | Accuracy + ablation cells, `--dataset DBLP` | `tables/S07_accuracy.csv`; `A_accuracy_DBLP.tex` |
| S8 | IMDB core accuracy | Accuracy + ablation cells, `--dataset IMDB` | `tables/S08_accuracy.csv`; `A_accuracy_IMDB.tex` |
| S9 | WIKI core accuracy | Accuracy + ablation cells, `--dataset WIKI` | `tables/S09_accuracy.csv`; `A_accuracy_WIKI.tex` |
| S10 | AUTHOR intermediate budgets | `run_accuracy.sh --dataset AUTHOR --budget A25 A50 A75` | `tables/S10_intermediate.csv`; `A_supplement_AUTHOR.tex` |
| S11 | DBLP intermediate budgets | `run_accuracy.sh --dataset DBLP --budget A25 A50 A75` | `tables/S11_intermediate.csv`; `A_supplement_DBLP.tex` |
| S12 | IMDB intermediate budgets | `run_accuracy.sh --dataset IMDB --budget A25 A50 A75` | `tables/S12_intermediate.csv`; `A_supplement_IMDB.tex` |
| S13 | WIKI intermediate budgets | `run_accuracy.sh --dataset WIKI --budget A25 A50 A75` | `tables/S13_intermediate.csv`; `A_supplement_WIKI.tex` |
| S14 | IMDB seeds 0/1 | Intermediate-budget IMDB accuracy + `bash scripts/run_supplement.sh` | `tables/S14_seeds.csv`; `A_IMDB_seed_comparison.tex` |
| S15 | Degradation relative to A100 | `bash scripts/run_accuracy.sh --method CLIQUE BOLT`, then collect | `tables/S15_degradation.csv`; `A_degradation.tex` |
| S16 | AUTHOR continuous ordering | Same core predictions; `collect_results.sh` | `tables/S16_continuous_order.csv`; `C_legacy_order_AUTHOR.tex` |
| S17 | DBLP continuous ordering | Same core predictions; `collect_results.sh` | `tables/S17_continuous_order.csv`; `C_legacy_order_DBLP.tex` |
| S18 | IMDB continuous ordering | Same core predictions; `collect_results.sh` | `tables/S18_continuous_order.csv`; `C_legacy_order_IMDB.tex` |
| S19 | WIKI continuous ordering | Same core predictions; `collect_results.sh` | `tables/S19_continuous_order.csv`; `C_legacy_order_WIKI.tex` |
| S20 | LB-conditioned mean/tail error | Same A0 predictions as Table 8 | `tables/S20_conditional.csv`; `C_lb_conditional.tex` |
| S21 | UB/count tail quantiles and violations | `bash scripts/collect_results.sh --source reference` | `tables/Table09_S21_upper_bounds.csv`; `C_bounds.tex` |
| S22 | Calibration tail/tie statistics | `bash scripts/run_calibration.sh` | `tables/S22_calibration.csv`; `C_zero.tex` |
| S23 | A1 retained-chain sensitivity | `bash scripts/run_costs.sh --phase sensitivity` | `tables/S23_sensitivity.csv`; `D_ratio.tex`; saved predictions in `manifests/sensitivity.json` |
| S24 | Label-cost repetitions | `bash scripts/run_costs.sh --phase labels` | fresh `tables/S24_label_repetitions.csv`; `E_labels.tex` |
| S25 | AUTHOR construction repetitions | `run_costs.sh --phase construction --dataset AUTHOR` | fresh `construction_repetitions.csv`; `E_repetitions_AUTHOR.tex` |
| S26 | DBLP construction repetitions | `run_costs.sh --phase construction --dataset DBLP` | fresh `construction_repetitions.csv`; `E_repetitions_DBLP.tex` |
| S27 | IMDB construction repetitions | `run_costs.sh --phase construction --dataset IMDB` | fresh `construction_repetitions.csv`; `E_repetitions_IMDB.tex` |
| S28 | WIKI construction repetitions | `run_costs.sh --phase construction --dataset WIKI` | fresh `construction_repetitions.csv`; `E_repetitions_WIKI.tex` |
| S29 | A1 full-chain warm construction | Included in `run_costs.sh --phase construction` | fresh repetitions with `kind=fullchain`; `E_A1_fullchain_warm.tex` |
| S30 | Standalone summary builds | `bash scripts/run_costs.sh --phase summaries` | `tables/S30_summaries.csv`; historical `S30_summary_components.csv`; `E_summary_components.tex` |

All abbreviated script names in the supplement table live in `scripts/` and should be invoked with `bash scripts/<name>`. `--source reference` recomputes numerical accuracy/order diagnostics from saved per-query predictions and reaggregates saved JOB samples. It does not remeasure hardware-specific historical construction/label/CPU costs. Those raw ledgers and their original displayed TeX are retained, while the corresponding `run_costs.sh` phase produces new measurements.

Figure 1 keeps the original post-hoc selected eight-node case and saved model predictions. Its audit independently verifies exact counts, bounds, and set containment on AUTHOR; it does not claim to recreate the original case-search process. Figure 2 has no random experiment to rerun.

The UB/count quantiles and violation counts are recalculated from per-query bounds. The exact-summary-hit percentage comes from the original aggregate summary audit: per-query hit identities were not retained in the historical query export, and empty markers are not interpreted as non-hits.
