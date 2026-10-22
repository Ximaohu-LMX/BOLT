# BOLT: Label-Efficient LIKE Cardinality Estimation

Code, frozen inputs, configurations, and experiment evidence for reproducing BOLT. The repository builds on [CLIQUE](https://github.com/sykwon/sigmod2025like) and retains its `src/`, `configs/`, `data/`, `res/`, `setting.sh`, `download.sh`, and model entry-point organization.

Current source version: `paper-v1.3`. [GitHub repository](https://github.com/Ximaohu-LMX/BOLT). Frozen artifact bundles are hosted in the [paper-v1.0 release](https://github.com/Ximaohu-LMX/BOLT/releases/tag/paper-v1.0); their checksums are unchanged.

Start with [REPRODUCE.md](REPRODUCE.md). The [figure and table index](docs/PAPER_MAP.md) covers all 5 figures, 13 main-paper tables, and 30 supplementary tables. [PROTOCOLS.md](docs/PROTOCOLS.md) explains the label budgets, evaluation population, and timing rules.

## Quick review: no GPU training

Clone the GitHub repository, then run the CPU checks:

```bash
git clone --branch paper-v1.3 https://github.com/Ximaohu-LMX/BOLT.git
cd BOLT
conda env create -f environment.yml
conda activate bolt
bash scripts/smoke_test.sh
bash run_all.sh --dry-run
```

The checks above require no paper data. Saved-result verification and training automatically download the frozen artifact bundles from GitHub Releases using the URLs in [manifests/artifacts.json](manifests/artifacts.json). The four original raw columns use CLIQUE's Google Drive downloads. No account, access token, or custom download host is required. Existing files are checked against their SHA-256 hashes and reused; see [docs/DATA.md](docs/DATA.md) for direct links and optional local/mirror overrides.

```bash
bash scripts/verify_results.sh --dataset AUTHOR
```

This verifies the frozen AUTHOR workload and saved predictions against their SHA-256 hashes, and recomputes the metrics. To check all four datasets and regenerate the paper displays:

```bash
bash scripts/verify_results.sh
bash scripts/audit_data.sh
bash scripts/render_paper.sh --source reference
```

Outputs are in `paper_results/generated/`. Saved predictions are evidence from the paper experiments; verifying them does **not** retrain the models. Historical timing evidence is identified separately from newly measured timings.

## Train and measure again

A CUDA GPU is required by the inherited training implementations. The paper used an RTX 2080 Ti, Python 3.9.25, and PyTorch 2.0.1 with CUDA 11.8. See [resources and dependencies](docs/RESOURCES.md).

```bash
# Inspect the complete experiment plan first.
bash run_all.sh --dry-run

# One independently reproducible cell.
bash scripts/run_accuracy.sh --dataset AUTHOR --budget A0 --method BOLT

# All model accuracy, ablation, calibration, and additional-seed experiments.
bash run_all.sh --suite main

# Entire workflow, including costs, sensitivity, and the 59-query JOB experiment.
bash run_all.sh
```

The full workflow is substantial. Run it on a machine reserved for the experiment. Use paper labels directly, for example `--method "UB+Order"` or `--method "UB+LB"`; portable identifiers `UB_ORDER` and `UB_LB` also work. Every completed training cell is skipped only after checking its inputs, configuration, code, environment, and output hashes. Interrupted or changed cells get a new attempt directory. Existing attempts remain available for inspection. `--force` requests a new attempt. `--epochs 1` is an explicitly exploratory override and cannot populate paper figures.

## Stepwise scripts

| Command | Purpose |
|---|---|
| `bash setting.sh` | Create `.venv` and install dependencies; alternatively use Conda above |
| `bash download.sh` | Download CLIQUE columns and frozen BOLT workloads/summaries |
| `bash scripts/prepare_data.sh` | Verify split disjointness and paired supervision masks |
| `bash scripts/audit_data.sh` | Recompute dataset, budget, and external-target statistics |
| `bash scripts/run_accuracy.sh` | CLIQUE/BOLT budget sweep and external baselines |
| `bash scripts/run_ablations.sh` | Bound/order ablations, LB+Order, matched λ=0 control |
| `bash scripts/run_calibration.sh` | Zero-label and endpoint-calibrated controls |
| `bash scripts/run_supplement.sh` | Additional IMDB seed experiment |
| `bash scripts/run_costs.sh --phase inference` | CPU singleton inference and storage |
| `bash scripts/run_costs.sh --phase labels` | PostgreSQL exact-label acquisition |
| `bash scripts/run_costs.sh --phase construction` | Cold/warm construction repetitions |
| `bash scripts/run_costs.sh --phase sensitivity` | A1 chain-retention sweep |
| `bash scripts/run_costs.sh --phase summaries` | Independent UB/LB summary construction |
| `bash scripts/run_job.sh` | Dedicated PostgreSQL, 16 model cells, and six-arm JOB timing |
| `bash scripts/collect_results.sh --source new` | Aggregate verified fresh model runs |
| `bash scripts/render_paper.sh --source reference` | Recompute saved-result displays and render figures |

## Repository layout

```text
configs/runs/             explicit configuration of each paper training cell
src/                      CLIQUE-derived estimators and BOLT components
repro/                    portable download, run, verification, and reporting code
scripts/                  stepwise bash entry points and figure renderers
manifests/                task lists, checksums, and artifact locations
paper_results/reference/  small immutable numerical ledgers and provenance
paper_results/table_sources/   exact final-paper table TeX
paper_results/figure_sources/  original figure wrappers and static architecture figure
job_experiments/           PostgreSQL estimation hook and JOB protocol
cpp/                      inherited optional CLIQUE workload utilities
docs/                     paper index, data, protocols, resources, provenance
tests/                    model semantics and reproduction integrity checks
data/, res/, artifacts/   downloaded/generated files; excluded from Git
```

Large data, checkpoints, and predictions are separate, checksum-pinned artifact bundles; see [data access](docs/DATA.md). Both transfer and extraction verify hashes. Method identifiers and figure labels follow the paper; see [method names](docs/PROTOCOLS.md#method-names).

The original chain-generation provenance is not sufficient to promise regeneration from a seed alone. Reproduction therefore starts from the published frozen chains and masks, as disclosed in the supplement. No unpublished local research directory is required.

Attribution and licensing status: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Validation performed for this release: [VALIDATION.md](docs/VALIDATION.md).
