# Reproduction guide

Run commands from the repository root. Each bash wrapper also works when invoked from a different directory. Set `BOLT_PYTHON=/path/to/python` to select an interpreter; otherwise wrappers use `.venv/bin/python` if present, then `python3` from the active environment.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate bolt
bash setting.sh --check
```

Alternatively, `BOLT_PYTHON=python3.9 bash setting.sh` creates a local virtual environment. Python 3.9 is the reference version; the setup script accepts 3.9–3.11. CPU verification does not need a CUDA driver. Full training uses CUDA. The JOB and label-cost scripts build a private PostgreSQL 14.5 and need `gcc`, `g++`, `make`, and `perl`; no system PostgreSQL is replaced. On Ubuntu, these tools are supplied by `build-essential` and `perl`.

```bash
bash scripts/smoke_test.sh          # CPU semantic/integrity tests
bash scripts/smoke_test.sh --train  # also train a tiny synthetic BOLT model on CUDA
```

Synthetic outputs stay in `res/smoke_*` and cannot be mistaken for paper results.

## 2. Frozen data and fast numerical verification

The default commands download original CLIQUE columns from Google Drive and all required BOLT bundles from the public [GitHub release](https://github.com/Ximaohu-LMX/BOLT/releases/tag/paper-v1.0). No environment override or GitHub login is needed. The downloader verifies both archive and extracted-file SHA-256 values before reuse. CPU smoke tests and the dry-run planner work without these bundles.

For offline reuse, optionally set `BOLT_ASSET_DIR=/path/to/bolt-artifact-bundles`. To use a mirror, set `BOLT_ASSET_BASE_URL` to an HTTP(S) directory containing the same filenames. Local bundles take priority over the mirror, which takes priority over the manifest URLs. See [docs/DATA.md](docs/DATA.md) for direct links.

```bash
bash download.sh
bash scripts/prepare_data.sh
bash scripts/verify_results.sh
bash scripts/audit_data.sh
bash scripts/run_job.sh --verify
bash scripts/render_paper.sh --source reference
```

The raw columns use the original CLIQUE Google Drive IDs. BOLT chains, train/validation/test splits, masks, LPLM prefix targets, and summaries come from the published frozen BOLT bundles. The original CLIQUE workloads are not substituted for the BOLT paper workloads. If data already exist, the downloader verifies them and skips transfer. The public URLs are configured by default; optional local and mirror sources must contain the same checksum-pinned files.

For a smaller first check, append `--dataset AUTHOR` to `download.sh`, `prepare_data.sh`, `verify_results.sh`, and `audit_data.sh`. Figure rendering requires the full matrix; do not use a dataset filter there.

Generated results:

- `paper_results/generated/verification.json`: main saved-prediction verification receipt.
- `paper_results/generated/verified_metrics.csv`: recomputed main metrics.
- `paper_results/generated/reference/tables/`: recomputed saved-result tables and clearly identified historical timing ledgers.
- `paper_results/generated/reference/paper_display_tex/`: exact final-paper table source, retained as historical display evidence.
- `paper_results/generated/reference/figs/`: Figures 1–5.
- `paper_results/generated/reference/job/`: JOB aggregation from all raw timing samples.
- `paper_results/generated/data/`: statistics recomputed from frozen datasets and masks.

Table formatting in generated CSVs is convenient for auditing and need not match the paper's typography. The original table TeX remains available for every display.

## 3. Fresh training

```bash
bash scripts/run_accuracy.sh --dataset AUTHOR --budget A0 --method BOLT
bash scripts/run_ablations.sh --dataset AUTHOR
bash scripts/run_calibration.sh --dataset AUTHOR
bash scripts/run_supplement.sh
```

`run_all.sh --suite main` executes all 162 non-JOB training/static-predictor cells, including the intermediate budgets and additional IMDB seed. `run_all.sh` also executes the cost and sensitivity suites and JOB. Cells run serially to keep GPU memory demands predictable.

```bash
bash run_all.sh --suite main --status
bash run_all.sh --suite main --dry-run
bash run_all.sh --suite accuracy --dataset AUTHOR --budget A0 --method BOLT --force
```

A success receipt is `res/runs/<task>/DONE.json`. It records input/output hashes and the environment. A later invocation checks the receipt and its files before printing `SKIP verified`. Changed files cause a new attempt. Failed attempts keep their log but do not produce a success receipt. Stale results are not admitted by `collect_results.sh --source new`.

```bash
bash scripts/run_accuracy.sh --dataset AUTHOR --budget A0 --method BOLT --epochs 1
```

This last command is a short exploration, not a paper reproduction. It will not create paper figures. Rerun without `--epochs` for the formal task.

After all required cells finish:

```bash
bash scripts/render_paper.sh --source new
```

The `new/` and `reference/` result directories are separate. Shared fixed bounds and the illustrative Figure 1/architecture Figure 2 remain fixed inputs. Timing tables require the cost/JOB commands; fresh training wall time is never reused as a paper timing measurement.

## 4. Costs, sensitivity, and JOB

```bash
bash scripts/run_costs.sh --phase labels --dataset AUTHOR
bash scripts/run_costs.sh --phase construction --dataset AUTHOR
bash scripts/run_costs.sh --phase inference --dataset AUTHOR
bash scripts/run_costs.sh --phase sensitivity --dataset AUTHOR
bash scripts/run_costs.sh --phase summaries --dataset AUTHOR
```

The inference script measures the published A100 checkpoints. It performs 100 warmup singleton predictions, then a full test pass on one CPU thread. It reports median and p95 latency plus serialized artifact sizes. Construction uses separate summary caches, three cold repetitions and two warm repetitions. Full-chain A1 warm measurements are additional runs. Summary construction has the paper's explicit 180 s LB / 300 s UB limits; a timeout is reported as a timeout, never a zero-duration successful build. Label acquisition uses eight PostgreSQL workers and three measured warm repetitions after a full warmup pass.

```bash
bash scripts/run_job.sh --setup-only       # download/import the original JOB database
bash scripts/run_job.sh --source reference # new SQL timings with the saved model estimates
bash scripts/run_job.sh                    # train the 16 JOB cells, then new SQL timings
```

JOB uses a dedicated cluster in `res/postgres/`, a repository-local socket, port 55432, and no TCP listener. Override `BOLT_PG_PORT` before initial cluster creation if needed. `BOLT_JOB_ARCHIVE=/path/to/imdb.tgz` reuses an existing archive after checking its SHA-256. The filename `postgresql-13.1.zip` is inherited from CLIQUE; the bundled source and built server are **PostgreSQL 14.5**.

The preflight checks database column identities, true counts, predicate-key coverage, negative/NULL/array semantics, baseline plan replay, and query-result equality. Timing then runs one warmup and ten interleaved repetitions for each of 59 queries and six arms. The hook replaces modeled single-table predicates; joins use PostgreSQL's native estimates. Stop only this dedicated server with:

```bash
python -m repro.postgres stop
```

Use a quiet machine for timing. The inherited JOB timing preflight refuses to start while a GPU experiment is active. A failed JOB timing attempt restarts as a new whole attempt; successful model training and database setup are reused.

## Method labels

Paper labels can be used directly in filters:

```bash
bash scripts/run_ablations.sh --dataset AUTHOR --budget A0 --method "UB+Order" "UB+LB"
bash scripts/run_calibration.sh --dataset AUTHOR --method "Order only" "Order + endpoints"
```

Configurations and directories use `UB_ORDER`, `UB_LB`, `ZERO_MID_ORDER`, and `ANCHORED_MID_ORDER`. The BOLT implementation is `main_BOLT.py` / `src/BOLT.py`. See [all method names](docs/PROTOCOLS.md#method-names). Existing versioned archives are reused through verified extraction into these current names.
