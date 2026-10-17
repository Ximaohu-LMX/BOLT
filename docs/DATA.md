# Data and artifacts

The source of the four raw columns is [CLIQUE's download script](https://github.com/sykwon/sigmod2025like/blob/main/download.sh). `download.sh` uses its original Google Drive IDs through `gdown`, extracts the raw column only, and checks the exact SHA-256 in `manifests/artifacts.json`.

The release supplies the final paper's frozen workload inputs separately. A new clone needs no file from the author's original research workspace. Release archives contain only listed regular files; path traversal and links are rejected during extraction. Both compressed archives and extracted files are checked.

| Bundle | Contents |
|---|---|
| `workloads-<DATASET>.tar.gz` | Frozen full/supervised/masked views, seven budgets, chain indices, label masks, validation/test sets, external prefix targets |
| `summaries.tar.gz` | Pruned and full N-gram caches, associated metadata, and lower-bound witness libraries |
| `predictions.tar.gz` | Aligned predictions for the 162 non-JOB cells, shared bounds, and 12 sensitivity cells |
| `checkpoints.tar.gz` | Available saved neural checkpoints for the main model/control experiments |
| `timing.tar.gz` | CPU raw latency samples and standalone summary-build evidence |
| `job-inputs.tar.gz` | Four exported JOB columns, workloads, summaries, 16 saved model cells, database identity, and patched PostgreSQL source |
| `job-evidence.tar.gz` | Raw JOB timings, aggregated results, and correctness evidence |

The task manifest associates configurations, predictions, checkpoints, and expected Q-error summaries. `manifests/source_provenance.json` retains original relative paths and hashes. Those old paths are historical identifiers, not runtime dependencies. Main downloads are about 1.6 GiB compressed in total; use the manifest's byte fields for the exact release sizes. Expanded data and duplicate per-method views need more disk space.

The separate original JOB database comes from [CWI](https://event.cwi.nl/da/job/imdb.tgz). Its expected size is 1,263,193,115 bytes and its SHA-256 is pinned in `artifacts/job/source_import.json`. The importer loads 21 new tables transactionally and never replaces tables in an existing user database.

`data/`, `res/`, `.cache/`, and `artifacts/` are excluded from version control. The downloaded raw columns remain subject to their original source terms. See [third-party notices](../THIRD_PARTY_NOTICES.md).
