# Data and artifacts

The source of the four raw columns is [CLIQUE's download script](https://github.com/sykwon/sigmod2025like/blob/main/download.sh). `download.sh` uses its original Google Drive IDs through `gdown`, extracts the raw column only, and checks the exact SHA-256 in `manifests/artifacts.json`.

The final paper's frozen workload inputs are separate artifact bundles. This review snapshot has no configured anonymous remote bundle host. The bundle URLs in both current and historical manifests are therefore `null`; all archive hashes, member hashes, sizes, and copy mappings are retained.

To use locally supplied bundles, set `BOLT_ASSET_DIR` to their directory. To use an anonymous download service, set `BOLT_ASSET_BASE_URL` to its HTTP(S) directory URL; the downloader appends each manifest bundle filename. The directory URL must not contain query parameters or a fragment. Local bundles take priority over the remote override, and verified cached or installed files are reused. A missing bundle with no configured source stops with a data-access message.

Reviewers can run the CPU smoke tests and dry-run planner without artifact bundles. Saved-prediction verification and paper training require the bundles below. Archives contain only listed regular files; path traversal and links are rejected during extraction. Both compressed archives and extracted files are checked.

| Bundle | Contents |
|---|---|
| `workloads-<DATASET>*.tar.gz` | Frozen full/supervised/masked views, seven budgets, chain indices, label masks, validation/test sets, external prefix targets |
| `summaries.tar.gz` | Pruned and full N-gram caches, associated metadata, and lower-bound witness libraries |
| `predictions.tar.gz` | Aligned predictions for the 162 non-JOB cells, shared bounds, and 12 sensitivity cells |
| `checkpoints.tar.gz` | Available saved neural checkpoints for the main model/control experiments |
| `timing.tar.gz` | CPU raw latency samples and standalone summary-build evidence |
| `job-inputs.tar.gz` | Four exported JOB columns, workloads, summaries, 16 saved model cells, database identity, and patched PostgreSQL source |
| `job-evidence.tar.gz` | Raw JOB timings, aggregated results, and correctness evidence |

The task manifest associates configurations, predictions, checkpoints, and expected Q-error summaries. `manifests/source_provenance.json` retains original relative paths and hashes. Those old paths are historical identifiers, not runtime dependencies. Release downloads are about 1.13 GiB compressed in total; use the manifest's byte fields for the exact release sizes. Expanded data and duplicate per-method views need more disk space.

The separate original JOB database comes from [CWI](https://event.cwi.nl/da/job/imdb.tgz). Its expected size is 1,263,193,115 bytes and its SHA-256 is pinned in `artifacts/job/source_import.json`. The importer loads 21 new tables transactionally and never replaces tables in an existing user database.

`data/`, `res/`, `.cache/`, and `artifacts/` are excluded from version control. The downloaded raw columns remain subject to their original source terms. See [third-party notices](../THIRD_PARTY_NOTICES.md).

The WIKI bundle stores 27 distinct file contents once and restores 78 byte-identical views through the manifest's explicit `copies` map. Restored views are independent regular files, not symbolic or hard links. Every restored file retains its original paper-input checksum. This reduces the WIKI archive from about 582 MiB to 249 MiB without changing its 105 extracted files.
