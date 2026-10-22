# Data and artifacts

The source of the four raw columns is [CLIQUE's download script](https://github.com/sykwon/sigmod2025like/blob/main/download.sh). `download.sh` uses its original Google Drive IDs through `gdown`, extracts the raw column only, and checks the exact SHA-256 in `manifests/artifacts.json`.

The four original source downloads are:

| Dataset | Original CLIQUE download |
|---|---|
| AUTHOR | [AUTHOR.tar.gz](https://drive.google.com/uc?id=1v5SZMep0puR3LEml7hwmMhBymBP8xlBw) |
| DBLP | [DBLP.tar.gz](https://drive.google.com/uc?id=11S8D_8CLSXAsFycaoU1x0ub6qIapo7CE) |
| IMDB | [IMDB.tar.gz](https://drive.google.com/uc?id=1r_A1RL-_qrJUReFqVN0ZjQ0JzkDwpn08) |
| WIKI | [WIKI.tar.gz](https://drive.google.com/uc?id=1j-JV04acH3vbCdiI8pnLqmOzx8g_0gtO) |

CLIQUE supplies the raw columns; its original training workloads do not contain BOLT's frozen chains, supervision masks, summaries, predictions, or checkpoints. These BOLT artifacts are publicly available in the [paper-v1.0 GitHub release](https://github.com/Ximaohu-LMX/BOLT/releases/tag/paper-v1.0). Both current and historical manifests point to these existing release assets. Archive hashes, member hashes, sizes, and copy mappings are unchanged. The downloader converts the historical archive paths to the current paper method names after verifying them.

| Bundle | Size (MiB) |
|---|---:|
| [`workloads-AUTHOR.tar.gz`](https://github.com/Ximaohu-LMX/BOLT/releases/download/paper-v1.0/workloads-AUTHOR.tar.gz) | 20.9 |
| [`summaries.tar.gz`](https://github.com/Ximaohu-LMX/BOLT/releases/download/paper-v1.0/summaries.tar.gz) | 88.2 |
| [`workloads-DBLP.tar.gz`](https://github.com/Ximaohu-LMX/BOLT/releases/download/paper-v1.0/workloads-DBLP.tar.gz) | 20.4 |
| [`workloads-IMDB.tar.gz`](https://github.com/Ximaohu-LMX/BOLT/releases/download/paper-v1.0/workloads-IMDB.tar.gz) | 190.5 |
| [`workloads-WIKI.compact.tar.gz`](https://github.com/Ximaohu-LMX/BOLT/releases/download/paper-v1.0/workloads-WIKI.compact.tar.gz) | 248.6 |
| [`predictions.tar.gz`](https://github.com/Ximaohu-LMX/BOLT/releases/download/paper-v1.0/predictions.tar.gz) | 130.1 |
| [`checkpoints.tar.gz`](https://github.com/Ximaohu-LMX/BOLT/releases/download/paper-v1.0/checkpoints.tar.gz) | 296.8 |
| [`timing.tar.gz`](https://github.com/Ximaohu-LMX/BOLT/releases/download/paper-v1.0/timing.tar.gz) | 1.9 |
| [`job-evidence.tar.gz`](https://github.com/Ximaohu-LMX/BOLT/releases/download/paper-v1.0/job-evidence.tar.gz) | 0.3 |
| [`job-inputs.tar.gz`](https://github.com/Ximaohu-LMX/BOLT/releases/download/paper-v1.0/job-inputs.tar.gz) | 160.1 |

No login or manual source configuration is needed:

```bash
# Raw columns, BOLT workloads, and summaries.
bash download.sh

# Also fetch saved predictions, checkpoints, timings, and JOB bundles.
bash download.sh --artifacts all

# Smaller first download.
bash download.sh --dataset AUTHOR
```

Each command verifies existing files and skips completed downloads. Repeating a command is safe.

Optional overrides: set `BOLT_ASSET_DIR` to a directory containing local copies of the bundles, or set `BOLT_ASSET_BASE_URL` to an HTTP(S) mirror directory. The downloader appends each manifest bundle filename; the directory URL must not contain query parameters or a fragment. Local bundles take priority over the mirror, which takes priority over the public manifest URL. Cached or installed files are reused only after verification.

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
