# Resources and dependencies

The paper's main GPU reference was an NVIDIA RTX 2080 Ti, Python 3.9.25, PyTorch 2.0.1+cu118, and CUDA 11.8. Explicit direct Python dependencies are pinned in `requirements.txt`; `environment.yml` selects the reference Python version. A compatible NVIDIA driver is needed for training. CPU metric verification, plotting, and published-checkpoint inference do not require CUDA execution.

Use Linux for the full suite: the runners use `fcntl` locks and CPU affinity. The private PostgreSQL build needs a C/C++ toolchain, GNU Make, and Perl. `bash setting.sh --check` reports the detected runtime and build tools. Optional inherited C++ workload tools additionally use ICU, RE2, Boost, and indicators; they are not needed to reproduce the frozen paper workloads, and `setting.sh` does not install system packages.

Allow at least 20 GiB of free disk for downloads and frozen artifacts, and substantially more (50 GiB or more is a practical starting allocation) for full training/checkpoints and JOB. This is a planning allowance, not a measured peak. The original JOB archive is another 1.18 GiB compressed, with a much larger imported database. RAM needs grow with the complete string column and full N-gram tables; IMDB/WIKI cold summary construction is the largest stage.

The full paper matrix has 162 main/static-control cells, 16 JOB model cells, additional construction repetitions, and sensitivity experiments. Expect a long run rather than a brief CI job. For scale, the published full-chain BOLT A1 warm construction medians were roughly 19 min (AUTHOR), 18 min (DBLP), 179 min (IMDB), and 224 min (WIKI) per run. Those are historical hardware-specific measurements, not promises for another machine. Do not multiply a quick one-epoch smoke time into a claimed full-training cost.

`--dataset`, `--budget`, and `--method` filters let reviewers start small. The normal runner is serial. Formal timing should run without competing jobs; changing concurrency changes the measurement protocol. `BOLT_BUILD_JOBS` controls compilation parallelism only, and defaults to four.
