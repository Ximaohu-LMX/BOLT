# Release validation

## Paper-name update

The current code uses BOLT, UB+Order, UB+LB, and the paper's Order control names throughout its entry points, configurations, workloads, reports, and JOB arms. Validation for this naming update:

- 51 tests passed, including migration, checksum failure, query preservation, paper-label filters, and checkpoint-path regression cases.
- All 935 installed artifact files passed their current manifest hashes; the 10 downloaded archives retain their original release checksums. Only nine metadata files required content conversion.
- All 312,186 numerical cells in the tracked reference CSVs were compared with `paper-v1.0` and are unchanged.
- All 166 generated CLI training commands passed the actual parsers; the other 12 LB+Order cells use their dedicated Python runner.
- The native `main_BOLT.py` completed two synthetic CUDA epochs, checkpoint reload, and held-out prediction. Published AUTHOR CLIQUE/BOLT checkpoints reproduced sampled non-exact predictions through the current classes.
- The full saved-result collection, 16 JOB model checks, and 3,540 measured JOB samples passed; Figures 1–5 were regenerated with current names.

The original first-release validation record follows.

## Initial release

The following checks were executed on the prepared standalone BOLT tree before the first release:

- A fresh temporary Git clone passed the CPU tests, downloaded/extracted the AUTHOR inputs, and verified the saved AUTHOR experiment matrix using only the published-format release files.
- The compact WIKI bundle restored all 105 original file hashes in an empty directory; damaged copies and unsafe copy mappings are covered by the tests.
- 42 CPU tests passed, covering bounds, label masking, chain sampling, LB+Order, LPLM tokenization, completion-receipt invalidation, archive paths, and metric conventions.
- A tiny synthetic BOLT model completed two CUDA training epochs, saved/reloaded its checkpoint, and produced all held-out predictions.
- Both CLIQUE and BOLT completed an exploratory one-epoch training run on the small JOB link-type column. Repeating the identical commands skipped both verified completed cells.
- All 162 non-JOB saved prediction cells passed query/truth/order alignment and Q-error checks. The four datasets passed split-disjointness and nested, paired-label-mask audits.
- The 12 sensitivity cells were recomputed from saved per-query predictions. All 16 JOB model files were checked; 3,540 measured JOB samples were reaggregated to the recorded summaries.
- All four CLIQUE download sources were downloaded independently and their raw-column hashes matched the paper inputs. Dataset statistics and LPLM target totals were recomputed from the released files. Figure 1's eight exact counts, bounds, and set-containment relations were checked against the original AUTHOR column.
- Figures 1–5 were generated locally from the standalone repository. Upper-bound quantiles/violations were recalculated; the unavailable per-query exact-hit identities are explicitly distinguished from the original aggregate exact-hit percentage.
- The bundled PostgreSQL source compiled to version 14.5 in an independent cluster. Actual hook tests passed for LIKE, NOT LIKE, equality, arrays, NULL handling, and five invalid-map cases.
- AUTHOR lower- and upper-bound summaries were freshly constructed and compared equal to the frozen summary objects. Both builds completed within their stated time limits.
- The CPU latency/storage path loaded the published AUTHOR CLIQUE/BOLT checkpoints and completed the full singleton test pass.
- Python compilation, Bash syntax, and generated training-command parsing checks passed.

These checks establish that the transferred artifact is internally consistent and that representative execution paths work. The entire 162-cell training matrix, full multi-dataset cost measurements, the 21-table JOB reimport, and a fresh complete 59-query SQL timing campaign were **not** rerun as part of repository packaging. Their original evidence is published separately, and executable commands are provided for a new full run. New model results may vary across CUDA/PyTorch/hardware environments; new timings are inherently machine-dependent.

The GitHub Actions configuration is supplied as `docs/ci/checks.yml`. The publishing credential did not have GitHub's `workflow` scope, so automatic CI is not enabled in this release. Once a credential with that permission is available, placing the template at `.github/workflows/checks.yml` enables the lightweight CPU tests, syntax checks, and dry-run planner. It does not run full training or database timing.
