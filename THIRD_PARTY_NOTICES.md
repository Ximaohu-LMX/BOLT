# Attribution and licensing status

BOLT modifies and extends **CLIQUE**, whose official artifact is available at <https://github.com/sykwon/sigmod2025like>. This repository retains its estimator infrastructure, external baseline implementations, C++ utilities, dataset download sources, and PostgreSQL integration foundation. BOLT adds lower-bound/order mechanisms, controlled experiments, and the portable reproduction layer. Historical internal names are documented rather than attributed to a different upstream project.

The PostgreSQL source archive retains the upstream PostgreSQL copyright/license files. The keyed estimation hook and experiment protocol are included in `job_experiments/`. Python packages and the vendored UTF-8 utilities retain their respective upstream terms and notices. Dataset ownership and terms are not transferred by this artifact.

No blanket repository-wide license is asserted here: the referenced CLIQUE repository did not expose a root license in the version inspected for this release. This notice records provenance and does not relicense inherited material. Consult the corresponding upstream terms before redistribution beyond their permitted scope.

Only the new `repro/` and bash interfaces are the recommended public reproduction workflow. Some inherited standalone research utilities remain for provenance; their historical CLI interfaces are not used by `run_all.sh`.
