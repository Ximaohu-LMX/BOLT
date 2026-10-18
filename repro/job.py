"""Reproduce the 59-query JOB experiment on an isolated PostgreSQL 14.5."""
import argparse
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from .common import ROOT, tasks, read_json, atomic_json, digest, identity, rows, verify_files
from .runner import lock, run, completed


def prepare_inputs(folder, source):
    from job_experiments.prediction import positive_predictions
    from job_experiments.protocol import workload
    metadata = read_json(ROOT / "artifacts/job/column_metadata.json")
    archived = read_json(ROOT / "artifacts/job/positive_predictions.json")
    predictions = archived if source == "reference" else {"TrueCard": archived["TrueCard"]}
    selected = [t for t in tasks() if t["suite"] == "job"]
    w = workload(ROOT / "likeBench/job_LPLM.sql")
    if source == "new":
        for t in selected:
            from .runner import status
            if status(t) != "COMPLETE":
                raise ValueError("Missing completed JOB training: " + t["id"])
            receipt = read_json(ROOT / "res/runs" / t["id"] / "DONE.json")
            values = {r["query"]: float(r["est"]) for r in rows(ROOT / receipt["result"]["prediction"])}
            arm = t["method"] + "_" + t["budget"]
            predictions.setdefault(arm, {})[t["dataset"]] = positive_predictions(w["test_patterns"][t["dataset"]], values, metadata[t["dataset"]])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "plan").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "likeBench/job_LPLM.sql", folder / "plan/job.sql")
    atomic_json(folder / "columns.json", metadata)
    atomic_json(folder / "positive_predictions.json", predictions)
    atomic_json(folder / "database_import/IMPORTED.json", read_json(ROOT / "artifacts/job/source_import.json"))
    atomic_json(folder / "SOURCE.json", dict(source=source, training_tasks=[t["id"] for t in selected]))


def run_all(force=False, source="new", setup_only=False):
    from .datasets import ensure, download_file
    from .postgres import setup
    from job_experiments import import_job, e2e
    ensure([], ["job-inputs"], raw=False)
    with lock(ROOT / "res/.training.lock"):
        if source == "new" and not setup_only:
            for t in tasks():
                if t["suite"] == "job":
                    run(t, force)
    base, service = setup()
    expected = read_json(ROOT / "artifacts/job/source_import.json")
    import_path = base / "database_import/IMPORTED.json"
    if not import_path.exists():
        archive = Path(os.getenv("BOLT_JOB_ARCHIVE", str(ROOT / ".cache/imdb.tgz")))
        download_file("https://event.cwi.nl/da/job/imdb.tgz", archive, expected["archive_sha256"])
        # A failed import transaction leaves only its diagnostic directory.
        target = base / "database_import"
        if target.exists():
            target.rename(base / ("failed_import_" + str(time.time_ns())))
        import_job.run(SimpleNamespace(service=service, archive=str(archive), output=str(target), expected_bytes=expected["archive_bytes"]))
    actual = read_json(import_path)
    if actual["archive_sha256"] != expected["archive_sha256"] or {k:v["rows"] for k,v in actual["tables"].items()} != {k:v["rows"] for k,v in expected["tables"].items()}:
        raise ValueError("JOB imported database differs from the frozen paper input")
    if setup_only:
        print("JOB database ready", flush=True)
        return
    root = ROOT / "res/job" / source
    prepare_inputs(root / "inputs", source)
    sources = [*sorted((ROOT / "job_experiments").glob("*.py")), ROOT / "job_experiments/pg_job_hook.h",
               ROOT / "repro/job.py", ROOT / "repro/postgres.py", *sorted((root / "inputs").rglob("*.json")), root / "inputs/plan/job.sql", base / "pg/bin/postgres"]
    spec = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    key = identity(spec)
    receipt = root / "DONE.json"
    if not force and completed(receipt, key):
        print("SKIP verified JOB", source, flush=True)
        return
    attempt = root / ("attempt_" + str(time.time_ns()))
    attempt.mkdir(parents=True)
    for name in ("pg", "pgdata", "build", "database_import"):
        (attempt / name).symlink_to(base / name, target_is_directory=True)
    shutil.copy2(base / "SERVER.json", attempt / "SERVER.json")
    e2e.run(SimpleNamespace(service=service, output=str(attempt), pgdata=str(base / "pgdata"), training_root=str(root / "inputs")))
    outputs = {str(p.relative_to(ROOT)): digest(p) for p in [attempt / "timings.jsonl", attempt / "RESULTS.json", attempt / "CONTRACT.json", attempt / "COMPLETE.json", attempt / "HOOK_SMOKE.json", attempt / "TRUECARD_CHECKS.json"]}
    summary = read_json(attempt / "RESULTS.json")
    if not all(a["complete"] for a in summary["arms"].values()):
        raise RuntimeError("JOB has failed/timed-out samples; raw evidence retained, no successful receipt written")
    verify_files(spec)
    atomic_json(receipt, dict(status="COMPLETE", identity=key, inputs=spec, outputs=outputs, source=source))
    from .common import write_csv
    write_csv(ROOT / "paper_results/generated/new/tables/Table13_JOB.csv", [dict(arm=k, **v) for k,v in summary["arms"].items()])


def verify_reference():
    from .datasets import ensure
    from job_experiments import e2e
    ensure([], ["job-inputs", "job-evidence"], raw=False)
    from .metrics import evaluate
    for task in tasks():
        if task["suite"] != "job":
            continue
        scores = evaluate(task, ROOT / task["predictions"])
        for k, v in task["reference"].items():
            key = "test_queries" if k == "test_rows" else k
            import math
            if not math.isclose(scores[key], v, rel_tol=3e-6, abs_tol=1e-7):
                raise ValueError((task["id"], k, scores[key], v))
    evidence = ROOT / "artifacts/job"
    old = read_json(evidence / "RESULTS.json")
    out = ROOT / "paper_results/generated/reference/job"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(evidence / "timings.jsonl", out / "timings.jsonl")
    summary = e2e.collect(out)
    if summary != old["arms"]:
        raise ValueError("JOB aggregation mismatch")
    from .common import write_csv
    write_csv(out.parent / "tables/Table13_JOB.csv", [dict(arm=k, **v) for k,v in summary.items()])
    print("VERIFIED 16 JOB models and 3540 measured timing samples", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["new", "reference"], default="new")
    p.add_argument("--force", action="store_true")
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    verify_reference() if a.verify else run_all(a.force, a.source, a.setup_only)


if __name__ == "__main__":
    main()
