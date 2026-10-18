"""Small, portable filesystem and provenance helpers."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("AUTHOR", "DBLP", "IMDB", "WIKI")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        yield from csv.DictReader(stream)


def labeled_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                query, count = line.rstrip("\r\n").rsplit(",", 1)
                yield query, float(count)


def write_csv(path, records):
    records = list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def tasks():
    result = read_json(ROOT / "manifests/tasks.json")
    job = ROOT / "manifests/job_tasks.json"
    return result + (read_json(job) if job.exists() else [])


def select_tasks(suite="all", datasets=None, budgets=None, methods=None, seed=None):
    from .names import method_id
    methods = {method_id(m) for m in methods} if methods else None
    return [t for t in tasks()
            if ((suite in ("all", "main") and t["suite"] != "job") or t["suite"] == suite)
            and (not datasets or t["dataset"] in datasets)
            and (not budgets or t["budget"] in budgets)
            and (not methods or t["method"] in methods)
            and (seed is None or t["seed"] == seed)]


def verify_files(files, root=ROOT):
    for relative, expected in files.items():
        path = Path(root) / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing or changed artifact: {relative}")
