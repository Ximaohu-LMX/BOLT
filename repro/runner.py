"""Execute independently resumable experiment cells with verified receipts."""
from contextlib import contextmanager
from functools import lru_cache
import fcntl
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from .common import ROOT, atomic_json, digest, identity, read_json, verify_files


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process holds {path}") from None
        yield


@lru_cache(maxsize=1)
def environment():
    import importlib.metadata
    packages = {}
    for name in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    return dict(python=platform.python_version(), packages=packages,
                cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES", "default"),
                threads=os.getenv("BOLT_THREADS", "1"))


def contract(task):
    inputs = [ROOT / task["config"], ROOT / f'data/{task["dataset"]}/{task["dataset"]}.txt']
    view = ROOT / f'data/{task["dataset"]}/training/{task["workload"]}'
    inputs += [view / name for name in ("train.txt", "valid.txt", "test.txt")]
    if task["method"] in ("BOLT", "BOLT_L0", "UB_ORDER", "LB_ORDER", "ZERO_MID_MONO", "ANCHORED_MID_MONO"):
        inputs.append(view / "chain_index.json")
    if task["method"] == "LPLM":
        inputs.append(view / "train_LPLM.txt")
    inputs += sorted((ROOT / "src").rglob("*.py"))
    inputs += sorted(ROOT.glob("main_*.py"))
    inputs += sorted((ROOT / "repro").glob("*.py"))
    inputs += sorted((ROOT / "a_route").glob("*.py"))
    inputs += sorted((ROOT / "configs").glob("*.csv"))
    hashes = {}
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path.relative_to(ROOT)}; run download.sh")
        hashes[str(path.relative_to(ROOT))] = digest(path)
    # Frozen summaries affect predictions and must be part of the identity too.
    for folder in (ROOT / f'res/{task["dataset"]}/Ntable', ROOT / f'res/{task["dataset"]}/freq_library'):
        for path in sorted(folder.iterdir()) if folder.exists() else []:
            if path.suffix not in (".pkl", ".yml"):
                continue
            hashes[str(path.relative_to(ROOT))] = digest(path)
    return dict(schema=1, task={k: task[k] for k in ("id", "dataset", "budget", "method", "seed", "workload")},
                inputs=hashes, environment=environment(), operation="train")


def completed(receipt, expected_identity):
    try:
        record = read_json(receipt)
        if record.get("status") != "COMPLETE" or record["identity"] != expected_identity or not record["outputs"]:
            return False
        verify_files(record["outputs"])
        return True
    except (OSError, ValueError, KeyError):
        return False


def status(task):
    receipt = ROOT / "res/runs" / task["id"] / "DONE.json"
    if not receipt.exists():
        return "PENDING"
    try:
        return "COMPLETE" if completed(receipt, identity(contract(task))) else "STALE"
    except FileNotFoundError:
        return "MISSING_INPUT"


def run(task, force=False, epochs=None):
    folder = ROOT / "res/runs" / task["id"]
    folder.mkdir(parents=True, exist_ok=True)
    with lock(folder / ".lock"):
        spec = contract(task)
        if epochs is not None:
            spec["epoch_override"] = epochs
        fingerprint = identity(spec)
        receipt = folder / "DONE.json"
        if not force and completed(receipt, fingerprint):
            print("SKIP verified", task["id"], flush=True)
            return read_json(receipt)
        attempt = folder / ("attempt_" + str(time.time_ns()))
        attempt.mkdir()
        atomic_json(attempt / "contract.json", spec)
        command = [sys.executable, "-m", "repro.worker", "--task", task["id"], "--output", str(attempt)]
        if epochs is not None:
            command += ["--epochs", str(epochs)]
        env = dict(os.environ)
        for name in list(env):
            if name in ("MODE", "EST", "DEBUG", "PYTHONPATH") or name.startswith(("EFFICIENCY_", "BIMO_")):
                env.pop(name)
        env.update(PYTHONHASHSEED=str(task["seed"]), PYTHONUNBUFFERED="1", SKIP_LEGACY_MONO_EVAL="1",
                   OMP_NUM_THREADS=os.getenv("BOLT_THREADS", "1"), MKL_NUM_THREADS=os.getenv("BOLT_THREADS", "1"),
                   OPENBLAS_NUM_THREADS=os.getenv("BOLT_THREADS", "1"))
        # Inherited CLIQUE may rewrite a cached pickle after checking its
        # pruning threshold. Isolate this behavior from the frozen release input.
        import shutil
        private_cache = attempt / 'summary_cache'
        for name in ('Ntable', 'freq_library'):
            original = ROOT / 'res' / task['dataset'] / name
            if original.exists():
                shutil.copytree(original, private_cache / task['dataset'] / name)
        env['EFFICIENCY_CACHE_ROOT'] = str(private_cache)
        atomic_json(attempt / "RUNNING.json", dict(command=command, identity=fingerprint, pid=os.getpid()))
        print("RUN", task["id"], "log:", attempt.relative_to(ROOT) / "train.log", flush=True)
        started = time.monotonic()
        try:
            with (attempt / "train.log").open("w") as stream:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = read_json(attempt / "result.json")
            verify_files(spec["inputs"])
            outputs = {name: digest(ROOT / name) for name in result["files"]}
            outputs[str((attempt / "result.json").relative_to(ROOT))] = digest(attempt / "result.json")
            record = dict(status="COMPLETE", identity=fingerprint, contract=spec, outputs=outputs,
                          result=result, elapsed_seconds=time.monotonic() - started,
                          timing_usable=False, epoch_override=epochs)
            atomic_json(receipt, record)
            shutil.rmtree(private_cache, ignore_errors=True)
            print("DONE", task["id"], flush=True)
            return record
        except BaseException as error:
            atomic_json(attempt / "FAILED.json", dict(type=type(error).__name__, message=str(error)))
            raise
