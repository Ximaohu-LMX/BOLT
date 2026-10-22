"""Download CLIQUE columns and the versioned BOLT paper artifacts."""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from urllib.parse import quote, urlsplit
from pathlib import Path, PurePosixPath
from .common import ROOT, DATASETS, digest, read_json, verify_files
from .archive_compat import paper_names, materialized_bytes


def safe_members(archive):
    for member in archive.getmembers():
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
            raise ValueError(f"Unsafe archive member: {member.name}")
        yield member


def artifact_url(bundle):
    base = os.getenv("BOLT_ASSET_BASE_URL", "").strip()
    if not base:
        return bundle.get("url")
    parsed = urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("BOLT_ASSET_BASE_URL must be an HTTP(S) directory URL without a query or fragment")
    return base.rstrip("/") + "/" + quote(bundle["name"], safe="")


def download_file(url, target, expected=None, local=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (expected is None or digest(target) == expected):
        return target
    temporary = target.with_name(target.name + ".part")
    if local:
        shutil.copyfile(Path(local) / target.name, temporary)
    else:
        if not url:
            raise ValueError(
                f"No download source is configured for {target.name}. "
                "Set BOLT_ASSET_DIR to a directory containing the artifact bundles, "
                "or BOLT_ASSET_BASE_URL to an HTTP(S) mirror directory. See docs/DATA.md."
            )
        request = urllib.request.Request(url, headers={"User-Agent": "BOLT-reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    if expected and digest(temporary) != expected:
        raise ValueError(f"Download checksum mismatch: {target.name}")
    os.replace(temporary, target)
    return target


def install_bundle(bundle, local=None):
    archive_bundle = bundle
    migrate = "archive_manifest" in bundle
    if migrate:
        archive_bundle = next(b for b in read_json(ROOT / bundle["archive_manifest"])["bundles"] if b["name"] == bundle["name"])
        if archive_bundle["sha256"] != bundle["sha256"]:
            raise ValueError("Archive identity differs from its frozen manifest")
        renamed = {paper_names(p) for p in archive_bundle["files"]}
        if len(renamed) != len(archive_bundle["files"]) or renamed != set(bundle["files"]):
            raise ValueError("Invalid paper-name archive mapping")
        expected_copies = {paper_names(k): paper_names(v) for k, v in archive_bundle.get("copies", {}).items()}
        if bundle.get("copies", {}) != expected_copies:
            raise ValueError("Invalid paper-name copy mapping")
    for name in bundle["files"]:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe artifact destination")
    copies = bundle.get("copies", {})
    for target, source in copies.items():
        names = (PurePosixPath(target), PurePosixPath(source))
        if any(p.is_absolute() or ".." in p.parts for p in names):
            raise ValueError("Unsafe copied artifact path")
        if target not in bundle["files"] or source not in bundle["files"] or source in copies:
            raise ValueError("Copied artifacts must refer to a canonical archive file")
        if bundle["files"][target] != bundle["files"][source]:
            raise ValueError("Copied artifact checksum differs from its source")
    try:
        verify_files(bundle["files"], ROOT)
        print("SKIP verified", bundle["name"], flush=True)
        return
    except (ValueError, OSError):
        pass
    archive = download_file(None if local else artifact_url(bundle), ROOT / ".cache/downloads" / bundle["name"],
                            bundle["sha256"], local)
    with tarfile.open(archive, "r:gz") as tar:
        members = list(safe_members(tar))
        actual = {m.name for m in members if m.isfile()}
        if actual != set(archive_bundle["files"]) - set(archive_bundle.get("copies", {})):
            raise ValueError("Archive manifest does not match its member list")
        for member in members:
            if not member.isfile():
                continue
            current_name = paper_names(member.name) if migrate else member.name
            destination = ROOT / current_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink() or any(p.is_symlink() for p in destination.parents if p != ROOT.parent):
                raise ValueError(f"Refusing symlink destination: {destination}")
            temporary = destination.with_name(destination.name + ".extracting")
            with tar.extractfile(member) as inp, temporary.open("wb") as out:
                shutil.copyfileobj(inp, out)
            if digest(temporary) != archive_bundle["files"][member.name]:
                raise ValueError(f"Extracted source checksum mismatch: {member.name}")
            if migrate and current_name.startswith(("artifacts/checkpoints/", "artifacts/timing/", "artifacts/job/")) and destination.suffix in (".json", ".jsonl", ".yaml", ".yml", ".csv"):
                before = temporary.read_bytes()
                after = materialized_bytes(member.name, before)
                if after != before:
                    temporary.write_bytes(after)
            if digest(temporary) != bundle["files"][current_name]:
                raise ValueError(f"Materialized checksum mismatch: {current_name}")
            os.replace(temporary, destination)
    for target, source in copies.items():
        destination = ROOT / target
        if destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
            raise ValueError("Refusing symlink copy destination")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".extracting")
        shutil.copyfile(ROOT / source, temporary)
        if digest(temporary) != bundle["files"][target]:
            raise ValueError("Copied artifact checksum mismatch")
        os.replace(temporary, destination)
    verify_files(bundle["files"], ROOT)
    print("READY", bundle["name"], flush=True)


def raw_column(dataset, manifest):
    item = manifest["raw_columns"][dataset]
    destination = ROOT / item["path"]
    if destination.is_file() and digest(destination) == item["sha256"]:
        print("SKIP verified raw", dataset, flush=True)
        return
    archive = ROOT / ".cache/downloads" / (dataset + ".tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        subprocess.run([sys.executable, "-m", "gdown", "https://drive.google.com/uc?id=" + item["google_drive_id"],
                        "-O", str(temporary)], check=True)
        os.replace(temporary, archive)
    # Extract only the original column. Upstream workloads are not the BOLT splits.
    with tarfile.open(archive, "r:gz") as tar:
        matches = [m for m in tar.getmembers() if m.isfile() and PurePosixPath(m.name).name == dataset + ".txt"]
        if len(matches) != 1:
            raise ValueError(f"Expected one {dataset}.txt in {archive}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        with tar.extractfile(matches[0]) as inp, temporary.open("wb") as out:
            shutil.copyfileobj(inp, out)
        if digest(temporary) != item["sha256"]:
            raise ValueError(f"The upstream {dataset} column differs from the paper; see docs/DATA.md")
        os.replace(temporary, destination)
    print("READY raw", dataset, flush=True)


def ensure(datasets=DATASETS, kinds=("workloads", "summaries"), raw=True, local=None):
    manifest = read_json(ROOT / "manifests/artifacts.json")
    if raw:
        for dataset in datasets:
            raw_column(dataset, manifest)
    for bundle in manifest["bundles"]:
        if bundle["kind"] in kinds and (not bundle.get("dataset") or bundle["dataset"] in datasets):
            install_bundle(bundle, local or os.getenv("BOLT_ASSET_DIR"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", "--dataset", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--artifacts", nargs="+", default=["workloads", "summaries"],
                        choices=["workloads", "summaries", "predictions", "checkpoints", "timing", "job-evidence", "job-inputs", "all"])
    parser.add_argument("--no-raw", action="store_true")
    args = parser.parse_args()
    kinds = args.artifacts
    if "all" in kinds:
        kinds = [b["kind"] for b in read_json(ROOT / "manifests/artifacts.json")["bundles"]]
    ensure(args.datasets, kinds, not args.no_raw)


if __name__ == "__main__":
    main()
