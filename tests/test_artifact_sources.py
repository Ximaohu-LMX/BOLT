"""Check public artifact defaults, local overrides, and download integrity."""
import functools
import hashlib
import io
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tarfile
import threading

import pytest

from repro import datasets
from repro.common import digest


def make_bundle(directory):
    payload = b"pattern,true\na%,3\n"
    path = directory / "workload.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("data/source.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    return dict(name=path.name, url=None, sha256=digest(path),
                files={"data/source.txt": hashlib.sha256(payload).hexdigest()}), payload


def no_network(*args, **kwargs):
    raise AssertionError("Unconfigured or local artifact access must not contact a server")


def test_unconfigured_source_fails_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.urllib.request, "urlopen", no_network)
    with pytest.raises(ValueError, match="No download source"):
        datasets.download_file(None, tmp_path / "bundle.tar.gz", "unavailable")
    assert not (tmp_path / "bundle.tar.gz").exists()
    assert not (tmp_path / "bundle.tar.gz.part").exists()


def test_verified_cache_needs_no_source(tmp_path, monkeypatch):
    target = tmp_path / "bundle.tar.gz"
    target.write_bytes(b"already verified")
    monkeypatch.setattr(datasets.urllib.request, "urlopen", no_network)
    assert datasets.download_file(None, target, digest(target)) == target
    with pytest.raises(ValueError, match="No download source"):
        datasets.download_file(None, target, "incorrect hash")
    assert target.read_bytes() == b"already verified"


def test_local_bundle_works_without_url_and_overrides_remote(tmp_path, monkeypatch):
    mirror = tmp_path / "bundles"
    mirror.mkdir()
    root = tmp_path / "checkout"
    root.mkdir()
    bundle, payload = make_bundle(mirror)
    monkeypatch.setattr(datasets, "ROOT", root)
    monkeypatch.setenv("BOLT_ASSET_BASE_URL", "not a valid remote URL")
    monkeypatch.setattr(datasets.urllib.request, "urlopen", no_network)
    datasets.install_bundle(bundle, str(mirror))
    assert (root / "data/source.txt").read_bytes() == payload
    datasets.install_bundle(bundle, str(mirror))


def test_remote_override_installs_and_rejects_corruption(tmp_path, monkeypatch):
    mirror = tmp_path / "bundles"
    mirror.mkdir()
    root = tmp_path / "checkout"
    root.mkdir()
    bundle, payload = make_bundle(mirror)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(mirror))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(datasets, "ROOT", root)
        monkeypatch.setenv("BOLT_ASSET_BASE_URL", f"http://127.0.0.1:{server.server_port}/")
        datasets.install_bundle(bundle)
        assert (root / "data/source.txt").read_bytes() == payload
        (root / "data/source.txt").unlink()
        (root / ".cache/downloads" / bundle["name"]).unlink()
        (mirror / bundle["name"]).write_bytes(b"not the frozen artifact")
        with pytest.raises(ValueError, match="Download checksum mismatch"):
            datasets.install_bundle(bundle)
        assert not (root / "data/source.txt").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("base", ["/relative/path", "file:///tmp/bundles", "https://example.invalid/data?token=x", "https://example.invalid/data#files"])
def test_remote_override_rejects_ambiguous_directory_urls(monkeypatch, base):
    monkeypatch.setenv("BOLT_ASSET_BASE_URL", base)
    with pytest.raises(ValueError, match="directory URL"):
        datasets.artifact_url({"name": "workload.tar.gz", "url": None})


def test_paper_manifests_have_public_downloads(monkeypatch):
    monkeypatch.delenv("BOLT_ASSET_BASE_URL", raising=False)
    root = Path(__file__).resolve().parents[1]
    manifests = [json.loads((root / name).read_text()) for name in
                 ("manifests/artifacts.json", "manifests/archive/paper-v1.0/artifacts.json")]
    release = "https://github.com/Ximaohu-LMX/BOLT/releases/download/paper-v1.0/"
    frozen = {bundle["name"]: bundle for bundle in manifests[1]["bundles"]}
    assert len(frozen) == 10
    assert {bundle["name"] for bundle in manifests[0]["bundles"]} == set(frozen)
    for manifest in manifests:
        for bundle in manifest["bundles"]:
            assert bundle["url"] == release + bundle["name"]
            assert datasets.artifact_url(bundle) == bundle["url"]
            assert (bundle["sha256"], bundle["bytes"]) == (
                frozen[bundle["name"]]["sha256"], frozen[bundle["name"]]["bytes"])
