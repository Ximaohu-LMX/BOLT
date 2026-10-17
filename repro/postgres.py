"""Build the pinned PostgreSQL and maintain a dedicated local experiment cluster."""
import getpass
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
from .common import ROOT, digest, identity, read_json, atomic_json


def setup():
    from .datasets import ensure
    ensure([], ["job-inputs"], raw=False)
    folder = ROOT / "res/postgres"
    folder.mkdir(parents=True, exist_ok=True)
    source_zip = ROOT / "likeBench/postgresql-13.1.zip"
    prefix, build = folder / "pg", folder / "build"
    spec = {str(p.relative_to(ROOT)): digest(p) for p in (source_zip, ROOT / "job_experiments/build_pg.py", ROOT / "job_experiments/pg_job_hook.h")}
    stamp = folder / "BUILD.json"
    if not (stamp.exists() and read_json(stamp).get("identity") == identity(spec) and (prefix / "bin/postgres").exists()):
        if (folder / "pgdata/PG_VERSION").exists():
            raise RuntimeError("PostgreSQL sources changed. Preserve res/postgres and use a fresh checkout for this version.")
        build.mkdir(exist_ok=True)
        with zipfile.ZipFile(source_zip) as z:
            for info in z.infolist():
                p = Path(info.filename)
                if p.is_absolute() or ".." in p.parts or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("Unsafe PostgreSQL source archive")
            z.extractall(build)
        source = next(p.parent for p in build.glob("*/configure"))
        # zipfile does not restore executable bits; configure invokes these helpers.
        for path in source.rglob("*"):
            if path.is_file() and (path.name in ("configure", "install-sh", "missing", "config.guess", "config.sub") or path.suffix in (".sh", ".pl")):
                path.chmod(path.stat().st_mode | 0o111)
        from job_experiments.build_pg import patch
        patch(source)
        with (folder / "build.log").open("w") as log:
            for cmd in (["bash", "configure", "--prefix=" + str(prefix), "--without-readline", "--without-zlib"],
                        ["make", "-j", os.getenv("BOLT_BUILD_JOBS", "4")], ["make", "install"]):
                subprocess.run(cmd, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
        atomic_json(stamp, dict(identity=identity(spec), inputs=spec, binary_sha256=digest(prefix / "bin/postgres")))
    elif digest(prefix / "bin/postgres") != read_json(stamp)["binary_sha256"]:
        raise ValueError("PostgreSQL binary changed")
    data = folder / "pgdata"
    socket = folder / "socket"
    socket.mkdir(exist_ok=True)
    port = int(os.getenv("BOLT_PG_PORT", "55432"))
    if not (data / "PG_VERSION").exists():
        subprocess.run([str(prefix / "bin/initdb"), "-D", str(data), "--encoding=UTF8", "--locale=C", "--auth=trust"], check=True)
        with (data / "postgresql.conf").open("a") as f:
            f.write(f"\n# BOLT dedicated local cluster\nlisten_addresses = ''\nport = {port}\nunix_socket_directories = '{socket}'\nmax_parallel_workers_per_gather = 0\njit = off\nshared_buffers = '128MB'\n")
    control = [str(prefix / "bin/pg_ctl"), "-D", str(data)]
    running = subprocess.run(control + ["status"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not running:
        subprocess.run(control + ["-l", str(folder / "server.log"), "-w", "start"], check=True)
    service_file = folder / "pg_service.conf"
    service_file.write_text(f"[bolt_local]\nhost={socket}\nport={port}\nuser={getpass.getuser()}\ndbname=postgres\n")
    service_file.chmod(0o600)
    os.environ["PGSERVICEFILE"] = str(service_file)
    import psycopg2
    conn = psycopg2.connect(service="bolt_local")
    with conn, conn.cursor() as cur:
        cur.execute("SELECT version(), current_setting('data_directory'), current_setting('lc_collate')")
        version, actual_data, locale = cur.fetchone()
        if Path(actual_data).resolve() != data.resolve() or "14.5" not in version:
            raise ValueError("The service does not point at the dedicated PostgreSQL 14.5 cluster")
    conn.close()
    atomic_json(folder / "SERVER.json", dict(version=version, data_directory=str(data), locale=locale, binary_sha256=digest(prefix / "bin/postgres")))
    return folder, "bolt_local"


def stop():
    folder = ROOT / "res/postgres"
    if (folder / "pgdata/postmaster.pid").exists():
        subprocess.run([str(folder / "pg/bin/pg_ctl"), "-D", str(folder / "pgdata"), "-m", "fast", "-w", "stop"], check=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["setup", "stop"])
    args = p.parse_args()
    setup() if args.action == "setup" else stop()
