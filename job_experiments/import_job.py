"""Import the original CWI JOB CSV archive into new tables only.

COPY uses the source's backslash CSV escaping directly, without truncate.py.
All 21 tables and their indexes become visible together in one transaction.
"""
import argparse
import os
import re
import tarfile
import time
from pathlib import Path
from .protocol import ROOT, require, sha, write_json


def archive_members(path, expected):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        files = [m for m in members if m.isfile()]
        require(all(m.isdir() or m.isfile() for m in members), 'archive links/devices rejected')
        names = [Path(m.name).name for m in files]
        require(len(names) == len(set(names)), 'duplicate archive filenames')
        csv_names = {t+'.csv' for t in expected}
        # The original CWI archive also ships its schema as documentation.
        # Never execute it: the pinned repository DDL remains authoritative.
        require(set(names) - {'schematext.sql'} == csv_names, 'unexpected archive CSV set')
        for m in files:
            require(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts, 'unsafe archive member')
        return {Path(m.name).stem:m.name for m in files if Path(m.name).name in csv_names}


def run(args):
    import psycopg2
    from psycopg2 import sql
    archive=Path(args.archive).resolve(); out=Path(args.output).resolve()
    require(not out.exists(), 'import receipt directory already exists')
    ddl=(ROOT/'likeBench/imdb_schema.sql').read_text()
    index_ddl=(ROOT/'likeBench/imdb_index.sql').read_text()
    tables=re.findall(r'CREATE TABLE ([a-z_]+)\s*\(',ddl)
    require(len(tables)==21 and len(set(tables))==21,'expected 21 JOB tables')
    require(archive.stat().st_size==args.expected_bytes,'incomplete archive size')
    members=archive_members(archive,tables)
    archive_hash=sha(archive)
    out.mkdir(parents=True)
    status=dict(status='STARTING', archive=str(archive), archive_sha256=archive_hash,
        archive_bytes=archive.stat().st_size, schema_sha256=sha(ROOT/'likeBench/imdb_schema.sql'),
        indexes_sha256=sha(ROOT/'likeBench/imdb_index.sql'), tables={}, pid=os.getpid(),
        started_unix=time.time(), source='https://event.cwi.nl/da/job/imdb.tgz')
    write_json(out/'status.json',status)
    conn=psycopg2.connect(service=args.service,connect_timeout=10)
    stage='job_stage_'+str(os.getpid())
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT current_database(), version()'); status['database'],status['version']=cur.fetchone()
            cur.execute('SELECT pg_advisory_xact_lock(%s)',(1786214903,))
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename=ANY(%s)",(tables,))
            require(not cur.fetchall(),'JOB tables already exist; no replacement allowed')
            cur.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(stage)))
            cur.execute(sql.SQL('SET LOCAL search_path TO {}, public').format(sql.Identifier(stage)))
            cur.execute('SET LOCAL max_parallel_workers_per_gather=0')
            cur.execute(ddl)
            status['status']='COPYING_UNCOMMITTED';write_json(out/'status.json',status)
            # Stream archive once; no expanded 3+ GB copy on disk required.
            with tarfile.open(archive,'r|gz') as tar:
                for member in tar:
                    if not member.isfile(): continue
                    if Path(member.name).name == 'schematext.sql': continue
                    table=Path(member.name).stem
                    require(table in tables and member.name==members[table],'unexpected member')
                    before=time.monotonic()
                    statement=sql.SQL("COPY {} FROM STDIN WITH (FORMAT csv, DELIMITER ',', QUOTE '\"', ESCAPE '\\')").format(sql.Identifier(table))
                    with tar.extractfile(member) as stream:
                        cur.copy_expert(statement.as_string(conn),stream,size=1024*1024)
                    require(cur.rowcount>=0,'COPY row count unavailable')
                    status['tables'][table]=dict(rows=cur.rowcount,copy_seconds=time.monotonic()-before)
                    write_json(out/'status.json',status)
                    print('COPIED',table,cur.rowcount,flush=True)
            require(set(status['tables'])==set(tables),'not all tables copied')
            status['status']='INDEXING_UNCOMMITTED';write_json(out/'status.json',status)
            cur.execute(index_ddl)
            for table in tables:
                cur.execute(sql.SQL('ANALYZE {}').format(sql.Identifier(table)))
                cur.execute(sql.SQL('ALTER TABLE {} SET SCHEMA public').format(sql.Identifier(table)))
            cur.execute(sql.SQL('DROP SCHEMA {}').format(sql.Identifier(stage)))
        conn.commit()
        status['status']='IMPORTED';status['completed_unix']=time.time()
        write_json(out/'status.json',status);write_json(out/'IMPORTED.json',status)
        print('IMPORTED',len(tables),'tables',flush=True)
    except BaseException as exc:
        conn.rollback()
        status['status']='FAILED_ROLLED_BACK';status['error_type']=type(exc).__name__
        write_json(out/'status.json',status)
        raise
    finally:
        conn.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--service',required=True);p.add_argument('--archive',required=True)
    p.add_argument('--output',required=True);p.add_argument('--expected-bytes',type=int,default=1263193115)
    run(p.parse_args())

if __name__=='__main__':main()
