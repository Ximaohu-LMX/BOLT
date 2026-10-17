"""Wait for the isolated import, freeze server metadata, then run JOB timing."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .protocol import ROOT,read_json,write_json,require,sha
from .pipeline import process_live
from .e2e import connect,run


def launch(args):
    out=Path(args.output).resolve()
    state=dict(status='RUNNING',phase='WAIT_IMPORT',pid=os.getpid(),started_unix=time.time())
    write_json(out/'launch_status.json',state)
    try:
        receipt=out/'database_import/IMPORTED.json'
        while not receipt.exists():
            status=out/'database_import/status.json'
            if status.exists():
                s=read_json(status);require(not s['status'].startswith('FAILED'),'import failed')
                state['import_status']=s['status'];state['import_tables']=len(s['tables'])
            require(process_live(args.import_pid),'import process ended without completion receipt')
            write_json(out/'launch_status.json',state);time.sleep(10)
        state['phase']='VACUUM_AND_FREEZE';write_json(out/'launch_status.json',state)
        conn=connect(args.service)
        try:
            with conn.cursor() as cur:
                cur.execute('VACUUM (ANALYZE)')
                cur.execute('SELECT version()');version=cur.fetchone()[0]
                require(version.startswith('PostgreSQL 14.5 '),'unexpected server version')
                cur.execute('SELECT name, setting, unit FROM pg_settings ORDER BY name')
                settings=cur.fetchall()
                cur.execute('SELECT current_database(), current_user');db,user=cur.fetchone()
            files=[out/'pg/bin/postgres',out/'pgdata/postgresql.conf',out/'build/PATCH.json']
            write_json(out/'SERVER.json',dict(version=version,database=db,user=user,pg_settings=settings,
                files={str(p):sha(p) for p in files},host=os.uname(),cpu=subprocess.check_output(['lscpu'],text=True),
                imported_sha256=sha(receipt),timing_instance='isolated Unix socket, no TCP listener'))
        finally:conn.close()
        state['phase']='E2E';write_json(out/'launch_status.json',state)
        run(args)
        state.update(status='COMPLETE',phase='COMPLETE',completed_unix=time.time());write_json(out/'launch_status.json',state)
    except BaseException as exc:
        state.update(status='FAILED',error=str(exc),failed_unix=time.time());write_json(out/'launch_status.json',state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--service',required=True);p.add_argument('--output',required=True)
    p.add_argument('--pgdata',required=True);p.add_argument('--training-root',required=True);p.add_argument('--import-pid',type=int,required=True)
    launch(p.parse_args())
