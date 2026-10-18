"""Frozen, serial JOB six-arm experiment; explicit correctness and raw timings."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import psycopg2
from psycopg2 import sql
from .protocol import ROOT,COLUMNS,require,read_json,write_json,sha,verify,normalize,workload
from .prediction import selectivity
from .database import export_column,identity
from a_route.scheduler import locks,GLOBAL_LOCKS,TIMING_LOCKS

ARMS=('PG','CLIQUE_A100','BOLT_A100','CLIQUE_A0','BOLT_A0','TrueCard')
SETTINGS={'max_parallel_workers_per_gather':'0','jit':'off','work_mem':'64MB','statement_timeout':'120000',
          'ml_joinest_enabled':'off','print_single_tbl_queries':'off','ml_cardest_enabled':'off'}


def connect(service):
    c=psycopg2.connect(service=service,connect_timeout=10);c.autocommit=True
    with c.cursor() as cur:
        for k,v in SETTINGS.items():cur.execute('SELECT set_config(%s,%s,false)',(k,v))
    return c


def trace_atoms(row,cur):
    if row['array']:
        require((row['operator'],row['use_or']) in [('=',True),('<>',False)],'unsupported array operator')
        cur.execute('SELECT %s::text[]',(row['literal'],));values=cur.fetchone()[0]
        require(values and all(isinstance(x,str) for x in values),'NULL/empty array rejected')
        require(len(set(values))==len(values),'duplicate array requires explicit dedup contract')
        return values
    require(row['operator'] in ('=','<>','~~','!~~'),'unknown operator')
    return [row['literal']]


def key_for(row):
    return ('%s|%s|%d|%d|%s'%(row['column'],row['operator'],row['array'],row['use_or'],row['literal'])).encode().hex()


def estimate_selectivity(row,values,predictions,metadata):
    count=sum(predictions[normalize(v)] for v in values)
    # Equality-array elements are disjoint, but learned counts may sum above N.
    clipped=min(count,metadata['nonnull'])
    negative=row['operator'] in ('<>','!~~')
    return selectivity(clipped,metadata['rows'],metadata['nonnull'],negative), count>metadata['nonnull']


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan['Plan'],sort_keys=True,separators=(',',':')).encode()).hexdigest()


def extract_trace(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def explain(cur,query,analyze=False):
    cur.execute('SET query_no=0')
    cur.execute('EXPLAIN ('+('ANALYZE, BUFFERS, ' if analyze else '')+'FORMAT JSON) '+query)
    return cur.fetchone()[0][0]


def setup_arm(cur,arm,qid):
    cur.execute('SET print_single_tbl_queries=off')
    cur.execute('SELECT set_config(%s,%s,false)',('ml_cardest_fname',f'q{qid:02d}_{arm}.map'))
    cur.execute('SELECT set_config(%s,%s,false)',('ml_cardest_enabled','off' if arm=='PG' else 'on'))
    cur.execute('SET query_no=0')


def load_assets(training_root):
    metadata_path = training_root / 'columns.json'
    prediction_path = training_root / 'positive_predictions.json'
    source_path = training_root / 'SOURCE.json'
    return read_json(metadata_path), read_json(prediction_path), {str(p): sha(p) for p in (metadata_path, prediction_path, source_path)}



def smoke(cur,pgdata,out):
    """Actual hook tests: key identity, negatives/arrays and malformed input rejection."""
    # Only in a rolled-back transaction; real JOB table remains unchanged.
    cur.execute('BEGIN')
    try:
        cur.execute('CREATE TEMP TABLE title(id int,title text)')
        cur.execute("INSERT INTO title VALUES (1,'abc'),(2,'abc'),(3,'xyz'),(4,''),(5,NULL)")
        queries=["SELECT * FROM title WHERE title LIKE 'a%'", "SELECT * FROM title WHERE title NOT LIKE 'a%'",
                 "SELECT * FROM title WHERE title IN ('abc','xyz')", "SELECT * FROM title WHERE title <> ''",
                 "SELECT * FROM title WHERE title = 'abc'"]
        checks=[]
        for i,q in enumerate(queries):
            cur.execute('SET ml_cardest_enabled=off');cur.execute('SET print_single_tbl_queries=on')
            path=pgdata/'job_trace.jsonl';path.unlink(missing_ok=True)
            explain(cur,q);trace=extract_trace(path);require(len(trace)==1,'smoke trace count')
            row=trace[0];require(row['key']==key_for(row),'C/Python key mismatch')
            values=trace_atoms(row,cur)
            pred={'a%':2,'abc':2,'xyz':1,'':1}
            value,_=estimate_selectivity(row,values,pred,dict(rows=5,nonnull=4))
            (pgdata/'smoke.map').write_text(row['key']+' '+format(value,'.17g')+'\n')
            cur.execute("SET ml_cardest_fname='smoke.map'");cur.execute('SET ml_cardest_enabled=on')
            path.unlink(missing_ok=True);explain(cur,q)
            injected=extract_trace(path);require(all(math.isclose(x['selected'],value,abs_tol=1e-15) for x in injected),'smoke mismatch')
            checks.append(dict(query=q,trace=injected))
        for name,content in [('missing',None),('malformed','garbage\n'),('duplicate',(row['key']+' .2\n')*2),('range',row['key']+' 1.2\n'),('unknown','00 .2\n')]:
            path=pgdata/'bad.map';path.unlink(missing_ok=True)
            if content is not None:path.write_text(content)
            cur.execute('SAVEPOINT badmap');cur.execute("SET ml_cardest_fname='bad.map'")
            try:explain(cur,queries[-1])
            except psycopg2.Error:cur.execute('ROLLBACK TO SAVEPOINT badmap')
            else:raise ValueError('invalid map did not fail: '+name)
        write_json(out/'HOOK_SMOKE.json',dict(status='PASS',checks=checks,negative_cases=5))
    finally:cur.execute('ROLLBACK')


def prepare(args,out,conn,status):
    root=Path(args.training_root).resolve();pgdata=Path(args.pgdata).resolve()
    metadata,predictions,inputs=load_assets(root)
    sqlpath=root/'plan/job.sql';w=workload(sqlpath)
    queries=[q.strip() for q in sqlpath.read_text().splitlines() if q.strip()]
    inputs[str(sqlpath)]=sha(sqlpath)
    for source in (out/'SERVER.json',out/'pgdata/postgresql.conf',out/'build/PATCH.json'):
        inputs[str(source)]=sha(source)
    inference=[dict(source=read_json(root/'SOURCE.json')['source'], note='All model estimates prepared offline; Table 13 excludes model inference. CPU model latency is measured separately by run_costs.sh.')]
    import yaml
    for done_path in sorted((root/'training/tasks').glob('*/DONE.json')):
        done=read_json(done_path)
        stat_path=next(Path(p) for p in done['files'] if p.endswith('/stat.yml'))
        stat=yaml.safe_load(stat_path.read_text())
        inference.append(dict(task=done['task']['id'],test_patterns=done['metrics']['test_rows'],
            recorded_batch_prediction_seconds=stat.get('time_est'),
            provenance=str(stat_path),note='historical training-run batch prediction timing, excludes database planning; not a new online latency benchmark'))
    write_json(out/'INFERENCE_ACCOUNTING.json',inference)
    with conn.cursor() as cur:
        smoke(cur,pgdata,out)
    status.update(phase='VERIFY_SNAPSHOT');write_json(out/'status.json',status)
    # Read-only repeatable snapshot for byte-identical four-column data audit.
    snapshot=psycopg2.connect(service=args.service);snapshot.set_session(readonly=True,isolation_level='REPEATABLE READ')
    try:
        for col in COLUMNS:require(export_column(snapshot,col)==metadata[col],'database column changed: '+col)
    finally:snapshot.close()
    import_info=read_json(out/'database_import/IMPORTED.json')
    original=read_json(root/'database_import/IMPORTED.json')
    require({k:v['rows'] for k,v in import_info['tables'].items()}=={k:v['rows'] for k,v in original['tables'].items()},'import row mismatch')
    require(import_info['archive_sha256']==original['archive_sha256'],'archive mismatch')
    # Independently validate every model test target on the actual timing server.
    true_checks={}
    with conn.cursor() as cur:
        cur.execute('SET ml_cardest_enabled=off');cur.execute('SET print_single_tbl_queries=off')
        for col in COLUMNS:
            table,attr=col.split('.');true_checks[col]={}
            for pattern in w['test_patterns'][col]:
                cur.execute(sql.SQL('SELECT count(*) FROM {} WHERE {} LIKE %s').format(sql.Identifier(table),sql.Identifier(attr)),(pattern,))
                n=cur.fetchone()[0];require(n==predictions['TrueCard'][col][pattern],'TrueCard mismatch')
                true_checks[col][pattern]=n
    write_json(out/'TRUECARD_CHECKS.json',true_checks)
    status.update(phase='VERIFY_INJECTION');write_json(out/'status.json',status)
    evidence=[];map_files={}
    with conn.cursor() as cur:
        for qid,query in enumerate(queries,1):
            setup_arm(cur,'PG',qid);cur.execute('SET print_single_tbl_queries=on')
            tracepath=pgdata/'job_trace.jsonl';tracepath.unlink(missing_ok=True)
            base=explain(cur,query);trace=extract_trace(tracepath)
            require(trace,'missing modeled predicates: '+str(qid))
            keyed={};seen=set();clipped=[]
            expected={(x['column'],x['pattern']) for x in w['predicates'] if x['sql_id']==qid}
            maps={a:{} for a in ARMS if a!='PG'};maps['REPLAY']={}
            for row in trace:
                require(row['key']==key_for(row),'C/Python key mismatch')
                values=trace_atoms(row,cur)
                for v in values:
                    identity_pair=(row['column'],normalize(v));require(identity_pair in expected,'unexpected predicate')
                    seen.add(identity_pair)
                keyed[row['key']]=row
                maps['REPLAY'][row['key']]=-1
                for arm in ARMS[1:]:
                    val,clip=estimate_selectivity(row,values,predictions[arm][row['column']],metadata[row['column']])
                    maps[arm][row['key']]=val
                    if clip:clipped.append(dict(arm=arm,key=row['key']))
            require(seen==expected,'uncovered SQL targets: '+str(expected-seen))
            for arm,values in maps.items():
                path=pgdata/f'q{qid:02d}_{arm}.map'
                path.write_text(''.join(k+' '+format(v,'.17g')+'\n' for k,v in sorted(values.items())))
                map_files[str(path)]=sha(path)
            checked={}
            for arm in ('REPLAY',)+ARMS[1:]:
                setup_arm(cur,arm,qid);cur.execute('SET print_single_tbl_queries=on');tracepath.unlink(missing_ok=True)
                plan=explain(cur,query);actual=extract_trace(tracepath)
                require({x['key'] for x in actual}==set(keyed),'predicate key coverage changed')
                for row in actual:
                    want=row['baseline'] if arm=='REPLAY' else maps[arm][row['key']]
                    require(math.isclose(row['selected'],want,abs_tol=1e-15),'injection value mismatch')
                if arm=='REPLAY':require(plan_hash(plan)==plan_hash(base),'PG replay plan changed')
                checked[arm]=dict(plan=plan,trace=actual)
            write_json(out/'verification'/f'q{qid:02d}.json',dict(baseline=base,baseline_trace=trace,checked=checked,clipped_arrays=clipped))
            evidence.append(dict(sql_id=qid,keys=len(keyed),clipped_arrays=clipped))
            status.update(verified_queries=qid);write_json(out/'status.json',status)
        # Full SELECT result equality, beyond EXPLAIN plan-only checks.
        status.update(phase='VERIFY_RESULTS');write_json(out/'status.json',status)
        for qid,query in enumerate(queries,1):
            result_by_arm={}
            for arm in ARMS:
                setup_arm(cur,arm,qid);cur.execute(query);rows=cur.fetchall()
                result_by_arm[arm]=rows
            require(all(v==result_by_arm['PG'] for v in result_by_arm.values()),'query result mismatch')
            write_json(out/'verification'/f'q{qid:02d}_results.json',result_by_arm)
            status.update(verified_results=qid);write_json(out/'status.json',status)
    # Fixed, interleaved order; no adaptation based on timings or accuracy.
    rng=random.Random(0);schedule=[]
    for trial in range(11):
        qids=list(range(1,60));rng.shuffle(qids)
        for qid in qids:
            arms=list(ARMS);rng.shuffle(arms)
            for arm in arms:schedule.append(dict(sql_id=qid,arm=arm,trial=trial,phase='warmup' if trial==0 else 'measured'))
    sources={str(p):sha(p) for p in (ROOT/'job_experiments').glob('*.py')}
    sources[str(ROOT/'job_experiments/pg_job_hook.h')]=sha(ROOT/'job_experiments/pg_job_hook.h')
    contract=dict(arms=ARMS,sql_count=59,repeats=10,warmup_per_arm_query=1,seed=0,
        timeout_ms=120000,cache_protocol='one full warmup pass then interleaved repeated warm steady-state runs; no OS cache drop',
        timing='EXPLAIN ANALYZE planning_ms + execution_ms; offline model inference excluded',
        settings=SETTINGS,database=identity(conn),inputs=inputs,maps=map_files,sources=sources,
        binary_sha256=sha(out/'pg/bin/postgres'),schedule=schedule,query_evidence=evidence,
        aggregation='retain all raw samples; Tukey 1.5 IQR on total_ms within each arm/query; failures retained and aggregates marked incomplete')
    write_json(out/'CONTRACT.json',contract);write_json(out/'PREFLIGHT_PASS.json',dict(status='PASS',contract_sha256=sha(out/'CONTRACT.json')))
    return queries,contract


def collect(out):
    records=[json.loads(x) for x in (out/'timings.jsonl').read_text().splitlines()]
    rows=[];by_arm=defaultdict(list)
    for arm in ARMS:
        for qid in range(1,60):
            group=[x for x in records if x['phase']=='measured' and x['arm']==arm and x['sql_id']==qid]
            valid=[x for x in group if x['status']=='OK'];values=[x['total_ms'] for x in valid]
            retained=[]
            if values:
                lo,hi=np.quantile(values,[.25,.75]);width=hi-lo
                retained=[x for x in valid if lo-1.5*width<=x['total_ms']<=hi+1.5*width]
            item=dict(arm=arm,sql_id=qid,attempts=len(group),success=len(valid),failed=len(group)-len(valid),complete=len(valid)==10,
                retained_trials=[x['trial'] for x in retained],excluded_trials=[x['trial'] for x in valid if x not in retained])
            if retained:
                for k in ('total_ms','planning_ms','execution_ms'):item[k]=float(np.mean([x[k] for x in retained]))
                item['unfiltered_mean_total_ms']=float(np.mean(values));item['median_total_ms']=float(np.median(values))
            rows.append(item);by_arm[arm].append(item)
    summary={a:dict(complete=all(x['complete'] for x in r),failed_attempts=sum(x['failed'] for x in r),
                    sum_query_mean_total_ms=sum(x['total_ms'] for x in r) if all(x['complete'] for x in r) else None)
             for a,r in by_arm.items()}
    pg={r['sql_id']:r for r in by_arm['PG']}
    for row in rows:
        b=pg[row['sql_id']]
        if row['complete'] and b['complete']:row['speedup_vs_pg']=b['total_ms']/row['total_ms']
    write_json(out/'RESULTS.json',dict(arms=summary,queries=rows))
    return summary


def run(args):
    out=Path(args.output).resolve();pgdata=Path(args.pgdata).resolve()
    require(not (out/'status.json').exists(),'existing E2E attempt; reconcile explicitly')
    status=dict(status='RUNNING',phase='PREFLIGHT',pid=os.getpid(),started_unix=time.time(),measured_done=0,measured_total=3540)
    write_json(out/'status.json',status)
    try:
        with locks(GLOBAL_LOCKS+TIMING_LOCKS+('clique_job_timing',)):
            import shutil
            if shutil.which('nvidia-smi'):
                gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
                require(not gpu.stdout.strip(),'GPU experiment active during timing')
            conn=connect(args.service)
            try:
                queries,contract=prepare(args,out,conn,status)
                require(all(sha(p)==h for p,h in contract['inputs'].items()),'changed input')
                require(all(sha(p)==h for p,h in contract['maps'].items()),'changed maps')
                with conn.cursor() as cur, (out/'timings.jsonl').open('x') as raw:
                    for index,item in enumerate(contract['schedule']):
                        status.update(phase=item['phase'].upper(),current=item,schedule_index=index)
                        write_json(out/'status.json',status)
                        setup_arm(cur,item['arm'],item['sql_id']);started=time.perf_counter()
                        record=dict(item,started_unix=time.time())
                        try:
                            plan=explain(cur,queries[item['sql_id']-1],True)
                            p=out/'plans'/f"{item['trial']:02d}_{item['sql_id']:02d}_{item['arm']}.json"
                            write_json(p,plan)
                            record.update(status='OK',planning_ms=plan['Planning Time'],execution_ms=plan['Execution Time'],
                                total_ms=plan['Planning Time']+plan['Execution Time'],total_cost=plan['Plan']['Total Cost'],plan=str(p),plan_sha256=sha(p))
                        except psycopg2.errors.QueryCanceled as exc:
                            record.update(status='TIMEOUT',error=str(exc),timeout_ms=120000)
                        except psycopg2.Error as exc:
                            record.update(status='ERROR',sqlstate=exc.pgcode,error=str(exc))
                            raw.write(json.dumps(record)+'\n');raw.flush();raise
                        record['client_wall_ms']=(time.perf_counter()-started)*1000
                        raw.write(json.dumps(record,allow_nan=False)+'\n');raw.flush()
                        if item['phase']=='measured':status['measured_done']+=1
                require(all(sha(p)==h for p,h in contract['inputs'].items()),'input changed during measurements')
                require(all(sha(p)==h for p,h in contract['maps'].items()),'map changed during measurements')
                require(all(sha(p)==h for p,h in contract['sources'].items()),'runner changed during measurements')
                summary=collect(out)
                status.update(status='COMPLETE',phase='COMPLETE',completed_unix=time.time(),summary=summary)
                write_json(out/'COMPLETE.json',dict(status=status,contract_sha256=sha(out/'CONTRACT.json'),timings_sha256=sha(out/'timings.jsonl'),results_sha256=sha(out/'RESULTS.json')))
                write_json(out/'status.json',status)
            finally:conn.close()
    except BaseException as exc:
        status.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),failed_unix=time.time());write_json(out/'status.json',status)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--service',required=True);p.add_argument('--output',required=True)
    p.add_argument('--pgdata',required=True);p.add_argument('--training-root',required=True)
    run(p.parse_args())
