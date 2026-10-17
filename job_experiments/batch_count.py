"""Reuse CLIQUE's LEADER-S exact batch counter, cross-check against PostgreSQL."""
import argparse
from collections import Counter
import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path
from .protocol import ROOT,COLUMNS,read_json,write_json,sha,verify,seal,require
from .cli import staged


def batch_counts(column_file, patterns, workspace, column, binary=ROOT/'main'):
    workspace=Path(workspace); db=workspace/'data'/column; query=db/'query'/'JOB'
    query.mkdir(parents=True)
    (db/(column+'.txt')).symlink_to(Path(column_file).resolve())
    histogram=Counter()
    with Path(column_file).open() as f:
        for line in f:histogram[len(line.rstrip('\n'))]+=1
    cards={}
    neural=[]
    for q in patterns:
        if all(c in '%_' for c in q):
            k=q.count('_'); cards[q]=sum(v for n,v in histogram.items() if (n>=k if '%' in q else n==k))
        else:neural.append(q)
    (query/'all.txt').write_text(''.join(q+'\n' for q in neural))
    if neural:
        with (workspace/'counter.log').open('w') as log:
            subprocess.run([str(binary),'LEADER-S',column,'JOB','all','0'],cwd=workspace,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        result=workspace/'res'/column/'JOB'/'data'/'all'/'0'/'LEADER-S.txt'
        with result.open(newline='') as f:
            rows=list(csv.reader(f))
        require(len(rows)==len(neural),'batch counter row mismatch')
        for expected,row in zip(neural,rows):
            require(len(row)==2 and row[0]==expected,'batch counter query mismatch')
            cards[expected]=int(row[1])
    require(set(cards)==set(patterns),'batch counter target mismatch')
    require(all(0<=v<=sum(histogram.values()) for v in cards.values()),'count range error')
    return cards


def run(args):
    from .database import connect,identity,export_column,count_patterns
    from a_route.scheduler import locks,GLOBAL_LOCKS,TIMING_LOCKS
    prepared=Path(args.prepared).resolve();info=verify(prepared,'PREPARED.json')
    require(info['status']=='PREPARED','preparation is not usable')
    def action(out):
        conn=connect(args.service,args.timeout_ms)
        checks={}
        try:
            require(identity(conn)==info['database'],'database identity changed')
            for column in COLUMNS:
                require(export_column(conn,column)==info['columns'][column],'column snapshot changed: '+column)
                patterns=read_json(prepared/column/'count_targets.json')
                cards=batch_counts(prepared/column/'column.txt',patterns,out/column,column)
                split=read_json(prepared/column/'split.json')
                others=sorted(set(patterns)-set(split['test']))
                sample=sorted(set(split['test'])|set(random.Random(0).sample(others,min(args.check_samples,len(others)))))
                reference=count_patterns(conn,column,sample)
                require(all(cards[q]==v for q,v in reference.items()),'PostgreSQL cross-check failed: '+column)
                checks[column]=dict(checked=len(sample),total=len(patterns),sample_seed=0)
                write_json(out/(column+'.json'),cards)
                write_json(out/(column+'.postgres_check.json'),reference)
                write_json(out/'progress.json',checks)
                print('COUNTED',column,len(patterns),'PG_CHECKED',len(sample),flush=True)
            seal(out,'COUNTS.json',dict(prepared_path=str(prepared),prepared_sha256=sha(prepared/'PREPARED.json'),
                 method='existing LEADER-S exact counter plus exact length histogram for wildcard-only patterns',
                 binary_sha256=sha(ROOT/'main'),adapter_sha256=sha(Path(__file__)),postgres_checks=checks,
                 accounting='all chain-node counts include offline diagnostic labels; PG cross-check is sampled'))
        finally:
            conn.rollback();conn.close()
    with locks(GLOBAL_LOCKS+TIMING_LOCKS+('clique_time_efficiency_label_v1',)):
        staged(args.output,action)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',required=True)
    p.add_argument('--output',required=True);p.add_argument('--service',required=True)
    p.add_argument('--timeout-ms',type=int,default=60000);p.add_argument('--check-samples',type=int,default=128)
    run(p.parse_args())
