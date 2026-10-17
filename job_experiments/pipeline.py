"""Authorized JOB import -> chain preparation -> counted views -> serial training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .protocol import ROOT,read_json,write_json,require,sha


def process_live(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.exists() and p.read_text().split()[2]!='Z'


def resume_training(args):
    """Resume only the frozen training stage; preserve the failed attempt."""
    import shutil
    from .protocol import verify
    folder=Path(args.folder).resolve()
    status_path=folder/'pipeline_status.json'
    prior=read_json(status_path)
    require(prior['status']=='FAILED' and prior['phase']=='TRAIN', 'only failed TRAIN may resume')
    require(prior['completed']==['IMPORT','EXPORT','PREPARE','COUNT','ASSEMBLE','FREEZE_TRAIN'], 'incomplete preparation')
    require(not process_live(prior['pid']) and not process_live(prior.get('child_pid',prior['pid'])), 'previous pipeline still active')
    verify(folder/'training','TRAIN_READY.json')
    archive=folder/'attempt_history'/('train_resume_'+str(time.time_ns()))
    archive.mkdir(parents=True)
    for name in ('pipeline_status.json','TRAIN.log','TRAIN.command.json','training/status.json'):
        source=folder/name
        if source.exists():
            dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    command=[sys.executable,'-u','-m','job_experiments.train','run','--folder',str(folder/'training'),'--resume']
    write_json(archive/'RESUME.json',dict(command=command,source_sha256={str(p):sha(p) for p in sorted((ROOT/'job_experiments').glob('*.py'))},training_manifest_sha256=sha(folder/'training/TRAIN_READY.json')))
    state=dict(status='RUNNING',phase='TRAIN',pid=os.getpid(),completed=prior['completed'],started_unix=prior['started_unix'],resumed_unix=time.time(),resume_record=str(archive/'RESUME.json'))
    write_json(status_path,state)
    try:
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
        with (folder/'TRAIN.resume.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;write_json(status_path,state)
            require(child.wait()==0,'stage failed: TRAIN resume')
        state['completed'].append('TRAIN');state.pop('child_pid',None)
        state.update(status='COMPLETE',phase='TRAINING_COMPLETE',completed_unix=time.time())
        write_json(status_path,state);write_json(folder/'PIPELINE_COMPLETE.json',state)
    except BaseException as exc:
        state.update(status='FAILED',error=str(exc),failed_unix=time.time())
        write_json(status_path,state)
        raise


def run(args):
    folder=Path(args.folder).resolve(); status_path=folder/'pipeline_status.json'
    require(not status_path.exists(),'pipeline already exists; reconcile rather than duplicate')
    folder.mkdir(parents=True,exist_ok=True)
    state=dict(status='RUNNING',phase='WAIT_DOWNLOAD',pid=os.getpid(),completed=[],started_unix=time.time())
    sources={str(p):sha(p) for p in sorted((ROOT/'job_experiments').glob('*.py'))}
    sources.update({str(ROOT/p):sha(ROOT/p) for p in ('src/util.py','src/gen_query_chain.py','likeBench/imdb_schema.sql','likeBench/imdb_index.sql')})
    write_json(folder/'PIPELINE_CONTRACT.json',dict(source_sha256=sources,service=args.service,
        planned_training_cells=16,training_seed=0,started_unix=time.time(),timing_usable=False))
    write_json(status_path,state)
    def stage(name,mod,options):
        require(all(sha(p)==v for p,v in sources.items()),'pipeline source changed since launch')
        state['phase']=name;write_json(status_path,state)
        cmd=[sys.executable,'-u','-m',mod,*map(str,options)]
        write_json(folder/(name+'.command.json'),dict(command=cmd))
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (folder/(name+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;write_json(status_path,state)
            require(child.wait()==0,'stage failed: '+name)
        state['completed'].append(name);state.pop('child_pid',None);write_json(status_path,state)
    try:
        download=folder/'import'/'DOWNLOAD.json'
        started=time.monotonic()
        while not download.exists():
            require(process_live(args.download_pid),'download process stopped without verified completion')
            require(time.monotonic()-started<7200,'download wait exceeded two hours')
            prefix=ROOT/'data/JOB_SOURCE/imdb.tgz.part'
            parts=list(prefix.parent.glob('imdb.range.*'))
            state['download_bytes']=prefix.stat().st_size+sum(p.stat().st_size for p in parts)
            state['download_expected_bytes']=1263193115;write_json(status_path,state)
            time.sleep(10)
        receipt=read_json(download)
        archive=ROOT/'data/JOB_SOURCE/imdb.tgz'
        require(sha(archive)==receipt['sha256'],'download archive changed')
        stage('IMPORT','job_experiments.import_job',['--service',args.service,'--archive',archive,'--output',folder/'database_import'])
        stage('EXPORT','job_experiments.cli',['export','--service',args.service,'--timeout-ms','300000','--output',folder/'assets'])
        stage('PREPARE','job_experiments.cli',['prepare','--config',folder/'plan/config.json','--sql',folder/'plan/job.sql','--assets',folder/'assets','--output',folder/'prepared'])
        require(read_json(folder/'prepared/PREPARED.json')['status']=='PREPARED','insufficient disjoint chains; inspect preparation statistics')
        stage('COUNT','job_experiments.batch_count',['--service',args.service,'--prepared',folder/'prepared','--output',folder/'counts'])
        stage('ASSEMBLE','job_experiments.cli',['assemble','--prepared',folder/'prepared','--counts',folder/'counts','--output',folder/'views'])
        stage('FREEZE_TRAIN','job_experiments.train',['freeze','--prepared',folder/'prepared','--views',folder/'views','--output',folder/'training'])
        stage('TRAIN','job_experiments.train',['run','--folder',folder/'training'])
        state['status']='COMPLETE';state['phase']='TRAINING_COMPLETE';state['completed_unix']=time.time()
        write_json(status_path,state)
        write_json(folder/'PIPELINE_COMPLETE.json',state)
    except BaseException as exc:
        state['status']='FAILED';state['error']=str(exc);state['failed_unix']=time.time()
        write_json(status_path,state)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',required=True)
    p.add_argument('--service');p.add_argument('--download-pid',type=int);p.add_argument('--resume-training',action='store_true')
    args=p.parse_args()
    if args.resume_training: resume_training(args)
    else:
        require(args.service and args.download_pid is not None,'service and download-pid required for a new pipeline')
        run(args)
