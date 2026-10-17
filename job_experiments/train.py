"""Frozen, serial JOB column training; shared resource locks with A/B routes."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .protocol import ROOT,COLUMNS,read_json,write_json,sha,verify,seal,require
from .cli import staged


def commands(python=sys.executable):
    result=[]
    # Smallest column first is an engineering gate, not selection by test error.
    for column in ('link_type.link','movie_companies.note','title.title','name.name'):
        for budget in ('A100','A0'):
            for method in ('CLIQUE','BIMO'):
                task=column+'_'+budget+'_'+method+'_S0'
                entry='main_CLIQUE.py' if method=='CLIQUE' else 'main_CLIQUE_bound_mono.py'
                command=[python,entry,'-d',column,'-w',budget+'_'+method,'--seed','0','--tag','JOB/'+task]
                if method=='BIMO': command+=['--lambda_mono','1.0','--mono_tolerance','0.1','--chain_sample_ratio','1.0']
                result.append(dict(id=task,column=column,budget=budget,method=method,command=command))
    return result


def freeze(prepared,views,output):
    from a_route.training import transform,FILES
    prepared,views=Path(prepared).resolve(),Path(views).resolve()
    pinfo=verify(prepared,'PREPARED.json');vinfo=verify(views,'VIEWS.json')
    require(vinfo['prepared_sha256']==sha(prepared/'PREPARED.json'),'views do not match data')
    cfg=read_json(prepared/'config.json')
    require(cfg['budgets']=={'A100':1.0,'A0':0.0},'training expects the two frozen budgets')
    def action(out):
        work=out/'workspace';work.mkdir()
        for folder in ('src','configs','a_route'):
            shutil.copytree(ROOT/folder,work/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('main_CLIQUE.py','main_CLIQUE_bound_mono.py'):
            shutil.copyfile(ROOT/name,work/name)
        for name in FILES:
            path=work/name;path.write_text(transform(path.read_text()))
        for col in COLUMNS:
            data=work/'data'/col; data.mkdir(parents=True)
            (data/(col+'.txt')).symlink_to(prepared/col/'column.txt')
            (data/'training').mkdir()
            for budget in ('A100','A0'):
                for method in ('CLIQUE','BIMO'):
                    source=views/col/(budget+'_'+method)
                    shutil.copytree(source,data/'training'/source.name)
        tasks=commands()
        batch_sizes={}
        for col in COLUMNS:
            lengths=[]
            for budget in ('A100','A0'):
                with (views/col/(budget+'_CLIQUE')/'train.txt').open() as f:
                    lengths.append(sum(1 for _ in f))
            require(min(lengths)>=2, 'not enough labeled examples for BatchNorm: '+col)
            choices=[b for b in range(64,1,-1) if all(n%b != 1 for n in lengths)]
            require(choices, 'no common non-singleton batch size: '+col)
            batch_sizes[col]=choices[0]
        for task in tasks:
            task['command']+=['--batch-size',str(batch_sizes[task['column']])]
        manifest=dict(tasks=tasks,batch_sizes=batch_sizes,prepared=str(prepared),prepared_sha256=sha(prepared/'PREPARED.json'),
            views=str(views),views_sha256=sha(views/'VIEWS.json'),columns=pinfo['columns'],
            evaluation_mode='validation_only_preserve_loader_rng',timing_usable=False,
            hyperparameters='inherited CLIQUE defaults and four-column overrides; BIMO lambda=1,tau=.1,r=1')
        seal(out,'TRAIN_READY.json',manifest)
        return manifest
    return staged(output,action)


def finalize_task(folder, frozen, task):
    import yaml
    from a_route.evaluate import metrics
    from .prediction import positive_predictions,selectivity
    work=folder/'workspace'
    target=folder/'tasks'/task['id']
    require(read_json(target/'command.json')==task, 'task command changed')
    base=work/'res'/task['column']/(task['budget']+'_'+task['method'])
    configs=list((base/'model'/'JOB'/task['id']).glob('*/0/config.yml'))
    require(len(configs)==1,'ambiguous output model')
    model_dir=configs[0].parent;model_name=model_dir.parent.name
    est=base/'estimation'/'JOB'/task['id']/model_name/'0'/'estimation.csv'
    stat=base/'stat'/'JOB'/task['id']/model_name/'0'/'stat.yml'
    config=yaml.safe_load(configs[0].read_text())
    require(config['seed']==0 and config['data_name']==task['column'] and config['workload']==task['budget']+'_'+task['method'],'model identity mismatch')
    test=work/'data'/task['column']/'training'/config['workload']/'test.txt'
    scores,preds=metrics(est,test)
    exact=read_json(test.parent/'exact_test.json')['patterns']
    merged=positive_predictions(list(preds)+list(exact),preds,frozen['columns'][task['column']])
    outputs={q:dict(cardinality=p,selectivity=selectivity(p,frozen['columns'][task['column']]['rows'],frozen['columns'][task['column']]['nonnull'])) for q,p in merged.items()}
    write_json(target/'positive_predictions.json',outputs)
    import torch
    checkpoint=torch.load(model_dir/'model.pth',map_location='cpu',weights_only=True)
    require(isinstance(checkpoint,dict) and checkpoint and all(torch.isfinite(v).all().item() for v in checkpoint.values()), 'invalid checkpoint tensors')
    recorded=yaml.safe_load(stat.read_text())
    import math
    require(math.isclose(scores['q_avg'],float(recorded['avg']),rel_tol=1e-7,abs_tol=1e-9),'stat/prediction mismatch')
    files=[configs[0],model_dir/'model.pth',est,stat,target/'positive_predictions.json']
    write_json(target/'DONE.json',dict(task=task,metrics=scores,files={str(p):sha(p) for p in files},timing_usable=False))

def run(folder, resume=False):
    import yaml
    from a_route.scheduler import locks,GLOBAL_LOCKS,TIMING_LOCKS
    from a_route.evaluate import metrics
    from .prediction import positive_predictions,selectivity
    folder=Path(folder).resolve();frozen=verify(folder,'TRAIN_READY.json');work=folder/'workspace'
    require(sha(Path(frozen['prepared'])/'PREPARED.json')==frozen['prepared_sha256'],'changed preparation')
    require(sha(Path(frozen['views'])/'VIEWS.json')==frozen['views_sha256'],'changed data views')
    verify(Path(frozen['prepared']),'PREPARED.json');verify(Path(frozen['views']),'VIEWS.json')
    previous=folder/'status.json'
    if previous.exists():
        require(resume,'existing queue state requires --resume')
        prior=read_json(previous)
        proc=Path('/proc')/str(prior['pid'])/'stat'
        require(not proc.exists() or proc.read_text().split()[2]=='Z', 'prior dispatcher still exists')
    with locks(GLOBAL_LOCKS+TIMING_LOCKS):
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        require(not gpu.stdout.strip(),'external GPU process active; refusing overlapping training')
        status={'status':'RUNNING','pid':os.getpid(),'completed':[],'total':len(frozen['tasks'])}
        try:
            for task in frozen['tasks']:
                # Hash only frozen source/input files, not generated caches/results.
                verify(folder,'TRAIN_READY.json')
                target=folder/'tasks'/task['id']
                if target.exists():
                    require(resume,'task attempt exists; explicit reconciliation needed')
                    if (target/'DONE.json').exists():
                        done=read_json(target/'DONE.json')
                        require(done['task']==task and all(sha(p)==v for p,v in done['files'].items()), 'completed task changed')
                    else:
                        require((target/'PROCESS_SUCCESS.json').exists(), 'unconfirmed attempt; refuse to relaunch')
                        receipt=read_json(target/'PROCESS_SUCCESS.json')
                        require(receipt['exit_code']==0 and receipt['command']==task['command'], 'invalid process success receipt')
                        finalize_task(folder,frozen,task)
                    status['completed'].append(task['id']);write_json(folder/'status.json',status)
                    continue
                target.mkdir(parents=True);write_json(target/'command.json',task)
                status['current']=task['id'];write_json(folder/'status.json',status)
                env={k:v for k,v in os.environ.items() if k not in {'MODE','EST','DEBUG','PYTHONPATH','EXPERIMENT_PROTOCOL_ID','FORMAL_RUN_ID'} and not k.startswith(('EFFICIENCY_','BIMO_'))}
                env.update(CUDA_VISIBLE_DEVICES='0',SKIP_LEGACY_MONO_EVAL='1',PYTHONUNBUFFERED='1',
                           PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                with (target/'stdout.log').open('w') as log:
                    proc=subprocess.Popen(task['command'],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
                    write_json(target/'process.json',dict(pid=proc.pid,started_unix=time.time()))
                    rc=proc.wait()
                    write_json(target/'PROCESS_SUCCESS.json' if rc==0 else target/'PROCESS_FAILED.json',dict(exit_code=rc,command=task['command'],pid=proc.pid))
                    require(rc==0,'training failed: '+task['id'])
                finalize_task(folder,frozen,task)
                status['completed'].append(task['id']);write_json(folder/'status.json',status)
            status['status']='COMPLETE';status.pop('current',None);write_json(folder/'status.json',status)
            write_json(folder/'TRAIN_COMPLETE.json',status)
        except BaseException as exc:
            status['status']='FAILED';status['error_type']=type(exc).__name__;write_json(folder/'status.json',status)
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze');f.add_argument('--prepared',required=True);f.add_argument('--views',required=True);f.add_argument('--output',required=True)
    r=sub.add_parser('run');r.add_argument('--folder',required=True);r.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.action=='freeze':freeze(a.prepared,a.views,a.output)
    else:run(a.folder,a.resume)
