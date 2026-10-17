"""CPU-only aligned metrics; continuous serving and integer order stay separate."""
import csv
import math
from itertools import zip_longest
import numpy as np
from .io import require

def metrics(estimation, test):
    predictions={}
    errors=[]
    with open(estimation, newline='', encoding='utf-8') as ef, open(test, encoding='utf-8') as tf:
        truth=(line.rstrip('\r\n').rsplit(',',1) for line in tf if line.strip())
        for saved, target in zip_longest(csv.DictReader(ef), truth):
            require(saved is not None and target is not None, 'prediction row count mismatch')
            q,c=target; c=float(c); p=float(saved['est'])
            require(saved['query']==q and float(saved['true'])==c, 'prediction query/true/order mismatch')
            require(q not in predictions and all(math.isfinite(x) for x in (c,p)) and c>=0 and p>=0, 'invalid/duplicate prediction')
            predictions[q]=p
            errors.append(max(max(c,1)/max(p,1),max(p,1)/max(c,1)))
    require(bool(errors), 'empty test')
    return dict(q_avg=float(np.mean(errors)), **{k:float(np.quantile(errors,p)) for k,p in
        [('q50',.5),('q95',.95),('q99',.99),('q100',1.)]}, test_rows=len(errors)), predictions

def compare_metrics(actual, expected):
    for name in ('q_avg','q50','q95','q99','q100','test_rows'):
        if name in expected:
            require(math.isclose(actual[name],float(expected[name]),rel_tol=1e-8,abs_tol=1e-8), 'metric mismatch: '+name)

def integer_order(chains,predictions):
    total=passed=ties=0
    for chain in chains:
        for a,b in zip(chain['patterns'],chain['patterns'][1:]):
            x,y=round(predictions[a]),round(predictions[b])
            total+=1;passed+=x<=y;ties+=x==y
    return dict(integer_pair_total=total,integer_order=passed/total if total else None,
                integer_tie_rate=ties/total if total else None)

def clip_values(true,pred,lower,upper):
    require(all(math.isfinite(v) for v in (true,pred,lower,upper)), 'nonfinite bound')
    require(0<=true<=upper and lower<=true and 0<=pred<=upper+1e-5*max(upper,1), 'unsafe bound/prediction')
    return max(pred,min(upper,max(0,lower)))
