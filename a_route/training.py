"""Auditable source transformation applied ONLY to an A-route code snapshot.

The historical deterministic test forward consumed one DataLoader base seed.
We preserve that RNG transition with an empty sequential loader; no test rows,
bounds or predictions are accessed during optimization. Tests execute all five
actual build_model paths on small CPU data before this transform is admitted.
"""
import ast
from .io import require

FILES=('src/CLIQUE.py','src/CLIQUE_bound.py','src/CLIQUE_mono.py','src/CLIQUE_bound_mono.py')

def skip_test(loader, model):
    import torch
    from torch.utils.data import SequentialSampler
    require(loader.num_workers==0 and loader.generator is None and isinstance(loader.sampler,SequentialSampler),
            'unsupported test-loader RNG contract')
    require(len(loader.dataset)==0, 'training test loader must be empty')
    model.eval()
    # Creating this iterator consumes the same single CPU RNG draw as the legacy
    # deterministic evaluation. Do not fetch examples or synthesize test metrics.
    iter(loader)


def transform(source):
    tree=ast.parse(source)
    methods=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='build_model']
    require(len(methods)==1,'expected exactly one build_model')
    method=methods[0]
    edits=[]
    # Drop test bound lookups, while keeping construction of an empty loader.
    for n in ast.walk(method):
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('test_data','test_data_bounded'):
            require(isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='append_bound', 'unexpected test preparation')
            edits.append((n.lineno,n.end_lineno,' '*n.col_offset+n.targets[0].id+' = []'))
    loops=[n for n in ast.walk(method) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch']
    require(len(loops)==1,'expected one epoch loop')
    body=loops[0].body
    starts=[i for i,n in enumerate(body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
            and any(isinstance(a,ast.Name) and a.id=='dl_test' for a in n.value.args)]
    require(len(starts)==1,'expected one epoch test evaluation')
    start=starts[0]
    ends=[i for i in range(start,len(body)) if isinstance(body[i],ast.If)
          and any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='best_test_score' for t in n.targets) for n in ast.walk(body[i]))]
    require(len(ends)==1,'unexpected best-test block')
    indent=' '*body[start].col_offset
    replacement=indent+'_a_route_skip_test(dl_test, model)\n'+indent+'test_loss = test_q_avg = test_q_max = None'
    edits.append((body[start].lineno,body[ends[0]].end_lineno,replacement))
    require(len(edits)==2,'test preparation shape changed')
    lines=source.splitlines()
    for first,last,text in sorted(edits,reverse=True):
        lines[first-1:last]=text.splitlines()
    result='from a_route.training import skip_test as _a_route_skip_test\n'+'\n'.join(lines)+'\n'
    ast.parse(result)
    return result
