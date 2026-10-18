"""Small correctness checks; optional two-epoch GPU training on synthetic data."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from .common import ROOT


def toy_train():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('--train requires a CUDA GPU; omit it for CPU smoke tests')
    with tempfile.TemporaryDirectory(prefix='bolt-smoke-') as tmp:
        root = Path(tmp)
        for name in ('src','configs','a_route'):
            shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT / 'main_BOLT.py', root / 'main_BOLT.py')
        data = root / 'data/TOY'
        view = data / 'training/smoke'
        view.mkdir(parents=True)
        db = ['abca','abcd','abba','aabc','bbca','bbbc','caba','cabc','ccba','ccca'] * 3
        (data / 'TOY.txt').write_text('\n'.join(db) + '\n')
        chains = [['abca','abc%','ab%','a%','%'], ['bbca','bbc%','bb%','b%','%'], ['caba','cab%','ca%','c%','%']]
        def cardinality(q):
            pattern = '^' + ''.join('.*' if c=='%' else '.' if c=='_' else re.escape(c) for c in q) + '$'
            return sum(re.fullmatch(pattern, s) is not None for s in db)
        targets = {q: (cardinality(q) if i in (0,len(chain)-1) else -1) for chain in chains for i,q in enumerate(chain)}
        for name, values in [('train',list(targets.items())), ('valid', [(q,cardinality(q)) for q in ['%a%','%b%','%c%']]), ('test', [(q,cardinality(q)) for q in ['%ca','%bc','a_c%']])]:
            (view / (name+'.txt')).write_text(''.join(f'{q},{n}\n' for q,n in values))
        (view / 'chain_index.json').write_text(json.dumps([dict(chain_id=i, patterns=c, is_anchor=[j in (0,len(c)-1) for j in range(len(c))]) for i,c in enumerate(chains)]))
        env = dict(os.environ, SKIP_LEGACY_ORDER_EVAL='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        env.pop('PYTHONPATH', None)
        for k in list(env):
            if k in ('MODE','EST','DEBUG') or k.startswith(('EFFICIENCY_','BOLT_PROFILE','BOLT_CACHE_CHAIN','BOLT_VERIFY_CHAIN','BOLT_EXPECTED_CHAIN')):
                env.pop(k)
        command = [sys.executable, 'main_BOLT.py', '--data-name','TOY','--workload','smoke', '--n-epoch','2', '--ch-es','4', '--rnn-hs','8', '--pred-hs','8', '--n-pred-layer','2', '--batch-size','8', '--N','2', '--no-batch-norm', '--lambda_order','5', '--order_tolerance','.1', '--lb_min_substr_len','2', '--lb_max_substr_len','3', '--lb_top_k_strings','5', '--lb_top_k_substrings','5', '--lb_min_freq_ratio','.0001']
        log = ROOT / 'res/smoke_train.log'
        log.parent.mkdir(exist_ok=True)
        with log.open('w') as f:
            subprocess.run(command, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        outputs = list((root/'res/TOY/smoke/estimation').rglob('estimation.csv'))
        if len(outputs) != 1 or len(list(csv.DictReader(outputs[0].open()))) != 3:
            raise ValueError('Toy training did not produce all held-out predictions')
        shutil.copy2(outputs[0], ROOT/'res/smoke_predictions.csv')
        print('PASS synthetic BOLT training, checkpoint reload, and three held-out predictions', flush=True)


def run(train=False):
    subprocess.run([sys.executable, '-m', 'pytest', str(ROOT/'tests'), '-q'], cwd=ROOT, check=True)
    if train:
        toy_train()
    print('PASS smoke tests. Synthetic outputs are excluded from paper metrics.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', action='store_true', help='Also run a tiny two-epoch synthetic CUDA training job')
    a = p.parse_args()
    run(a.train)
