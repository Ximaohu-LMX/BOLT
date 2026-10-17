"""Independent label, construction, CPU inference, and summary-build measurements."""
import argparse
import csv
import gzip
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from .common import ROOT, DATASETS, tasks, read_json, atomic_json, digest, identity, rows, labeled_rows, write_csv, verify_files
from .runner import lock, completed, contract, environment


def measurement(name, spec, action, force=False):
    folder = ROOT / "res/costs" / name
    folder.mkdir(parents=True, exist_ok=True)
    spec = dict(spec, code={str(p.relative_to(ROOT)): digest(p) for p in [*sorted((ROOT / "repro").glob("*.py")), *sorted((ROOT / "src").rglob("*.py"))]}, environment=environment())
    key = identity(spec)
    receipt = folder / "DONE.json"
    if not force and completed(receipt, key):
        print("SKIP verified", name, flush=True)
        return read_json(receipt)["result"]
    attempt = folder / ("attempt_" + str(time.time_ns()))
    attempt.mkdir()
    print("MEASURE", name, attempt.relative_to(ROOT), flush=True)
    result, files = action(attempt)
    atomic_json(attempt / "result.json", result)
    files = list(files) + [attempt / "result.json"]
    atomic_json(receipt, dict(status="COMPLETE", identity=key, contract=spec, result=result,
                outputs={str(p.relative_to(ROOT)): digest(p) for p in files}))
    return result


def load_model(task):
    import torch
    import src.util as ut
    from src.CLIQUE import CLIQUEestimator
    from src.CLIQUE_bound_mono import CLIQUEestimator_bound_mono
    c = read_json(ROOT / task["config"])
    db = ut.read_strings(str(ROOT / f'data/{task["dataset"]}/{task["dataset"]}.txt'))
    char_dict, _ = ut.gen_char_dict(db)
    cls = CLIQUEestimator if task["method"] == "CLIQUE" else CLIQUEestimator_bound_mono
    model = cls()
    count = ROOT / f'res/{task["dataset"]}/Ntable/count_N{c["N"]}_ME{c["max_entry_ratio"]}.pkl'
    checkpoint = ROOT / task["checkpoint"]
    lb = ROOT / f'res/{task["dataset"]}/freq_library/lib_s500_sub200_len5-15_freq0.0001_full.pkl'
    model.set_params(N=c['N'], PT=c['PT'], seed=c['seed'], char_dict=char_dict, ch_es=c['ch_es'], n_rnn=1,
        packed=c['packed'], rnn_hs=c['rnn_hs'], pred_hs=c['pred_hs'], n_pred_layer=c['n_pred_layer'],
        lr=c['learning_rate'], l2=c['l2'], n_epoch=c['n_epoch'], bs=c['batch_size'], patience=c['patience'],
        last_flip=True, sw=None, count_path=str(count), model_path=str(checkpoint), bound_in=c['bound_in'],
        pack_all=c['pack_all'], dynamicPT=c['dynamicPT'], max_entry_ratio=c['max_entry_ratio'], batch_norm=c['batch_norm'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.n_db = len(db)
    with count.open('rb') as f:
        model.hashtable, model.PTs = pickle.load(f)
    if task['method'] == 'BOLT':
        with lb.open('rb') as f:
            model.substr_library = pickle.load(f)
    model.eval()
    return model, checkpoint, count, lb


def inference(datasets, force=False):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    available = sorted(os.sched_getaffinity(0))
    cpu = available[-1]
    os.sched_setaffinity(0, {cpu})
    result = []
    try:
        for t in tasks():
            if t['dataset'] not in datasets or t['budget'] != 'A100' or t['method'] not in ('CLIQUE', 'BOLT') or t['seed'] != 0:
                continue
            def action(out):
                model, model_path, count_path, lb_path = load_model(t)
                queries = [q for q,c in labeled_rows(ROOT / f'data/{t["dataset"]}/training/{t["workload"]}/test.txt')]
                latencies = []
                with torch.inference_mode():
                    for q in queries[:100]:
                        model.estimate([q], to_cuda=False, is_tqdm=False)
                    for q in queries:
                        begin = time.perf_counter_ns()
                        v = model.estimate([q], to_cuda=False, is_tqdm=False)
                        elapsed = time.perf_counter_ns() - begin
                        if len(v) != 1 or not np.isfinite(float(v[0])):
                            raise ValueError('Invalid measured prediction')
                        latencies.append(elapsed)
                path = out / 'latency.csv.gz'
                with gzip.open(path, 'wt') as f:
                    w = csv.writer(f); w.writerow(['query_index', 'latency_ns']); w.writerows(enumerate(latencies))
                r = dict(dataset=t['dataset'], method=t['method'], queries=len(queries), warmup=100, passes=1, device='CPU', threads=1, cpu=cpu,
                         p50_ms=float(np.quantile(latencies, .5)/1e6), p95_ms=float(np.quantile(latencies, .95)/1e6),
                         model_bytes=model_path.stat().st_size, ub_bytes=count_path.stat().st_size,
                         lb_bytes=lb_path.stat().st_size if t['method']=='BOLT' else 0,
                         source='published checkpoint; newly measured latency')
                return r, [path]
            result.append(measurement('inference_' + t['id'], dict(task=contract(t), checkpoint_sha256=digest(ROOT / t['checkpoint']), cpu=cpu), action, force))
    finally:
        os.sched_setaffinity(0, available)
    write_csv(ROOT / 'paper_results/generated/new/tables/Table12_inference.csv', result)


def construction(datasets, force=False, sensitivity=False):
    from .worker import command_for
    from .metrics import evaluate
    records = []
    cases = []
    by_id = {t['id']: t for t in tasks()}
    for ds in datasets:
        if sensitivity:
            if ds == 'IMDB':
                continue
            for gamma in (0, .25, .5, 1):
                task = by_id[f'{ds}_A1_' + ('UB_LB' if gamma == 0 else 'BOLT') + '_S0']
                cases.append((task, gamma, 'warm', 0, 'sensitivity'))
        else:
            for budget, method in [('A100','CLIQUE'), ('A1','BOLT'), ('A0','BOLT')]:
                task = by_id[f'{ds}_{budget}_{method}_S0']
                gamma = 1 if method == 'CLIQUE' else (.5 if ds == 'DBLP' and budget == 'A0' else .25)
                for cache, repeats in [('cold',3), ('warm',2)]:
                    cases.extend((task, gamma, cache, repeat, 'construction') for repeat in range(repeats))
            cases.extend((by_id[f'{ds}_A1_BOLT_S0'], 1, 'warm', repeat, 'fullchain') for repeat in range(2))
    for task, gamma, cache, repeat, kind in cases:
        name = f'{kind}_{task["id"]}_g{gamma}_{cache}_r{repeat}'
        spec = dict(task=contract(task), gamma=gamma, cache=cache, repeat=repeat, protocol='wall minus measured post-training phases')
        def action(out):
            cache_root = out / 'summary_cache'
            cache_root.mkdir()
            if cache == 'warm':
                for dirname in ('Ntable','freq_library'):
                    shutil.copytree(ROOT / 'res' / task['dataset'] / dirname, cache_root / task['dataset'] / dirname)
            tag = 'BOLT_COST/' + name + '/' + out.name
            command = command_for(task, tag, overrides={'chain_sample_ratio': gamma})
            env = dict(os.environ)
            for key in list(env):
                if key in ('MODE','EST','DEBUG','PYTHONPATH') or key.startswith(('EFFICIENCY_','BIMO_')):
                    env.pop(key)
            phase = out / 'phases.jsonl'
            env.update(PYTHONHASHSEED='0', PYTHONUNBUFFERED='1', SKIP_LEGACY_MONO_EVAL='1', EFFICIENCY_PROFILE='1', EFFICIENCY_TIMING_FILE=str(phase), EFFICIENCY_CACHE_ROOT=str(cache_root), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            atomic_json(out / 'command.json', dict(argv=command, cache_protocol=cache, gamma=gamma))
            before = time.perf_counter()
            with (out / 'train.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            wall = time.perf_counter() - before
            events = [__import__('json').loads(line) for line in phase.read_text().splitlines() if line.strip()]
            train = next(e for e in events if e['phase'].startswith('train_or_model_load'))
            final = next(e for e in reversed(events) if e['phase']=='result_persistence_complete')
            nonlabel = wall - (final['elapsed_since_start_seconds'] - train['elapsed_since_start_seconds'])
            base = ROOT / f'res/{task["dataset"]}/{task["workload"]}'
            configs = list((base/'model'/tag).glob('*/0/config.yml'))
            if len(configs) != 1:
                raise ValueError('Missing measured model')
            suffix = configs[0].parent.relative_to(base / 'model')
            predictions = base / 'estimation' / suffix / 'estimation.csv'
            values = evaluate(task, predictions)
            record = dict(dataset=task['dataset'], method=task['method'], budget=task['budget'], gamma=gamma, cache=cache, repeat=repeat,
                          kind=kind, wall_seconds=wall, nonlabel_build_seconds=nonlabel, **values)
            return record, [phase, predictions, configs[0], configs[0].with_name('model.pth'), out/'command.json']
        records.append(measurement(name, spec, action, force))
    target = 'S23_sensitivity' if sensitivity else 'construction_repetitions'
    write_csv(ROOT / 'paper_results/generated/new/tables' / (target+'.csv'), records)
    if not sensitivity:
        import numpy as np
        groups = {}
        for r in records:
            key = tuple(r[k] for k in ('dataset','method','budget','gamma','cache','kind'))
            groups.setdefault(key, []).append(r)
        write_csv(ROOT / 'paper_results/generated/new/tables/Table11_construction.csv', [dict(zip(('dataset','method','budget','gamma','cache','kind'),k), runs=len(v), seconds_median=float(np.median([x['nonlabel_build_seconds'] for x in v])), q_avg=float(np.median([x['q_avg'] for x in v]))) for k,v in groups.items()])


def summary_child(dataset, component, output):
    import src.util as ut
    from src.lower_bound import build_freq_string_library
    db_path = ROOT / f'data/{dataset}/{dataset}.txt'
    db = ut.read_strings(str(db_path))
    ratio = dict(AUTHOR=4.8, DBLP=3.12, IMDB=1.675, WIKI=4.35)[dataset]
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[-1]})
    before = time.perf_counter()
    if component == 'UB':
        result = list(ut.filter_infrequent_entry_max_entry_each_len(ut.extendedNgramTable(db, 5, 1), ratio))
        reference = ROOT / f'res/{dataset}/Ntable/count_N5_ME{ratio}.pkl'
    else:
        result = build_freq_string_library(db, top_k_strings=500, top_k_substrings=200, min_substr_len=5, max_substr_len=15, min_freq_ratio=.0001, full_scan=True, seed=42)
        reference = ROOT / f'res/{dataset}/freq_library/lib_s500_sub200_len5-15_freq0.0001_full.pkl'
    output = Path(output)
    with output.with_suffix('.pkl').open('wb') as f:
        pickle.dump(result, f)
    elapsed = time.perf_counter() - before
    with reference.open('rb') as f:
        equal = result == pickle.load(f)
    atomic_json(output, dict(dataset=dataset, component=component, status='COMPLETE' if equal else 'CONTENT_MISMATCH', seconds=elapsed, content_equal=equal,
                            bytes=output.with_suffix('.pkl').stat().st_size, cpu=list(os.sched_getaffinity(0)), protocol='resident column; includes serialization, excludes input read and equality comparison'))
    if not equal:
        raise ValueError('Fresh summary differs from the frozen summary')


def summaries(datasets, force=False):
    records = []
    for ds in datasets:
        for component in ('LB','UB'):
            timeout = 180 if component == 'LB' else 300
            def action(out):
                target = out / 'summary.json'
                with (out/'build.log').open('w') as log:
                    try:
                        subprocess.run([sys.executable, '-m', 'repro.costs', '--child', ds, component, str(target)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
                        value = read_json(target)
                        return value, [target, target.with_suffix('.pkl'), out/'build.log']
                    except subprocess.TimeoutExpired:
                        # A completed bounded attempt is evidence, not a zero-second build.
                        value = dict(dataset=ds, component=component, status='TIMEOUT', limit_seconds=timeout)
                        return value, [out/'build.log']
            records.append(measurement(f'summary_{ds}_{component}', dict(dataset_sha256=digest(ROOT/f'data/{ds}/{ds}.txt'), component=component, timeout=timeout), action, force))
    write_csv(ROOT / 'paper_results/generated/new/tables/S30_summaries.csv', records)


def count_batch(spec):
    service, table, queries = spec
    import psycopg2
    from psycopg2 import sql
    conn = psycopg2.connect(service=service)
    conn.autocommit = True
    results = []
    with conn.cursor() as cur:
        cur.execute('SET max_parallel_workers_per_gather=0')
        cur.execute('SET jit=off')
        for query, expected in queries:
            before = time.perf_counter()
            cur.execute(sql.SQL('SELECT count(*) FROM {}.{} WHERE value LIKE %s').format(sql.Identifier('bolt_labels'), sql.Identifier(table)), (query,))
            actual = cur.fetchone()[0]
            seconds = time.perf_counter() - before
            if actual != expected:
                raise ValueError(f'PostgreSQL label mismatch: {table} {query!r}: {actual} != {expected}')
            results.append((query, actual, seconds))
    conn.close()
    return results


def labels(datasets, force=False):
    import psycopg2
    from psycopg2 import sql
    from concurrent.futures import ProcessPoolExecutor
    import src.util as ut
    from .postgres import setup
    folder, service = setup()
    results = []
    for ds in datasets:
        raw = ROOT / f'data/{ds}/{ds}.txt'
        stamp = folder / f'labels_{ds}.json'
        raw_sha = digest(raw)
        conn = psycopg2.connect(service=service)
        if not stamp.exists():
            with tempfile.TemporaryFile(mode='w+') as csvfile:
                w = csv.writer(csvfile, lineterminator='\n')
                w.writerows((v,) for v in ut.read_strings(str(raw)))
                csvfile.seek(0)
                with conn, conn.cursor() as cur:
                    cur.execute('CREATE SCHEMA IF NOT EXISTS bolt_labels')
                    cur.execute(sql.SQL('CREATE TABLE {}.{} (value text NOT NULL)').format(sql.Identifier('bolt_labels'), sql.Identifier(ds)))
                    cur.copy_expert(sql.SQL('COPY {}.{} FROM STDIN WITH (FORMAT csv)').format(sql.Identifier('bolt_labels'),sql.Identifier(ds)).as_string(conn), csvfile)
                    cur.execute(sql.SQL('ANALYZE {}.{}').format(sql.Identifier('bolt_labels'),sql.Identifier(ds)))
            atomic_json(stamp, dict(raw_sha256=raw_sha))
        if read_json(stamp)['raw_sha256'] != raw_sha:
            raise ValueError('Label database is stale; use a fresh res/postgres directory')
        conn.close()
        for budget in ('A100','A1','A0','validation'):
            path = ROOT / f'data/{ds}/training/{"A100" if budget=="validation" else budget}_CLIQUE/{"valid" if budget=="validation" else "train"}.txt'
            query_list = list(labeled_rows(path))
            def action(out):
                chunks = [(service, ds, query_list[i::8]) for i in range(8)]
                records = []
                for repeat in range(-1, 3):
                    before = time.perf_counter()
                    with ProcessPoolExecutor(max_workers=8) as pool:
                        counted = [r for group in pool.map(count_batch, chunks) for r in group]
                    elapsed = time.perf_counter() - before
                    write_csv(out / f'queries_{repeat}.csv', [dict(query=q, true=n, seconds=s) for q,n,s in counted])
                    records.append(dict(dataset=ds, budget=budget, repeat=repeat, phase='warmup' if repeat<0 else 'measured', seconds=elapsed, queries=len(counted), workers=8))
                return records, list(out.glob('queries_*.csv'))
            value = measurement(f'labels_{ds}_{budget}', dict(raw=raw_sha, queries=digest(path), workers=8, warmups=1, repeats=3, postgres=digest(folder/'pg/bin/postgres')), action, force)
            results.extend(value)
    write_csv(ROOT / 'paper_results/generated/new/tables/S24_label_repetitions.csv', results)
    import numpy as np
    write_csv(ROOT / 'paper_results/generated/new/tables/Table10_labels.csv', [dict(dataset=ds, budget=budget, median_seconds=float(np.median([r['seconds'] for r in results if r['dataset']==ds and r['budget']==budget and r['phase']=='measured']))) for ds in datasets for budget in ('A100','A1','A0','validation')])


def run_all(datasets=DATASETS, force=False, phase='all'):
    from .datasets import ensure
    ensure(datasets, ['workloads','summaries'] + (['checkpoints'] if phase in ('all','inference') else []), raw=True)
    with lock(ROOT / 'res/.training.lock'), lock(ROOT / 'res/.timing.lock'):
        for name, action in [('inference', inference), ('labels', labels), ('construction', construction), ('summaries', summaries)]:
            if phase in ('all', name):
                action(datasets, force)
        if phase in ('all','sensitivity'):
            construction(datasets, force, sensitivity=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['all','labels','construction','inference','summaries','sensitivity'], default='all')
    p.add_argument('--dataset', '--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    p.add_argument('--force', action='store_true')
    p.add_argument('--child', nargs=3, metavar=('DATASET','COMPONENT','OUTPUT'))
    a = p.parse_args()
    summary_child(*a.child) if a.child else run_all(a.dataset, a.force, a.phase)


if __name__ == '__main__':
    main()
