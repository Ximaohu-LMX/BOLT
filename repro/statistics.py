"""Dataset, supervision-budget and external-target statistics from frozen inputs."""
import argparse
import numpy as np
from .common import ROOT, DATASETS, read_json, labeled_rows, atomic_json, write_csv


def generate(datasets=DATASETS):
    from .datasets import ensure
    ensure(datasets, ['workloads'], raw=True)
    from .prepare import audit
    budgets = audit(datasets)
    out = ROOT / 'paper_results/generated/data'
    write_csv(out/'Table04_S01_budgets.csv', budgets)
    columns, distributions, shifts, external = [], [], [], []
    for ds in datasets:
        with (ROOT/f'data/{ds}/{ds}.txt').open() as f:
            values = [s.rstrip('\n') for s in f]
        columns.append(dict(dataset=ds, rows=len(values), alphabet=len(set(''.join(values))), mean_string_length=float(np.mean([len(s) for s in values]))))
        del values
        base = ROOT/f'data/{ds}/training'
        info = read_json(base/'FULL_DISJOINT_V1/split_info.json')
        for split in ('train','valid','test'):
            values = list(labeled_rows(base/f'A100_BOLT/{split}.txt'))
            chains = read_json(base/'FULL_DISJOINT_V1/chain_index.json') if split == 'train' else info[split+'_chain_index']
            patterns = [q for q,c in values]
            # Final WIKI exclusion is already reflected in the public test file.
            allowed = set(patterns)
            chain_lengths = [sum(q in allowed for q in c['patterns']) for c in chains]
            unique_queries = len(patterns)
            patterns = [q for c in chains for q in c["patterns"] if q in allowed]
            distributions.append(dict(dataset=ds, split=split, unique_queries=unique_queries, chains=len(chains),
                mean_chain_length=float(np.mean(chain_lengths)), mean_pattern_length=float(np.mean([len(q) for q in patterns])),
                mean_literal_length=float(np.mean([len(q.replace('%','').replace('_','')) for q in patterns])),
                underscore_pct=float(np.mean(['_' in q for q in patterns])*100)))
        for budget in ('A100','A10','A1','A0'):
            cards = [c for q,c in labeled_rows(base/f'{budget}_CLIQUE/train.txt')]
            shifts.append(dict(dataset=ds,budget=budget,mean=float(np.mean(cards)),q50=float(np.quantile(cards,.5)),q95=float(np.quantile(cards,.95))))
        counts = {}
        for budget in ('A0','A100'):
            file = base/f'{budget}_CLIQUE/train_LPLM.txt'
            # Each line is pattern + comma + a space-separated vector of
            # conditional prefix targets; patterns themselves may contain commas.
            total = 0
            with file.open() as f:
                for line in f:
                    if line.strip():
                        total += len(line.rstrip('\n').rsplit(',',1)[1].split())
            counts[budget] = total
        external.append(dict(dataset=ds,LPLM_targets_A0=counts['A0'],LPLM_targets_A100=counts['A100'],LBS_calibration=100,validation=len(list(labeled_rows(base/'A100_CLIQUE/valid.txt')))))
    write_csv(out/'Table03_columns.csv', columns)
    write_csv(out/'S02_distributions.csv', distributions)
    write_csv(out/'S03_label_cardinality.csv', shifts)
    write_csv(out/'S05_external_targets.csv', external)
    print('DATA STATISTICS ->', out.relative_to(ROOT), flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset','--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    generate(p.parse_args().dataset)
