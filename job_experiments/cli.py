"""Run with python -m job_experiments.cli; each mutation creates a new stage."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path
from .protocol import COLUMNS, ROOT, read_json, write_json, workload, sha, seal, verify, require


def staged(output, action):
    output = Path(output).resolve()
    require(not output.exists(), 'output already exists; choose a new stage directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=output.name+'.building-', dir=output.parent))
    try:
        result = action(tmp)
        tmp.rename(output)
        return result
    except BaseException:
        shutil.rmtree(tmp)
        raise


def validate_cfg(cfg):
    require(cfg['protocol'] == 'job_chain_v1', 'unknown protocol')
    require(type(cfg['seed']) is int, 'integer seed required')
    for key in ('candidate_chains_per_column', 'max_chain_length', 'compress_to', 'max_word_len', 'min_word_len', 'min_segment_len'):
        require(type(cfg[key]) is int and cfg[key] > 0, 'invalid '+key)
    require(cfg['max_chain_length'] >= 2 and cfg['compress_to'] >= 2, 'chains require two nodes')
    require(0 < cfg['validation_chain_fraction'] < 1, 'invalid validation fraction')
    require(set(cfg['budgets']) <= {'A100', 'A0'} and cfg['budgets'], 'supported budgets: A100/A0')
    require(all(v == (1. if k == 'A100' else 0.) for k,v in cfg['budgets'].items()), 'budget mismatch')
    require(cfg['query_types'] == ['SUBSTR', 'PREFIX', 'SUFFIX'], 'unexpected root family')
    require(len(cfg['query_type_weights']) == 3 and all(v >= 0 for v in cfg['query_type_weights'])
            and sum(cfg['query_type_weights']) > 0, 'invalid root weights')


def plan(args):
    cfg = read_json(args.config); validate_cfg(cfg)
    w = workload(args.sql)
    return dict(status='PLANNED', protocol=cfg, sql_sha256=w['sql_sha256'], sql_count=w['sql_count'],
                columns={c: dict(test_unique=len(w['test_patterns'][c]), candidate_chain_attempts=cfg['candidate_chains_per_column']) for c in COLUMNS},
                planned_model_cells=len(COLUMNS)*len(cfg['budgets'])*2,
                training_started=False, readiness='requires exported JOB assets, counts, model worker and database integration')


def freeze(args):
    result = plan(args)
    def action(out):
        write_json(out/'config.json', result['protocol'])
        write_json(out/'workload.json', workload(args.sql))
        shutil.copyfile(args.sql, out/'job.sql')
        source_files = ['src/gen_query_chain.py', 'src/util.py'] + [str(p.relative_to(ROOT)) for p in sorted((ROOT/'job_experiments').glob('*.py'))]
        seal(out, 'PLAN.json', dict(result, source_sha256={p:sha(ROOT/p) for p in source_files}))
        return dict(status='PLAN_FROZEN', **{k:result[k] for k in ('sql_count','columns','planned_model_cells','training_started')})
    return staged(args.output, action)


def export(args):
    from .database import connect, identity, export_column
    def action(out):
        conn = connect(args.service, args.timeout_ms)
        try:
            metadata = dict(database=identity(conn), columns={})
            for c in COLUMNS:
                metadata['columns'][c] = export_column(conn, c, out/(c+'.jsonl'))
            seal(out, 'EXPORT.json', metadata)
        finally:
            conn.rollback(); conn.close()
        return dict(status='EXPORTED', columns=metadata['columns'])
    return staged(args.output, action)


def prepare(args):
    from .chains import generate
    cfg = read_json(args.config); validate_cfg(cfg)
    assets = Path(args.assets).resolve(); asset_info = verify(assets, 'EXPORT.json')
    w = workload(args.sql)
    def action(out):
        write_json(out/'config.json', cfg)
        write_json(out/'workload.json', w)
        shutil.copyfile(args.sql, out/'job.sql')
        stats, usable = {}, True
        for column in COLUMNS:
            meta = asset_info['columns'][column]
            values, total, nulls = [], 0, 0
            with (assets/(column+'.jsonl')).open() as stream:
                for line in stream:
                    value = json.loads(line)['value']
                    total += 1
                    if value is None: nulls += 1
                    else: values.append(value)
            require(total == meta['rows'] and nulls == meta['nulls'], 'export row mismatch')
            require(len(values) == meta['nonnull'] and values, 'empty non-NULL model column')
            require(all(isinstance(v, str) and '\r' not in v and '\n' not in v for v in values),
                    'line-based model format cannot preserve CR/LF; explicit encoding required: '+column)
            split = generate(values, w['test_patterns'][column], cfg)
            folder = out/column; folder.mkdir()
            # Duplicates and empty strings retained. NULL is excluded only from
            # model data; full-table denominator remains in EXPORT metadata.
            (folder/'column.txt').write_text(''.join(v+'\n' for v in values))
            write_json(folder/'split.json', split)
            targets = sorted(set(split['test']) | {p for n in ('train','valid') for c in split[n] for p in c['patterns']})
            write_json(folder/'count_targets.json', targets)
            stats[column] = split['statistics']; usable &= split['usable']
        sources = {p: sha(ROOT/p) for p in ('src/gen_query_chain.py','src/util.py',
                    'job_experiments/chains.py','job_experiments/protocol.py','job_experiments/cli.py')}
        seal(out, 'PREPARED.json', dict(status='PREPARED' if usable else 'INSUFFICIENT_DISJOINT_CHAINS',
            assets_path=str(assets), assets_manifest_sha256=sha(assets/'EXPORT.json'),
            database=asset_info['database'], columns=asset_info['columns'], source_sha256=sources,
            statistics=stats, training_started=False))
        return dict(status='PREPARED' if usable else 'INSUFFICIENT_DISJOINT_CHAINS', statistics=stats)
    return staged(args.output, action)


def count(args):
    from .database import connect, identity, export_column, count_patterns
    prepared = Path(args.prepared).resolve(); info = verify(prepared, 'PREPARED.json')
    require(info['status'] == 'PREPARED', 'insufficient disjoint chains; inspect preparation statistics')
    def action(out):
        conn = connect(args.service, args.timeout_ms)
        try:
            require(identity(conn) == info['database'], 'database identity changed since export')
            for column in COLUMNS:
                require(export_column(conn, column) == info['columns'][column], 'column snapshot changed: '+column)
                targets = read_json(prepared/column/'count_targets.json')
                counts = count_patterns(conn, column, targets)
                write_json(out/(column+'.json'), counts)
            seal(out, 'COUNTS.json', dict(prepared_path=str(prepared), prepared_sha256=sha(prepared/'PREPARED.json'),
                method='PostgreSQL exact LIKE, read-only repeatable-read snapshot',
                accounting='all chain-node counts include offline diagnostic labels; not label-materialization savings'))
        finally:
            conn.rollback(); conn.close()
        return dict(status='COUNTED')
    return staged(args.output, action)


def assemble(args):
    from .chains import make_views
    prepared, counts = Path(args.prepared).resolve(), Path(args.counts).resolve()
    info, receipt = verify(prepared, 'PREPARED.json'), verify(counts, 'COUNTS.json')
    require(info['status'] == 'PREPARED', 'preparation not usable')
    require(receipt['prepared_sha256'] == sha(prepared/'PREPARED.json'), 'counts belong to a different preparation')
    cfg = read_json(prepared/'config.json')
    def action(out):
        summary = {}
        for col in COLUMNS:
            split = read_json(prepared/col/'split.json')
            cards = read_json(counts/(col+'.json'))
            if '' in cards:
                require(cards[''] == info['columns'][col]['empty_strings'], 'empty-string summary mismatch')
            views = make_views(split, cards, info['columns'][col]['nonnull'], cfg['budgets'], cfg['seed'])
            summary[col] = {}
            for budget, view in views.items():
                summary[col][budget] = {k:view[k] for k in ('label_count','unique_train','realized_rate')}
                for method in ('CLIQUE', 'BIMO'):
                    folder = out/col/(budget+'_'+method); folder.mkdir(parents=True)
                    for name, rows in (('train',view[method]), ('valid',view['valid']), ('test',view['test'])):
                        (folder/(name+'.txt')).write_text(''.join('%s,%d\n' % row for row in rows))
                    write_json(folder/'exact_test.json', dict(patterns=view['exact_test'],
                        reason='empty pattern bypasses encoder; shared exact empty-string summary count'))
                    if method == 'BIMO': write_json(folder/'chain_index.json', view['chain_index'])
                    write_json(folder/'sample_info.json', dict(protocol=cfg['protocol'], column=col,
                        budget=budget, model=method, seed=cfg['seed'], **summary[col][budget]))
        seal(out, 'VIEWS.json', dict(status='DATA_READY', prepared_path=str(prepared),
            prepared_sha256=sha(prepared/'PREPARED.json'), counts_path=str(counts),
            counts_sha256=sha(counts/'COUNTS.json'), statistics=summary,
            training_started=False, remaining='column model worker, summary semantics audit, injection slots and isolated timing'))
        return dict(status='DATA_READY', statistics=summary, training_started=False)
    return staged(args.output, action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    for command in ('plan','freeze','export','prepare','count','assemble','audit'):
        p = subs.add_parser(command)
        if command in ('plan','freeze','prepare'):
            p.add_argument('--config', default=str(ROOT/'configs/job/chains.json'))
            p.add_argument('--sql', default=str(ROOT/'likeBench/job_LPLM.sql'))
        if command in ('export','count'):
            p.add_argument('--service', required=True)
            p.add_argument('--timeout-ms', type=int, default=60000)
        if command == 'prepare': p.add_argument('--assets', required=True)
        if command in ('count','assemble'): p.add_argument('--prepared', required=True)
        if command == 'assemble': p.add_argument('--counts', required=True)
        if command not in ('plan','audit'): p.add_argument('--output', required=True)
        if command == 'audit':
            p.add_argument('--directory', required=True)
            p.add_argument('--receipt', required=True, choices=['PLAN.json','EXPORT.json','PREPARED.json','COUNTS.json','VIEWS.json'])
    args = parser.parse_args()
    if args.command == 'audit':
        info = verify(Path(args.directory), args.receipt)
        result = dict(status='HASH_VERIFIED', receipt=args.receipt, files=len(info['files']))
    else:
        result = globals()[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
