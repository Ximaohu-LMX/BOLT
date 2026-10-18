"""Reuse the project's Phase 1 / degradation / compression algorithms."""
import random
from collections import Counter
from .protocol import normalize, require


def generate(values, test_patterns, cfg):
    from src.gen_query_chain import string_to_initial_pattern, compress_chain
    from src.util import generate_degradation_chain

    # Stable ordering fixes list(set(...))'s process-dependent seed behavior.
    pool = sorted({v for v in values if len(v) >= cfg['min_word_len']
                   and not any(c in v for c in '%_\\\r\n')})
    require(pool, 'no eligible chain source strings')
    rng = random.Random(cfg['seed'])
    test = set(map(normalize, test_patterns))
    candidates, seen, rejected = [], set(), Counter()
    for attempt in range(cfg['candidate_chains_per_column']):
        source = rng.choice(pool)
        word = source
        if len(word) > cfg['max_word_len']:
            start = rng.randint(0, len(word)-cfg['max_word_len'])
            word = word[start:start+cfg['max_word_len']]
        qtype = rng.choices(cfg['query_types'], weights=cfg['query_type_weights'], k=1)[0]
        head = string_to_initial_pattern(word, qtype, max_internal_pct=cfg['max_internal_pct'],
                                         min_segment_len=cfg['min_segment_len'], rng=rng)
        seed = rng.randint(0, 2**31)
        raw = generate_degradation_chain(head, seed=seed, **{k: cfg[k] for k in (
            'max_chain_length', 'degrade_char_prob', 'degrade_shrink_prob',
            'degrade_under_to_pct_prob', 'degrade_insert_pct_prob',
            'max_wildcard_ratio', 'min_literal_len')})
        raw = compress_chain(raw, cfg['compress_to'], **{k: cfg[k] for k in (
            'min_literal_len', 'compress_power_alpha', 'compress_power_max')})
        pats = list(dict.fromkeys(map(normalize, raw)))
        key = tuple(pats)
        if test.intersection(pats): rejected['test_conflict_whole_chain'] += 1; continue
        if len(pats) < 2: rejected['short_chain'] += 1; continue
        if key in seen: rejected['duplicate_chain'] += 1; continue
        seen.add(key)
        candidates.append(dict(chain_id=attempt, patterns=pats, source=source,
                               root=head, degradation_seed=seed, query_type=qtype))
    rng.shuffle(candidates)
    n_valid = max(1, int(len(candidates)*cfg['validation_chain_fraction']))
    valid = candidates[:n_valid]
    valid_patterns = {p for c in valid for p in c['patterns']}
    train = []
    for c in candidates[n_valid:]:
        if valid_patterns.intersection(c['patterns']):
            rejected['validation_conflict_whole_chain'] += 1
        else:
            train.append(c)
    train_patterns = {p for c in train for p in c['patterns']}
    require(not (train_patterns & valid_patterns or train_patterns & test or valid_patterns & test),
            'split intersection')
    return dict(train=train, valid=valid, test=sorted(test),
                statistics=dict(source_unique=len(pool), attempts=cfg['candidate_chains_per_column'],
                                accepted_train_chains=len(train), accepted_valid_chains=len(valid),
                                train_unique=len(train_patterns), valid_unique=len(valid_patterns),
                                test_unique=len(test), rejected=dict(rejected)),
                usable=bool(train and valid_patterns and test))


def make_views(split, cards, n_nonnull, budgets, seed):
    from src.util import build_chain_train_data
    from src.gen_query_chain import _sample_anchor_mask

    needed = set(split['test']) | {p for name in ('train', 'valid') for c in split[name] for p in c['patterns']}
    require(set(cards) == needed, 'missing or unexpected exact-count targets')
    require(all(type(c) is int and 0 <= c <= n_nonnull for c in cards.values()), 'invalid cardinality')
    if '%' in cards:
        require(cards['%'] == n_nonnull, 'all-percent count must exclude NULL')
    for name in ('train', 'valid'):
        for chain in split[name]:
            require(all(cards[a] <= cards[b] for a, b in zip(chain['patterns'], chain['patterns'][1:])),
                    'chain cardinality inversion: '+str(chain['chain_id']))
    output = {}
    valid = sorted({p for c in split['valid'] for p in c['patterns']})
    for budget, rate in budgets.items():
        rng = random.Random(seed)
        idx = [dict(chain_id=c['chain_id'], patterns=c['patterns'],
                    is_anchor=_sample_anchor_mask(c['patterns'], rate, rng)) for c in split['train']]
        rows, _ = build_chain_train_data(idx, cards, n_nonnull)
        # Synchronize occurrence masks with the actual unique-query supervision.
        labeled = {p for p, count in rows if count >= 0}
        for c in idx:
            c['is_anchor'] = [p in labeled for p in c['patterns']]
        cli = [(p, c) for p, c in rows if c >= 0]
        output[budget] = dict(BOLT=rows, CLIQUE=cli, chain_index=idx,
                              valid=[(p, cards[p]) for p in valid],
                              test=[(p, cards[p]) for p in split['test'] if p != ''],
                              exact_test={p: cards[p] for p in split['test'] if p == ''},
                              label_count=len(labeled), unique_train=len(rows),
                              realized_rate=len(labeled)/len(rows))
    return output
