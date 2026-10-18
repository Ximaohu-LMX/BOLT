"""Unbounded, chain-batched control for Figure 1 (no summary access).

Reuse the NO-BOUND encoder/head and its numerical count floor of one. This
floor is an output convention, not a query-specific lower-bound synopsis.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
import src.util as ut
from src.CLIQUE_no_bound import CLIQUEestimator_no_bound


def chain_batches(rows, chains, batch_size, seed):
    cards = {}
    for q, c in rows:
        if q in cards and cards[q] != c:
            raise ValueError('Conflicting visible targets: ' + q)
        cards[q] = c
    chain_patterns = {q for ch in chains for q in ch['patterns']}
    if chain_patterns != set(cards):
        raise ValueError('Training queries and chain queries must agree')
    rng = np.random.RandomState(seed)
    batches = []
    queries, targets, tight, loose = [], [], [], []
    for idx in rng.permutation(len(chains)):
        ps = chains[int(idx)]['patterns']
        if not ps:
            raise ValueError('Empty chain')
        offset = len(queries)
        queries.extend(ps)
        targets.extend(cards[q] for q in ps)  # Never restore a hidden '%' target.
        tight.extend(range(offset, offset + len(ps) - 1))
        loose.extend(range(offset + 1, offset + len(ps)))
        if len(queries) >= batch_size:
            batches.append(dict(queries=queries, targets=targets, tight=tight, loose=loose))
            queries, targets, tight, loose = [], [], [], []
    if queries:
        if len(queries) == 1 and batches:  # BatchNorm requires two training rows.
            batches[-1]['queries'].extend(queries)
            batches[-1]['targets'].extend(targets)
        else:
            batches.append(dict(queries=queries, targets=targets, tight=tight, loose=loose))
    return batches


def encode(queries, char_dict, device):
    ps = [ut.canonicalize_like_query(q, is_last_flip=True) for q in queries]
    encoded = ut.string_encoding(ps, char_dict)
    lengths = [len(x) for x in encoded]
    padded = ut.keras_pad_sequences(encoded, padding='post')
    return torch.as_tensor(padded, dtype=torch.long, device=device), lengths


def objective(z, targets, tight, loose, supervised_weight=1., lambda_order=5., tolerance=.1):
    """One forward pass; supervision uses visible targets only.

    log1p(served_count) == max(z, log(2)) for the inherited NO-BOUND head.
    Compute this stably without exponentiating during the monotonic loss.
    """
    if supervised_weight < 0 or lambda_order < 0 or tolerance < 0:
        raise ValueError('Loss weights and tolerance must be nonnegative')
    zero = z.sum() * 0
    mask = targets >= 0
    sup = F.mse_loss(z[mask], torch.log1p(targets[mask])) if supervised_weight and mask.any() else zero
    if tight:
        log_counts = z.clamp_min(math.log(2.))
        mono = F.relu(log_counts[tight] - log_counts[loose] - math.log1p(tolerance)).mean()
    else:
        mono = zero
    return supervised_weight * sup + lambda_order * mono, sup, mono


def make_model(cfg, char_dict):
    model = CLIQUEestimator_no_bound()
    model.set_params(cfg['N'], cfg['PT'], cfg['seed'], char_dict, cfg['ch_es'], 1,
                     False, cfg['rnn_hs'], cfg['pred_hs'], cfg['n_pred_layer'],
                     cfg['learning_rate'], cfg['l2'], cfg['n_epoch'], cfg['batch_size'],
                     cfg['patience'], True, None, None, None, False, False,
                     cfg['dynamicPT'], cfg['max_entry_ratio'], cfg['batch_norm'])
    return model


def predict(model, queries, char_dict, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    values = []
    with torch.no_grad():
        for start in range(0, len(queries), batch_size):
            z = model.logit(encode(queries[start:start + batch_size], char_dict, device))
            pred = torch.expm1(z.double()).clamp_min(1.)
            if not torch.isfinite(pred).all():
                raise FloatingPointError('Nonfinite unbounded prediction; no silent clipping')
            values.extend(pred.cpu().tolist())
    return np.asarray(values)
