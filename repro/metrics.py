"""Paper metrics: continuous Q-error and ties-to-even integer order diagnostics."""
import math
import numpy as np
from .common import ROOT, DATASETS, rows, read_json, labeled_rows, write_csv, atomic_json


def qerrors(truth, predicted):
    truth, predicted = np.asarray(truth, dtype=float), np.asarray(predicted, dtype=float)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("Empty or mismatched predictions")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all() or (truth < 0).any() or (predicted < 0).any():
        raise ValueError("Invalid prediction/truth values")
    a, b = np.maximum(truth, 1), np.maximum(predicted, 1)
    return np.maximum(a / b, b / a)


def summarize(truth, predicted):
    qe = qerrors(truth, predicted)
    return dict(q_avg=float(qe.mean()), q50=float(np.quantile(qe, .5)), q95=float(np.quantile(qe, .95)),
                q99=float(np.quantile(qe, .99)), q100=float(qe.max()), test_queries=len(qe))


def test_chains(dataset):
    # This is the final held-out occurrence population, including the WIKI exclusion.
    return read_json(ROOT / f"artifacts/test_chains/{dataset}.json")


def ordering(predictions, chains):
    pairs = [(a, b) for chain in chains for a, b in zip(chain, chain[1:])]
    if not pairs:
        raise ValueError("No held-out adjacent pairs")
    a = np.asarray([predictions[x] for x, y in pairs])
    b = np.asarray([predictions[y] for x, y in pairs])
    ra, rb = np.rint(a), np.rint(b)
    differences = np.log1p(a) - np.log1p(b)
    nontrivial = [chain for chain in chains if len(chain) > 1]
    chain_slack, chain_tolerant = [], []
    for chain in nontrivial:
        x = np.log1p([predictions[q] for q in chain])
        delta = x[:-1] - x[1:]
        chain_slack.append(bool(np.all(delta <= 1e-6)))
        chain_tolerant.append(bool(np.all(delta <= np.log1p(.1) + 1e-6)))
    return dict(R_int=float(np.mean(ra <= rb)), R_tie=float(np.mean(ra == rb)),
                pair_count=len(pairs), raw_zero_slack=float(np.mean(a <= b)),
                pair_slack=float(np.mean(differences <= 1e-6)), pair_tolerant=float(np.mean(differences <= np.log1p(.1)+1e-6)),
                chain_slack=float(np.mean(chain_slack)), chain_tolerant=float(np.mean(chain_tolerant)), chain_count=len(nontrivial))


def evaluate(task, path):
    records = list(rows(path))
    truth = list(labeled_rows(ROOT / f'data/{task["dataset"]}/training/{task["workload"]}/test.txt'))
    if [(r["query"], float(r["true"])) for r in records] != truth:
        raise ValueError("Prediction order, coverage, or ground truth differs: " + task["id"])
    if len({r["query"] for r in records}) != len(records):
        raise ValueError("Duplicate test query")
    preds = {r["query"]: float(r["est"]) for r in records}
    result = summarize([c for q, c in truth], list(preds.values()))
    if task["dataset"] in DATASETS:
        result.update(ordering(preds, test_chains(task["dataset"])))
    return result


def verify_reference(selected):
    report = []
    for task in selected:
        result = evaluate(task, ROOT / task["predictions"])
        for name, expected in task["reference"].items():
            if not math.isclose(result[name], expected, rel_tol=2e-6, abs_tol=1e-7):
                raise ValueError(f'{task["id"]}: {name} differs from the reference')
        report.append(dict(id=task["id"], dataset=task["dataset"], budget=task["budget"],
                           method=task["method"], seed=task["seed"], **result))
        print("VERIFIED", task["id"], flush=True)
    write_csv(ROOT / "paper_results/generated/verified_metrics.csv", report)
    atomic_json(ROOT / "paper_results/generated/verification.json", dict(status="PASS", tasks=len(report),
                 source="archived paper predictions; no new model training"))
    return report
