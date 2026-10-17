"""Audit frozen paper splits and endpoint-anchored label masks."""
from .common import ROOT, DATASETS, read_json, labeled_rows, atomic_json, verify_files


def audit(datasets=DATASETS):
    manifest = read_json(ROOT / "manifests/artifacts.json")
    for bundle in manifest["bundles"]:
        if bundle["kind"] == "workloads" and bundle.get("dataset") in datasets:
            verify_files(bundle["files"])
    report = []
    for dataset in datasets:
        base = ROOT / f"data/{dataset}/training"
        pool = dict(labeled_rows(base / "A100_BIMO/train.txt"))
        valid = dict(labeled_rows(base / "A100_BIMO/valid.txt"))
        test = dict(labeled_rows(base / "A100_BIMO/test.txt"))
        if set(pool) & set(valid) or set(pool) & set(test) or set(valid) & set(test):
            raise ValueError(f"Overlapping splits in {dataset}")
        if any(c <= 0 for c in test.values()):
            raise ValueError("The final paper test population requires positive counts")
        chains = read_json(base / "A0_BIMO/chain_index.json")
        endpoints = {q for c in chains for q in (c["patterns"][0], c["patterns"][-1])}
        previous = set()
        for budget in ("A0", "A1", "A10", "A25", "A50", "A75", "A100"):
            masked = dict(labeled_rows(base / f"{budget}_BIMO/train.txt"))
            supervised = dict(labeled_rows(base / f"{budget}_CLIQUE/train.txt"))
            revealed = {q: c for q, c in masked.items() if c >= 0}
            if set(masked) != set(pool) or revealed != supervised:
                raise ValueError(f"Method supervision mismatch: {dataset}/{budget}")
            if any(pool[q] != c for q, c in revealed.items()) or not endpoints <= set(revealed):
                raise ValueError("Changed labels or missing chain endpoints")
            if not previous <= set(revealed):
                raise ValueError("Label masks are not nested")
            if budget == "A0" and set(revealed) != endpoints:
                raise ValueError("A0 must reveal exactly the endpoint identity union")
            previous = set(revealed)
            report.append(dict(dataset=dataset, budget=budget, pool=len(pool), labels=len(revealed),
                               rho=len(revealed) / len(pool), validation=len(valid), test=len(test)))
        if dataset == "WIKI" and len(test) != 70043:
            raise ValueError("Unexpected WIKI test revision")
        print("PREPARED", dataset, len(pool), len(test), flush=True)
    atomic_json(ROOT / "res/preparation.json", dict(status="PASS", budgets=report))
    return report
