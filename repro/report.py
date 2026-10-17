"""Recompute numerical displays and render figures without mixing run sources."""
import csv
import importlib.util
import shutil
import sys
from pathlib import Path
import numpy as np
from .common import ROOT, DATASETS, read_json, rows, write_csv, atomic_json, digest
from .metrics import evaluate, qerrors

ALIASES = {"BOLT": "BIMO", "UB_LB": "BOUND", "UB_ORDER": "MONO", "BOLT_L0": "BIMO-L0"}


def collect(selected, source="reference"):
    from .datasets import ensure
    datasets = list(dict.fromkeys(t["dataset"] for t in selected))
    ensure(datasets, ["workloads", "predictions"], raw=False)
    out = ROOT / "paper_results/generated" / source
    reference = ROOT / "paper_results/reference"
    records, paths, scope = [], {}, []
    ledger = {(r["dataset"], r["budget"]): r for r in rows(reference / "budget_ledger.csv")}
    for task in selected:
        if source == "reference":
            path = ROOT / task["predictions"]
        else:
            from .runner import status
            if status(task) != "COMPLETE":
                raise ValueError("No verified completed run for " + task["id"])
            receipt = read_json(ROOT / "res/runs" / task["id"] / "DONE.json")
            if receipt.get("epoch_override"):
                raise ValueError("Exploratory epoch overrides cannot populate paper figures")
            path = ROOT / receipt["result"]["prediction"]
        values = evaluate(task, path)
        if source == "reference":
            for key, expected in task["reference"].items():
                if not np.isclose(values[key], expected, rtol=2e-6, atol=1e-7):
                    raise ValueError((task["id"], key, values[key], expected))
        cfg = read_json(ROOT / task["config"])
        budget = ledger.get((task["dataset"], task["budget"]), {})
        r = dict(run_id=task["id"], dataset=task["dataset"], budget=task["budget"],
                 method=ALIASES.get(task["method"], task["method"]), seed=str(task["seed"]),
                 family="control" if task["method"] == "BOLT_L0" else "core",
                 **values, label_mask_id="anchor_seed_0", test_sha256=digest(ROOT / f'data/{task["dataset"]}/training/{task["workload"]}/test.txt'),
                 revealed_unique=budget.get("revealed_unique", budget.get("labels", "")),
                 output_convention="continuous served counts", checkpoint_selection=("train_mono" if "MID_MONO" in task["method"] else "not_applicable" if task["method"] in ("ZERO_LB","ZERO_MID") else "validation_q_error"),
                 chain_ratio=str(cfg.get("chain_sample_ratio", 1)), protocol="fixed-seed-current",
                 lower_bound_mode="witness-library", lambda_mono=0 if task["method"] == "BOLT_L0" else cfg.get("lambda_mono", 0))
        records.append(r)
        paths[task["id"]] = path
        scope.append(dict(task=task["id"], prediction_sha256=digest(path)))
    # LBS uses the same calibration and predictions at every label budget.
    for r in list(records):
        if r["method"] == "LBS" and r["budget"] == "A100":
            records.append(dict(r, budget="A0", run_id=r["run_id"] + "_shared_calibration"))
    core = [r for r in records if r["budget"] not in ("A25", "A50", "A75")]
    intermediate = [r for r in records if r["budget"] in ("A25", "A50", "A75")]
    write_csv(out / "results/run_metrics.csv", core)
    write_csv(out / "results/supplement_metrics.csv", intermediate)
    write_csv(out / "all_metrics.csv", records)
    write_csv(out / "tables/Table05_accuracy.csv", [r for r in records if r["budget"] in ("A0", "A100") and r["method"] in ("CLIQUE", "BIMO", "LPLM", "DREAM", "LBS")])
    write_csv(out / "tables/Table06_order.csv", [r for r in records if r["budget"] == "A0" and r["method"] in ("CLIQUE", "BOUND", "MONO", "BIMO", "LB_ORDER", "NO_BOUND", "BIMO-L0")])
    write_csv(out / "tables/Table07_calibration.csv", [r for r in records if r["method"].startswith("ZERO_") or r["method"] == "ANCHORED_MID_MONO"])
    conditional = []
    for task in selected:
        if task["budget"] != "A0" or task["method"] not in ("CLIQUE", "UB_LB", "UB_ORDER", "BOLT", "LB_ORDER"):
            continue
        bounds = {r["query"]: float(r["lower_bound"]) for r in rows(ROOT / f'artifacts/bounds/{task["dataset"]}.csv.gz')}
        predictions = list(rows(paths[task["id"]]))
        errors = qerrors([float(r["true"]) for r in predictions], [float(r["est"]) for r in predictions])
        covered = np.array([bounds[r["query"]] > 0 for r in predictions])
        for yes in (False, True):
            mask = covered == yes
            conditional.append(dict(dataset=task["dataset"], method=task["method"], stratum="LB>0" if yes else "LB=0",
                                   queries=int(mask.sum()), share=float(mask.mean()), q_avg=float(errors[mask].mean()), q99=float(np.quantile(errors[mask], .99))))
    write_csv(out / "tables/Table08_lower_bound.csv", conditional)
    # Setup and historical timing tables are immutable evidence, not fresh measurements.
    if source == "reference":
        mapping = {"Table03_workloads": "pattern_distribution_statistics", "Table04_budgets": "budget_ledger",
                   "Table09_upper_bound": "upper_bound_diagnostics", "Table10_labels": "time_efficiency_label_runs",
                   "Table11_construction": "construction_display", "Table12_inference": "inference_display"}
        for name, input_name in mapping.items():
            write_csv(out / "tables" / (name + ".csv"), rows(reference / (input_name + ".csv")))
    for i, ds in enumerate(DATASETS, 6):
        write_csv(out / "tables" / f"S{i:02d}_accuracy.csv", [r for r in core if r["dataset"] == ds and r["method"] in ("NO_BOUND", "CLIQUE", "BOUND", "MONO", "BIMO")])
        write_csv(out / "tables" / f"S{i+4:02d}_intermediate.csv", [r for r in intermediate if r["dataset"] == ds and r["seed"] == "0"])
    write_csv(out / "tables/S14_seeds.csv", [r for r in intermediate if r["dataset"] == "IMDB"])
    write_csv(out / "tables/S20_conditional.csv", conditional)
    write_csv(out / "tables/S22_calibration.csv", [r for r in records if r["method"].startswith("ZERO_") or r["method"] == "ANCHORED_MID_MONO"])
    # Diagnostics can be recalculated for any newly trained prediction set.
    for i, ds in enumerate(DATASETS, 16):
        write_csv(out / "tables" / f"S{i:02d}_continuous_order.csv", [r for r in core if r["dataset"] == ds and r["method"] in ("NO_BOUND", "CLIQUE", "BOUND", "MONO", "BIMO")])
    indexed = {(r['dataset'],r['budget'],r['method'],r['seed']):r for r in records}
    degradation = []
    for r in records:
        key=(r['dataset'],'A100',r['method'],r['seed'])
        if r['method'] in ('CLIQUE','BIMO') and r['budget'] != 'A100' and key in indexed:
            full=indexed[key]
            degradation.append(dict(dataset=r['dataset'], method=r['method'], budget=r['budget'], seed=r['seed'], mean_factor=r['q_avg']/full['q_avg'], q99_factor=r['q99']/full['q99']))
    write_csv(out/'tables/S15_degradation.csv',degradation)
    ub_rows=[]
    for ds in datasets:
        bounds=list(rows(ROOT/f'artifacts/bounds/{ds}.csv.gz'))
        upper=np.asarray([float(r['upper_bound']) for r in bounds])
        true=np.asarray([float(r['true_count']) for r in bounds])
        database_rows=next(float(r['database_rows']) for r in rows(reference/'dataset_string_statistics.csv') if r['dataset']==ds)
        ratio=upper/true
        audit=next(r for r in rows(reference/'upper_bound_diagnostics.csv') if r['dataset']==ds)
        # The original query export did not retain per-query exact-hit identities.
        # Its audited aggregate is available; missing markers must not become zeros.
        exact_hit_pct=float(audit['exact_table_hit_pct'])
        ub_rows.append(dict(dataset=ds,test_queries=len(bounds),ub_q50=float(np.median(upper)),ub_database_q50=float(np.median(upper)/database_rows),
            ub_card_q50=float(np.quantile(ratio,.5)),ub_card_q95=float(np.quantile(ratio,.95)),ub_card_q99=float(np.quantile(ratio,.99)),
            exact_hit_pct=exact_hit_pct,exact_hit_source='original aggregate summary audit; per-query identities unavailable',violations=int(np.sum(upper<true))))
    write_csv(out/'tables/Table09_S21_upper_bounds.csv',ub_rows)
    if source == 'reference':
        # Retain the original displays and detailed timing evidence under an
        # explicitly historical path; these are never labeled fresh measurements.
        shutil.copytree(ROOT/'paper_results/table_sources',out/'paper_display_tex',dirs_exist_ok=True)
        mapping={'S01_budgets':'budget_ledger','S02_distributions':'pattern_distribution_statistics',
                 'S03_label_cardinality':'budget_label_distribution_statistics','S04_configuration':'configuration',
                 'S24_label_repetitions':'time_efficiency_label_runs','S29_fullchain':'a1_fullchain_timing_runs'}
        for name,src in mapping.items():
            write_csv(out/'tables'/(name+'.csv'),rows(reference/(src+'.csv')))
        timing=list(rows(reference/'time_efficiency_runs.csv'))
        for i,ds in enumerate(DATASETS,25):
            write_csv(out/'tables'/f'S{i:02d}_construction_repetitions.csv',[r for r in timing if r['dataset']==ds])
        from .datasets import ensure
        ensure([],['timing','job-evidence','job-inputs'],raw=False)
        write_csv(out/'tables/S30_summary_components.csv',read_json(ROOT/'artifacts/timing/round3_summary_builds.json'))
        sensitivity=[]
        for task in read_json(ROOT/'manifests/sensitivity.json'):
            if task['dataset'] not in datasets:
                continue
            values=evaluate(task,ROOT/task['predictions'])
            for key,want in task['reference'].items():
                if not np.isclose(values[key],want,rtol=2e-6,atol=1e-7):
                    raise ValueError('Sensitivity reference mismatch')
            sensitivity.append(dict(dataset=task['dataset'],budget='A1',gamma=task['gamma'],**values))
        write_csv(out/'tables/S23_sensitivity.csv',sensitivity)
        from .job import verify_reference
        verify_reference()
    atomic_json(out / "PROVENANCE.json", dict(source=source, predictions=scope,
                 historical_timing=source == "reference", note="Reference and new runs are stored in separate directories. Frozen table TeX is in paper_results/table_sources."))
    print("COLLECTED", len(records), "rows ->", out.relative_to(ROOT), flush=True)
    return records


def render(source="reference"):
    out = ROOT / "paper_results/generated" / source
    script = ROOT / "scripts/render_layout_figures.py"
    spec = importlib.util.spec_from_file_location("render_layout_figures", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PAPER = out
    (out / "figs").mkdir(parents=True, exist_ok=True)
    module.main()
    for original, public in [('F3A_quantile_ranges.pdf','F3_accuracy.pdf'), ('F4_absolute_ablations.pdf','F4_ablations.pdf'), ('F5_matched_loss_bars.pdf','F5_loss.pdf')]:
        shutil.copy2(out/'figs'/original, out/'figs'/public)
    # Figure 1 is a fixed empirical case, distinct from the full test matrix.
    spec = importlib.util.spec_from_file_location("render_motivation_figure", ROOT / "scripts/render_motivation_figure.py")
    motivation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(motivation)
    motivation.render(case_dir=ROOT / "paper_results/reference/figure1_filled_case", output=out / "figs/F1_motivation.pdf")
    shutil.copy2(ROOT / "paper_results/figure_sources/F2_framework.pdf", out / "figs/F2_framework.pdf")
    print("RENDERED Figures 1–5 in", out.relative_to(ROOT) / "figs", flush=True)
