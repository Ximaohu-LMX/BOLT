"""BOLT stepwise and complete reproduction entry point."""
import argparse
from .common import ROOT, DATASETS, select_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="all", choices=["all", "main", "accuracy", "ablations", "calibration", "supplement", "costs", "job"])
    parser.add_argument("--stage", default="run", choices=["run", "prepare", "verify", "collect", "render", "smoke"])
    parser.add_argument("--dataset", "--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--budget", "--budgets", nargs="+")
    parser.add_argument("--method", "--methods", nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, help="Exploratory override, explicitly recorded; not a paper reproduction")
    parser.add_argument("--source", choices=["reference", "new"], default="reference", help="Source for collect/render")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    selected = select_tasks(args.suite, args.dataset, args.budget, args.method, args.seed)
    if args.dry_run:
        for t in selected:
            print(t["id"], t["suite"], t["config"])
        if args.suite in ("all", "costs", "job"):
            print("Additional suite:", args.suite, "(costs: PostgreSQL label counts/build/CPU inference; JOB: isolated PostgreSQL 14.5)")
        print("Planned training cells:", len(selected))
        return
    if args.status:
        from .runner import status
        for t in selected:
            print(status(t), t["id"])
        return
    if args.stage == "smoke":
        from .smoke import run
        run()
        return
    if args.suite == "costs" and args.stage == "run":
        from .costs import run_all
        run_all(args.dataset or DATASETS, args.force)
        return
    if args.suite == "job" and args.stage == "run":
        from .job import run_all
        run_all(args.force)
        return
    if not selected:
        parser.error("No experiments match the selected filters")
    datasets = list(dict.fromkeys(t["dataset"] for t in selected))
    from .datasets import ensure
    if args.stage in ("render", "collect"):
        from .report import collect, render
        collect(selected, args.source)
        if args.stage == "render":
            render(args.source)
        return
    ensure(datasets, ["workloads", "summaries"] + (["predictions"] if args.stage == "verify" else []), raw=args.stage == "run")
    from .prepare import audit
    audit(datasets)
    if args.stage == "prepare":
        return
    if args.stage == "verify":
        from .metrics import verify_reference
        verify_reference(selected)
        return
    from .runner import lock, run
    with lock(ROOT / "res/.training.lock"):
        for task in selected:
            run(task, args.force, args.epochs)
    if args.epochs is not None:
        print("Exploratory runs complete; paper aggregation skipped because --epochs was overridden.")
        return
    from .report import collect, render
    collect(selected, "new")
    if args.suite in ("main", "all") and not any((args.dataset, args.budget, args.method, args.seed is not None, args.epochs)):
        render("new")
    if args.suite == "all":
        from .costs import run_all as costs
        from .job import run_all as job
        from .statistics import generate
        generate(datasets)
        costs(datasets, args.force)
        job(args.force)


if __name__ == "__main__":
    main()
