"""Translate paper task configurations to the inherited CLIQUE entry points."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import yaml
from .common import ROOT, tasks, read_json, atomic_json
from .metrics import evaluate

ENTRIES = {"CLIQUE": "main_CLIQUE.py", "BOLT": "main_CLIQUE_bound_mono.py",
           "UB_LB": "main_CLIQUE_bound.py", "UB_ORDER": "main_CLIQUE_mono.py",
           "NO_BOUND": "main_no_bound.py", "BOLT_L0": "main_CLIQUE_bound_mono.py",
           "DREAM": "main_DREAM.py", "LPLM": "main_LPLM.py", "LBS": "main_LBS.py"}
BOUND_KEYS = ("lb_min_substr_len", "lb_max_substr_len", "lb_top_k_strings", "lb_top_k_substrings", "lb_min_freq_ratio")


def command_for(task, tag, epochs=None, overrides=None):
    cfg = read_json(ROOT / task["config"])
    cfg.update(overrides or {})
    method = task["method"]
    config_group = method if method in ("DREAM", "LPLM", "LBS") else "CLIQUE"
    defaults = yaml.safe_load((ROOT / f"configs/{config_group}/default.yml").read_text())
    entry = ENTRIES.get(method, "main_zero_label.py")
    command = [sys.executable, entry]
    values = dict(cfg, tag=tag, data_name=task["dataset"], workload=task["workload"], seed=task["seed"])
    if epochs is not None:
        values["n_epoch"] = epochs
    for key, default in defaults.items():
        value = values.get(key, default)
        option = "--" + key.replace("_", "-")
        if isinstance(default, bool):
            command.append(option if value else "--no-" + key.replace("_", "-"))
        elif value is not None:
            command += [option, str(value)]
    if method in ("BOLT", "UB_LB", "BOLT_L0", "ZERO_LB", "ZERO_MID", "ZERO_MID_MONO", "ANCHORED_MID_MONO"):
        for key in BOUND_KEYS:
            command += ["--" + key, str(cfg[key])]
    if method in ("BOLT", "UB_ORDER", "BOLT_L0", "ZERO_LB", "ZERO_MID", "ZERO_MID_MONO", "ANCHORED_MID_MONO"):
        command += ["--lambda_mono", str(0 if method == "BOLT_L0" else cfg.get("lambda_mono", 5)),
                    "--mono_tolerance", str(cfg.get("mono_tolerance", .1))]
    if method in ("BOLT", "BOLT_L0"):
        command += ["--chain_sample_ratio", str(cfg.get("chain_sample_ratio", 1))]
    if method == "BOLT_L0":
        command += ["--force-chain-training"]
    if method.startswith("ZERO_") or method == "ANCHORED_MID_MONO":
        variant = {"ZERO_LB": "LB", "ZERO_MID": "MID", "ZERO_MID_MONO": "MID_MONO", "ANCHORED_MID_MONO": "ANCHORED_MID_MONO"}[method]
        command += ["--zero-variant", variant, "--selection-policy", "train_mono"]
    return command


def run(task, output, epochs=None):
    os.chdir(ROOT)
    if task["method"] == "LB_ORDER":
        from .lb_order import train
        prediction, files = train(task, output, epochs)
    else:
        tag = "BOLT_RUN/" + task["id"] + "/" + output.name
        command = command_for(task, tag, epochs)
        atomic_json(output / "command.json", dict(argv=command))
        if task['suite'] == 'job':
            # Preserve the published JOB training contract: validation selection,
            # no epoch-wise test access, same historical DataLoader RNG draw.
            snapshot = output / 'code'
            import shutil
            snapshot.mkdir()
            for folder in ('src', 'configs', 'a_route'):
                shutil.copytree(ROOT / folder, snapshot / folder)
            for path in ROOT.glob('main_*.py'):
                shutil.copy2(path, snapshot / path.name)
            for name in ('data', 'res'):
                (snapshot / name).symlink_to(ROOT / name, target_is_directory=True)
            from a_route.training import FILES, transform
            for name in FILES:
                path = snapshot / name
                path.write_text(transform(path.read_text()))
            subprocess.run(command, cwd=snapshot, check=True)
        else:
            subprocess.run(command, cwd=ROOT, check=True)
        base = ROOT / f'res/{task["dataset"]}/{task["workload"]}'
        configs = list((base / "model" / tag).glob(f'*/{task["seed"]}/config.yml'))
        if len(configs) != 1:
            raise ValueError(f"Ambiguous/missing output config: {configs}")
        suffix = configs[0].parent.relative_to(base / "model")
        prediction = base / "estimation" / suffix / "estimation.csv"
        files = [configs[0], prediction, base / "stat" / suffix / "stat.yml"]
        files += [p for p in configs[0].parent.iterdir() if p.name in ("model.pth", "model.pkl", "initialization.json", "selection_trace.json")]
    scores = evaluate(task, prediction)
    atomic_json(output / "result.json", dict(task=task["id"], metrics=scores,
                 prediction=str(prediction.relative_to(ROOT)), files=[str(p.relative_to(ROOT)) for p in files],
                 source="new training", epoch_override=epochs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    selected = next(t for t in tasks() if t["id"] == args.task)
    run(selected, args.output, args.epochs)


if __name__ == "__main__":
    main()
