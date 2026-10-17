"""Portable runner for the paper's upper-bound removal control."""
from pathlib import Path
import numpy as np
import torch
from src.lb_order import make_model
from src.lower_bound import load_freq_library
import src.util as ut
from .common import ROOT, read_json, labeled_rows, write_csv, atomic_json


class Writer:
    def __init__(self, output):
        self.output, self.history = output, []

    def add_scalars(self, tag, values, global_step):
        if tag == "ACC" and "valid" in values:
            self.history.append(dict(epoch=global_step, validation_q=float(values["valid"])))
            write_csv(self.output / "history.csv", self.history)


def train(task, output, epochs=None):
    cfg = read_json(ROOT / task["config"])
    if epochs is not None:
        cfg["n_epoch"] = epochs
    ds = task["dataset"]
    view = ROOT / f'data/{ds}/training/{task["workload"]}'
    db = ut.read_strings(str(ROOT / f"data/{ds}/{ds}.txt"))
    chars, _ = ut.gen_char_dict(db)
    model = make_model(cfg, chars, Writer(output), str(output / "model.pth"))
    model.n_db = len(db)
    model.substr_library = load_freq_library(str(ROOT / f"res/{ds}/freq_library/lib_s500_sub200_len5-15_freq0.0001_full.pkl"))
    model.chain_index = read_json(view / "chain_index.json")
    train_rows = list(labeled_rows(view / "train.txt"))
    valid = list(labeled_rows(view / "valid.txt"))
    test = list(labeled_rows(view / "test.txt"))
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # This is the final LB+Order run's validation-only diagnostic callback.
    model.build(ds, db, train_rows, valid, valid)
    model.eval()
    values = model.estimate([q for q, c in test])
    prediction = output / "predictions.csv"
    write_csv(prediction, [dict(query=q, true=c, est=float(p)) for (q, c), p in zip(test, values)])
    atomic_json(output / "config.json", cfg)
    return prediction, [prediction, output / "model.pth", output / "config.json", output / "history.csv"]
