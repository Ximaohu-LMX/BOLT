#!/usr/bin/env python3
"""Recompute strict/tolerant monotonicity from one frozen held-out pair set.

This is a pure post-processing pass: it reads canonical estimation.csv files and
the held-out test_chain_index from FULL_DISJOINT_V1.  It never loads a model and
never changes the append-only experiment summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METHODS = ("NO_BOUND", "CLIQUE", "UB_ORDER", "UB_LB", "BOLT")
BUDGETS = ("A100", "A10", "A1", "A0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        type=Path,
        default=ROOT / "manifests" / "canonical_results_seed0.csv",
    )
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=ROOT / "manifests" / "unified_monotonicity_seed0.csv",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / "docs" / "统一单调性后处理.md",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def pair_set_sha256(chains: list[dict]) -> str:
    digest = hashlib.sha256()
    for chain_number, chain in enumerate(chains):
        for pair_number, (tight, loose) in enumerate(
            zip(chain["patterns"][:-1], chain["patterns"][1:])
        ):
            digest.update(str(chain_number).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(pair_number).encode("ascii"))
            digest.update(b"\0")
            digest.update(tight.encode("utf-8"))
            digest.update(b"\0")
            digest.update(loose.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def load_estimations(path: Path) -> tuple[dict[str, float], dict[str, int]]:
    predictions: dict[str, float] = {}
    true_cards: dict[str, int] = {}
    for line_number, row in enumerate(read_csv(path), start=2):
        query = row["query"]
        if query in predictions:
            raise ValueError(f"duplicate estimation query: {path}:{line_number}: {query!r}")
        prediction = float(row["est"])
        if not math.isfinite(prediction) or prediction < 0:
            raise ValueError(f"invalid prediction: {path}:{line_number}: {prediction}")
        predictions[query] = prediction
        true_cards[query] = int(float(row["true"]))
    return predictions, true_cards


def evaluate(
    chains: list[dict], predictions: dict[str, float], tolerance: float
) -> dict[str, int | float]:
    tau = math.log1p(tolerance)
    pair_total = 0
    pair_strict_ok = 0
    pair_tolerant_ok = 0
    chain_total = 0
    chain_strict_ok = 0
    chain_tolerant_ok = 0
    max_log_violation = 0.0

    for chain in chains:
        patterns = chain.get("patterns", [])
        if len(patterns) < 2:
            continue
        missing = [pattern for pattern in patterns if pattern not in predictions]
        if missing:
            raise ValueError(
                f"held-out chain contains {len(missing)} patterns absent from estimation; "
                f"first={missing[0]!r}"
            )

        strict_chain_ok = True
        tolerant_chain_ok = True
        for tight, loose in zip(patterns[:-1], patterns[1:]):
            tight_log = math.log1p(predictions[tight])
            loose_log = math.log1p(predictions[loose])
            violation = tight_log - loose_log
            strict_ok = violation <= 1e-6
            tolerant_ok = violation <= tau + 1e-6
            pair_total += 1
            pair_strict_ok += int(strict_ok)
            pair_tolerant_ok += int(tolerant_ok)
            strict_chain_ok = strict_chain_ok and strict_ok
            tolerant_chain_ok = tolerant_chain_ok and tolerant_ok
            max_log_violation = max(max_log_violation, violation)

        chain_total += 1
        chain_strict_ok += int(strict_chain_ok)
        chain_tolerant_ok += int(tolerant_chain_ok)

    if pair_total == 0 or chain_total == 0:
        raise ValueError("held-out chain set contains no comparable pairs")
    return {
        "pair_total": pair_total,
        "mono_pair_satisfied": pair_strict_ok,
        "mono_pair_score": pair_strict_ok / pair_total,
        "t_mono_pair_satisfied": pair_tolerant_ok,
        "t_mono_pair_score": pair_tolerant_ok / pair_total,
        "chain_total": chain_total,
        "mono_chain_satisfied": chain_strict_ok,
        "mono_chain_score": chain_strict_ok / chain_total,
        "t_mono_chain_satisfied": chain_tolerant_ok,
        "t_mono_chain_score": chain_tolerant_ok / chain_total,
        "max_log_violation": max_log_violation,
    }


def validate_canonical(rows: list[dict[str, str]]) -> tuple[list[str], dict]:
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    index = {}
    for row in rows:
        key = (row["dataset"], row["budget"], row["method"])
        if key in index:
            raise ValueError(f"duplicate canonical row: {key}")
        index[key] = row
    expected = {
        (dataset, budget, method)
        for dataset in datasets
        for budget in BUDGETS
        for method in METHODS
    }
    if set(index) != expected or len(rows) != len(expected):
        raise ValueError("canonical input is not the complete 4-budget x 5-method matrix")
    if any(row.get("seed") != "0" or row.get("anchor_seed") != "0" for row in rows):
        raise ValueError("canonical input is not the frozen seed-0/anchor-seed-0 matrix")
    return datasets, index


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_report(
    rows: list[dict[str, object]],
    tolerance: float,
    canonical_path: Path,
    csv_path: Path,
) -> str:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["budget"]), str(row["method"]))].append(row)

    lines = [
        "# 统一 mono / t_mono 后处理",
        "",
        "状态：**完成；作为 seed-0 的统一单调性主口径**",
        f"最后生成：{date.today().isoformat()}",
        f"容忍参数：`τ={tolerance}`",
        "",
        "## 1. 统一口径",
        "",
        "本次不重新训练，也不调用模型生成新的退化模式。对每个数据集，直接使用 "
        "`FULL_DISJOINT_V1/split_info.json` 中冻结的 held-out `test_chain_index`；五种方法、四个预算均在完全相同的有序相邻 pair 上读取已有 `estimation.csv` 预测并复算。",
        "",
        "- `mono`：`log(1+pred_tight) <= log(1+pred_loose) + 1e-6`，与原始基数空间的严格次序等价；",
        f"- `t_mono`：允许 `log(1+{tolerance})` 的 dead zone；",
        "- `pair score`：满足条件的相邻边比例；",
        "- `whole-chain score`：一条链的所有相邻边都满足时才计为通过；",
        "- 主文优先使用 `pair score`；whole-chain 对链长更敏感，放附录。",
        "",
        "这套口径补齐了 NO_BOUND、CLIQUE、UB_LB 原 summary 中缺失的 `t_mono`。原 summary 的 point-generated MonoM 仍保留为历史辅助指标，不与本表混用，也不回写 append-only summary。",
        "",
        "## 2. 四数据集宏平均",
        "",
        "下表每格均为四个数据集的算术平均；完整逐数据集数值见机器表。",
        "",
        "| Budget | Method | strict pair | t_mono pair | strict chain | t_mono chain |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for budget in BUDGETS:
        for method in METHODS:
            items = grouped[(budget, method)]
            lines.append(
                f"| {budget} | {method} | "
                f"{mean([float(x['mono_pair_score']) for x in items]):.4f} | "
                f"{mean([float(x['t_mono_pair_score']) for x in items]):.4f} | "
                f"{mean([float(x['mono_chain_score']) for x in items]):.4f} | "
                f"{mean([float(x['t_mono_chain_score']) for x in items]):.4f} |"
            )

    dataset_pairs = {}
    for row in rows:
        dataset_pairs.setdefault(
            str(row["dataset"]),
            (int(row["pair_total"]), int(row["chain_total"]), str(row["test_pair_set_sha256"])),
        )
    lines.extend(
        [
            "",
            "## 3. 固定测试集规模",
            "",
            "| Dataset | adjacent pairs | chains | pair-set SHA-256 |",
            "|---|---:|---:|---|",
        ]
    )
    for dataset, (pairs, chains, digest) in dataset_pairs.items():
        lines.append(f"| {dataset} | {pairs:,} | {chains:,} | `{digest}` |")

    lines.extend(
        [
            "",
            "## 4. 使用边界",
            "",
            "该结果衡量已有 held-out 链上的次序一致性；它不等于训练损失，也不证明所有可能 LIKE 模式上的全局单调性。test chain 由既有退化规则产生，因此不改变当前退化模式。",
            "",
            "可复现入口：",
            "",
            f"- canonical 输入：`{canonical_path.relative_to(ROOT)}`；",
            f"- 机器结果：`{csv_path.relative_to(ROOT)}`；",
            "- 生成命令：`python3 analyze_unified_monotonicity.py`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative")
    canonical_rows = read_csv(args.canonical)
    datasets, index = validate_canonical(canonical_rows)
    output_rows: list[dict[str, object]] = []

    for dataset in datasets:
        split_path = ROOT / "data" / dataset / "training" / "FULL_DISJOINT_V1" / "split_info.json"
        chains = load_json(split_path)["test_chain_index"]
        digest = pair_set_sha256(chains)
        expected_patterns = {
            pattern for chain in chains for pattern in chain.get("patterns", [])
        }
        reference_cards = None
        reference_test_hash = None

        for budget in BUDGETS:
            for method in METHODS:
                row = index[(dataset, budget, method)]
                predictions, true_cards = load_estimations(ROOT / row["est_path"])
                if set(predictions) != expected_patterns:
                    raise ValueError(
                        f"estimation query set mismatch: {dataset}/{budget}/{method}: "
                        f"missing={len(expected_patterns - set(predictions))}, "
                        f"extra={len(set(predictions) - expected_patterns)}"
                    )
                if reference_cards is None:
                    reference_cards = true_cards
                    reference_test_hash = row["test_sha256"]
                elif true_cards != reference_cards or row["test_sha256"] != reference_test_hash:
                    raise ValueError(f"test truth mismatch: {dataset}/{budget}/{method}")

                metrics = evaluate(chains, predictions, args.tolerance)
                output_rows.append(
                    {
                        "dataset": dataset,
                        "budget": budget,
                        "method": method,
                        "seed": row["seed"],
                        "anchor_seed": row["anchor_seed"],
                        "tolerance": args.tolerance,
                        "pair_source": "FULL_DISJOINT_V1/test_chain_index",
                        "test_pair_set_sha256": digest,
                        "test_sha256": row["test_sha256"],
                        **metrics,
                        "legacy_point_generated_mono_score": row.get("mono_score", ""),
                        "legacy_point_generated_t_mono_score": row.get("t_mono_score", ""),
                        "est_path": row["est_path"],
                        "data_id": row["data_id"],
                        "code_commit": row["code_commit"],
                    }
                )

    write_csv(args.csv_out, output_rows)
    report = build_report(output_rows, args.tolerance, args.canonical, args.csv_out)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report, encoding="utf-8")
    print(f"wrote {len(output_rows)} rows: {args.csv_out.relative_to(ROOT)}")
    print(f"wrote report: {args.report_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
