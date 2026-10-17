"""
CLIQUE_bound_mono.py  —  上下界 + 单调性约束的 CLIQUE 模型

合并两个改进：
  1. 下界约束 (from CLIQUE_bound):
     pred = LB + (UB - LB) * sigmoid(logit)
  2. 单调性约束 (from CLIQUE_mono):
     chain-aware batching + hinge monotonicity loss

继承关系: CLIQUEestimator_bound_mono -> CLIQUEestimator_bound -> CLIQUEestimator
  - logit / forward / estimate / append_bound / build_lower_bound_table: 来自 CLIQUE_bound
  - build_model: 重写为 chain-aware 训练 + 双 loss（来自 CLIQUE_mono 的逻辑）

放置位置: src/CLIQUE_bound_mono.py

用法:
  python main_CLIQUE_bound_mono.py -d AUTHOR -w SEMO --seed 0 --lambda_mono 1.0
"""

import os
import json
import time
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import src.util as ut
from src.CLIQUE_bound import (
    CLIQUEestimator_bound,
    CLIQUEdataset_bound,
    collate_fn_hybrid_bound,
    evaluate_bound_model,
    get_bound_dl_test,
    _batch2cuda_bound,
)
from src.model import msle_loss
from src.util import hinge_monotonicity_loss


def _append_final_chain_batch(batches, final_batch):
    """Append the tail batch without ever leaving BatchNorm a singleton.

    Chain-aware batching keeps every chain intact, so its final remainder is not
    governed by ``total_rows % batch_size``.  Ratio sampling can also represent
    retained labels as singleton, edge-free chains.  If shuffling leaves exactly
    one such row at the end, merge it into the preceding batch instead of
    dropping a labeled row or sending a size-one batch through BatchNorm.
    """
    if not final_batch["queries"]:
        return
    if len(final_batch["queries"]) != 1 or not batches:
        batches.append(final_batch)
        return

    previous_batch = batches[-1]
    offset = len(previous_batch["queries"])
    for field in ("queries", "cards", "upper_bounds", "lower_bounds"):
        previous_batch[field].extend(final_batch[field])
    previous_batch["pair_tight_idx"].extend(
        offset + idx for idx in final_batch["pair_tight_idx"]
    )
    previous_batch["pair_loose_idx"].extend(
        offset + idx for idx in final_batch["pair_loose_idx"]
    )


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class CLIQUEestimator_bound_mono(CLIQUEestimator_bound):
    """
    上下界 + 单调性约束的 CLIQUE 模型。

    新增属性（在 bound 基础上）:
        lambda_mono: float, 单调性 loss 权重
        chain_index: list of dict, 链结构索引
    """

    def __init__(self):
        super().__init__()
        self.lambda_mono = 1.0
        self.mono_tolerance = 0.0
        self.chain_index = None
        # Keep the chain-aware batching/forward path even when lambda_mono is
        # zero.  This is used by the strict B1 matched-batch control.  The
        # default remains the historical lambda=0 -> BOUND fallback.
        self.force_chain_training = False
        self.training_path = None
        self.lb_chain_conflict_summary = None
        self.bimo_profile_enabled = _env_flag("BIMO_PROFILE")
        self.bimo_profile_epoch = _env_int("BIMO_PROFILE_EPOCH", 5)
        self.bimo_profile_exit = _env_flag("BIMO_PROFILE_EXIT", default=True)
        self.bimo_profile_startup_timings = {}
        self.bimo_profile_context = {}
        self.bimo_cache_chain_bounds = _env_flag(
            "BIMO_CACHE_CHAIN_BOUNDS", default=True
        )
        self.bimo_verify_chain_bounds_cache = _env_flag(
            "BIMO_VERIFY_CHAIN_BOUNDS_CACHE"
        )
        self.bimo_expected_chain_bounds_hash = os.getenv(
            "BIMO_EXPECTED_CHAIN_BOUNDS_HASH", ""
        ).strip()
        self.chain_pattern_bounds_cache = None
        self._bimo_profile_current_epoch = None
        self._bimo_profile_last_bounds_stats = {}
        self._bimo_profile_prepare_stats = {}
        self._bimo_profile_epoch_stats = {}
        self._bimo_profile_prev_bounds_hash = None
        self._bimo_profile_bounds_hash_by_epoch = {}
        self._bimo_profile_tracked_query = None
        self._bimo_profile_tracked_counts = {"ub": 0, "lb": 0}

    def _profile_now(self):
        if self.bimo_profile_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def chain_training_enabled(self):
        """Return whether build_model must use the chain-aware training path."""
        if self.chain_index is None:
            if self.force_chain_training:
                raise ValueError(
                    "force_chain_training requires a non-empty chain index"
                )
            return False
        return self.lambda_mono != 0 or self.force_chain_training

    def _hash_chain_pattern_bounds(self, pattern_bounds):
        hasher = hashlib.sha256()
        for pat, (ub, lb) in sorted(pattern_bounds.items()):
            hasher.update(str(pat).encode("utf-8", errors="replace"))
            hasher.update(b"\t")
            hasher.update(repr(float(ub)).encode("ascii"))
            hasher.update(b"\t")
            hasher.update(repr(float(lb)).encode("ascii"))
            hasher.update(b"\n")
        return hasher.hexdigest()

    def _collect_chain_patterns(self, chain_index):
        all_chain_patterns = set()
        for chain_info in chain_index:
            for pat in chain_info["patterns"]:
                all_chain_patterns.add(pat)
        return all_chain_patterns

    def _precompute_chain_pattern_bounds_cache(self, chain_index):
        all_chain_patterns = self._collect_chain_patterns(chain_index)
        profile = self.bimo_profile_enabled
        start = time.perf_counter()
        ub_time = 0.0
        lb_time = 0.0

        if profile and self._bimo_profile_tracked_query is None:
            self._bimo_profile_tracked_query = next(
                (pat for pat in sorted(all_chain_patterns) if pat != "%"),
                None,
            )

        cache = {}
        for pat in all_chain_patterns:
            if pat == "%":
                cache[pat] = (self.n_db, self.n_db)
                continue

            t0 = time.perf_counter() if profile else None
            ub = self.get_bound(pat)
            if profile:
                ub_time += time.perf_counter() - t0

            t0 = time.perf_counter() if profile else None
            lb = self.get_effective_lower_bound_for_query(pat, ub)
            if profile:
                lb_time += time.perf_counter() - t0
                if pat == self._bimo_profile_tracked_query:
                    self._bimo_profile_tracked_counts["ub"] += 1
                    self._bimo_profile_tracked_counts["lb"] += 1

            cache[pat] = (ub, lb)

        elapsed = time.perf_counter() - start
        cache_hash = self._hash_chain_pattern_bounds(cache)
        self.chain_pattern_bounds_cache = cache

        if profile:
            self.bimo_profile_startup_timings[
                "precompute_chain_bounds_cache"
            ] = elapsed
            self.bimo_profile_startup_timings[
                "precompute_chain_bounds_ub_lookup"
            ] = ub_time
            self.bimo_profile_startup_timings[
                "precompute_chain_bounds_lb_lookup"
            ] = lb_time

        print("\nBIMO chain bounds cache")
        print(f"  enabled: {self.bimo_cache_chain_bounds}")
        print(f"  unique chain patterns: {len(cache)}")
        print(f"  precompute time: {elapsed:.4f}s")
        print(f"  chain bounds hash: {cache_hash}")
        if self.bimo_verify_chain_bounds_cache:
            verify_start = time.perf_counter()
            uncached = {}
            for pat in all_chain_patterns:
                if pat == "%":
                    uncached[pat] = (self.n_db, self.n_db)
                else:
                    ub = self.get_bound(pat)
                    lb = self.get_effective_lower_bound_for_query(pat, ub)
                    uncached[pat] = (ub, lb)
            uncached_hash = self._hash_chain_pattern_bounds(uncached)
            mismatches = []
            for pat in sorted(all_chain_patterns):
                if cache.get(pat) != uncached.get(pat):
                    mismatches.append((pat, cache.get(pat), uncached.get(pat)))
                    if len(mismatches) >= 5:
                        break
            print("  cache verification:")
            print(f"    uncached recompute time: {time.perf_counter() - verify_start:.4f}s")
            print(f"    cache_hash: {cache_hash}")
            print(f"    uncached_hash: {uncached_hash}")
            print(f"    cache_equals_uncached: {cache_hash == uncached_hash and not mismatches}")
            if mismatches:
                print(f"    first_mismatches: {mismatches}")
        if (
            self.bimo_expected_chain_bounds_hash
            and cache_hash != self.bimo_expected_chain_bounds_hash
        ):
            raise RuntimeError(
                "BIMO chain bounds cache hash mismatch: "
                f"expected {self.bimo_expected_chain_bounds_hash}, got {cache_hash}"
            )
        return cache

    def _bound_profile_forward_count(self):
        data_name = self.bimo_profile_context.get("data_name")
        workload = self.bimo_profile_context.get("workload")
        if not data_name or not workload or not workload.endswith("_BIMO"):
            return None, None

        bound_workload = workload[:-5] + "_BOUND"
        train_path = os.path.join(
            "data", data_name, "training", bound_workload, "train.txt"
        )
        if not os.path.exists(train_path):
            return bound_workload, None

        with open(train_path, "r", encoding="utf-8") as f:
            return bound_workload, sum(1 for line in f if line.strip())

    def _print_bimo_profile_report(self, epoch, n_epoch):
        stats = self._bimo_profile_epoch_stats.get(epoch, {})
        prepare_stats = self._bimo_profile_prepare_stats.get(epoch, {})
        stage_times = stats.get("stage_times", {})
        prepare_time = stats.get("prepare_chain_batches", 0.0)
        batch_total = stats.get("batch_loop_total", 0.0)
        epoch_total = prepare_time + batch_total
        pct_base = max(epoch_total, 1e-12)

        print("\n" + "=" * 72)
        print(f"BIMO PROFILE REPORT (epoch {epoch})")
        print("=" * 72)

        print("\n[one_time_setup]")
        for name, value in self.bimo_profile_startup_timings.items():
            print(f"  {name:36s} {value:12.4f}s")
        print("  note: build_count appears twice if main pre-build_count and build() both run.")

        bounds_info = self._bimo_profile_bounds_hash_by_epoch.get(epoch, {})
        same_prev = bounds_info.get("same_as_previous")
        same_text = "N/A"
        if same_prev is True:
            same_text = "YES"
        elif same_prev is False:
            same_text = "NO"
        unique_hashes = {
            item.get("hash")
            for item in self._bimo_profile_bounds_hash_by_epoch.values()
            if item.get("hash")
        }

        print("\n[repeatability_checks]")
        print(f"  chain_bounds_hash_epoch_{epoch}: {bounds_info.get('hash', 'N/A')}")
        print(f"  same_as_previous_epoch: {same_text}")
        print(f"  unique_chain_bounds_hashes_seen: {len(unique_hashes)}")
        cache_active = (
            self.bimo_cache_chain_bounds and self.chain_pattern_bounds_cache is not None
        )
        print(f"  chain_bounds_cache_active: {cache_active}")
        print(
            "  tracked_query: "
            f"{self._bimo_profile_tracked_query if self._bimo_profile_tracked_query else 'N/A'}"
        )
        print(
            "  tracked_query_ub_calls_observed: "
            f"{self._bimo_profile_tracked_counts.get('ub', 0)}"
        )
        print(
            "  tracked_query_lb_calls_observed: "
            f"{self._bimo_profile_tracked_counts.get('lb', 0)}"
        )
        print(
            "  projected_calls_if_no_early_stop: "
            f"{1 if cache_active else n_epoch} per tracked chain query"
        )
        if cache_active:
            print(
                "  conclusion: LB/UB lookup is served from the precomputed "
                "chain-bounds cache inside epochs."
            )
        else:
            print(
                "  conclusion: LB/UB lookup is outside the batch loop, but "
                "chain bounds are recomputed once per epoch."
            )
        print(
            "  conclusion: pair indices are built in _prepare_chain_batches; "
            "the batch loop only reuses them as index lists."
        )

        print("\n[prepare_chain_batches_breakdown]")
        for name in [
            "card_dict",
            "collect_chain_patterns",
            "chain_bounds_total",
            "chain_bounds_ub_lookup",
            "chain_bounds_lb_lookup",
            "chain_order_shuffle",
            "batch_materialize",
            "pair_index_build",
            "prepare_other",
        ]:
            value = prepare_stats.get(name, 0.0)
            print(f"  {name:36s} {value:12.4f}s {100 * value / pct_base:8.2f}%")
        print(
            f"  unique_chain_patterns: {prepare_stats.get('unique_chain_patterns', 0)}"
        )
        print(f"  chain_pairs: {prepare_stats.get('chain_pairs', 0)}")
        print(f"  chain_batches: {prepare_stats.get('chain_batches', 0)}")

        print("\n[epoch_time_accounting]")
        print(
            "  epoch_total = prepare_chain_batches "
            f"({prepare_time:.4f}s) + batch_loop ({batch_total:.4f}s) "
            f"= {epoch_total:.4f}s"
        )
        print("  note: epoch_total excludes validation/test after the train batch loop.")
        rows = [("prepare_chain_batches", prepare_time)]
        rows.extend((name, value) for name, value in stage_times.items())
        rows.sort(key=lambda item: item[1], reverse=True)
        print("  stage                            seconds      pct")
        for name, value in rows:
            print(f"  {name:30s} {value:10.4f} {100 * value / pct_base:8.2f}%")

        bimo_labeled = stats.get("bimo_forward_labeled", 0)
        bimo_unlabeled = stats.get("bimo_forward_unlabeled", 0)
        bimo_total = bimo_labeled + bimo_unlabeled
        bound_workload, bound_forward = self._bound_profile_forward_count()
        print("\n[forward_sample_counts]")
        print(f"  BIMO labeled samples per epoch:   {bimo_labeled}")
        print(f"  BIMO unlabeled chain patterns:    {bimo_unlabeled}")
        print(f"  BIMO total forward rows:          {bimo_total}")
        print(
            "  note: forward is timed once over all BIMO rows; it is not "
            "split into extra labeled/unlabeled forwards."
        )
        if bound_forward is None:
            print(f"  BOUND workload inferred:          {bound_workload or 'N/A'}")
            print("  BOUND total forward rows:         N/A")
            print("  BIMO/BOUND forward-row ratio:     N/A")
        else:
            ratio = bimo_total / max(bound_forward, 1)
            print(f"  BOUND workload inferred:          {bound_workload}")
            print(f"  BOUND total forward rows:         {bound_forward}")
            print(f"  BIMO/BOUND forward-row ratio:     {ratio:.4f}x")

        a_repeat = (
            prepare_stats.get("chain_bounds_total", 0.0)
            + prepare_stats.get("pair_index_build", 0.0)
        )
        b_compute = (
            stage_times.get("forward", 0.0)
            + stage_times.get("mono_loss", 0.0)
            + stage_times.get("backward", 0.0)
        )
        print("\n[ab_split]")
        print(
            "  A_redundant_cacheable = chain_bounds_total + pair_index_build "
            f"= {a_repeat:.4f}s ({100 * a_repeat / pct_base:.2f}%)"
        )
        print(
            "  B_intrinsic_chain_compute = forward + mono_loss + backward "
            f"= {b_compute:.4f}s ({100 * b_compute / pct_base:.2f}%)"
        )
        print(
            "  A_redundant_cacheable_pct is the measured upper bound for "
            "caching away the repeated chain-bound/pair-index work."
        )
        print("=" * 72 + "\n")

    def _build_chain_pattern_bounds(self, all_chain_patterns):
        """
        为链内所有模式查询上界和下界。

        返回:
            dict: {pattern: (upper_bound, lower_bound)}
        """
        profile = self.bimo_profile_enabled
        ub_time = 0.0
        lb_time = 0.0
        total_start = time.perf_counter() if profile else None
        if self.bimo_cache_chain_bounds and self.chain_pattern_bounds_cache is not None:
            pattern_bounds = {}
            missing_patterns = []
            for pat in all_chain_patterns:
                if pat in self.chain_pattern_bounds_cache:
                    pattern_bounds[pat] = self.chain_pattern_bounds_cache[pat]
                else:
                    missing_patterns.append(pat)

            if missing_patterns:
                raise KeyError(
                    "BIMO chain bounds cache missing patterns: "
                    f"{missing_patterns[:5]} (total={len(missing_patterns)})"
                )

            if profile:
                self._bimo_profile_last_bounds_stats = {
                    "total": time.perf_counter() - total_start,
                    "ub_lookup": 0.0,
                    "lb_lookup": 0.0,
                    "n_patterns": len(all_chain_patterns),
                    "n_non_root_patterns": sum(
                        1 for pat in all_chain_patterns if pat != "%"
                    ),
                    "cache_hits": len(pattern_bounds),
                }
            return pattern_bounds

        if profile and self._bimo_profile_tracked_query is None:
            self._bimo_profile_tracked_query = next(
                (pat for pat in sorted(all_chain_patterns) if pat != "%"),
                None,
            )

        pattern_bounds = {}
        for pat in all_chain_patterns:
            if pat == "%":
                pattern_bounds[pat] = (self.n_db, self.n_db)
            else:
                if profile:
                    t0 = time.perf_counter()
                    ub = self.get_bound(pat)
                    ub_time += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    lb = self.get_effective_lower_bound_for_query(pat, ub)
                    lb_time += time.perf_counter() - t0
                    if pat == self._bimo_profile_tracked_query:
                        self._bimo_profile_tracked_counts["ub"] += 1
                        self._bimo_profile_tracked_counts["lb"] += 1
                else:
                    ub = self.get_bound(pat)
                    lb = self.get_effective_lower_bound_for_query(pat, ub)
                pattern_bounds[pat] = (ub, lb)
        if profile:
            self._bimo_profile_last_bounds_stats = {
                "total": time.perf_counter() - total_start,
                "ub_lookup": ub_time,
                "lb_lookup": lb_time,
                "n_patterns": len(all_chain_patterns),
                "n_non_root_patterns": sum(
                    1 for pat in all_chain_patterns if pat != "%"
                ),
            }
        return pattern_bounds

    def _prepare_chain_batches(self, train_data, chain_index, char_dict, bs):
        """
        构建 chain-aware batches（带上下界）。

        每个 batch 包含若干条完整链的所有模式。
        有标签的模式参与 MSLE loss，所有相邻对参与单调性 loss。

        返回:
            list of dict: 每个 batch 包含:
                - "queries": list of str
                - "cards": list of float (-1 表示无标签)
                - "upper_bounds": list of float
                - "lower_bounds": list of float
                - "pair_tight_idx": list of int
                - "pair_loose_idx": list of int
        """
        profile = self.bimo_profile_enabled
        epoch = self._bimo_profile_current_epoch
        prepare_start = time.perf_counter() if profile else None

        # pattern -> card
        t0 = time.perf_counter() if profile else None
        card_dict = {}
        for query, card in train_data:
            card_dict[query] = card
        card_dict["%"] = self.n_db
        card_dict_time = time.perf_counter() - t0 if profile else 0.0

        # 所有链内模式查上下界
        t0 = time.perf_counter() if profile else None
        all_chain_patterns = self._collect_chain_patterns(chain_index)
        collect_patterns_time = time.perf_counter() - t0 if profile else 0.0
        pattern_bounds = self._build_chain_pattern_bounds(all_chain_patterns)
        if profile:
            bounds_hash = self._hash_chain_pattern_bounds(pattern_bounds)
            previous_hash = self._bimo_profile_prev_bounds_hash
            self._bimo_profile_bounds_hash_by_epoch[epoch] = {
                "hash": bounds_hash,
                "same_as_previous": (
                    None if previous_hash is None else bounds_hash == previous_hash
                ),
            }
            self._bimo_profile_prev_bounds_hash = bounds_hash
            if (
                self.bimo_expected_chain_bounds_hash
                and bounds_hash != self.bimo_expected_chain_bounds_hash
            ):
                raise RuntimeError(
                    "BIMO chain bounds epoch hash mismatch: "
                    f"expected {self.bimo_expected_chain_bounds_hash}, got {bounds_hash}"
                )

        # 打乱链顺序
        t0 = time.perf_counter() if profile else None
        chain_order = list(range(len(chain_index)))
        np.random.shuffle(chain_order)
        chain_order_time = time.perf_counter() - t0 if profile else 0.0

        batches = []
        n_total_pairs = 0
        n_lb_violation = 0
        n_infeasible = 0
        batch_materialize_time = 0.0
        pair_index_time = 0.0

        current_batch = {
            "queries": [], "cards": [],
            "upper_bounds": [], "lower_bounds": [],
            "pair_tight_idx": [], "pair_loose_idx": [],
        }

        for ci in chain_order:
            chain_info = chain_index[ci]
            patterns = chain_info["patterns"]
            is_anchor = chain_info["is_anchor"]

            offset = len(current_batch["queries"])

            t0 = time.perf_counter() if profile else None
            for i, (pat, anchor) in enumerate(zip(patterns, is_anchor)):
                if pat in card_dict:
                    card = card_dict[pat]
                elif pat == "%":
                    card = self.n_db
                else:
                    card = -1

                ub, lb = pattern_bounds.get(pat, (self.n_db, 0))

                current_batch["queries"].append(pat)
                current_batch["cards"].append(float(card))
                current_batch["upper_bounds"].append(float(ub))
                current_batch["lower_bounds"].append(float(lb))
            if profile:
                batch_materialize_time += time.perf_counter() - t0

            # 链内相邻对
            t0 = time.perf_counter() if profile else None
            for i in range(len(patterns) - 1):
                _, lb_tight = pattern_bounds.get(patterns[i], (self.n_db, 0))
                ub_loose, lb_loose = pattern_bounds.get(
                    patterns[i + 1], (self.n_db, 0)
                )

                if lb_tight > lb_loose:
                    n_lb_violation += 1
                if lb_tight > ub_loose:
                    n_infeasible += 1
                n_total_pairs += 1

                current_batch["pair_tight_idx"].append(offset + i)
                current_batch["pair_loose_idx"].append(offset + i + 1)
            if profile:
                pair_index_time += time.perf_counter() - t0

            if len(current_batch["queries"]) >= bs:
                batches.append(current_batch)
                current_batch = {
                    "queries": [], "cards": [],
                    "upper_bounds": [], "lower_bounds": [],
                    "pair_tight_idx": [], "pair_loose_idx": [],
                }

        _append_final_chain_batch(batches, current_batch)

        if self.lb_chain_conflict_summary is None:
            lb_violation_rate = n_lb_violation / max(n_total_pairs, 1)
            infeasible_rate = n_infeasible / max(n_total_pairs, 1)
            self.lb_chain_conflict_summary = {
                "total_pairs": n_total_pairs,
                "lb_violation": n_lb_violation,
                "lb_violation_rate": lb_violation_rate,
                "infeasible": n_infeasible,
                "infeasible_rate": infeasible_rate,
            }
            print("\nLB chain conflict diagnostics")
            print(f"  total pairs: {n_total_pairs}")
            print(
                f"  LB_tight > LB_loose: {n_lb_violation} "
                f"({100 * lb_violation_rate:.4f}%)"
            )
            print(
                f"  LB_tight > UB_loose: {n_infeasible} "
                f"({100 * infeasible_rate:.4f}%)"
            )

        if profile:
            bounds_stats = self._bimo_profile_last_bounds_stats
            prepare_total = time.perf_counter() - prepare_start
            known_prepare = (
                card_dict_time
                + collect_patterns_time
                + bounds_stats.get("total", 0.0)
                + chain_order_time
                + batch_materialize_time
                + pair_index_time
            )
            self._bimo_profile_prepare_stats[epoch] = {
                "card_dict": card_dict_time,
                "collect_chain_patterns": collect_patterns_time,
                "chain_bounds_total": bounds_stats.get("total", 0.0),
                "chain_bounds_ub_lookup": bounds_stats.get("ub_lookup", 0.0),
                "chain_bounds_lb_lookup": bounds_stats.get("lb_lookup", 0.0),
                "chain_order_shuffle": chain_order_time,
                "batch_materialize": batch_materialize_time,
                "pair_index_build": pair_index_time,
                "prepare_other": max(prepare_total - known_prepare, 0.0),
                "unique_chain_patterns": bounds_stats.get("n_patterns", 0),
                "chain_pairs": n_total_pairs,
                "chain_batches": len(batches),
                "total_batch_queries": sum(
                    len(batch["queries"]) for batch in batches
                ),
            }

        return batches

    def _encode_batch(self, batch_dict, char_dict):
        """
        将 batch_dict 编码为模型可接受的 tensor 格式。

        返回:
            xs: tuple (encoded_seqs, lens, upper_bounds, lower_bounds)
            ys: Tensor of cards
            pair_tight_idx: list of int
            pair_loose_idx: list of int
        """
        patterns = [
            ut.canonicalize_like_query(q, is_last_flip=True)
            for q in batch_dict["queries"]
        ]
        encoded = ut.string_encoding(patterns, char_dict)
        lens = [len(e) for e in encoded]
        padded = ut.keras_pad_sequences(encoded, padding="post")

        xs_tensor = torch.LongTensor(padded)
        ub_tensor = torch.FloatTensor(batch_dict["upper_bounds"])
        lb_tensor = torch.FloatTensor(batch_dict["lower_bounds"])
        ys_tensor = torch.FloatTensor(batch_dict["cards"])

        xs = (xs_tensor, lens, ub_tensor, lb_tensor)
        return xs, ys_tensor, batch_dict["pair_tight_idx"], batch_dict["pair_loose_idx"]

    def build_model(self, train_data, valid_data, test_data):
        """
        Chain-aware 训练 + MSLE + 单调性 loss + 上下界约束。
        """
        model = self
        lr = self.lr
        l2 = self.l2
        n_epoch = self.n_epoch
        bs = self.bs
        patience = self.patience
        sw = self.sw
        seed = self.seed
        char_dict = self.char_dict
        model_path = self.model_path
        lambda_mono = self.lambda_mono
        mono_tolerance = getattr(self, "mono_tolerance", 0.0)
        chain_index = self.chain_index
        label_free_selection = getattr(self, "label_free_selection", False)
        selection_trace = []

        # The strict B1 control deliberately stays on this chain-aware path at
        # lambda=0.  Ordinary lambda=0 calls preserve the historical fallback.
        if not self.chain_training_enabled():
            self.training_path = "bound_fallback"
            labeled_train_data = ut.labeled_training_rows(train_data)
            print(
                "No chain index or lambda_mono=0, falling back to "
                f"CLIQUE_bound training with {len(labeled_train_data)}/"
                f"{len(train_data)} labeled rows"
            )
            return super().build_model(
                labeled_train_data, valid_data, test_data
            )

        self.training_path = (
            "chain_strict_zero_mono"
            if self.force_chain_training and lambda_mono == 0
            else "chain_joint"
        )

        # 记录 train 的下界覆盖率；valid/test 用标准 bound dataloader 评估。
        t0 = time.perf_counter() if self.bimo_profile_enabled else None
        self.record_lower_bound_coverage("train", train_data)
        if self.bimo_profile_enabled:
            self.bimo_profile_startup_timings["train_lb_coverage"] = (
                time.perf_counter() - t0
            )
        t0 = time.perf_counter() if self.bimo_profile_enabled else None
        valid_data_bounded = self.append_bound(valid_data, split_name="valid")
        if self.bimo_profile_enabled:
            self.bimo_profile_startup_timings["append_bound_valid"] = (
                time.perf_counter() - t0
            )
        t0 = time.perf_counter() if self.bimo_profile_enabled else None
        test_data_bounded = self.append_bound(test_data, split_name="test")
        if self.bimo_profile_enabled:
            self.bimo_profile_startup_timings["append_bound_test"] = (
                time.perf_counter() - t0
            )

        t0 = time.perf_counter() if self.bimo_profile_enabled else None
        dl_valid = get_bound_dl_test(valid_data_bounded, bs, char_dict)
        dl_test = get_bound_dl_test(test_data_bounded, bs, char_dict)
        if self.bimo_profile_enabled:
            self.bimo_profile_startup_timings["valid_test_dataloader_build"] = (
                time.perf_counter() - t0
            )

        ut.set_seed(seed)
        if self.bimo_cache_chain_bounds:
            self._precompute_chain_pattern_bounds_cache(chain_index)
        else:
            print("\nBIMO chain bounds cache")
            print("  enabled: False")
            print("  using old path: chain bounds are recomputed each epoch")

        ut.set_seed(seed)
        t0 = time.perf_counter() if self.bimo_profile_enabled else None
        model.cuda()
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
        if self.bimo_profile_enabled:
            self.bimo_profile_startup_timings["model_cuda_and_optimizer"] = (
                time.perf_counter() - t0
            )
        bs_step = 0
        best_val_score = float("inf")
        best_epoch = -1
        best_test_score = -1
        epochs_since_improvement = 0
        build_time = 0
        n_labeled = sum(1 for _, card in train_data if card >= 0)
        n_unlabeled = len(train_data) - n_labeled

        print(f"\n{'=' * 60}")
        print(f"Training: Lower Bound + Monotonicity")
        print(f"  lambda_mono = {lambda_mono}")
        print(f"  mono_tolerance = {mono_tolerance}")
        print(f"  chains = {len(chain_index)}")
        print(f"  train samples = {len(train_data)} total")
        print(f"  labeled train samples = {n_labeled}")
        print(f"  unlabeled train samples = {n_unlabeled}")
        print(f"  label_free_selection = {label_free_selection}")
        print(f"  force_chain_training = {self.force_chain_training}")
        print(f"  training_path = {self.training_path}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, n_epoch + 1):
            start_time = time.time()
            model.train()

            # 每 epoch 重建 chain-aware batches
            np.random.seed(seed + epoch)
            self._bimo_profile_current_epoch = epoch
            prepare_start = time.perf_counter() if self.bimo_profile_enabled else None
            chain_batches = self._prepare_chain_batches(
                train_data, chain_index, char_dict, bs
            )
            if self.bimo_profile_enabled:
                self._bimo_profile_epoch_stats.setdefault(epoch, {})
                self._bimo_profile_epoch_stats[epoch][
                    "prepare_chain_batches"
                ] = time.perf_counter() - prepare_start

            epoch_msle_loss = 0.0
            epoch_mono_loss = 0.0
            n_batches = 0
            profile_epoch = (
                self.bimo_profile_enabled and epoch == self.bimo_profile_epoch
            )
            if profile_epoch:
                stage_times = {
                    "data_loading": 0.0,
                    "lb_ub_lookup": 0.0,
                    "mono_pair_build": 0.0,
                    "forward": 0.0,
                    "msle_loss": 0.0,
                    "mono_loss": 0.0,
                    "backward": 0.0,
                    "optimizer_step": 0.0,
                    "other": 0.0,
                }
                batch_loop_start = self._profile_now()
                bimo_forward_labeled = 0
                bimo_forward_unlabeled = 0

            for batch_dict in tqdm(
                chain_batches,
                total=len(chain_batches),
                desc=f"[Epoch {epoch:3d}/{n_epoch:3d}]",
            ):
                batch_start = self._profile_now() if profile_epoch else None
                stage_sum_before = sum(stage_times.values()) if profile_epoch else 0.0
                optim.zero_grad()

                t0 = self._profile_now() if profile_epoch else None
                xs, ys, pair_tight_idx, pair_loose_idx = self._encode_batch(
                    batch_dict, char_dict
                )

                # GPU
                xs = (xs[0].cuda(), xs[1], xs[2].cuda(), xs[3].cuda())
                ys = ys.cuda()
                if profile_epoch:
                    stage_times["data_loading"] += self._profile_now() - t0

                # 前向（走 CLIQUE_bound 的 logit: LB + (UB-LB)*sigmoid）
                t0 = self._profile_now() if profile_epoch else None
                preds = model(xs)
                if profile_epoch:
                    stage_times["forward"] += self._profile_now() - t0

                # ---- MSLE loss（仅有标签样本）----
                t0 = self._profile_now() if profile_epoch else None
                labeled_mask = ys >= 0
                if labeled_mask.any():
                    loss_msle = msle_loss(ys[labeled_mask], preds[labeled_mask])
                else:
                    loss_msle = torch.tensor(0.0, device=preds.device)
                if profile_epoch:
                    n_labeled_batch = int(labeled_mask.detach().sum().cpu())
                    bimo_forward_labeled += n_labeled_batch
                    bimo_forward_unlabeled += int(ys.numel()) - n_labeled_batch
                    stage_times["msle_loss"] += self._profile_now() - t0

                # ---- 单调性 loss（链内相邻对）----
                if pair_tight_idx and pair_loose_idx:
                    t0 = self._profile_now() if profile_epoch else None
                    tight_idx = torch.LongTensor(pair_tight_idx)
                    loose_idx = torch.LongTensor(pair_loose_idx)
                    if profile_epoch:
                        stage_times["mono_pair_build"] += self._profile_now() - t0

                    t0 = self._profile_now() if profile_epoch else None
                    preds_tight = preds[tight_idx]
                    preds_loose = preds[loose_idx]
                    loss_mono = hinge_monotonicity_loss(
                        preds_tight,
                        preds_loose,
                        tolerance=mono_tolerance,
                    )
                    if profile_epoch:
                        stage_times["mono_loss"] += self._profile_now() - t0
                else:
                    t0 = self._profile_now() if profile_epoch else None
                    loss_mono = torch.tensor(0.0, device=preds.device)
                    if profile_epoch:
                        stage_times["mono_loss"] += self._profile_now() - t0

                # ---- 总 loss ----
                loss = loss_msle + lambda_mono * loss_mono
                t0 = self._profile_now() if profile_epoch else None
                loss.backward()
                if profile_epoch:
                    stage_times["backward"] += self._profile_now() - t0
                t0 = self._profile_now() if profile_epoch else None
                optim.step()
                if profile_epoch:
                    stage_times["optimizer_step"] += self._profile_now() - t0

                epoch_msle_loss += float(loss_msle.detach().cpu())
                epoch_mono_loss += float(loss_mono.detach().cpu())
                n_batches += 1
                bs_step += 1

                sw.add_scalars("Loss", {
                    "train_msle": float(loss_msle.detach().cpu()),
                    "train_mono": float(loss_mono.detach().cpu()),
                    "train_total": float(loss.detach().cpu()),
                }, global_step=bs_step)
                if profile_epoch:
                    known_batch = sum(stage_times.values()) - stage_sum_before
                    batch_elapsed = self._profile_now() - batch_start
                    stage_times["other"] += max(batch_elapsed - known_batch, 0.0)

            avg_msle = epoch_msle_loss / max(n_batches, 1)
            avg_mono = epoch_mono_loss / max(n_batches, 1)
            if profile_epoch:
                self._bimo_profile_epoch_stats[epoch]["stage_times"] = stage_times
                self._bimo_profile_epoch_stats[epoch]["batch_loop_total"] = (
                    self._profile_now() - batch_loop_start
                )
                self._bimo_profile_epoch_stats[epoch][
                    "bimo_forward_labeled"
                ] = bimo_forward_labeled
                self._bimo_profile_epoch_stats[epoch][
                    "bimo_forward_unlabeled"
                ] = bimo_forward_unlabeled
                self._print_bimo_profile_report(epoch, n_epoch)
                if self.bimo_profile_exit:
                    raise SystemExit(0)

            if label_free_selection:
                selection_trace.append({'epoch': epoch, 'train_mono': float(avg_mono),
                                        'train_msle': float(avg_msle)})
                end_time = time.time()
                build_time += end_time - start_time

                print(f"  Epoch {epoch:3d}: msle={avg_msle:.4f} mono={avg_mono:.4f} "
                      f"selection_mono={avg_mono:.4f}")

                if avg_mono < best_val_score:
                    best_val_score = avg_mono
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_path)
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if epochs_since_improvement >= patience:
                    print(f"Early stopping triggered.")
                    break

                continue

            # ---- Validation ----
            losses, q_errs, estimates = evaluate_bound_model(model, dl_valid)
            loss_ = sum(losses) / len(losses)
            avg_q_err_ = sum(q_errs) / len(q_errs)
            max_q_err_ = max(q_errs)

            sw.add_scalars("Loss", {"valid": loss_}, global_step=bs_step)
            sw.add_scalars("ACC", {"valid": avg_q_err_}, global_step=bs_step)
            sw.add_scalars("ACC", {"valid_max": max_q_err_}, global_step=bs_step)

            end_time = time.time()
            build_time += end_time - start_time

            print(f"  Epoch {epoch:3d}: msle={avg_msle:.4f} mono={avg_mono:.4f} "
                  f"val_q_err={avg_q_err_:.4f} val_max={max_q_err_:.2f}")

            q_err_ = avg_q_err_
            if q_err_ < best_val_score:
                best_val_score = q_err_
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # ---- Test ----
            losses, q_errs, estimates = evaluate_bound_model(model, dl_test)
            avg_q_err_ = sum(q_errs) / len(q_errs)
            max_q_err_ = max(q_errs)
            sw.add_scalars("ACC", {"test": avg_q_err_}, global_step=bs_step)
            sw.add_scalars("ACC", {"test_max": max_q_err_}, global_step=bs_step)

            if epochs_since_improvement == 0:
                best_test_score = avg_q_err_

            if epochs_since_improvement >= patience:
                print(f"Early stopping triggered.")
                break

        if label_free_selection:
            self.selection_best_epoch = int(best_epoch)
            self.selection_best_score = float(best_val_score)
            self.selection_epochs_completed = int(epoch)
            with open(os.path.join(os.path.dirname(model_path), 'selection_trace.json'), 'w') as handle:
                json.dump({'selection_policy': 'train_mono', 'best_epoch': best_epoch,
                           'best_score': best_val_score, 'patience': patience,
                           'epochs': selection_trace}, handle, indent=2)
        print(f"{best_val_score = }, {best_test_score = }, {patience = }, {best_epoch = }")
        model.load_state_dict(torch.load(model_path))
        return build_time

    def build(self, data_name, db, train_data, valid_data, test_data):
        """BIMO build path with optional profiling probes; training logic is unchanged."""
        seed = self.seed
        self.n_db = len(db)
        np.random.seed(seed)

        t0 = time.perf_counter() if self.bimo_profile_enabled else None
        build_time_count = self.build_count(data_name)
        if self.bimo_profile_enabled:
            self.bimo_profile_startup_timings["build_count_inside_build"] = (
                time.perf_counter() - t0
            )
        print(f"{build_time_count = }")

        if self.use_lower_bound:
            full_scan = getattr(self, "full_scan", True)
            t0 = time.perf_counter() if self.bimo_profile_enabled else None
            self.build_lower_bound_table(
                data_name,
                db,
                top_k_strings=getattr(self, "lb_top_k_strings", 500),
                top_k_substrings=getattr(self, "lb_top_k_substrings", 500),
                min_substr_len=getattr(self, "lb_min_substr_len", 5),
                max_substr_len=getattr(self, "lb_max_substr_len", 10),
                min_freq_ratio=getattr(self, "lb_min_freq_ratio", 0.0005),
                full_scan=full_scan,
            )
            if self.bimo_profile_enabled:
                self.bimo_profile_startup_timings[
                    "build_lower_bound_table"
                ] = time.perf_counter() - t0

        build_time_model = self.build_model(train_data, valid_data, test_data)
        build_time = build_time_count + build_time_model
        return build_time
