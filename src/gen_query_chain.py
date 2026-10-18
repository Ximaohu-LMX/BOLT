"""
gen_query_chain.py  —  SEMO 链式查询生成器

三种生成模式：

  Mode 1: --generate  完全独立生成
    从数据集字符串出发，随机采样生成初始模式，Phase2退化构建链，
    按 8:1:1 切分 train/valid/test。不依赖任何已有 workload。

    python gen_query_chain.py -d AUTHOR --generate --num_chains 10000 --anchor_ratio 0.5
    python calculate_cardinality.py --query_file ... --output_file ...
    python gen_query_chain.py -d AUTHOR --fill_card --dst_workload SEMO
    python main_UB_ORDER.py -d AUTHOR -w SEMO --seed 0 --lambda_order 1.0

  Mode 2: --extend  基于已有 workload 扩展
    从原 workload 的 train.txt 提取查询作为链头，退化生成链数据，
    test/valid 从原 workload 复制。

    python gen_query_chain.py -d AUTHOR --extend --src_workload CLIQUE
    python calculate_cardinality.py --query_file ... --output_file ...
    python gen_query_chain.py -d AUTHOR --fill_card --dst_workload SEMO

    目录结构：
      data/AUTHOR/training/SEMO/
        ├── train.txt          # 链式训练数据（query,card 格式）
        ├── valid.txt          # 从原 workload 复制
        ├── test.txt           # 从原 workload 复制
        ├── chain_index.json   # 链结构索引（训练循环用）
        ├── chain_query.txt    # 需要查基数的模式列表
        └── chain_card.txt     # 基数查询结果（tab 分隔）

  Mode 3: --chain_only  纯正则化
    只在原 workload 目录中生成 chain_index.json，不修改任何训练文件。
    训练时用原始数据 + 单调性约束。

    python gen_query_chain.py -d AUTHOR --chain_only --chain_length 3
    python main_UB_ORDER.py -d AUTHOR -w CLIQUE --seed 0 --lambda_order 1.0

  回填基数（所有模式共用）：
    python gen_query_chain.py -d AUTHOR --fill_card --dst_workload SEMO
"""

import numpy as np
import random
import os
import sys
import json
import shutil
import argparse
import math
from tqdm import tqdm

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.util import (
    read_strings,
    save_query,
    canonicalize_like_query,
    Qtype,
    read_training,
    generate_degradation_chain,
    sample_anchors,
    load_chain_index,
    build_chain_train_data,
)


# =====================================================
#  Phase 1: 从字符串生成初始模式
# =====================================================

def string_to_initial_pattern(word, query_type, max_internal_pct=1,
                               min_segment_len=1, rng=None):
    """
    从一个字符串生成初始 LIKE 模式（Phase 1）。

    过程：
      1. 根据 query_type 确定首尾 % 位置
      2. 在字符串内部随机选 0~max_internal_pct 个位置插入 %
      3. 组装成 LIKE 模式

    参数:
        word: str, 数据集中的原始字符串
        query_type: str, 'SUBSTR'/'PREFIX'/'SUFFIX'
        max_internal_pct: int, 最多插入几个内部 %（默认1，即最多2段）
        min_segment_len: int, 每个片段的最小长度
        rng: random.Random

    返回:
        str: 初始模式，如 '%John%Smith%' 或 'March%'
    """
    if rng is None:
        rng = random.Random()

    n = len(word)
    if n == 0:
        return None

    # 确定内部 % 插入点数量
    max_possible = max(0, (n // max(min_segment_len + 1, 2)) - 1)
    num_internal = rng.randint(0, min(max_internal_pct, max_possible))

    # 选择内部插入位置（确保每段 >= min_segment_len）
    internal_positions = []
    if num_internal > 0:
        available = list(range(min_segment_len, n - min_segment_len + 1))
        rng.shuffle(available)
        for pos in available:
            if len(internal_positions) >= num_internal:
                break
            # 确保和已选位置以及首尾保持距离
            all_pos = sorted(internal_positions + [0, n])
            if all(abs(pos - p) >= min_segment_len for p in all_pos):
                internal_positions.append(pos)

    internal_positions = sorted(internal_positions)

    # 构建模式
    segments = []
    prev = 0
    for pos in internal_positions:
        segments.append(word[prev:pos])
        prev = pos
    segments.append(word[prev:])

    # 过滤空段
    segments = [s for s in segments if s]
    if not segments:
        return None

    # 按 query_type 加首尾 %
    if query_type == 'SUBSTR':
        pattern = '%' + '%'.join(segments) + '%'
    elif query_type == 'PREFIX':
        pattern = '%'.join(segments) + '%'
    elif query_type == 'SUFFIX':
        pattern = '%' + '%'.join(segments)
    else:
        pattern = '%'.join(segments)

    # 清理连续 %%
    while '%%' in pattern:
        pattern = pattern.replace('%%', '%')

    return pattern


def compress_chain(chain, max_len, min_literal_len=0,
                   compress_power_alpha=0.8,
                   compress_power_max=3.0):
    """
    压缩链到指定长度，保留头尾，并对长链尾部加密采样。
    过滤掉 literal 长度 <= min_literal_len 的模式（被 N-gram 表覆盖）。

    参数:
        chain: list of str
        max_len: int, 目标长度
        min_literal_len: int, 过滤阈值（0=不过滤）
        compress_power_alpha: 动态 power 强度，越大越偏向链尾
        compress_power_max: power 上限

    返回:
        list of str
    """
    # 先过滤短模式（保留链头）
    if min_literal_len > 0 and len(chain) > 1:
        from src.util import _get_literal_length
        filtered = [chain[0]]  # 始终保留链头
        for pat in chain[1:]:
            if _get_literal_length(pat) > min_literal_len:
                filtered.append(pat)
        chain = filtered if len(filtered) >= 2 else chain[:2]

    n = len(chain)
    if n <= max_len:
        return chain

    # 始终保留首尾
    if max_len <= 2:
        return [chain[0], chain[-1]]

    # 动态尾部加密采样：
    # p=1 等距；p>1 时 y = 1 - (1-x)^p 让采样点更靠近链尾。
    ratio = max(n / max(max_len, 1), 1.0)
    power = 1.0 + max(compress_power_alpha, 0.0) * math.log(ratio)
    power = min(max(power, 1.0), max(compress_power_max, 1.0))

    raw_indices = []
    for i in range(max_len):
        x = i / (max_len - 1)
        y = 1.0 - ((1.0 - x) ** power)
        raw_indices.append(round(y * (n - 1)))

    indices = sorted(set(int(idx) for idx in raw_indices if 0 <= idx < n))
    if 0 not in indices:
        indices.insert(0, 0)
    if n - 1 not in indices:
        indices.append(n - 1)
    indices = sorted(set(indices))
    return [chain[i] for i in indices]


# =====================================================
#  Mode 1: 完全独立生成 (--generate)
# =====================================================

def generate_independent_data(
    data_name,
    dst_workload="SEMO",
    num_chains=10000,
    max_chain_length=20,
    compress_to=10,
    compress_power_alpha=0.8,
    compress_power_max=3.0,
    anchor_ratio=1.0,
    max_internal_pct=1,
    min_segment_len=1,
    substr_prob=0.5,
    prefix_prob=0.25,
    suffix_prob=0.25,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    degrade_char_prob=0.4,
    degrade_shrink_prob=0.5,
    degrade_under_to_pct_prob=0.0,
    degrade_insert_pct_prob=0.1,
    max_wildcard_ratio=0.2,
    chain_seed=42,
    max_word_len=30,
    min_word_len=1,
    min_literal_len=0,
):
    """
    完全独立生成训练/验证/测试数据。

    流程:
      1. 从数据集加载字符串
      2. 随机采样字符串，生成初始模式（Phase 1）
      3. 对每个初始模式做 Phase 2 退化
      4. 压缩链
      5. 等距选锚点
      6. 按比例切分 train/valid/test
      7. 输出文件
    """
    db_path = f"data/{data_name}/{data_name}.txt"
    dst_dir = f"data/{data_name}/training/{dst_workload}"
    os.makedirs(dst_dir, exist_ok=True)

    # 加载数据集
    db = read_strings(db_path)
    db = [s for s in db if len(s) >= min_word_len and '_' not in s and '%' not in s]
    db = list(set(db))  # 去重
    n_db = len(db)

    rng = random.Random(chain_seed)
    np.random.seed(chain_seed)

    print(f"{'=' * 60}")
    print(f"INDEPENDENT DATA GENERATION (Mode: --generate)")
    print(f"{'=' * 60}")
    print(f"  Dataset:       {data_name} ({n_db} unique strings)")
    print(f"  Output:        {dst_dir}")
    print(f"  Num chains:    {num_chains}")
    print(f"  Max chain len: {max_chain_length} -> compress to {compress_to}")
    print(f"  Compress:      alpha={compress_power_alpha} power_max={compress_power_max}")
    print(f"  Anchor ratio:  {anchor_ratio:.0%}")
    print(f"  Query types:   SUBSTR={substr_prob:.0%} PREFIX={prefix_prob:.0%} "
          f"SUFFIX={suffix_prob:.0%}")
    print(f"  Degrade probs: char_to_={degrade_char_prob} shrink={degrade_shrink_prob} "
          f"insert_pct={degrade_insert_pct_prob} under_to_pct={degrade_under_to_pct_prob}")
    print(f"  Wildcards:     max_wildcard_ratio={max_wildcard_ratio}")
    print(f"  Split:         train={train_ratio:.0%} valid={valid_ratio:.0%} "
          f"test={test_ratio:.0%}")
    print(f"{'=' * 60}")

    # query type 分布
    type_weights = [substr_prob, prefix_prob, suffix_prob]
    type_names = ['SUBSTR', 'PREFIX', 'SUFFIX']

    # ---- Step 1-5: 生成所有链 ----
    all_chains = []  # list of list of dict
    type_stats = {'SUBSTR': 0, 'PREFIX': 0, 'SUFFIX': 0}

    print(f"\nGenerating {num_chains} chains...")
    for i in tqdm(range(num_chains), desc="Chains"):
        # 随机采样字符串
        word = rng.choice(db)
        if len(word) > max_word_len:
            # 随机截取子串
            start = rng.randint(0, len(word) - max_word_len)
            word = word[start:start + max_word_len]

        # 选择 query type
        qtype = rng.choices(type_names, weights=type_weights, k=1)[0]
        type_stats[qtype] += 1

        # Phase 1: 生成初始模式
        initial_pattern = string_to_initial_pattern(
            word, qtype, max_internal_pct=max_internal_pct,
            min_segment_len=min_segment_len, rng=rng,
        )
        if initial_pattern is None:
            continue

        # Phase 2: 退化链
        chain = generate_degradation_chain(
            initial_pattern,
            max_chain_length=max_chain_length,
            seed=rng.randint(0, 2**31),
            degrade_char_prob=degrade_char_prob,
            degrade_shrink_prob=degrade_shrink_prob,
            degrade_under_to_pct_prob=degrade_under_to_pct_prob,
            degrade_insert_pct_prob=degrade_insert_pct_prob,
            max_wildcard_ratio=max_wildcard_ratio,
            min_literal_len=min_literal_len,
        )

        # 压缩（同时过滤短模式）
        chain = compress_chain(
            chain,
            compress_to,
            min_literal_len=min_literal_len,
            compress_power_alpha=compress_power_alpha,
            compress_power_max=compress_power_max,
        )

        # 锚点采样
        chain_with_anchors = sample_anchors(
            chain, anchor_ratio=anchor_ratio, seed=rng.randint(0, 2**31),
        )

        all_chains.append(chain_with_anchors)

    print(f"  Generated {len(all_chains)} chains")
    print(f"  Type distribution: {type_stats}")

    # ---- Step 6: 切分 train/valid/test ----
    rng.shuffle(all_chains)
    n_total = len(all_chains)
    n_test = int(n_total * test_ratio)
    n_valid = int(n_total * valid_ratio)
    n_train = n_total - n_test - n_valid

    train_chains = all_chains[:n_train]
    valid_chains = all_chains[n_train:n_train + n_valid]
    test_chains = all_chains[n_train + n_valid:]

    print(f"  Split: train={n_train} valid={n_valid} test={n_test}")

    # ---- Step 7: 保存 ----

    # 收集需要查基数的 pattern
    all_anchor_patterns = set()
    for chain in all_chains:
        for item in chain:
            if item["pattern"] != "%":
                all_anchor_patterns.add(item["pattern"])

    # chain_query.txt（所有需要查基数的模式）
    query_file = os.path.join(dst_dir, "chain_query.txt")
    sorted_patterns = sorted(all_anchor_patterns)
    with open(query_file, "w") as f:
        for p in sorted_patterns:
            f.write(p + "\n")

    # chain_index.json（只记录训练集的链，valid/test 不需要链结构）
    chain_index = []
    for cid, chain in enumerate(train_chains):
        chain_info = {
            "chain_id": cid,
            "patterns": [item["pattern"] for item in chain],
            "is_anchor": [item["is_anchor"] for item in chain],
        }
        chain_index.append(chain_info)

    chain_index_file = os.path.join(dst_dir, "chain_index.json")
    with open(chain_index_file, "w", encoding="utf-8") as f:
        json.dump(chain_index, f, indent=2, ensure_ascii=False)

    # split_info.json（记录 valid/test 链的锚点，用于回填时生成 valid/test.txt）
    split_info = {
        "train_chains": n_train,
        "valid_chains": n_valid,
        "test_chains": n_test,
        # Gate A provenance: keep the exact generation parameters inside the
        # immutable split artifact. Legacy split_info files lack this block and
        # must therefore remain explicitly unverified.
        "generation": {
            "num_chains": num_chains,
            "generated_chains": n_total,
            "chain_length": max_chain_length,
            "compress_to": compress_to,
            "compress_power_alpha": compress_power_alpha,
            "compress_power_max": compress_power_max,
            "anchor_ratio": anchor_ratio,
            "max_internal_pct": max_internal_pct,
            "min_segment_len": min_segment_len,
            "substr_prob": substr_prob,
            "prefix_prob": prefix_prob,
            "suffix_prob": suffix_prob,
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
            "degrade_char_prob": degrade_char_prob,
            "degrade_shrink_prob": degrade_shrink_prob,
            "degrade_under_to_pct_prob": degrade_under_to_pct_prob,
            "degrade_insert_pct_prob": degrade_insert_pct_prob,
            "max_wildcard_ratio": max_wildcard_ratio,
            "chain_seed": chain_seed,
            "max_word_len": max_word_len,
            "min_word_len": min_word_len,
            "min_literal_len": min_literal_len,
        },
        "valid": [],
        "test": [],
        "valid_chain_index": [],
        "test_chain_index": [],
    }
    for chain in valid_chains:
        split_info["valid_chain_index"].append({
            "patterns": [item["pattern"] for item in chain],
            "is_anchor": [item["is_anchor"] for item in chain],
        })
        for item in chain:
            if item["is_anchor"]:
                split_info["valid"].append(item["pattern"])
    for chain in test_chains:
        split_info["test_chain_index"].append({
            "patterns": [item["pattern"] for item in chain],
            "is_anchor": [item["is_anchor"] for item in chain],
        })
        for item in chain:
            if item["is_anchor"]:
                split_info["test"].append(item["pattern"])

    # 去重
    split_info["valid"] = sorted(set(split_info["valid"]))
    split_info["test"] = sorted(set(split_info["test"]))

    split_info_file = os.path.join(dst_dir, "split_info.json")
    with open(split_info_file, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    # 统计
    total_patterns = sum(len(c) for c in all_chains)
    n_anchors = sum(sum(1 for item in c if item["is_anchor"]) for c in all_chains)
    avg_chain_len = total_patterns / max(len(all_chains), 1)

    print(f"\n{'=' * 60}")
    print(f"GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Chains:            {len(all_chains)}")
    print(f"  Total patterns:    {total_patterns}")
    print(f"  Avg chain length:  {avg_chain_len:.1f}")
    print(f"  Anchors:           {n_anchors} ({100*n_anchors/max(total_patterns,1):.1f}%)")
    print(f"  Unique queries:    {len(sorted_patterns)}")
    print(f"  Train anchors:     {sum(sum(1 for it in c if it['is_anchor']) for c in train_chains)}")
    print(f"  Valid queries:     {len(split_info['valid'])}")
    print(f"  Test queries:      {len(split_info['test'])}")
    print(f"")
    print(f"Output files:")
    print(f"  {chain_index_file}")
    print(f"  {query_file}")
    print(f"  {split_info_file}")
    print(f"")
    print(f"Next steps:")
    print(f"  1. python calculate_cardinality.py \\")
    print(f"       --query_file {query_file} \\")
    print(f"       --output_file {os.path.join(dst_dir, 'chain_card.txt')} \\")
    print(f"       --db_name string_db --table_name {data_name.lower()}")
    print(f"  2. python gen_query_chain.py -d {data_name} --fill_card "
          f"--dst_workload {dst_workload}")
    print(f"{'=' * 60}")


# =====================================================
#  Mode 2: 基于已有 workload 扩展 (--extend)
# =====================================================

def generate_extend_data(
    data_name,
    src_workload="CLIQUE",
    dst_workload="SEMO",
    chain_length=20,
    compress_to=10,
    compress_power_alpha=0.8,
    compress_power_max=3.0,
    anchor_ratio=1.0,
    degrade_char_prob=0.4,
    degrade_shrink_prob=0.5,
    degrade_under_to_pct_prob=0.0,
    degrade_insert_pct_prob=0.1,
    max_wildcard_ratio=0.2,
    chain_seed=42,
    min_literal_len=0,
):
    """
    基于已有 workload 的训练查询生成链数据。
    test/valid 从原 workload 复制。
    """
    src_dir = f"data/{data_name}/training/{src_workload}"
    dst_dir = f"data/{data_name}/training/{dst_workload}"
    os.makedirs(dst_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"EXTEND DATA GENERATION (Mode: --extend)")
    print(f"{'=' * 60}")
    print(f"  Source:     {src_dir}")
    print(f"  Dest:       {dst_dir}")
    print(f"  Chain len:  {chain_length} -> compress to {compress_to}")
    print(f"  Compress:   alpha={compress_power_alpha} power_max={compress_power_max}")
    print(f"  Anchor:     {anchor_ratio:.0%}")
    print(f"  Degrade:    char_to_={degrade_char_prob} shrink={degrade_shrink_prob} "
          f"insert_pct={degrade_insert_pct_prob} under_to_pct={degrade_under_to_pct_prob}")
    print(f"  Wildcards:  max_wildcard_ratio={max_wildcard_ratio}")
    print(f"{'=' * 60}")

    # 复制 test/valid
    for split in ["test.txt", "valid.txt", "pack_simple.txt"]:
        src_file = os.path.join(src_dir, split)
        dst_file = os.path.join(dst_dir, split)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            with open(dst_file) as f:
                n_lines = sum(1 for _ in f)
            print(f"  Copied {split}: {n_lines} lines")

    # 加载链头
    train_path = f"data/{data_name}/training/{src_workload}/train.txt"
    train_data = read_training(train_path)
    chain_heads = [query for query, card in train_data]
    print(f"  Chain heads: {len(chain_heads)}")

    # 生成退化链
    rng = random.Random(chain_seed)
    all_chains = []
    all_patterns = set()

    print(f"\nGenerating degradation chains...")
    for head_query in tqdm(chain_heads, desc="Chains"):
        chain = generate_degradation_chain(
            head_query,
            max_chain_length=chain_length,
            seed=rng.randint(0, 2**31),
            degrade_char_prob=degrade_char_prob,
            degrade_shrink_prob=degrade_shrink_prob,
            degrade_under_to_pct_prob=degrade_under_to_pct_prob,
            degrade_insert_pct_prob=degrade_insert_pct_prob,
            max_wildcard_ratio=max_wildcard_ratio,
            min_literal_len=min_literal_len,
        )
        chain = compress_chain(
            chain,
            compress_to,
            min_literal_len=min_literal_len,
            compress_power_alpha=compress_power_alpha,
            compress_power_max=compress_power_max,
        )

        chain_with_anchors = sample_anchors(
            chain, anchor_ratio=anchor_ratio, seed=rng.randint(0, 2**31),
        )
        all_chains.append(chain_with_anchors)

        for item in chain_with_anchors:
            if item["pattern"] != "%":
                all_patterns.add(item["pattern"])

    # 保存 chain_index.json
    chain_index = []
    for cid, chain in enumerate(all_chains):
        chain_index.append({
            "chain_id": cid,
            "patterns": [item["pattern"] for item in chain],
            "is_anchor": [item["is_anchor"] for item in chain],
        })

    chain_index_file = os.path.join(dst_dir, "chain_index.json")
    with open(chain_index_file, "w", encoding="utf-8") as f:
        json.dump(chain_index, f, indent=2, ensure_ascii=False)

    # chain_query.txt
    query_file = os.path.join(dst_dir, "chain_query.txt")
    sorted_patterns = sorted(all_patterns)
    with open(query_file, "w") as f:
        for p in sorted_patterns:
            f.write(p + "\n")

    total_patterns = sum(len(c) for c in all_chains)
    n_anchors = sum(sum(1 for it in c if it["is_anchor"]) for c in all_chains)

    print(f"\n{'=' * 60}")
    print(f"EXTEND GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Chains: {len(all_chains)}, Patterns: {total_patterns}, "
          f"Anchors: {n_anchors}, Unique queries: {len(sorted_patterns)}")
    print(f"  Next: calculate cardinality -> fill_card")
    print(f"{'=' * 60}")


# =====================================================
#  Mode 3: 纯正则化 (--chain_only)
# =====================================================

def generate_chain_only(
    data_name,
    src_workload="CLIQUE",
    chain_length=20,
    compress_to=None,
    compress_power_alpha=0.8,
    compress_power_max=3.0,
    degrade_char_prob=0.4,
    degrade_shrink_prob=0.5,
    degrade_under_to_pct_prob=0.0,
    degrade_insert_pct_prob=0.1,
    max_wildcard_ratio=0.2,
    chain_seed=42,
    min_literal_len=0,
):
    """
    只在原 workload 目录中生成 chain_index.json，不修改任何训练文件。
    训练时 train/valid/test 全用原始的，chain_index 只提供单调性约束。
    """
    src_dir = f"data/{data_name}/training/{src_workload}"

    print(f"{'=' * 60}")
    print(f"CHAIN-ONLY MODE (Mode: --chain_only, pure regularization)")
    print(f"{'=' * 60}")
    print(f"  Workload:       {src_workload}")
    print(f"  Directory:      {src_dir}")
    print(f"  Chain len:      {chain_length}"
          + (f" -> compress to {compress_to}" if compress_to else ""))
    if compress_to:
        print(f"  Compress:       alpha={compress_power_alpha} power_max={compress_power_max}")
    print(f"  Min literal:    {min_literal_len} (0=no filter)")
    print(f"  Degrade:        char_to_={degrade_char_prob} shrink={degrade_shrink_prob} "
          f"insert_pct={degrade_insert_pct_prob} under_to_pct={degrade_under_to_pct_prob}")
    print(f"  Wildcards:      max_wildcard_ratio={max_wildcard_ratio}")
    print(f"  NOTE: train/valid/test.txt will NOT be modified")
    print(f"{'=' * 60}")

    # 加载链头
    train_path = f"data/{data_name}/training/{src_workload}/train.txt"
    train_data = read_training(train_path)
    chain_heads = [query for query, card in train_data]
    print(f"  Chain heads: {len(chain_heads)}")

    rng = random.Random(chain_seed)
    all_chains = []

    print(f"\nGenerating degradation chains...")
    for head_query in tqdm(chain_heads, desc="Chains"):
        chain = generate_degradation_chain(
            head_query,
            max_chain_length=chain_length,
            seed=rng.randint(0, 2**31),
            degrade_char_prob=degrade_char_prob,
            degrade_shrink_prob=degrade_shrink_prob,
            degrade_under_to_pct_prob=degrade_under_to_pct_prob,
            degrade_insert_pct_prob=degrade_insert_pct_prob,
            max_wildcard_ratio=max_wildcard_ratio,
            min_literal_len=min_literal_len,
        )
        if compress_to:
            chain = compress_chain(
                chain,
                compress_to,
                min_literal_len=min_literal_len,
                compress_power_alpha=compress_power_alpha,
                compress_power_max=compress_power_max,
            )
        all_chains.append(chain)

    # 保存 chain_index.json（is_anchor 全 False）
    chain_index = []
    for cid, chain in enumerate(all_chains):
        chain_index.append({
            "chain_id": cid,
            "patterns": chain,
            "is_anchor": [False] * len(chain),
        })

    chain_index_file = os.path.join(src_dir, "chain_index.json")
    with open(chain_index_file, "w", encoding="utf-8") as f:
        json.dump(chain_index, f, indent=2, ensure_ascii=False)

    total_patterns = sum(len(c) for c in all_chains)
    train_set = set(chain_heads)
    chain_set = set(p for c in all_chains for p in c)

    print(f"\n{'=' * 60}")
    print(f"CHAIN-ONLY COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Chains: {len(all_chains)}, Total patterns: {total_patterns}")
    print(f"  Overlap with train: {len(chain_set & train_set)}")
    print(f"  New patterns: {len(chain_set - train_set)} (mono loss only)")
    print(f"  Output: {chain_index_file}")
    print(f"  Train: python main_UB_ORDER.py -d {data_name} -w {src_workload} "
          f"--seed 0 --lambda_order 1.0")
    print(f"{'=' * 60}")


# =====================================================
#  回填基数 → 生成 train.txt / valid.txt / test.txt
# =====================================================

def fill_cardinality(data_name, dst_workload="SEMO", card_file=None):
    """
    读取基数计算结果，结合 chain_index.json（和 split_info.json），
    生成 train.txt / valid.txt / test.txt。

    支持两种情况：
      - --generate 模式：有 split_info.json，生成 train/valid/test
      - --extend 模式：无 split_info.json，只生成 train.txt
    """
    dst_dir = f"data/{data_name}/training/{dst_workload}"
    if card_file is None:
        card_file = os.path.join(dst_dir, "chain_card.txt")
    chain_index_file = os.path.join(dst_dir, "chain_index.json")
    split_info_file = os.path.join(dst_dir, "split_info.json")

    print(f"{'=' * 60}")
    print(f"FILL CARDINALITY")
    print(f"  Card file:    {card_file}")
    print(f"  Chain index:  {chain_index_file}")
    print(f"{'=' * 60}")

    # 加载基数
    card_dict = {}
    with open(card_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line == "":
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) == 2:
                pat, card = parts[0], int(parts[1])
                card_dict[pat] = card
    print(f"  Loaded {len(card_dict)} cardinalities")

    # 加载链索引
    chain_index = load_chain_index(chain_index_file)
    print(f"  Loaded {len(chain_index)} chains")

    # n_db
    db_path = f"data/{data_name}/{data_name}.txt"
    db = read_strings(db_path)
    n_db = len(db)

    # 构建 train.txt（有标签 + 无标签）
    all_data, chain_pairs = build_chain_train_data(chain_index, card_dict, n_db)

    train_path = os.path.join(dst_dir, "train.txt")
    with open(train_path, "w") as f:
        for pat, card in all_data:
            f.write(f"{pat},{card}\n")

    n_labeled = sum(1 for _, c in all_data if c >= 0)
    n_unlabeled = sum(1 for _, c in all_data if c < 0)
    print(f"\n  train.txt: {len(all_data)} samples "
          f"({n_labeled} labeled, {n_unlabeled} unlabeled)")

    # 如果是 --generate 模式，有 split_info.json，生成 valid/test
    if os.path.exists(split_info_file):
        with open(split_info_file, "r") as f:
            split_info = json.load(f)

        for split_name in ["valid", "test"]:
            patterns = split_info.get(split_name, [])
            split_path = os.path.join(dst_dir, f"{split_name}.txt")
            with open(split_path, "w") as f:
                for pat in patterns:
                    card = card_dict.get(pat, 0)
                    f.write(f"{pat},{card}\n")
            print(f"  {split_name}.txt: {len(patterns)} samples")
    else:
        print(f"  No split_info.json found (--extend mode), "
              f"valid/test already copied from source")

    # 基数统计
    cards = [c for _, c in all_data if c >= 0]
    if cards:
        print(f"\n  Cardinality stats (labeled):")
        print(f"    Min: {min(cards)}, Max: {max(cards)}, Avg: {sum(cards)/len(cards):.1f}")
        print(f"    Zero: {sum(1 for c in cards if c == 0)}")

    # 格式验证
    try:
        verify = read_training(train_path)
        print(f"  Format verification: OK ({len(verify)} samples)")
    except Exception as e:
        print(f"  Format verification FAILED: {e}")

    print(f"\nReady to train:")
    print(f"  python main_UB_ORDER.py -d {data_name} -w {dst_workload} --seed 0")


# =====================================================
#  命令行入口
# =====================================================

def fill_cardinality_baseline(data_name, dst_workload="SEMO", card_file=None):
    """
    Generate supervised baseline files from the same chain stage.

    This writes train.txt with only labeled anchor patterns so main_CLIQUE.py
    can run on the same anchor sampling stage used by main_UB_ORDER.py.
    """
    dst_dir = f"data/{data_name}/training/{dst_workload}"
    if card_file is None:
        card_file = os.path.join(dst_dir, "chain_card.txt")
    chain_index_file = os.path.join(dst_dir, "chain_index.json")
    split_info_file = os.path.join(dst_dir, "split_info.json")

    print(f"{'=' * 60}")
    print(f"FILL CARDINALITY BASELINE")
    print(f"  Card file:    {card_file}")
    print(f"  Chain index:  {chain_index_file}")
    print(f"{'=' * 60}")

    card_dict = {}
    with open(card_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line == "":
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) == 2:
                pat, card = parts[0], int(parts[1])
                card_dict[pat] = card
    print(f"  Loaded {len(card_dict)} cardinalities")

    db_path = f"data/{data_name}/{data_name}.txt"
    db = read_strings(db_path)
    n_db = len(db)

    chain_index = load_chain_index(chain_index_file)
    print(f"  Loaded {len(chain_index)} chains")

    train_data = []
    seen = set()
    missing = 0
    for chain_info in chain_index:
        for pat, anchor in zip(chain_info["patterns"], chain_info["is_anchor"]):
            if not anchor or pat in seen:
                continue
            seen.add(pat)
            if pat == "%":
                train_data.append((pat, n_db))
            elif pat in card_dict:
                train_data.append((pat, card_dict[pat]))
            else:
                missing += 1

    train_path = os.path.join(dst_dir, "train.txt")
    with open(train_path, "w") as f:
        for pat, card in train_data:
            f.write(f"{pat},{card}\n")
    print(f"\n  train.txt: {len(train_data)} labeled samples")
    if missing:
        print(f"  WARNING: skipped {missing} anchor patterns missing from card file")

    if os.path.exists(split_info_file):
        with open(split_info_file, "r") as f:
            split_info = json.load(f)

        for split_name in ["valid", "test"]:
            patterns = split_info.get(split_name, [])
            split_data = []
            missing_split = 0
            for pat in patterns:
                if pat == "%":
                    split_data.append((pat, n_db))
                elif pat in card_dict:
                    split_data.append((pat, card_dict[pat]))
                else:
                    missing_split += 1

            split_path = os.path.join(dst_dir, f"{split_name}.txt")
            with open(split_path, "w") as f:
                for pat, card in split_data:
                    f.write(f"{pat},{card}\n")
            print(f"  {split_name}.txt: {len(split_data)} labeled samples")
            if missing_split:
                print(f"  WARNING: skipped {missing_split} missing {split_name} patterns")
    else:
        print(f"  No split_info.json found (--extend mode); keeping existing valid/test files")

    try:
        verify = read_training(train_path)
        print(f"  Format verification: OK ({len(verify)} train samples)")
    except Exception as e:
        print(f"  Format verification FAILED: {e}")

    print(f"\nReady to train baseline:")
    print(f"  python main_CLIQUE.py -d {data_name} -w {dst_workload} --seed 0")


def _load_card_dict(card_file):
    card_dict = {}
    with open(card_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line == "":
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) == 2:
                card_dict[parts[0]] = int(parts[1])
    return card_dict


def _write_labeled_patterns(path, patterns, card_dict, n_db):
    seen = set()
    written = 0
    missing = 0
    with open(path, "w", encoding="utf-8") as f:
        for pat in patterns:
            if pat in seen:
                continue
            seen.add(pat)
            if pat == "%":
                card = n_db
            elif pat in card_dict:
                card = card_dict[pat]
            else:
                missing += 1
                continue
            f.write(f"{pat},{card}\n")
            written += 1
    return written, missing


def _flatten_chain_patterns(chains):
    patterns = []
    for chain in chains:
        patterns.extend(chain.get("patterns", []))
    return patterns


def _sample_anchor_mask(patterns, anchor_ratio, rng):
    if not patterns:
        return []

    mask = []
    n = len(patterns)
    for i, pat in enumerate(patterns):
        if pat == "%":
            mask.append(False)
        elif i == 0 or i == n - 1:
            mask.append(True)
        else:
            mask.append(anchor_ratio >= 1.0 or rng.random() < anchor_ratio)

    if sum(mask) < 2 and len(patterns) > 2:
        candidates = [i for i, pat in enumerate(patterns) if pat != "%" and not mask[i]]
        if candidates:
            mask[rng.choice(candidates)] = True
    return mask


def fill_cardinality_sampled_pair(data_name, src_workload="FULL",
                                  dst_workload="A50",
                                  card_file=None, anchor_ratio=0.5,
                                  anchor_seed=0):
    """
    Fill one sampled experiment pair from a full-cardinality chain workload.

    Inputs are read from data/<data_name>/training/<src_workload>/:
      - chain_index.json: full train chains
      - split_info.json: full valid/test chains
      - chain_card.txt: full cardinalities for all chain patterns

    Outputs are written to:
      - data/<data_name>/training/<dst_workload>_UB_ORDER/
      - data/<data_name>/training/<dst_workload>_CLIQUE/
    """
    src_dir = f"data/{data_name}/training/{src_workload}"
    mono_workload = f"{dst_workload}_UB_ORDER"
    clique_workload = f"{dst_workload}_CLIQUE"
    mono_dir = f"data/{data_name}/training/{mono_workload}"
    clique_dir = f"data/{data_name}/training/{clique_workload}"
    os.makedirs(mono_dir, exist_ok=True)
    os.makedirs(clique_dir, exist_ok=True)

    if card_file is None:
        card_file = os.path.join(src_dir, "chain_card.txt")
    chain_index_file = os.path.join(src_dir, "chain_index.json")
    split_info_file = os.path.join(src_dir, "split_info.json")

    print(f"{'=' * 60}")
    print("FILL SAMPLED PAIR")
    print(f"  Source:       {src_dir}")
    print(f"  Mono output:  {mono_dir}")
    print(f"  CLIQUE output: {clique_dir}")
    print(f"  Card file:    {card_file}")
    print(f"  Anchor ratio: {anchor_ratio:.0%}")
    print(f"  Anchor seed:  {anchor_seed}")
    print(f"{'=' * 60}")

    card_dict = _load_card_dict(card_file)
    train_chains_full = load_chain_index(chain_index_file)
    with open(split_info_file, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    valid_chains = split_info.get("valid_chain_index", [])
    test_chains = split_info.get("test_chain_index", [])
    if not valid_chains:
        valid_chains = [{"patterns": split_info.get("valid", [])}]
    if not test_chains:
        test_chains = [{"patterns": split_info.get("test", [])}]

    db_path = f"data/{data_name}/{data_name}.txt"
    n_db = len(read_strings(db_path))
    rng = random.Random(anchor_seed)

    sampled_chain_index = []
    anchor_mask_records = []
    for cid, chain_info in enumerate(train_chains_full):
        patterns = chain_info["patterns"]
        mask = _sample_anchor_mask(patterns, anchor_ratio, rng)
        sampled_chain_index.append({
            "chain_id": cid,
            "patterns": patterns,
            "is_anchor": mask,
        })
        anchor_mask_records.append({
            "chain_id": cid,
            "patterns": patterns,
            "is_anchor": mask,
        })

    # UB_ORDER: full chains, sampled labels; unsampled chain points get card=-1.
    shutil.copy2(card_file, os.path.join(mono_dir, "chain_card.txt"))
    with open(os.path.join(mono_dir, "chain_index.json"), "w", encoding="utf-8") as f:
        json.dump(sampled_chain_index, f, indent=2, ensure_ascii=False)

    all_data, _ = build_chain_train_data(sampled_chain_index, card_dict, n_db)
    mono_train_path = os.path.join(mono_dir, "train.txt")
    with open(mono_train_path, "w", encoding="utf-8") as f:
        for pat, card in all_data:
            f.write(f"{pat},{card}\n")

    # CLIQUE: reuse UB_ORDER's labeled rows so both workloads have exactly the
    # same unique patterns and cardinalities after duplicate aggregation.
    clique_train_path = os.path.join(clique_dir, "train.txt")
    clique_labeled_rows = [(pat, card) for pat, card in all_data if card >= 0]
    with open(clique_train_path, "w", encoding="utf-8") as f:
        for pat, card in clique_labeled_rows:
            f.write(f"{pat},{card}\n")
    clique_train_written = len(clique_labeled_rows)

    valid_patterns = _flatten_chain_patterns(valid_chains)
    test_patterns = _flatten_chain_patterns(test_chains)
    for out_dir in [mono_dir, clique_dir]:
        valid_written, valid_missing = _write_labeled_patterns(
            os.path.join(out_dir, "valid.txt"), valid_patterns, card_dict, n_db
        )
        test_written, test_missing = _write_labeled_patterns(
            os.path.join(out_dir, "test.txt"), test_patterns, card_dict, n_db
        )

    with open(os.path.join(mono_dir, "anchor_mask.json"), "w", encoding="utf-8") as f:
        json.dump(anchor_mask_records, f, indent=2, ensure_ascii=False)
    shutil.copy2(os.path.join(mono_dir, "anchor_mask.json"),
                 os.path.join(clique_dir, "anchor_mask.json"))

    sampled_split_info = {
        "source_workload": src_workload,
        "anchor_ratio": anchor_ratio,
        "anchor_seed": anchor_seed,
        "valid_chains": len(valid_chains),
        "test_chains": len(test_chains),
    }
    for out_dir in [mono_dir, clique_dir]:
        with open(os.path.join(out_dir, "sample_info.json"), "w", encoding="utf-8") as f:
            json.dump(sampled_split_info, f, indent=2, ensure_ascii=False)

    n_labeled = sum(1 for _, card in all_data if card >= 0)
    n_unlabeled = sum(1 for _, card in all_data if card < 0)
    print(f"\nMONO train: {len(all_data)} total ({n_labeled} labeled, {n_unlabeled} unlabeled)")
    print(f"CLIQUE train: {clique_train_written} labeled")
    print(f"Valid/test: {valid_written} valid, {test_written} test")
    if valid_missing or test_missing:
        print(f"WARNING: skipped {valid_missing} valid and {test_missing} test missing patterns")

    print(f"\nReady to train:")
    print(f"  python main_UB_ORDER.py -d {data_name} -w {mono_workload} --seed 0 --lambda_order 1.0")
    print(f"  python main_CLIQUE.py -d {data_name} -w {clique_workload} --seed 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLIQUE chain query generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Three generation modes:

  Mode 1: --generate (fully independent)
    python gen_query_chain.py -d AUTHOR --generate --num_chains 10000

  Mode 2: --extend (from existing workload)
    python gen_query_chain.py -d AUTHOR --extend --src_workload CLIQUE

  Mode 3: --chain_only (pure regularization)
    python gen_query_chain.py -d AUTHOR --chain_only --chain_length 3

  Fill cardinality (after calculate_cardinality.py):
    python gen_query_chain.py -d AUTHOR --fill_card --dst_workload SEMO
        """,
    )

    # 数据集
    parser.add_argument("--data_name", "-d", type=str, required=True)
    parser.add_argument("--src_workload", type=str, default="CLIQUE")
    parser.add_argument("--dst_workload", type=str, default="SEMO")

    # 模式选择（互斥）
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--generate", action="store_true",
                            help="Mode 1: fully independent generation")
    mode_group.add_argument("--extend", action="store_true",
                            help="Mode 2: extend from existing workload")
    mode_group.add_argument("--chain_only", action="store_true",
                            help="Mode 3: pure regularization (chain_index only)")
    mode_group.add_argument("--fill_card", action="store_true",
                            help="Fill cardinality into train/valid/test.txt")
    mode_group.add_argument("--fill_card_baseline", action="store_true",
                            help="Fill labeled-only train/valid/test.txt for main_CLIQUE.py baseline")
    mode_group.add_argument("--fill_card_sampled_pair", action="store_true",
                            help="Fill sampled CLIQUE/UB_ORDER workload pair from a full-cardinality chain workload")

    # --generate 参数
    parser.add_argument("--num_chains", type=int, default=20000)
    parser.add_argument("--max_word_len", type=int, default=30)
    parser.add_argument("--min_word_len", type=int, default=1)
    parser.add_argument("--max_internal_pct", type=int, default=1,
                        help="Max internal %% insertions per pattern (default: 1)")
    parser.add_argument("--min_segment_len", type=int, default=1)
    parser.add_argument("--substr_prob", type=float, default=0.5)
    parser.add_argument("--prefix_prob", type=float, default=0.25)
    parser.add_argument("--suffix_prob", type=float, default=0.25)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # 链参数（所有模式共用）
    parser.add_argument("--chain_length", type=int, default=100)
    parser.add_argument("--compress_to", type=int, default=10)
    parser.add_argument("--compress_power_alpha", type=float, default=0.8,
                        help="Dynamic tail-dense compression strength.")
    parser.add_argument("--compress_power_max", type=float, default=3.0,
                        help="Maximum power for tail-dense chain compression.")
    parser.add_argument("--anchor_ratio", type=float, default=0.5)
    parser.add_argument("--min_literal_len", type=int, default=3,
                        help="Stop degradation when literal chars <= this value. "
                             "Set to N (e.g. 5) to stop at N-gram boundary. 0=no filter")
    parser.add_argument("--degrade_char_prob", type=float, default=0.3)
    parser.add_argument("--degrade_shrink_prob", type=float, default=0.4)
    parser.add_argument("--degrade_under_to_pct_prob", type=float, default=0.0,
                        help="Deprecated compatibility knob: '_' -> '%%'. Prefer keeping this at 0.")
    parser.add_argument("--degrade_insert_pct_prob", type=float, default=0.1,
                        help="Relative probability of inserting %% inside a literal token.")
    parser.add_argument("--max_wildcard_ratio", type=float, default=0.2,
                        help="Maximum ratio of %%/_ wildcards to pattern length; <=0 disables.")
    parser.add_argument("--chain_seed", type=int, default=42)

    # 回填
    parser.add_argument("--card_file", type=str, default=None)
    parser.add_argument("--anchor_seed", type=int, default=0,
                        help="Seed for sampling labeled anchors during fill_card_sampled_pair")

    args = parser.parse_args()

    if args.generate:
        generate_independent_data(
            data_name=args.data_name,
            dst_workload=args.dst_workload,
            num_chains=args.num_chains,
            max_chain_length=args.chain_length,
            compress_to=args.compress_to,
            compress_power_alpha=args.compress_power_alpha,
            compress_power_max=args.compress_power_max,
            anchor_ratio=args.anchor_ratio,
            max_internal_pct=args.max_internal_pct,
            min_segment_len=args.min_segment_len,
            substr_prob=args.substr_prob,
            prefix_prob=args.prefix_prob,
            suffix_prob=args.suffix_prob,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            degrade_char_prob=args.degrade_char_prob,
            degrade_shrink_prob=args.degrade_shrink_prob,
            degrade_under_to_pct_prob=args.degrade_under_to_pct_prob,
            degrade_insert_pct_prob=args.degrade_insert_pct_prob,
            max_wildcard_ratio=args.max_wildcard_ratio,
            chain_seed=args.chain_seed,
            max_word_len=args.max_word_len,
            min_word_len=args.min_word_len,
            min_literal_len=args.min_literal_len,
        )

    elif args.extend:
        generate_extend_data(
            data_name=args.data_name,
            src_workload=args.src_workload,
            dst_workload=args.dst_workload,
            chain_length=args.chain_length,
            compress_to=args.compress_to,
            compress_power_alpha=args.compress_power_alpha,
            compress_power_max=args.compress_power_max,
            anchor_ratio=args.anchor_ratio,
            degrade_char_prob=args.degrade_char_prob,
            degrade_shrink_prob=args.degrade_shrink_prob,
            degrade_under_to_pct_prob=args.degrade_under_to_pct_prob,
            degrade_insert_pct_prob=args.degrade_insert_pct_prob,
            max_wildcard_ratio=args.max_wildcard_ratio,
            chain_seed=args.chain_seed,
            min_literal_len=args.min_literal_len,
        )

    elif args.chain_only:
        generate_chain_only(
            data_name=args.data_name,
            src_workload=args.src_workload,
            chain_length=args.chain_length,
            compress_to=args.compress_to if args.compress_to != 10 else None,
            compress_power_alpha=args.compress_power_alpha,
            compress_power_max=args.compress_power_max,
            degrade_char_prob=args.degrade_char_prob,
            degrade_shrink_prob=args.degrade_shrink_prob,
            degrade_under_to_pct_prob=args.degrade_under_to_pct_prob,
            degrade_insert_pct_prob=args.degrade_insert_pct_prob,
            max_wildcard_ratio=args.max_wildcard_ratio,
            chain_seed=args.chain_seed,
            min_literal_len=args.min_literal_len,
        )

    elif args.fill_card:
        fill_cardinality(
            data_name=args.data_name,
            dst_workload=args.dst_workload,
            card_file=args.card_file,
        )

    elif args.fill_card_baseline:
        fill_cardinality_baseline(
            data_name=args.data_name,
            dst_workload=args.dst_workload,
            card_file=args.card_file,
        )

    elif args.fill_card_sampled_pair:
        fill_cardinality_sampled_pair(
            data_name=args.data_name,
            src_workload=args.src_workload,
            dst_workload=args.dst_workload,
            card_file=args.card_file,
            anchor_ratio=args.anchor_ratio,
            anchor_seed=args.anchor_seed,
        )

    else:
        parser.print_help()
        print("\nPlease specify a mode: --generate, --extend, --chain_only, --fill_card, --fill_card_baseline, or --fill_card_sampled_pair")
