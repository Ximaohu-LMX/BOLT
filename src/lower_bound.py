"""
lower_bound.py  —  基于高频字符串的下界计算模块

核心思路：
  维护一个高频字符串库（数据集中出现次数最多的完整字符串/子串），
  对查询模式 P，找出所有满足 S LIKE P 的高频串 S，
  取其中基数最大的作为 Card(P) 的严格下界。

  数学保证：
    如果 S LIKE P 为真，则数据集中每一行等于 S 的行都匹配 P，
    所以 Card(P) >= Count(S in dataset)。

集成到 CLIQUE：
  原始: pred = upper_bound * sigmoid(logit)       范围 [0, UB]
  改后: pred = LB + (UB - LB) * sigmoid(logit)   范围 [LB, UB]

放置位置: src/lower_bound.py
"""

import re
import os
import json
import pickle
import time
import numpy as np
from collections import Counter
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional


# =====================================================
#  1. LIKE 匹配
# =====================================================

def like_match(pattern: str, string: str) -> bool:
    """
    判断 string LIKE pattern 是否为真（SQL LIKE 语义）。

    '%' 匹配任意长度（含0）的任意字符序列
    '_' 匹配恰好一个任意字符
    """
    regex_parts = []
    for ch in pattern:
        if ch == '%':
            regex_parts.append('.*')
        elif ch == '_':
            regex_parts.append('.')
        elif ch in r'\.^$*+?{}[]|()':
            regex_parts.append('\\' + ch)
        else:
            regex_parts.append(ch)
    regex = '^' + ''.join(regex_parts) + '$'
    return re.match(regex, string, re.DOTALL) is not None


# =====================================================
#  2. 高频字符串库构建
# =====================================================

def _dedup_and_select(counts: dict, top_k: int, mode: str = 'infix',
                      merge_threshold: float = 0.1) -> dict:
    """
    选择高频子串，温和去重。

    去重规则：
      当两个串存在包含关系（短串是长串的子串/前缀/后缀），
      且长串的基数与短串的基数相差不超过 merge_threshold（默认10%），
      则短串冗余——保留长串，去除短串。

      原因：基数相差 <10% 说明多出来的字符几乎没有过滤掉额外的行，
      长串在下界查询中能匹配到的查询严格多于短串（更多 segment 能在长串中找到），
      所以长串严格更优。

      当基数差距大（如 Rob=2100 vs Robert=1523，差 27%），保留两者，
      因为短串 Rob 在某些查询上提供更大的下界。

    参数:
        counts: dict, {string: count}
        top_k: int
        mode: str, 'prefix' / 'suffix' / 'infix'
        merge_threshold: float, 基数差异比例阈值（默认 0.1 = 10%）

    返回:
        dict: {string: count}
    """
    # 按长度降序排列（长串先入选），同长度按频率降序
    candidates = sorted(counts.items(), key=lambda x: (-len(x[0]), -x[1]))

    selected = {}  # {string: count}

    for cand, count in candidates:
        if len(selected) >= top_k * 2:  # 预留空间，最后再截断
            break
        selected[cand] = count

    # 去重：扫描所有已选对，标记冗余的短串
    to_remove = set()

    selected_list = sorted(selected.items(), key=lambda x: -len(x[0]))

    for i, (long_str, long_count) in enumerate(selected_list):
        if long_str in to_remove:
            continue
        for j in range(i + 1, len(selected_list)):
            short_str, short_count = selected_list[j]
            if short_str in to_remove:
                continue

            # 检查包含关系
            is_contained = False
            if mode == 'prefix':
                is_contained = long_str.startswith(short_str)
            elif mode == 'suffix':
                is_contained = long_str.endswith(short_str)
            elif mode == 'infix':
                is_contained = short_str in long_str

            if not is_contained:
                continue

            # 检查基数差异：短串基数一定 >= 长串基数
            # 差异比例 = (short_count - long_count) / short_count
            if short_count == 0:
                continue
            diff_ratio = (short_count - long_count) / short_count

            if diff_ratio <= merge_threshold:
                # 基数差距小，短串冗余，去除短串保留长串
                to_remove.add(short_str)

    # 移除冗余串
    for s in to_remove:
        del selected[s]

    # 按频率降序取 top_k
    sorted_items = sorted(selected.items(), key=lambda x: (-x[1], -len(x[0])))
    return dict(sorted_items[:top_k])


def build_freq_string_library(db: List[str],
                               top_k_strings: int = 500,
                               top_k_substrings: int = 500,
                               min_substr_len: int = 6,
                               max_substr_len: int = 15,
                               min_freq_ratio: float = 0.0005,
                               sample_ratio: float = 0.1,
                               full_scan: bool = True,
                               seed: int = 42) -> dict:
    """
    从数据集构建高频字符串库。

    三种位置的统计语义完全分离：
      - prefix: s[:l]，固定首位，统计"以 S 开头的行数"
      - suffix: s[-l:]，固定末位，统计"以 S 结尾的行数"
      - infix:  s[1:-1] 的严格内部位置，统计"包含 S（非首非尾）的行数"
      - exact:  完整行的重复计数

    参数:
        full_scan: bool, True=Phase2 用全表精确计数，False=采样估计（默认）
        min_freq_ratio: float, 最低频率阈值
        sample_ratio: float, Phase1/Phase2 采样比例（full_scan=True 时忽略）

    返回:
        dict: {'exact': {str: count}, 'prefix': {...}, 'suffix': {...}, 'infix': {...}}
    """
    import random
    rng = random.Random(seed)
    n_db = len(db)
    min_freq = max(2, int(n_db * min_freq_ratio))

    print(f"Building frequent string library...")
    print(f"  Dataset size: {n_db}")
    print(f"  Min frequency: {min_freq} ({min_freq_ratio:.2%} of {n_db})")
    print(f"  Full scan: {full_scan}")

    # ---- Part 1: 高频完整字符串 ----
    string_counter = Counter(db)
    top_strings = string_counter.most_common(top_k_strings)
    freq_exact = {s: count for s, count in top_strings if count >= min_freq}
    print(f"  Frequent complete strings: {len(freq_exact)} (count >= {min_freq})")

    # ---- Part 2: 高频子串 ----
    # Phase 1: 小采样发现候选子串
    sample_size = max(5000, min(int(n_db * sample_ratio), n_db))
    sample = rng.sample(db, sample_size) if sample_size < n_db else db

    prefix_counter = Counter()
    suffix_counter = Counter()
    infix_counter = Counter()

    for s in sample:
        s_len = len(s)
        seen_infix = set()
        for l in range(min_substr_len, min(max_substr_len + 1, s_len + 1)):
            # 前缀：固定首位
            prefix_counter[s[:l]] += 1
            # 后缀：固定末位
            suffix_counter[s[s_len - l:]] += 1
            # infix：严格内部位置（排除首位和末位起始）
            for i in range(1, s_len - l):
                sub = s[i:i + l]
                if sub not in seen_infix:
                    seen_infix.add(sub)
                    infix_counter[sub] += 1

    n_cand = top_k_substrings * 3
    pfx_cands = [s for s, _ in prefix_counter.most_common(n_cand)]
    sfx_cands = [s for s, _ in suffix_counter.most_common(n_cand)]
    ifx_cands = [s for s, _ in infix_counter.most_common(n_cand)]

    # Phase 2: 精确计数（全表扫描或采样）
    if full_scan:
        scan_data = db
        scale = 1.0
        print(f"  Phase 2: full scan ({n_db} rows)")
    else:
        phase2_size = max(10000, min(int(n_db * max(sample_ratio * 2, 0.2)), n_db))
        scan_data = rng.sample(db, phase2_size) if phase2_size < n_db else db
        scale = n_db / len(scan_data)
        print(f"  Phase 2: sampling ({len(scan_data)} rows, scale={scale:.2f})")

    print(f"  Counting prefix candidates: {len(pfx_cands)}")
    pfx_counts = {}
    for cand in pfx_cands:
        c = sum(1 for s in scan_data if len(s) >= len(cand) and s[:len(cand)] == cand)
        count = int(c * scale) if not full_scan else c
        if count >= min_freq:
            pfx_counts[cand] = count

    print(f"  Counting suffix candidates: {len(sfx_cands)}")
    sfx_counts = {}
    for cand in sfx_cands:
        c = sum(1 for s in scan_data if len(s) >= len(cand) and s[-len(cand):] == cand)
        count = int(c * scale) if not full_scan else c
        if count >= min_freq:
            sfx_counts[cand] = count

    print(f"  Counting infix candidates: {len(ifx_cands)}")
    ifx_counts = {}
    for cand in ifx_cands:
        c = sum(1 for s in scan_data if cand in s)
        count = int(c * scale) if not full_scan else c
        if count >= min_freq:
            ifx_counts[cand] = count

    print(f"  Above threshold: prefix={len(pfx_counts)}, "
          f"suffix={len(sfx_counts)}, infix={len(ifx_counts)}")

    # ---- 去重 ----
    # 位置感知去重（同类型内）
    freq_prefix = _dedup_and_select(pfx_counts, top_k_substrings, mode='prefix')
    freq_suffix = _dedup_and_select(sfx_counts, top_k_substrings, mode='suffix')
    freq_infix = _dedup_and_select(ifx_counts, top_k_substrings, mode='infix')

    # 跨类型去冗余：如果 infix 串的频率远低于同串在 prefix/suffix 中的频率，
    # 说明它主要出现在首尾位置，infix 价值不大，去除
    infix_to_remove = []
    for s, count in freq_infix.items():
        pfx_count = freq_prefix.get(s, 0)
        sfx_count = freq_suffix.get(s, 0)
        # 如果 prefix 或 suffix 中的计数 >= infix 计数的 80%，
        # 说明该子串主要出现在边界，infix 计数膨胀不可靠
        if pfx_count >= count * 0.8 or sfx_count >= count * 0.8:
            infix_to_remove.append(s)
    for s in infix_to_remove:
        del freq_infix[s]
    if infix_to_remove:
        print(f"  Cross-type dedup: removed {len(infix_to_remove)} infix entries "
              f"(dominated by prefix/suffix)")

    substr_library = {
        'exact': freq_exact,
        'prefix': freq_prefix,
        'suffix': freq_suffix,
        'infix': freq_infix,
    }

    total = sum(len(v) for v in substr_library.values())
    print(f"  Library: exact={len(freq_exact)}, prefix={len(freq_prefix)}, "
          f"suffix={len(freq_suffix)}, infix={len(freq_infix)}, total={total}")
    print(f"  Substring length range: [{min_substr_len}, {max_substr_len}] "
          f"(N-gram covers <= {min_substr_len - 1})")

    return substr_library


# =====================================================
#  3. 下界查询
# =====================================================

def _segments_match_in_string(string: str, segments: List[str]) -> bool:
    """
    检查 segments 是否按顺序出现在 string 中（子串型 LIKE 匹配的核心）。
    segments 中的 '_' 匹配任意单个字符。
    """
    pos = 0
    for seg in segments:
        found = False
        while pos + len(seg) <= len(string):
            match = True
            for i, ch in enumerate(seg):
                if ch == '_':
                    continue
                if string[pos + i] != ch:
                    match = False
                    break
            if match:
                pos += len(seg)
                found = True
                break
            pos += 1
        if not found:
            return False
    return True


def _get_like_segments(query: str) -> List[str]:
    core = query.strip('%')
    return [seg for seg in core.split('%') if seg]


def _suffix_after_last_percent(query: str) -> str:
    return query.rsplit('%', 1)[-1]


def _has_literal(text: str) -> bool:
    return any(ch not in '%_' for ch in text)


def _right_extension_safe(query: str) -> bool:
    """
    Whether appending arbitrary characters to a matching witness can still be
    absorbed by the query. A trailing literal after the last % fixes the row
    suffix and is therefore not right-extension safe.
    """
    return not _has_literal(_suffix_after_last_percent(query))


def _prefix_entry_safe(query: str, entry: str) -> bool:
    """
    Whether all rows starting with entry are guaranteed to match query.

    This is a conservative sufficient condition, not a complete LIKE
    implication solver.
    """
    if not like_match(query, entry):
        return False

    if query.startswith('%'):
        # If containing entry is enough, starting with entry is also enough.
        return _infix_entry_safe(query, entry)

    # Rows represented by a prefix entry may have arbitrary extra suffix.
    return _right_extension_safe(query)


def _suffix_entry_safe(query: str, entry: str) -> bool:
    """
    Whether all rows ending with entry are guaranteed to match query.

    Conservative sufficient condition.
    """
    if not like_match(query, entry):
        return False

    if query.endswith('%'):
        # If containing entry is enough, ending with entry is also enough.
        return _infix_entry_safe(query, entry)

    # Rows represented by a suffix entry may have arbitrary extra prefix.
    return query.startswith('%')


def _infix_entry_safe(query: str, entry: str) -> bool:
    """
    Whether all rows containing entry are guaranteed to match query.

    Infix entries say nothing about row boundaries, so the query must have no
    fixed row-start requirement and no fixed literal suffix. Patterns such as
    %Gree%_ are safe if entry itself matches them: any row containing "Green"
    also contains a witness for "Gree" followed by at least one character.
    """
    if not query.startswith('%') or not _right_extension_safe(query):
        return False
    return like_match(query, entry)


def get_lower_bound(query: str, substr_library: dict, min_literal_len: int = 3) -> int:
    """
    对查询模式 P 计算严格下界。

    每个库的计数语义不同，必须匹配使用：
      - exact 库 count(S) = "行等于 S 的数量"
        → S LIKE P 为真时，count(S) 是下界（任何查询类型都适用）
      - prefix 库 count(S) = "行以 S 开头的数量"
        → 仅当查询也要求以 S 开头时才是下界（查询无前导 %）
      - suffix 库 count(S) = "行以 S 结尾的数量"
        → 仅当查询也要求以 S 结尾时才是下界（查询无尾随 %/_）
      - infix 库 count(S) = "行包含 S 的数量"
        → 仅当查询是纯子串型（首尾都是 %）时才是下界

    LB > UB 的情况不应出现——如果出现，在 CLIQUE_bound 的 append_bound 中会 min(lb, ub)。

    对短查询（literal 字符数 < min_literal_len）跳过。
    """
    literal_len = sum(1 for ch in query if ch not in '%_')
    if literal_len < min_literal_len:
        return 0

    # 判断查询边界
    is_prefix_query = not query.startswith('%')   # 查询要求固定前缀
    is_suffix_query = not query.endswith('%')      # 查询要求固定后缀（末尾不是%）
    # 注意：%abc%_ 末尾是 _，is_suffix_query = True

    best_lb = 0

    # ---- exact 库：S 是完整行，任何查询类型都适用 ----
    for s, count in substr_library.get('exact', {}).items():
        if count <= best_lb:
            continue
        if like_match(query, s):
            best_lb = count

    # ---- prefix 库：count = "以 S 开头的行数" ----
    # 必须保证所有以 S 开头的行都匹配 query。
    for s, count in substr_library.get('prefix', {}).items():
        if count <= best_lb:
            continue
        if _prefix_entry_safe(query, s):
            best_lb = count

    # ---- suffix 库：count = "以 S 结尾的行数" ----
    # 必须保证所有以 S 结尾的行都匹配 query。
    for s, count in substr_library.get('suffix', {}).items():
        if count <= best_lb:
            continue
        if _suffix_entry_safe(query, s):
            best_lb = count

    # ---- infix 库：count = "包含 S 的行数" ----
    # 仅对首尾都是 % 的纯包含/顺序查询安全。
    for s, count in substr_library.get('infix', {}).items():
        if count <= best_lb:
            continue
        if _infix_entry_safe(query, s):
            best_lb = count

    return best_lb


# =====================================================
#  4. 批量下界计算
# =====================================================

def batch_get_lower_bounds(queries: List[str],
                           substr_library: dict,
                           min_literal_len: int = 3,
                           show_progress: bool = True) -> List[int]:
    """批量计算查询列表的下界。"""
    bounds = []
    iterator = tqdm(queries, desc="[Lower Bounds]") if show_progress else queries
    for q in iterator:
        lb = get_lower_bound(q, substr_library, min_literal_len)
        bounds.append(lb)
    return bounds


# =====================================================
#  5. 持久化
# =====================================================

def save_freq_library_txt(substr_library: dict, filepath: str):
    """
    保存高频字符串库为可读 txt 文件，便于人工检查。

    格式:
      === EXACT (N entries) ===
      count  string
      === PREFIX (N entries, len >= X) ===
      count  len  string
      ...
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for category in ['exact', 'prefix', 'suffix', 'infix']:
            entries = substr_library.get(category, {})
            # 按基数降序
            sorted_entries = sorted(entries.items(), key=lambda x: -x[1])
            f.write(f"{'=' * 60}\n")
            f.write(f"  {category.upper()} ({len(sorted_entries)} entries)\n")
            if sorted_entries:
                lengths = [len(s) for s, _ in sorted_entries]
                f.write(f"  Length range: [{min(lengths)}, {max(lengths)}]\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"{'Count':>8s}  {'Len':>4s}  String\n")
            f.write(f"{'-'*8}  {'-'*4}  {'-'*40}\n")
            for s, count in sorted_entries:
                f.write(f"{count:8d}  {len(s):4d}  {s}\n")
            f.write(f"\n")
    print(f"Saved frequency library (txt) to {filepath}")


def save_freq_library(substr_library: dict, filepath: str):
    """保存高频字符串库。"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(substr_library, f)
    print(f"Saved frequency library to {filepath}")


def load_freq_library(filepath: str) -> dict:
    """加载高频字符串库。"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)



# =====================================================
#  6. 测试
# =====================================================

def _run_tests():
    """验证下界逻辑的正确性。"""
    print("=" * 60)
    print("Lower Bound Tests")
    print("=" * 60)

    test_library = {
        'exact': {
            'hello world': 10,
            'hello there': 5,
            'goodbye': 3,
        },
        'prefix': {
            'hello': 15,
            'hel': 20,
            'good': 8,
            'goodbye world': 2,
        },
        'suffix': {
            'world': 12,
            'there': 7,
            'ello': 18,
            'Green': 130,
        },
        'infix': {
            'ello': 18,
            'ood': 9,
            'hello': 15,
            'bye': 6,
            'interesting': 25,
            'Green': 130,
        },
    }

    tests = [
        ('%ello%',       18, "infix 'ello'"),
        ('hel%',         20, "prefix 'hel'"),
        ('%world',       12, "suffix 'world'"),
        ('hello world',  10, "exact 'hello world'"),
        ('%int%rest%',   25, "infix 'interesting' matches segments [int, rest]"),
        ('%xyz%',         0, "no match"),
        ('%a%',           0, "skipped (too short)"),
        ('%h_llo%',      15, "infix 'hello' via _ wildcard"),
        ('good%',         8, "prefix 'good'"),
        ('%i%ter%ng%',   25, "infix 'interesting' matches [i, ter, ng]"),
        ('%Gree%_',     130, "infix 'Green' is safe: % before _ absorbs extra suffix"),
        ('good%world',     0, "prefix 'goodbye world' is unsafe for fixed suffix"),
        ('%hello',         0, "prefix/infix cannot guarantee fixed suffix"),
        ('%good%',        9, "infix 'ood' not 'good'? Actually infix 'hello'=15 no... check"),
    ]

    passed = 0
    for query, expected, desc in tests[:-1]:  # skip last custom one below
        lb = get_lower_bound(query, test_library)
        status = "PASS" if lb == expected else "FAIL"
        print(f"  {status}: {query:20s} -> LB={lb:4d}  (expect {expected}, {desc})")
        if lb == expected:
            passed += 1

    # Test %good%: infix 里没有 'good'，但有 'ood'(9)
    # 'ood' 满足 _segments_match_in_string('ood', ['good'])? 不，'good' 不在 'ood' 中
    # 但 exact 里有 'goodbye'(3)，like_match('%good%', 'goodbye') = True
    lb = get_lower_bound('%good%', test_library)
    expected = 8  # from prefix 'good' (like_match('%good%', 'good') = True, count=8 > goodbye's 3)
    status = "PASS" if lb == expected else "FAIL"
    print(f"  {status}: {'%good%':20s} -> LB={lb:4d}  (expect {expected}, prefix 'good')")
    if lb == expected:
        passed += 1

    total = len(tests)
    print(f"\n  {passed}/{total} passed")

    # 验证 like_match 本身
    print(f"\n  like_match sanity checks:")
    assert like_match('%abc%', 'xabcy') == True
    assert like_match('%abc%', 'abc') == True
    assert like_match('%abc%', 'ab') == False
    assert like_match('abc%', 'abcdef') == True
    assert like_match('abc%', 'xabc') == False
    assert like_match('%abc', 'xyzabc') == True
    assert like_match('%a_c%', 'xabcy') == True
    assert like_match('%a_c%', 'xacy') == False  # len mismatch: a_c needs 3 chars
    assert like_match('%a_c%', 'xadcy') == True
    print(f"  All like_match checks passed!")


if __name__ == '__main__':
    _run_tests()
