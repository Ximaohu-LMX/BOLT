import time
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import string as _string
import itertools
import re
import pickle
import torch
import sys
import math
import argparse
import json
import socket
import platform
from packaging import version
from tensorboardX import SummaryWriter
from datetime import datetime
import warnings
import os
import csv
import fcntl
import hashlib
import subprocess
import yaml
import pandas as pd
from typing import Iterable
from collections import defaultdict
from copy import deepcopy
from functools import partial

esc_sc = "\u02C2"  # escape start character
esc_ec = "\u02C3"  # escape end character
esc_ac = "\u203C"  # escape any character
like_query_wildcard = r"""%_[]^-\()"""
regex_query_wildcard = r"$+*.[]|?\()"
punctuations = _string.punctuation + r"""—¨©«®°´»‒‘’‚,“”„•…″"""
min_word_len = 2
max_word_len = 10
# region query pattern gen

_aug_func_dict = None


class Qtype:
    SUBSTR = 0
    PREFIX = 1
    SUFFIX = 2

    @classmethod
    def iter(cls):
        return iter([cls.SUBSTR, cls.PREFIX, cls.SUFFIX])


class Ptype:
    SIMPLE = 0
    CORE = 1
    SIMPLE_B = 2
    CORE_B = 3
    SIMPLE_M = 4
    CORE_M = 5
    NORMAL = 6
    NORMAL_B = 7
    NORMAL_M = 8

    @classmethod
    def iter(cls):
        return iter(
            [
                cls.SIMPLE,
                cls.CORE,
                cls.SIMPLE_B,
                cls.CORE_B,
                cls.SIMPLE_M,
                cls.CORE_M,
                cls.NORMAL,
                cls.NORMAL_B,
                cls.NORMAL_M,
            ]
        )


class passingdict(dict):
    @staticmethod
    def __missing__(key):
        return key


def get_aug_func_dict(trim=False):
    global _aug_func_dict
    _aug_func_dict = {
        "q": None,
        "pr": augment_query_prefix,
        "de": lambda x: augment_query_delimiter(x, False, trim),
        "pl": lambda x: augment_query_placeholder(x, False, trim),
        "gq": None,
        "ru": None,
    }
    return _aug_func_dict


def save_augmented_queries(qrys, aug_func, exp_path):
    os.makedirs(os.path.dirname(exp_path), exist_ok=True)
    aqrys = augment_queries(qrys, aug_func)
    with open(exp_path, "w") as f:
        print(f"n_qry: {len(qrys)}, n_aqry: {len(aqrys)} path:", exp_path)
        for aqry in aqrys:
            f.write(aqry + "\n")
    print(aqrys[:3])


def augment_queries(patterns, aug_func):
    expanded_patterns = set()
    for pattern in patterns:
        for exp_pat in aug_func(pattern):
            expanded_patterns.add(exp_pat)

    return list(sorted(expanded_patterns))


def augment_query_prefix(query):
    aug_queries = []
    for i in range(len(query)):
        aug_queries.append(query[: i + 1])
    return aug_queries


def augment_query_delimiter(query, unique=True, trim=False):
    aug_queries = []
    if trim:
        assert query[0] == "%" and query[-1] != "%"
        query += "%"

    assert query[0] == "%" and query[-1] == "%"
    if unique:
        if query[0] == "%":
            aug_queries.append("%")
    for i in range(len(query)):
        prefix = query[: i + 1]
        if prefix[-1] == "%":
            aug_query = prefix
            if unique:
                continue
        else:
            aug_query = prefix + "%"
        aug_queries.append(aug_query)
    if trim:
        aug_queries = aug_queries[:-1]
    return aug_queries


def augment_query_placeholder(query, unique=True, trim=False):
    aug_queries = []
    if trim:
        assert query[0] == "%" and query[-1] != "%"
        query += "%"
    assert query[0] == "%" and query[-1] == "%"
    query_len = len(query)
    placeholder_list = ["%" if ch == "%" else "_" for ch in query]
    for i in range(query_len):
        placeholder_list[i] = query[i]
        aug_query = "".join(placeholder_list)
        aug_query = normalize_like_query(aug_query, trim)
        if unique:
            if len(aug_queries) == 0 or aug_queries[-1] != aug_query:
                aug_queries.append(aug_query)
        else:
            aug_queries.append(aug_query)
    if trim:
        aug_queries = aug_queries[:-1]

    return aug_queries


def normalize_like_query(query, trim=False):
    #         '_'
    #         / \
    #   '%'   v |
    # 0 --->   1   ---> 2
    #         ^ |
    #         \ /
    #         '%'
    assert query[-1] == "%"
    char_list = list(query)
    state = 0
    for i in reversed(range(len(query))):
        ch = query[i]
        if state == 0:
            if ch == "%":
                state = 1
            else:
                break
        elif state == 1:
            if ch == "%":
                char_list.pop(i)
            elif ch == "_":
                state = 1
            else:
                state = 2
                break

    return "".join(char_list)


def gen_all_like_patterns_with_two_subwords(sentences, max_len, suppress2=False):
    patterns = set()
    for sentence in tqdm(sentences):
        words = sentence.split(" ")
        for word in words:
            for i, j in zip(*np.triu_indices(len(word) + 1, k=1)):
                if (j - i) + 2 > max_len:
                    continue
                pattern = "%" + word[i:j] + "%"
                if len(pattern) <= max_len:
                    patterns.add(pattern)

                if not suppress2:
                    remain_word = word[j:]
                    for m, n in zip(*np.triu_indices(len(remain_word) + 1, k=1)):
                        if (j - i) + (m - n) + 3 > max_len:
                            continue
                        pattern = "%" + word[i:j] + \
                            "%" + remain_word[m:n] + "%"
                        if len(pattern) <= max_len:
                            patterns.add(pattern)

    patterns = sorted(patterns)
    return patterns


def gen_like_pattern_with_two_subwords(
    sentences, max_len, seed=None, repeat=1, **kwargs
):
    random.seed(seed)
    # gen pattern %s1%s2%
    patterns = set()
    phrases = set()
    max_len_subwords = max_len - 3  # three %
    for sentence in sentences:
        words = sentence.split()
        count_long_word = 0
        for i in range(len(words)):
            word = words[i]
            if len(word) > max_len_subwords:
                count_long_word += 1
                phrases.add(word)
        if count_long_word == 0:
            count_long_phrase = 0
            for i in range(len(words) - 1):
                phrase = " ".join(words[i: i + 1])
                if len(phrase) > max_len_subwords:
                    count_long_phrase += 1
                    phrases.add(phrase)

    for word in phrases:
        assert len(word) > max_len_subwords, word
        min_delete_len = len(word) - max_len_subwords  # min_delete_len >= 1
        for _ in range(repeat):
            (i,) = random.sample(
                range(1, max_len_subwords), 1
            )  # select i characters (word[0], word[1], ... , word[i-1])
            (j,) = random.sample(
                range(i, max_len_subwords), 1
            )  # select max_len_subwords - j

            # skip max_delete_len characters (word[i], word[i+1], ... word[i+(max_delete_len-1)])
            j += min_delete_len

            pattern = "%" + word[:i] + "%" + word[j:] + "%"
            patterns.add(pattern)
    patterns = list(sorted(patterns))
    return patterns


def pattern_gen2(sentences, pat_type="sub", max_len=None, **kwargs):
    patterns = set()
    word_tuples = set()
    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                word_tuples.add((words[i], words[j]))

    for words in word_tuples:
        word1, word2 = words
        for i in range(len(word1)):
            for j in range(i + 1, len(word1) + 1):
                for k in range(len(word2)):
                    for l in range(k + 1, len(word2) + 1):
                        if (j - i) + (l - k) > max_len:
                            continue
                        pattern = "%" + word1[i:j] + "%" + word2[k:l] + "%"
                        patterns.add(pattern)
    patterns = list(sorted(patterns))
    return patterns


def pattern_gen3(sentences, pat_type="sub", max_len=None, **kwargs):
    patterns = set()
    word_tuples = set()
    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                word_tuples.add((words[i], words[j]))

    for words in word_tuples:
        word1, word2 = words
        for i in range(len(word1)):
            for j in range(i + 1, len(word1) + 1):
                for k in range(len(word2)):
                    for l in range(k + 1, len(word2) + 1):
                        if (j - i) + (l - k) + 3 > max_len:
                            continue
                        pattern = "%" + word1[i:j] + "%" + word2[k:l] + "%"
                        patterns.add(pattern)
    patterns = list(sorted(patterns))
    return patterns


def pattern_gen(sentences, pat_type, n_pat=1, max_len=None, **kwargs):
    if n_pat == 2:
        assert pat_type == "sub"
        return pattern_gen3(sentences, "sub", max_len)
    patterns = set()

    strings = set()
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            strings.add(word)

    if pat_type == "sub":
        for string in strings:
            for i in range(len(string)):
                for j in range(i + 1, len(string) + 1):
                    if max_len is not None:
                        if (j - i) > max_len:
                            continue
                    pat = "%" + string[i:j] + "%"
                    patterns.add(pat)
    elif pat_type == "prf":
        for string in strings:
            for i in range(1, len(string) + 1):
                if max_len is not None:
                    if i > max_len:
                        continue
                pat = string[:i] + "%"
                patterns.add(pat)
    elif pat_type == "suf":
        for string in strings:
            for i in range(1, len(string) + 1):
                if max_len is not None:
                    if i > max_len:
                        continue
                pat = "%" + string[-i:]
                patterns.add(pat)
    patterns = list(sorted(patterns))
    return patterns


def trim_words(words, min_word_len, max_word_len):
    words = list(
        filter(
            lambda word: len(word) >= min_word_len
            and len(word) <= max_word_len,
            words,
        )
    )
    return words


def refine_words(words, min_word_len, max_word_len, max_n_under):
    # min_word_len restricts only for words[1:-1]
    # It returns refined words
    # The first word is a prefix of sentence
    # The last word is a suffix of sentence
    # Each word may contain underscores

    def rstrip_prefix(first_word):
        if first_word.count("_") > max_n_under:
            indices = []
            for i in range(len(first_word)):
                if first_word[i] == "_":
                    indices.append(i)
            first_word = first_word[: indices[max_n_under - 1] + 1]
        return first_word

    def lstrip_suffix(last_word):
        if last_word.count("_") > max_n_under:
            indices = []
            for i in range(len(last_word)):
                if last_word[i] == "_":
                    indices.append(i)
            last_word = last_word[indices[-max_n_under]:]
        return last_word

    global like_query_wildcard
    global regex_query_wildcard
    global punctuations

    # replace like query wildcards of words with underscore
    words = [
        "".join(["_" if ch in like_query_wildcard else ch for ch in word])
        for word in words
    ]

    # replace regex wildcards of words with underscore
    words = [
        "".join(["_" if ch in regex_query_wildcard else ch for ch in word])
        for word in words
    ]

    # replace punctuations of words with underscore
    words = [
        "".join(["_" if ch in punctuations else ch for ch in word]) for word in words
    ]

    # replace space of words with underscore
    words = ["".join(["_" if ch == " " else ch for ch in word])
             for word in words]
    assert len(words) > 0, words

    # remove too short or long word

    if len(words) > 1:
        first_word = words[0]
        if len(first_word) > max_word_len:
            first_word = first_word[:max_word_len]
        first_word = rstrip_prefix(first_word)

        words_inter = list(
            filter(
                lambda word: len(word) >= min_word_len
                and len(word) <= max_word_len
                and word.count("_") <= max_n_under,
                words[1:-1],
            )
        )
        last_word = words[-1]
        last_word = lstrip_suffix(last_word)

        if len(last_word) > max_word_len:
            last_word = last_word[-max_word_len:]

        if last_word.count("_") > max_n_under:
            last_word = last_word[last_word.rfind("_", max_n_under - 1):]
        words = [first_word]
        words.extend(words_inter)
        words.append(last_word)
    else:
        word = words[0]
        if len(word) > max_word_len or word.count("_") > max_n_under:
            if len(word) >= max_word_len * 2:
                sp = max_word_len
            else:
                sp = len(word) // 2
            first_word = word[:sp]
            first_word = rstrip_prefix(first_word)
            second_word = word[-sp:]
            second_word = lstrip_suffix(second_word)
            words = [first_word, second_word]

    return words


def filter_words(words, min_word_len=None, max_word_len=None):
    global like_query_wildcard
    global regex_query_wildcard
    global punctuations

    # remove some words having like query wildcard
    words = list(
        filter(lambda word: all(x not in like_query_wildcard for x in word), words)
    )

    # remove some words having regex wildcard
    words = list(
        filter(lambda word: all(x not in regex_query_wildcard for x in word), words)
    )

    # remove some words consisting of only punctuations
    words = list(filter(lambda word: any(
        x not in punctuations for x in word), words))

    # remove starting and ending punctuations
    words = [x.strip(punctuations) for x in words]

    # remove too short or long word
    if min_word_len is not None:
        words = list(filter(lambda word: len(word) >= min_word_len, words))

    if max_word_len is not None:
        words = list(filter(lambda word: len(word) <= max_word_len, words))

    return words


def general_query_pat(
    strings,
    n_token=1,
    max_n_under=0,
    min_word_len=2,
    max_word_len=5,
    seed=None,
    **kwargs,
):
    raise DeprecationWarning

    def gen_operator(n_op, max_n_under):
        # rejection sampling
        while True:
            op_list = []
            for i in range(n_op):
                is_percent = np.random.randint(0, 2)
                if i == 0 or i == n_op - 1:
                    n_under = np.random.randint(0, max_n_under + 1)
                else:
                    if max_n_under == 0:
                        is_percent = 1
                    if is_percent == 0:
                        n_under = np.random.randint(1, max_n_under + 1)
                    else:
                        n_under = np.random.randint(0, max_n_under + 1)

                op = "_" * n_under
                if is_percent:
                    op = "%" + op
                op_list.append(op)
            if sum([len(x) for x in op_list]) > 0:
                break
        return op_list

    np.random.seed(seed)
    qrys = []
    n_op = n_token + 1
    for string in strings:
        words = string.split()
        words = filter_words(words, min_word_len, max_word_len)

        n_word = len(words)
        if n_word < n_token:
            continue
        token_idx = sorted(np.random.choice(
            range(n_word), size=n_token, replace=False))
        tokens = [words[idx] for idx in token_idx]
        op_list = gen_operator(n_op, max_n_under)
        assert len(op_list) == n_op, op_list
        qry = op_list[0] + \
            "".join([x + y for x, y in zip(tokens, op_list[1:])])
        assert "%" in qry or "_" in qry, (qry, op_list, tokens)
        qrys.append(qry)

    qrys = sorted(set(qrys))
    return qrys


def is_wild_string(word):
    return all([(x == "_" or x == "%") for x in word])


def get_num_betas(qry):
    parsed = parse_like_query(qry)
    count = 0
    for word in parsed:
        if not is_wild_string(word):
            count += 1
    return count


def get_join_alphas(qry):
    parsed = parse_like_query(qry)
    if is_wild_string(parsed[-1]):
        parsed = parsed[:-1]
    if is_wild_string(parsed[0]):
        parsed = parsed[1:]

    # print(parsed)

    output = []
    for token in parsed:
        if is_wild_string(token):
            output.append(token)
    return output


def compile_LIKE_query(qry):
    query_pat = "".join(
        "(.*?)" if ch == "%" else "." if ch == "_" else re.escape(ch)
        for ch in qry
    )
    qry_compiled = re.compile(query_pat)
    return qry_compiled


def eval_compiled_LIKE_query(qry_re, rec):
    return qry_re.fullmatch(rec)


def parse_like_query(qry, split_beta=False):
    parsed = []
    curr = qry[0]
    is_wild = curr == "_" or curr == "%"
    for ch in qry[1:]:
        if ch == "_" or ch == "%":
            if is_wild:
                curr += ch
            else:
                parsed.append(curr)
                is_wild = True
                curr = ch
        else:
            if is_wild:
                parsed.append(curr)
                is_wild = False
                curr = ch
            else:
                curr += ch
    parsed.append(curr)
    if split_beta:
        parsed_bak = parsed
        parsed = []
        for token in parsed_bak:
            if "%" in token or "_" in token:
                parsed.append(token)
            else:
                parsed.extend(list(token))
    return parsed


def flip_last_canonicalized_like_query(qry):
    is_percent = False
    n_under = 0
    for i in reversed(range(len(qry))):
        ch = qry[i]
        if ch == '%':
            is_percent = True
        elif ch == '_':
            n_under += 1
        else:
            break
    if is_percent:
        return qry[:(i+1)] + ('_' * n_under + '%')
    else:
        return qry


def canonicalize_like_query(qry, is_last_flip=False):
    parsed = parse_like_query(qry)

    out_tokens = []
    for token in parsed:
        if "_" in token or "%" in token:
            if "%" in token:
                new_token = "%"
            else:
                new_token = ""
            new_token += "_" * token.count("_")
            out_tokens.append(new_token)
        else:
            out_tokens.append(token)
    if is_last_flip and "%" in out_tokens[-1]:
        out_tokens[-1] = "_" * (len(out_tokens[-1]) - 1) + "%"
    return "".join(out_tokens)


def split_string_remaining_ending_spaces(string):
    words = string.split()
    ending = string[len(string.rstrip()):]
    starting = string[: -len(string.lstrip())]
    # print(starting, len(starting))
    if len(words) > 0:
        if len(starting) > 0:
            words[0] = starting + words[0]
        if len(ending) > 0:
            words[-1] = words[-1] + ending
    else:  # one word consisting of only spaces
        words.append(ending)
    return words


def replace_under(token, max_n_under):
    n_under = token.count("_")
    max_n_under = min(len(token), max_n_under)

    # rejection sampling
    alpha = 2
    assert n_under <= max_n_under
    if max_n_under - n_under == 0:
        return token
    while True:
        n_under_sample = np.random.zipf(alpha) - 1
        if n_under_sample <= (max_n_under - n_under):
            break
    indices = []
    for i in range(len(token)):
        if token[i] != "_":
            indices.append(i)
    replace_idx_list = np.random.choice(indices, n_under_sample)
    ch_list = list(token)
    for idx in replace_idx_list:
        ch_list[idx] = "_"
    token = "".join(ch_list)
    return token


def refine_data_string2words(data_string, sample_q_type, n_token, min_word_len=2, max_word_len=5):
    # if fail return None
    words = split_string_remaining_ending_spaces(data_string)
    if len(words) == 0:
        return
    if sample_q_type == Qtype.SUBSTR:
        words = trim_words(words, min_word_len, max_word_len)
    elif sample_q_type == Qtype.PREFIX:
        prefix = words[0][:max_word_len]
        words = trim_words(words[1:], min_word_len, max_word_len)
        words.insert(0, prefix)
    elif sample_q_type == Qtype.SUFFIX:
        suffix = words[-1][-max_word_len:]
        words = trim_words(words[:-1], min_word_len, max_word_len)
        words.append(suffix)
    if sample_q_type == Qtype.PREFIX and len(words) == 1:
        return words[:1]
    elif sample_q_type == Qtype.SUFFIX and len(words) == 1:
        return words[-1:]

    if len(words) >= n_token:
        return words


def general_query_pat2_gen_query_from_words(
    words, sample_q_type, max_n_under, n_token
):
    n_word = len(words)
    if sample_q_type == Qtype.SUBSTR:
        token_idx = sorted(np.random.choice(
            range(n_word), size=n_token, replace=False))
        tokens = [words[idx] for idx in token_idx]
        tokens = [replace_under(token, max_n_under) for token in tokens]
        qry = "%" + "%".join(tokens) + "%"
        qry = canonicalize_like_query(qry, is_last_flip=True)
        if qry[0] != "%" or qry[-1] != "%":
            return

    elif sample_q_type == Qtype.PREFIX:
        token_idx = sorted(
            np.random.choice(range(1, n_word), size=n_token - 1, replace=False)
        )
        tokens = [words[0]]
        tokens.extend([words[idx] for idx in token_idx])
        tokens = [replace_under(token, max_n_under) for token in tokens]
        qry = "%".join(tokens) + "%"
        qry = canonicalize_like_query(qry, is_last_flip=True)
        if qry[-1] != "%":
            return

    elif sample_q_type == Qtype.SUFFIX:
        token_idx = sorted(
            np.random.choice(range(n_word - 1),
                             size=n_token - 1, replace=False)
        )
        tokens = [words[idx] for idx in token_idx]
        tokens.append(words[-1])
        tokens = [replace_under(token, max_n_under) for token in tokens]
        qry = "%" + "%".join(tokens)
        qry = canonicalize_like_query(qry, is_last_flip=True)
        if qry[0] != "%":
            return
    else:
        raise ValueError(sample_q_type)

    op_tokens, tokens = wild_string_tokenizer(
        canonicalize_like_query(qry, True), "%")
    if len(tokens) != n_token:
        return
    qry = canonicalize_like_query(qry)

    return qry


def general_query_pat2_gen_query(
    string, sample_q_type, max_n_under, n_token, min_word_len=2, max_word_len=5,
    is_refine_words=False
):
    # words = string.split()

    words = split_string_remaining_ending_spaces(string)

    if len(words) == 0:
        return
    words_bak = words
    if is_refine_words:
        words = refine_words(words, min_word_len, max_word_len, max_n_under)
    else:
        if sample_q_type == Qtype.SUBSTR:
            words = trim_words(words, min_word_len, max_word_len)
        elif sample_q_type == Qtype.PREFIX:
            prefix = words[0][:max_word_len]
            words = trim_words(words[1:], min_word_len, max_word_len)
            words.insert(0, prefix)
        elif sample_q_type == Qtype.SUFFIX:
            suffix = words[-1][-max_word_len:]
            words = trim_words(words[:-1], min_word_len, max_word_len)
            words.append(suffix)
    if not all([x.count("_") <= max_n_under for x in words]):
        return
    assert all([x.count("_") <= max_n_under for x in words]
               ), (string, words_bak, words)

    if sample_q_type == Qtype.SUBSTR:
        words_bak = words
        words = []
        for word in words_bak:
            if len(word) > 0:
                words.append(word)
    elif sample_q_type == Qtype.PREFIX:
        if len(words[0]) == 0:
            return
    elif sample_q_type == Qtype.SUFFIX:
        if len(words[-1]) == 0:
            return

    n_word = len(words)
    if n_word < n_token:
        return

    if sample_q_type == Qtype.SUBSTR:
        token_idx = sorted(np.random.choice(
            range(n_word), size=n_token, replace=False))
        tokens = [words[idx] for idx in token_idx]
        tokens = [replace_under(token, max_n_under) for token in tokens]
        qry = "%" + "%".join(tokens) + "%"
        qry = canonicalize_like_query(qry, is_last_flip=True)
        if qry[0] != "%" or qry[-1] != "%":
            return

    elif sample_q_type == Qtype.PREFIX:
        token_idx = sorted(
            np.random.choice(range(1, n_word), size=n_token - 1, replace=False)
        )
        tokens = [words[0]]
        tokens.extend([words[idx] for idx in token_idx])
        tokens = [replace_under(token, max_n_under) for token in tokens]
        qry = "%".join(tokens) + "%"
        qry = canonicalize_like_query(qry, is_last_flip=True)
        if qry[-1] != "%":
            return

    elif sample_q_type == Qtype.SUFFIX:
        token_idx = sorted(
            np.random.choice(range(n_word - 1),
                             size=n_token - 1, replace=False)
        )
        tokens = [words[idx] for idx in token_idx]
        tokens.append(words[-1])
        tokens = [replace_under(token, max_n_under) for token in tokens]
        qry = "%" + "%".join(tokens)
        qry = canonicalize_like_query(qry, is_last_flip=True)
        if qry[0] != "%":
            return
    else:
        raise ValueError(sample_q_type)

    op_tokens, tokens = wild_string_tokenizer(
        canonicalize_like_query(qry, True), "%")
    if len(tokens) != n_token:
        return
    qry = canonicalize_like_query(qry)

    return qry


def general_query_pat2(
    strings,
    max_n_token=2,
    max_n_under=2,
    min_word_len=2,
    max_word_len=5,
    sub_r=0.5,
    pre_r=0.25,
    suf_r=0.25,
    seed=None,
    **kwargs,
):
    np.random.seed(seed)
    assert sub_r + pre_r + suf_r == 1, (sub_r, pre_r, suf_r)
    q_ratio = [sub_r, pre_r, suf_r]
    sample_q_types = np.random.choice(
        3, (len(strings), max_n_token), p=q_ratio)

    qrys = []
    for sid, string in tqdm(enumerate(strings), total=len(strings)):
        for n_token in range(1, max_n_token + 1):
            sample_q_type = sample_q_types[sid][n_token - 1]
            qry = general_query_pat2_gen_query(
                string, sample_q_type, max_n_under, n_token
            )
            if qry is not None:
                qrys.append(qry)
    qrys = sorted(set(qrys))
    return qrys


def query_n_pat(strings, max_word_len, max_n_pat=1, seed=None, **kwargs):
    np.random.seed(seed)
    qrys = []
    for string in strings:
        words = string.split()
        words = filter_words(words, min_word_len, max_word_len)

        n_word = len(words)
        if n_word == 0:
            continue
        n_pat = np.random.randint(1, min(max_n_pat, n_word) + 1)
        pat_idx = sorted(np.random.choice(
            range(n_word), size=n_pat, replace=False))
        pats = [words[idx] for idx in pat_idx]
        qry = "%" + "%".join(pats) + "%"
        qrys.append(qry)

    qrys = sorted(set(qrys))
    return qrys


# endregion

# region (load|stat|preproc|measure)


def keras_pad_sequences(
    sequences, maxlen=None, dtype="int32", padding="pre", truncating="pre", value=0.0
):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError(
                "`sequences` must be a list of iterables. "
                "Found non-iterable: " + str(x)
            )

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    unicode_dtype = getattr(np, "unicode_", np.str_)
    is_dtype_str = np.issubdtype(
        dtype, np.str_) or np.issubdtype(dtype, unicode_dtype)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError(
            "`dtype` {} is not compatible with `value`'s type: {}\n"
            "You should set `dtype=object` for variable length strings.".format(
                dtype, type(value)
            )
        )

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(
                'Truncating type "%s" ' "not understood" % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                "Shape of sample %s of sequence at position %s "
                "is different from expected shape %s"
                % (trunc.shape[1:], idx, sample_shape)
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def compute_n_placeholders(string):
    string = string[1:] + "a"
    res = list(np.cumsum([x != "%" for x in reversed(string)])[::-1])
    return res


def get_union_char_set(db, alpha_list=None):
    special_chars = ["[PAD]", "[UNK]", "%", "_", "[", "]", "^", "-"]
    if alpha_list is not None:
        special_chars = ["[PAD]", "[UNK]"]
        special_chars.extend(alpha_list)
        special_chars.extend(["[", "]", "^", "-"])
    char_set_db = char_set_from_db(db, sort_by_freq=True)
    for special_char in special_chars:
        # assert special_char not in char_set_db
        if special_char in char_set_db:
            char_set_db.remove(special_char)

    char_set = special_chars + list(char_set_db)
    n_special_char = len(special_chars)

    return char_set, n_special_char


def train_valid_test_split(input_list, p_test, p_valid, seed, p_train=None):
    # total: 1.0 + p_test + p_valid
    train_valid, test = train_test_split(
        input_list, test_size=p_test / (1.0 + p_test + p_valid), random_state=seed
    )
    train, valid = train_test_split(
        train_valid, test_size=p_valid / (1.0 + p_valid), random_state=seed)
    if p_train is not None:
        train = clip_string_by_its_len(train, p_train)
    return train, valid, test


def train_valid_test_split_test_first(input_list, p_test, p_valid, seed, p_train=None):
    train_valid, test = train_test_split(
        input_list, test_size=p_test, random_state=seed
    )
    train, valid = train_test_split(
        train_valid, test_size=p_valid, random_state=seed)
    if p_train is not None:
        train = clip_string_by_its_len(train, p_train)
    return train, valid, test


def clip_string_by_its_len(x, prob):
    assert isinstance(prob, float)
    if len(x) == 0 or prob == 1:
        return x
    group = {}
    for rec in x:
        query = rec
        if len(query) not in group:
            group[len(query)] = []
        group[len(query)].append(rec)

    output = []
    for length, g in group.items():
        prob2 = max(prob, 1 / len(g))
        selected, discarded = train_test_split(
            g, train_size=prob2, random_state=0)
        output.extend(selected)
    return output


# def annotate_cardinalities_to_query_steps(qry_steps, res_dict, aug=False):
#     data = []
#     for step, qry_step in enumerate(qry_steps, start=1):
#         data.append(annotate_cardinalities_to_queries(qry_step, res_dict, aug=aug, step=step))
#     return data


def annotate_cards_to_queries(queries, res_dict, aug_func=None, trim=False):
    res = []
    for query in queries:
        if aug_func is not None:
            aug_queries = aug_func(query)
            cards = []
            for aug_qry in aug_queries:
                if trim:
                    cards.append(res_dict[aug_qry])
                else:
                    cards.append(res_dict[aug_qry])

        else:
            if trim:
                cards = res_dict[query + "%"]
            else:
                cards = res_dict[query]
        res.append([query, cards])
    return res


# def annotate_cardinalities_to_queries(queries, res_dict, aug=False, step=1):
#     """
#     Parameters
#     ----------
#     """
#     res = []
#     for query in queries:
#         if aug and step == 2:
#             sb_list = query.strip("%").split("%")
#             patterns = []
#             for i in range(len(sb_list)):
#                 pattern = "%" + "%".join(sb_list[:i+1]) + "%"
#                 patterns.append(pattern)
#             cards = [res_dict[pattern] for pattern in patterns]
#             res.append([query, len(query), cards])
#         else:
#             res.append([query, len(query), res_dict[query]])
#     return res


def split_query_in_steps(queries, is_train, n_step):
    """
    Parameters
    ----------
    n_step : # of learning steps

    Examples
    --------
    >>> X = ["%Top%Gun%"]
    >>> split_query_in_steps(X, True, 2)
    >>> [[%Top%], [%Top%Gun%]]
    """
    assert n_step >= 1 and n_step <= 2
    if n_step == 1:
        return queries

    step1 = []
    step2 = []

    if is_train:
        for qry in queries:
            sb_list = qry.strip("%").split("%")
            n_sb = len(sb_list)
            assert n_sb >= 1 and n_sb <= 2
            if n_sb == 1:  # sb1
                step1.append(qry)
            elif n_sb == 2:  # sb2
                step1.append(f"%{sb_list[0]}%")  # sb21
                step2.append(qry)
    else:
        for qry in queries:
            sb_list = qry.strip("%").split("%")
            n_sb = len(sb_list)
            assert n_sb >= 1 and n_sb <= 2
            if n_sb == 1:  # sb1
                step1.append(qry)
                step2.append(qry)
            elif n_sb == 2:  # sb2
                step1.append(f"%{sb_list[0]}%")  # sb21
                step2.append(qry)

    step1 = sorted(set(step1))
    step2 = sorted(set(step2))
    assert len(step1) >= 1

    return [step1, step2]


def save_query(patterns, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        print("path:", save_path)
        for pattern in patterns:
            f.write(pattern + "\n")
    print(patterns[:3])


def _parse_pattern_card_line(line):
    line = line.rstrip("\n\r")
    parts = line.rsplit(",", 1)
    if len(parts) != 2:
        raise ValueError(f"Malformed training line: {line[:120]}")
    return parts[0], parts[1]


def get_qc_dict(data_name, workload):
    training_path = f"data/{data_name}/training/{workload}/pack_simple.txt"
    qc_dict = {}
    with open(training_path) as f:
        for line in f:
            pat, card = _parse_pattern_card_line(line)
            card = int(card)
            qc_dict[pat] = card
    return qc_dict


def get_cf_dict(data_name):
    file_path = f"data/{data_name}/{data_name}.txt"
    db = read_strings(file_path)
    n = len(db)
    prefix_count = defaultdict(float)
    substr_count = defaultdict(float)
    suffix_count = defaultdict(float)
    occurence_count = defaultdict(float)
    # db = db[:3]
    # print(db)
    for rec in db:
        prefix_count[rec[0]] += 1
        suffix_count[rec[-1]] += 1
        for ch in rec:
            occurence_count[ch] += 1
        for ch in set(rec):
            substr_count[ch] += 1

    def normalize(input_dict):
        for k, v in input_dict.items():
            input_dict[k] = v / n

    normalize(prefix_count)
    normalize(substr_count)
    normalize(suffix_count)
    normalize(occurence_count)

    return [prefix_count, substr_count, suffix_count, occurence_count]


def gen_char_dict(db):
    char_set, n_special_char = get_union_char_set(db)
    char_dict = char_dict_from_char_set(char_set)
    return char_dict, n_special_char


def get_training_data(
    data_name,
    workload,
    p_train=1.0,
):
    db, train_data, valid_data, test_data = load_training_files(
        data_name, workload)
    char_dict, n_special_char = gen_char_dict(db)
    train_data = clip_string_by_its_len(train_data, p_train)

    return char_dict, n_special_char, train_data, valid_data, test_data


def _training_target_values(target):
    """Return scalar or packed training targets as a flat iterable."""
    if isinstance(target, (list, tuple, np.ndarray)):
        return target
    return (target,)


def assert_no_unlabeled_training_targets(train_data, context="standard training"):
    """Reject card=-1 rows before they can enter a supervised loss."""
    bad_rows = []
    for index, row in enumerate(train_data):
        if any(float(value) < 0 for value in _training_target_values(row[1])):
            bad_rows.append(index)
            if len(bad_rows) == 5:
                break
    if bad_rows:
        raise ValueError(
            f"{context} requires non-negative targets; "
            f"found unlabeled rows at indices {bad_rows}"
        )


def labeled_training_rows(train_data):
    """Return only rows whose scalar or packed targets are all labeled."""
    return [
        row
        for row in train_data
        if all(float(value) >= 0 for value in _training_target_values(row[1]))
    ]


def get_training_data_AstridEach(
    data_name,
    p_train=1.0,
):
    train_data_triple, valid_data_triple, test_data_triple = load_training_files_AstridEach(
        data_name)
    train_data_triple = [clip_string_by_its_len(
        x, p_train) for x in train_data_triple]

    return train_data_triple, valid_data_triple, test_data_triple


def string_encoding(strings, char_dict):
    output = []
    for string in strings:
        encoded = [char_dict[x] for x in string]
        output.append(encoded)
    return output


def indices_decoding(indices, char_dict):
    output = ""
    char_set = {v: k for k, v in char_dict.items()}
    for index in indices:
        output += char_set[index]
    return output


def char_frequecy_from_db(db, char_set=None):
    if char_set is None:
        char_set = char_set_from_db(db)

    char_freq = dict()
    for rid, record in enumerate(db):
        for char in record:
            if char not in char_freq:
                char_freq[char] = 0
            char_freq[char] += 1
    return char_freq


def alpha_list_from_queries(queries):
    char_set = set()
    for query in queries:
        for char in parse_like_query(query):
            if "%" in char or "_" in char:
                char_set.add(char)
    return sorted(char_set)


def char_set_from_db(db, sort_by_freq=False):
    # """

    # :param db:
    # :param max_char: if max_char is None, return all characters. Otherwise, return at most max_char characters by their
    #     frequencies
    # :return:
    # """
    # if max_char:
    #     char_dict = dict()
    #     for rid, record in enumerate(db):
    #         for char in record:
    #             if char not in char_dict:
    #                 char_dict[char] = 0
    #             char_dict[char] += 1
    #     # frequency order
    #     char_set = list(zip(*sorted(char_dict.items(), key=lambda x: x[1], reverse=True)[:max_char]))[0]
    # else:
    char_set = set()
    for record in db:
        for char in record:
            char_set.add(char)
    char_set = sorted(char_set)  # alphabetical order
    if sort_by_freq:
        char_freq = char_frequecy_from_db(db)
        # print(f"{char_set[:10] = }")
        # char_set = list(zip(*sorted(char_freq.items(), key=lambda x: x[1], reverse=True)))[0]
        char_set = sorted(char_freq.keys(),
                          key=lambda x: char_freq[x], reverse=True)
        # print(f"{char_set[:10] = }")

    return char_set


def char_dict_from_db(db):
    char_set, n_special_char = get_union_char_set(db)
    char_dict = char_dict_from_char_set(char_set)
    return char_dict


def char_dict_from_char_set(char_set):
    # """

    # Args:
    #     db: list of strings

    # Returns:
    #     dictionary whose key and value are character and index, respectively.
    #     The index 0 is kept for [PAD] token.
    #     The index 1 is also kept for [UNK] token.
    # """
    # char_set = char_set_from_db(db)
    # char_dict = dict()
    # for i, char in enumerate(char_set):
    #     char_dict[char] = i + 3  # for [PAD], [UNK], [%]
    # return char_dict
    char_dict = dict()
    for i, char in enumerate(char_set):
        char_dict[char] = i

    return char_dict


def compare_two_dictionaries(dict1, dict2):
    dict1_keys = set(dict1.keys())
    dict2_keys = set(dict2.keys())
    if len(dict1_keys) != len(dict2_keys):
        return False
    if dict1_keys != dict2_keys:
        return False
    for key in dict1_keys:
        if dict1[key] != dict2[key]:
            return False
    return True


def mean_Q_error(input, target, reduction="mean", mask=None):
    if not isinstance(input, Iterable):
        input = [input]
    if not isinstance(target, Iterable):
        target = [target]
    input = np.array(input, dtype=float)
    target = np.array(target, dtype=float)

    input[input < 1.0] = 1.0
    target[target < 1.0] = 1.0
    q_errs = np.max(np.array([input / target, target / input]), axis=0)
    if mask is not None:
        q_errs *= mask

    if reduction == "mean":
        if mask is not None:
            output = np.sum(q_errs) / np.sum(mask)
        else:
            output = float(np.mean(q_errs))
    elif reduction == "max":
        output = np.max(q_errs)
    else:
        output = q_errs
    return output


def batch2cuda(batch, to_cuda=True):
    # ys should locate at the last position
    # only tensor to cuda
    xs = []
    ys = batch[-1]
    for i in range(len(batch) - 1):
        elem = batch[i]
        if isinstance(elem, torch.Tensor) and to_cuda:
            elem = elem.cuda()
        xs.append(elem)
    xs = tuple(xs)
    if isinstance(ys, torch.Tensor) and to_cuda:
        ys = ys.cuda()
    return xs, ys


def read_training(training_path):
    queries = []
    with open(training_path) as f:
        for line in f:
            pat, card = _parse_pattern_card_line(line)
            card = int(card)
            # res_dict[pat] = int(card)
            queries.append((pat, card))
    return queries


def read_training_SED(training_path, delta=3):
    queries = []
    with open(training_path) as f:
        for id, line in enumerate(f):
            if id == 0:
                continue
            line = line.rstrip("\n\r")
            parts = line.rsplit(",", delta + 1)
            if len(parts) != delta + 2:
                raise ValueError(f"Malformed SED training line in {training_path}: {line[:120]}")
            pat = parts[0]
            cards = [int(x) for x in parts[1:]]
            # res_dict[pat] = int(card)
            queries.append((pat, *cards))
    return queries


def read_strings(filepath):
    lines = []
    with open(filepath) as f:
        for line in f:
            lines.append(line.rstrip('\n'))
    return lines


def load_training_files(data_name, workload):
    # delta for samplingD
    training_files = []
    db = []
    with open(f"data/{data_name}/{data_name}.txt") as f:
        for line in f.readlines():
            db.append(line.rstrip("\n"))

    queries_all = []
    for d_type in ["train", "valid", "test"]:
        training_path = f"data/{data_name}/training/{workload}/{d_type}.txt"
        print(f"{d_type = :10s}{training_path = }")
        queries = read_training(training_path)
        queries_all.append(queries)

        training_files = [db, *queries_all]

    return training_files


def load_training_files_AstridEach(data_name):
    # delta for samplingD
    training_files = []
    for d_type in ["train", "valid", "test"]:
        data_triple = []
        for fn_desc in ["prefix", "suffix", "substring"]:
            training_path = f"data/{data_name}/training/Astrid/{fn_desc}/{d_type}.txt"
            print(f"{d_type = :10s}{training_path = }")
            queries = read_training(training_path)
            data_triple.append(queries)

        training_files.append(data_triple)

    return training_files


def pattern_with_q_type2query_string(pattern, q_type):
    if q_type == Qtype.SUBSTR:
        return "%" + pattern + "%"
    elif q_type == Qtype.PREFIX:
        return pattern + "%"
    elif q_type == Qtype.SUFFIX:
        return "%" + pattern
    else:
        raise ValueError


def find_like_query_type_idx(query):
    query = canonicalize_like_query(query, is_last_flip=True)
    wilds, norms = wild_string_tokenizer(query, wilds="%")

    n_norm = len(norms)

    if "%" in wilds[0] and "%" in wilds[-1]:
        return (n_norm - 1) * 3
    elif "%" in wilds[-1]:
        return (n_norm - 1) * 3 + 1
    elif "%" in wilds[0]:
        return (n_norm - 1) * 3 + 2
    else:
        raise TypeError(wilds, norms)


def wild_string_tokenizer(qry, wilds="_%"):
    tokens = []
    op_tokens = []
    ch = qry[0]
    is_op = any([ch == wilds for wilds in wilds])
    start = 0
    length = 0
    if not is_op:
        op_tokens.append("")

    for i, ch in enumerate(qry):
        if any([ch == wilds for wilds in wilds]):
            if is_op:
                length += 1
            else:
                token = qry[start: start + length]
                tokens.append(token)
                start = i
                length = 1
                is_op = True
        else:
            if is_op:
                token = qry[start: start + length]
                op_tokens.append(token)
                start = i
                length = 1
                is_op = False
            else:
                length += 1

    token = qry[start: start + length]
    if is_op:
        op_tokens.append(token)
    else:
        tokens.append(token)
        op_tokens.append("")

    return op_tokens, tokens


def merge_positional_lists(pl1, pl2, w):
    """
    For (id1, pos1) in pl1 and (id2, pos2) in pl2,
    if id1==id2 and (pos1 + w - 1) < pos2


    Parameters
    ----------
    pl1 : list of (id, list of pos)
        first positional inverted list
    pl2 : list of (id, list of pos)
        second positional inverted list
    w : int
        length of substring of pl1

    Return
    ------

    list of (id, list of pos)
        merged positional inverted list

    """
    ml = []  # merge list

    idx1 = 0
    idx2 = 0

    while idx1 < len(pl1) and idx2 < len(pl2):
        id1, pos_list1 = pl1[idx1]
        id2, pos_list2 = pl2[idx2]
        if id1 == id2:
            pos1 = pos_list1[0]
            for i, pos2 in enumerate(pos_list2):
                if pos1 + w - 1 < pos2:
                    ml.append((id2, pos_list2[i:]))
                    break
            idx1 += 1
            idx2 += 1
        elif id1 < id2:
            idx1 += 1
        else:
            idx2 += 1

    return ml


def human_readable_size(size, decimal_places=2):
    # for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def get_pickle_size(py_obj):
    # bytes
    return len(pickle.dumps(py_obj))


def get_torch_model_size(model):
    # bytes
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size


def find_parity_bool_array(arr, mask=None):
    arr_i = arr.reshape(-1, 1)
    arr_j = arr.reshape(1, -1)
    if mask is not None:
        return (arr_i < arr_j)[mask]
    else:
        return arr_i < arr_j


def get_non_equal_mask(arr):
    arr_i = arr.reshape(-1, 1)
    arr_j = arr.reshape(1, -1)
    mask = arr_i != arr_j
    return mask


def parity_score(c_true, c_est):
    mask = get_non_equal_mask(c_true)
    p_true = find_parity_bool_array(c_true, mask)
    p_est = find_parity_bool_array(c_est, mask)
    matched_arr = p_true == p_est

    matched = matched_arr.sum()
    total = len(matched_arr)
    return matched / total


def save_estimator_stat(
    stat_path, q_errs, build_time=None, est_time=None, model_size=None, count_size=None, neural_size=None,
):
    os.makedirs(os.path.dirname(stat_path), exist_ok=True)
    out_stat_dict = {}
    out_stat_dict["time"] = build_time
    out_stat_dict["time_est"] = est_time
    out_stat_dict["model_size"] = model_size
    out_stat_dict["avg"] = float(np.mean(q_errs))
    if count_size is not None:
        out_stat_dict["count_size"] = count_size
    if neural_size is not None:
        out_stat_dict["neural_size"] = neural_size

    for percent in range(1, 101):
        out_stat_dict[f"q{percent:03d}"] = float(
            np.percentile(q_errs, percent))
    with open(stat_path, "w") as f:
        yaml.dump(out_stat_dict, f)
    print(f"{out_stat_dict['avg'] = }, {out_stat_dict['q050'] = },{out_stat_dict['q075'] = },{out_stat_dict['q095'] = },{out_stat_dict['q100'] = }")
    print(f"{stat_path = }")


def split_string_by_length(string, length):
    return [
        string[i * length: (i + 1) * length]
        for i in range(math.ceil(len(string) / length))
    ]


def bisection_method(
    func, y_target, s_init=0.1, tolerance=1e-6, max_iterations=100, verbose=False
):
    assert s_init > 0
    assert y_target > 0
    f_s = func(s_init)
    if f_s > y_target:
        s, e = 0, s_init
    elif f_s == y_target:
        return s
    else:
        # Finding the second interval endpoint b automatically for a monotonically increasing function
        e = s_init
        f_e = func(e)
        iteration = 0
        x_of_max_y = 0
        max_y = 0
        max_n_faliures = 5
        n_faliure = 0
        while (
            f_e < y_target and n_faliure < max_n_faliures and iteration < max_iterations
        ):
            s = e
            e *= 2  # Doubling b until func(e) > target
            f_e = func(e)
            if f_e > max_y:
                n_faliure = 0
                max_y = f_e
                x_of_max_y = e
            else:
                n_faliure += 1
            iteration += 1
            if verbose:
                print(f"{iteration = }, {e = }, {f_e = }, {n_faliure = }")
        if func(e) < y_target:  # faile to find func(x) > target
            return x_of_max_y

    iteration = 0
    m_prev = e
    m = (s + e) / 2
    f_m = func(m)
    if f_m == y_target:
        return m  # Found exact root
    if f_m > y_target:
        e = m
    else:
        s = m
    while (
        f_m > y_target
        or (f_m < y_target * (1 - tolerance) and (abs(m - m_prev) > 1e-6))
    ) and iteration < max_iterations:
        m_prev = m
        m = (s + e) / 2
        f_m = func(m)
        if f_m == y_target:
            return m  # Found exact root
        if f_m > y_target:
            e = m
        else:
            s = m
        if verbose:
            print(
                f"{iteration = }, {y_target = }, {f_m = }, {m = :.6f}, {m_prev = :.6f}"
            )
        iteration += 1

    return m


def get_rel_path_for_symlink(src_path, target_path):
    target_path = os.path.dirname(target_path)
    common_path = os.path.commonpath([src_path, target_path])
    src_remain = src_path[len(common_path):].strip(os.path.sep)
    target_remain = target_path[len(common_path):].strip(os.path.sep)
    # os.path.split(target_remain)
    if len(target_remain) == 0:
        back_pathes = []
    else:
        back_pathes = [".."] * len(target_remain.split(os.path.sep))
    forward_pathes = src_remain.split(os.path.sep)
    rel_path = os.path.join(*back_pathes, *forward_pathes)
    # print(f"{common_path = }, {src_remain = }, {target_remain = }")
    # print(f"{back_pathes = }, {forward_pathes = }")
    # print(f"{common_path = }, {src_remain = }, {target_remain = }, {rel_path = }")
    return rel_path


def save_rel_symlink(src_path, target_path):
    rel_path = get_rel_path_for_symlink(src_path, target_path)
    if os.path.islink(target_path):
        os.remove(target_path)
    else:
        assert not os.path.exists(target_path)
    os.symlink(rel_path, target_path)


def add_prefix_augmented(train_data, qc_dict, last_flip=False):
    queries = set()
    output = []
    for query, card in tqdm(train_data):
        query_cano = canonicalize_like_query(query, last_flip)
        for query_len in range(1, len(query) + 1):
            sub_query = query_cano[:query_len]
            sub_query = canonicalize_like_query(sub_query)
            # sub_query = query[:query_len]
            # cards.append(card)
            if sub_query not in queries:
                queries.add(sub_query)
                card = qc_dict[sub_query]
                output.append([sub_query, card])
    return output


# endregion

# region param search


def make_partial(func, target_param, **param_inits):
    if target_param in param_inits:
        del param_inits[target_param]

    partial_func = partial(func, **param_inits)
    # print(f"{func = }")
    # print(f"{param_inits = }")
    # print(f"{partial_func = }")

    return lambda x: partial_func(**{target_param: x})


def param_search_general(target_value, func, target_param, s_init=0.1, **param_inits):
    partial_func = make_partial(func, target_param, **param_inits)
    # print(partial_func)
    # print(partial_func(1.0))

    x = bisection_method(
        partial_func, target_value, tolerance=1e-2, verbose=False, s_init=s_init
    )
    return x


# def param_search_table(target_value, target_param, **param_inits):
#     pass


def param_search_table_max_entry(dname, target_value, N):
    # target_value: the size of table
    # target_param: ME

    # param_inits = {"dname": dname, "N": N, "cached_out": False}

    def table_size(max_entry_ratio):
        hashtable, PTs, build_time = getCachedPrunedExtendedNgramTableMaxEntry(
            dname, N, max_entry_ratio, cached_out=False, verbose=False
        )
        return get_pickle_size([hashtable, PTs])

    max_entry_ratio = param_search_general(
        target_value, table_size, "max_entry_ratio")

    return max_entry_ratio


def get_param_setting_dict(search_dict):
    df = pd.read_csv("configs/comp_df_params.csv", index_col=0)
    searched_idx = None
    for k, v in search_dict.items():
        if searched_idx is None:
            searched_idx = (df[k] == v)
        else:
            searched_idx &= (df[k] == v)
    row = df[searched_idx]
    # print(search_dict)
    # print(row)
    assert len(row) <= 1
    if len(row) == 1:
        row = row.iloc[0]
        return row
    else:
        return None


def set_params_setting_dict(args, param_setting_dict):
    args = vars(args)
    for i in range(1, 3):
        param = param_setting_dict['model_param_name_'+str(i)]
        val = param_setting_dict['model_param_value_'+str(i)]
        if isinstance(val, float):
            val = float(f"{val:.3f}")
        if isinstance(val, np.int_):
            val = int(val)
        # print(f"{param, val, type(param), type(val) = }")

        # print(args, param, val)
        if isinstance(param, str):
            assert isinstance(param, str), param
            param = param.replace('-', '_')
            args[param] = val
    args = argparse.Namespace(**args)
    return args


def add_just_params(args):
    if 'ratio' in args and args.ratio != 1:
        ratio = args.ratio
        args.model_name += f'-{int(ratio*100):03d}'
    else:
        ratio = 1.0

    search_dict = {
        'data': args.data_name,
        'workload': args.workload,
        'model': args.model_name,
        'ratio': ratio
    }

    param_setting_dict = get_param_setting_dict(search_dict)
    if param_setting_dict is None:
        return args
    else:
        args = set_params_setting_dict(args, param_setting_dict)
        return args
# endregion

# region args


def get_parser(default_configs: dict, abbr_dict, choices_dict):
    parser = argparse.ArgumentParser()
    for key, val in default_configs.items():
        parser_key = key.replace("_", "-")
        keyword = "--" + parser_key
        choices = choices_dict[key] if key in choices_dict else None

        keywords = [keyword]
        if key in abbr_dict:
            abbr = "-" + abbr_dict[key]
            keywords.append(abbr)

        if type(val) == bool:
            parser.add_argument(keyword, action="store_true")
            no_keyword = keyword.replace("--", "--no-")
            parser.add_argument(no_keyword, dest=key, action="store_false")
            default_bool_dict = {key: val}
            parser.set_defaults(**default_bool_dict)
        else:
            parser.add_argument(*keywords, default=val,
                                type=type(val), choices=choices)
    return parser


def check_args(args, ignore_list=None):
    args_dict = vars(args)
    if ignore_list is None:
        ignore_list = []
    for key, val in args_dict.items():
        key = f"--{key}".replace("_", "-")
        if key not in ignore_list:
            assert val is not None, f"{key} is not given"


def args2exp_key(args, ignore_list=None, model_name=None):
    args_dict = vars(args)
    exp_key = ""
    # ignore_list = [opt.replace('--', '').replace('-', '_') for opt in ignore_list]
    if ignore_list is None:
        ignore_list = []

    for key in sorted(args_dict.keys()):
        val = args_dict[key]
        key = f"--{key}".replace("_", "-")
        if key not in ignore_list:
            key = "".join([tkn[0].upper() + tkn[1:] for tkn in key.split("_")])
            key = key[0].lower() + key[1:]
            exp_key += f" {key} {val}"
    exp_key = exp_key[1:]
    if model_name is not None:
        exp_key = model_name + " " + exp_key

    return exp_key


def args2exp_name(args, ignore_list=None, model_name=None):
    args_dict = vars(args)
    exp_key = ""
    if ignore_list is None:
        ignore_list = []
    ignore_list = [opt.replace("--", "").replace("-", "_")
                   for opt in ignore_list]

    for key in sorted(args_dict.keys()):
        if key not in ignore_list:
            val = args_dict[key]
            key = "".join([tkn[0].upper() + tkn[1:] for tkn in key.split("_")])
            key = key[0].lower() + key[1:]
            exp_key += f"_{key}_{val}"
    exp_key = exp_key[1:]
    if model_name is not None:
        exp_key = model_name + "_" + exp_key

    return exp_key


def get_parser_with_ignore_opt_list_alg():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    all_subparsers = []
    ignore_opt_list = []

    psql_parser = subparsers.add_parser("psql")
    all_subparsers.append(psql_parser)

    regex_parser = subparsers.add_parser("regex")
    all_subparsers.append(regex_parser)

    inv_parser = subparsers.add_parser("inv")
    all_subparsers.append(inv_parser)

    pinv_parser = subparsers.add_parser("pinv")
    all_subparsers.append(pinv_parser)

    # common parameters
    for subparser in all_subparsers:
        subparser: argparse._ActionsContainer = subparser
        subparser.add_argument("-d", "--dname", type=str, help="data name")
        # subparser.add_argument("-mp", "--max-n-pat", type=str, help="maximum patterns in like queries")
        # subparser.add_argument("-mu", "--max-n-under", type=int, default=-1,
        #                        help="maximum underscores in each operators")
        # subparser.add_argument("-pt", "--pack-type", type=str, help="packed learning type")
        subparser.add_argument("-q", "--query-key", type=str, help="query key")
        subparser.add_argument("-t", "--trial", type=int, help="trial id")

        ow_option = "--overwrite"
        ignore_opt_list.append(ow_option)
        subparser.add_argument(
            "-o",
            ow_option,
            action="store_true",
            help="run this code even if it has already been done",
        )

    if len(sys.argv) <= 1:
        parser.print_help()
        exit()
    model_name = sys.argv[1]
    return parser, ignore_opt_list, model_name


def get_parser_with_ignore_opt_list_model():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    all_subparsers = []
    ignore_opt_list = []

    astrid_parser = subparsers.add_parser("astrid")
    all_subparsers.append(astrid_parser)

    clique_parser = subparsers.add_parser("clique")
    all_subparsers.append(clique_parser)
    clique_parser.add_argument(
        "--prfx", default=False, type=bool, help="seperate model"
    )
    clique_parser.add_argument(
        "--n-rnn", type=int, help="number of encoder layers")

    # common parameters
    for subparser in all_subparsers:
        subparser: argparse._ActionsContainer = subparser
        subparser.add_argument("--dname", type=str, help="data name")
        subparser.add_argument("--p-train", type=float,
                               help="ratio of train data")
        subparser.add_argument("--p-val", type=float,
                               help="ratio of valid data")
        subparser.add_argument("--p-test", type=float,
                               help="ratio of test data")
        subparser.add_argument("--bs", type=int, help="batch size")
        subparser.add_argument("--lr", type=float, help="learning rate")
        subparser.add_argument("--l2", type=float, help="l2 regularizer")
        subparser.add_argument("--clip-gr", type=float,
                               help="cliping gradient")
        subparser.add_argument("--seed", type=int, help="random seed")
        subparser.add_argument(
            "--pred-layer", type=int, help="number of predict layers"
        )
        subparser.add_argument(
            "--pred-hs", type=int, help="hidden size of predict layers"
        )
        subparser.add_argument(
            "--cs", type=int, help="cell size of encode layers")
        subparser.add_argument("--ch-es", type=int,
                               help="character embedding size")
        subparser.add_argument("--max-epoch", type=int, help="maximum epoch")
        subparser.add_argument("--max-l", type=int,
                               help="maximum length of query")
        subparser.add_argument(
            "--repeat",
            type=str,
            help="repeatition of sampling queries from each record",
        )
        subparser.add_argument(
            "--patience", type=int, help="patience for training a neural network"
        )
        subparser.add_argument("--pattern", type=str,
                               help="like query patterns")

        ow_option = "--overwrite"
        ignore_opt_list.append(ow_option)
        subparser.add_argument(
            ow_option,
            action="store_true",
            help="run this code even if it has already been done",
        )

    if len(sys.argv) <= 1:
        parser.print_help()
        exit()
    model_name = sys.argv[1]
    return parser, ignore_opt_list, model_name


def parse_all_model():
    parser, ignore_opt_list, model_name = get_parser_with_ignore_opt_list_model()
    args = parser.parse_args()
    check_args(args, ignore_opt_list)

    exp_name = args2exp_name(args, ignore_opt_list, model_name)
    if len(exp_name) > 255:
        warnings.warn(f"exp_name is too long. [{len(exp_name)}:{exp_name}]")
    exp_key = args2exp_key(args, ignore_opt_list, model_name)

    return model_name, args, exp_name, exp_key


def parse_all_alg():
    parser, ignore_opt_list, alg_name = get_parser_with_ignore_opt_list_alg()
    args = parser.parse_args()
    check_args(args, ignore_opt_list)

    exp_name = args2exp_name(args, ignore_opt_list, alg_name)
    if len(exp_name) > 255:
        warnings.warn(f"exp_name is too long. [{len(exp_name)}:{exp_name}]")
    exp_key = args2exp_key(args, ignore_opt_list, alg_name)

    return alg_name, args, exp_name, exp_key


def get_common_abbr_dict():
    abbr_dict = {
        "data_name": "d",
        "workload": "w",
    }
    return abbr_dict


def get_common_choices_dict():
    choices_dict = {
        "data_name": ["TOY", "DBLP", "IMDB", "WIKI", "GENE", "AUTHOR",
                      "DBLP-AN", "IMDb-AN", "IMDb-MT", "TPCH-PN",
                      "link_type.link", "movie_companies.note",
                      "name.name", "title.title",
                      ],
    }
    return choices_dict


def get_seed():
    return 0


def get_common_pathes_for_estimator(data_name, model, workload, trial):
    db_path = f"data/{data_name}/{data_name}.txt"
    model_config_path = os.path.join(
        f"res/{data_name}/{workload}/model/{model}/{trial}", f"config.yml"
    )
    model_path = os.path.join(
        f"res/{data_name}/{workload}/model/{model}/{trial}", f"model.pkl")
    est_path = os.path.join(
        f"res/{data_name}/{workload}/estimation/{model}/{trial}", f"estimation.csv")
    stat_path = os.path.join(
        f"res/{data_name}/{workload}/stat/{model}/{trial}", f"stat.yml")
    est_time_path = os.path.join(
        f"res/{data_name}/{workload}/estimation/{model}/{trial}", f"time.csv"
    )

    if "EST" in model:
        model_path = model_path.replace("pkl", "txt")
    elif "Astrid" in model or "E2E" in model or "DREAM" in model or "LPLM" in model or "CLIQUE" in model:
        model_path = model_path.replace("pkl", "pth")
    elif "LBS" in model:
        pass
    else:
        print(model)
        raise ValueError("check model extension")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(est_path), exist_ok=True)
    os.makedirs(os.path.dirname(stat_path), exist_ok=True)

    print(f"{db_path = }")
    print(f"{model_config_path = }")
    print(f"{model_path = }")
    print(f"{est_path = }")
    print(f"{est_time_path = }")
    print(f"{stat_path = }")

    return db_path, model_config_path, model_path, est_path, est_time_path, stat_path


def get_count_path_for_sampling_estimator(dname, model):
    save_count_path = os.path.join(
        f"res/{dname}/estimation/{model}", f"count.csv"
    )
    os.makedirs(os.path.dirname(save_count_path), exist_ok=True)
    print(f"{save_count_path = }")
    return save_count_path


def get_triplet_dirpath_for_Astrid(dname, max_str_size=None):
    triplet_dirpath = os.path.join(f"res/{dname}/triplets")
    if max_str_size is not None:
        triplet_dirpath += f'/ms{max_str_size}'
    os.makedirs(triplet_dirpath, exist_ok=True)
    print(f"{triplet_dirpath = }")
    return triplet_dirpath


def get_embedding_model_dirpath_for_Astrid(dname, emb_dim=None):
    embedding_dirpath = os.path.join(
        f"res/{dname}/model/triplets"
    )
    if emb_dim is not None:
        embedding_dirpath += f"/e{emb_dim}"
    os.makedirs(embedding_dirpath, exist_ok=True)
    print(f"{embedding_dirpath = }")
    return embedding_dirpath


def get_word_vector_path_for_E2Eestimator(dname, min_count, tag):
    wv_path = os.path.join(
        f"res/{dname}/model/{tag}", f"wordvectors_mc_{min_count}.kv"
    )
    os.makedirs(os.path.dirname(wv_path), exist_ok=True)
    print(f"{wv_path = }")
    return wv_path


def get_count_path_for_HybridEstimator(
    dname, PT, N, dynamicPT, max_entry_ratio, makedir=True
):
    cache_root = os.getenv("EFFICIENCY_CACHE_ROOT")
    dataset_root = (
        os.path.join(cache_root, dname)
        if cache_root
        else os.path.join("res", dname)
    )
    if dynamicPT:
        assert max_entry_ratio is not None
        count_path = os.path.join(
            dataset_root, "Ntable", f"count_N{N}_ME{max_entry_ratio}.pkl"
        )
    else:
        count_path = os.path.join(
            dataset_root, "Ntable", f"count_N{N}_PT{PT}.pkl")
    if makedir:
        os.makedirs(os.path.dirname(count_path), exist_ok=True)
        print(f"{count_path = }")
    return count_path


def get_summary_writer(dname, model, workload, seed):
    sw_path = os.path.join(f"res/{dname}/{workload}/log/{model}/{seed}")
    print(f"{sw_path = }")
    sw = SummaryWriter(sw_path)
    return sw


def get_sampling_model_name(is_adapt=False, is_greek=False):
    if is_adapt:
        assert is_greek
        model_name = f"EST_B"
    else:
        if is_greek:
            model_name = f"EST_M"
        else:
            model_name = f"EST_S"
    return model_name


def save_estimated_cards(test_queries, test_cards, test_estimations, est_path):
    os.makedirs(os.path.dirname(est_path), exist_ok=True)
    header = ["query", "true", "est", "q-error"]
    q_errs = mean_Q_error(test_cards, test_estimations, reduction="none")
    data = list(zip(test_queries, test_cards, test_estimations, q_errs))

    df = pd.DataFrame(data, columns=header)
    print(f"{est_path = }")
    df.to_csv(est_path, index=False)


def save_estimation_times(
    test_queries, test_cards, test_estimations, test_estimation_times, est_time_path
):
    os.makedirs(os.path.dirname(est_time_path), exist_ok=True)
    header = ["query", "true", "est", "q-error", "est_time"]
    q_errs = mean_Q_error(test_cards, test_estimations, reduction="none")
    data = list(
        zip(test_queries, test_cards, test_estimations,
            q_errs, test_estimation_times)
    )

    df = pd.DataFrame(data, columns=header)
    df.to_csv(est_time_path, index=False)


def save_count_infos(
    test_queries, test_cards, test_estimations, count_info, save_count_path
):
    os.makedirs(os.path.dirname(save_count_path), exist_ok=True)
    q_errors = mean_Q_error(test_cards, test_estimations, reduction="none")

    stats_count = [
        [
            test_queries[i],
            test_cards[i],
            test_estimations[i],
            q_errors[i],
            count_info[i][0],
            count_info[i][1],
            count_info[i][2],
        ]
        for i in range(len(test_queries))
    ]

    os.makedirs(os.path.dirname(save_count_path), exist_ok=True)
    df = pd.DataFrame(
        stats_count, columns=["query", "true", "est", "q-error", "n", "m", "k"]
    )
    df.to_csv(save_count_path)


# endregion


# region torch


def find_k_index_in_encoded_strings(strings, value, k):
    """_summary_

    Parameters
    ----------
    strings : torch.LongTensor (2D)
        encoded strings
    value : int
        a value to find
    k : int
        k-th position from starting position

    Returns
    -------
    torch.Longtensor (1D)
        k-th index whose entry is equal to the value
    Examples
    --------
    >>> X = [[2, 3, 4, 2, 5, 2], [2, 4, 2, 5, 2, 0]]  # ["%ab%c%", "%b%c%"]
    >>> X = torch.IntTensor(X)
    >>> find_k_index_in_encoded_strings(X, 2, 2)
    >>> tensor([3, 2], dtype=torch.int32)
    """
    return (strings == value).nonzero(as_tuple=True)[1].reshape(-1, 3)[:, k - 1].cpu()


def set_seed(seed):
    """set seed for numpy and torch

    Parameters
    ----------
    seed : int or None
        seed for randomness
    """
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


# endregion


# region Ngram table
def count_token(engram_dict, token):
    if token not in engram_dict:
        engram_dict[token] = 0
    engram_dict[token] += 1


def filter_infrequent_entry_PT(engram_dict, PT):
    output = {}
    for k, v in engram_dict.items():
        if v >= PT:
            output[k] = v
    return output


def generator_replaced_string(string):
    """
        This function considers replacement operation only.
    :param string:
    :return:
    """

    # N = len(string)
    # total = 2 ** N
    # i = 0
    # while i < total:
    #
    #     i += 1
    global esc_ac

    def idx2onoff(x):
        output = []
        curr = x
        for _ in range(n):
            output.append(curr % 2 == 1)
            curr //= 2
        return reversed(output)

    # assert esc_ac not in string
    # if "?" in string:
    #     string = string.replace("?", esc_ac)
    # assert "?" not in string
    n = len(string)
    for idx in range(2**n):
        output = []
        onoff = idx2onoff(idx)
        for i, is_change in enumerate(onoff):
            if is_change:
                # output += '?'
                output.append(esc_ac)
            else:
                # output += string[i]
                output.append(string[i])
        yield "".join(output)


def extendedNgramTable(db, N, PT):
    hashtable = {}
    hashtable[""] = len(db)  # length 0

    for rid, record in tqdm(enumerate(db), total=len(db)):
        if len(record) <= N:
            for sub_token in generator_replaced_string(record):
                count_token(hashtable, esc_sc + sub_token + esc_ec)
        length = len(record)
        for l in range(1, min(N, length) + 1):
            distinct_tokens = set()
            for s in range(length - l + 1):
                token = record[s: s + l]

                for sub_token in generator_replaced_string(token):
                    distinct_tokens.add(sub_token)
                    if s == 0:  # prefix
                        count_token(hashtable, esc_sc + sub_token)
                    if s + l == len(record):  # suffix
                        count_token(hashtable, sub_token + esc_ec)
            for token in distinct_tokens:
                count_token(hashtable, token)

    if PT > 1:
        hashtable = filter_infrequent_entry_PT(hashtable, PT)

    return hashtable


def getCachedExtendedNgramTable(output_path, db=None, N=None, verbose=True):
    time_path = output_path.replace(".pkl", ".yml")
    if verbose:
        print(f"{time_path = }")

    if not os.path.exists(output_path) or not os.path.exists(time_path):
        start_time = time.time()
        hashtable = extendedNgramTable(db, N, 1)
        end_time = time.time()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(hashtable, f)
        build_time = end_time - start_time

        out_dict = {"time": build_time}
        with open(time_path, "w") as f:
            yaml.safe_dump(out_dict, f)

    with open(output_path, "rb") as f:
        hashtable = pickle.load(f)
    with open(time_path) as f:
        out_dict = yaml.safe_load(f)
        build_time = out_dict["time"]
    return hashtable, build_time


def extendedNgramTable_per_len(engram_dict):
    engram_dict_per_len = {}

    for key, val in engram_dict.items():
        token = key.lstrip(esc_sc).rstrip(esc_ec)
        token_len = len(token)
        if token_len not in engram_dict_per_len:
            engram_dict_per_len[token_len] = {}
        # print(f"{key = }, {val = }, {token = }")
        engram_dict_per_len[len(token)][key] = val
    return engram_dict_per_len


def filter_infrequent_entry_max_entry(engram_dict, max_entry):
    sorted_counts = list(
        sorted(engram_dict.items(), key=lambda x: x[1], reverse=True))
    if max_entry < len(sorted_counts):
        # element next to the max_entry-th element
        PT = sorted_counts[max_entry][1] + 1
    else:
        PT = 1
    pruned_dict = filter_infrequent_entry_PT(engram_dict, PT)
    # print(f"{len(pruned_dict) = }, {max_entry = }, {PT = }")
    return pruned_dict, PT


def filter_infrequent_entry_max_entry_each_len(engram_dict, max_entry_ratio):
    hashtable_per_len = extendedNgramTable_per_len(engram_dict)
    if 2 in hashtable_per_len:
        default_entry = len(hashtable_per_len[2])
    else:
        default_entry = len(hashtable_per_len[1])
    max_entry = int(max_entry_ratio * default_entry)
    # print(f"{max_entry = }")

    output_dict = {}
    PTs = []
    for length, each_dict in hashtable_per_len.items():
        each_dict, PT = filter_infrequent_entry_max_entry(each_dict, max_entry)
        for k, v in each_dict.items():
            output_dict[k] = v

        PTs.append(PT)
    return output_dict, PTs


full_table_cache = {}
db_cache = {}


def getCachedPrunedExtendedNgramTableMaxEntry(
    dname, N, max_entry_ratio=None, cached_out=True, verbose=True
):
    global db_cache
    global full_table_cache
    if dname not in db_cache:
        db_path = f"data/{dname}/{dname}.txt"
        db_cache[dname] = read_strings(db_path)
    db = db_cache[dname]

    output_path = get_count_path_for_HybridEstimator(
        dname,
        PT=None,
        N=N,
        dynamicPT=True,
        max_entry_ratio=max_entry_ratio,
        makedir=False,
    )
    assert f"ME{max_entry_ratio}" in output_path, output_path

    if dname not in full_table_cache:
        full_table_path = output_path.replace(f"ME{max_entry_ratio}", "PT1")
        full_table_cache[dname] = getCachedExtendedNgramTable(
            full_table_path, db=db, N=N, verbose=verbose
        )

    full_table, full_build_time = full_table_cache[dname]

    time_path = output_path.replace(".pkl", ".yml")
    if verbose:
        print(f"{time_path = }")

    is_saved = os.path.exists(output_path) and os.path.exists(time_path)

    if is_saved:
        with open(output_path, "rb") as f:
            hashtable, PTs = pickle.load(f)
        with open(time_path) as f:
            out_dict = yaml.safe_load(f)
            build_time = out_dict["time"]
        is_saved = (PTs[1] != 2)

    if not is_saved:
        start_time = time.time()
        hashtable, PTs = filter_infrequent_entry_max_entry_each_len(
            full_table, max_entry_ratio
        )

        end_time = time.time()
        build_time = full_build_time + (end_time - start_time)

        if cached_out:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump([hashtable, PTs], f)

            out_dict = {"time": build_time}
            with open(time_path, "w") as f:
                yaml.safe_dump(out_dict, f)

    return hashtable, PTs, build_time


def getCachedPrunedExtendedNgramTable(
    data_name, N=None, PT=None, cached_out=True, verbose=True
):
    global db_cache
    global full_table_cache

    if data_name not in db_cache:
        db_path = f"data/{data_name}/{data_name}.txt"
        db_cache = read_strings(db_path)
    db = db_cache

    output_path = get_count_path_for_HybridEstimator(
        data_name,
        PT=PT,
        N=N,
        dynamicPT=False,
        max_entry_ratio=None,
        makedir=False,
    )

    assert f"PT{PT}" in output_path, output_path

    if data_name not in full_table_cache:
        full_table_path = output_path.replace(f"PT{PT}", "PT1")

        full_table_cache[data_name] = getCachedExtendedNgramTable(
            full_table_path, db=db, N=N, verbose=verbose
        )

    full_table, full_build_time = full_table_cache[data_name]

    time_path = output_path.replace(".pkl", ".yml")

    if verbose:
        print(f"{time_path = }")

    is_saved = os.path.exists(output_path) and os.path.exists(time_path)

    if PT > 1 and not is_saved:
        start_time = time.time()
        hashtable = filter_infrequent_entry_PT(full_table, PT)
        end_time = time.time()
        build_time = full_build_time + (end_time - start_time)

        if cached_out:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(hashtable, f)
            out_dict = {"time": build_time}
            with open(time_path, "w") as f:
                yaml.safe_dump(out_dict, f)

    if is_saved:
        with open(output_path, "rb") as f:
            hashtable = pickle.load(f)
        with open(time_path) as f:
            out_dict = yaml.safe_load(f)
            build_time = out_dict["time"]
    return hashtable, build_time


# endregion

"""
新增  —  模式链生成工具函数

提供：
  1. degrade_like_pattern()     单步退化操作
  2. generate_degradation_chain()  从一个 CLIQUE 格式查询生成完整退化链
  3. sample_anchors()           链内锚点采样
  4. ChainBatchGenerator        chain-aware batch 生成器（训练用）
  5. hinge_monotonicity_loss()  单调性 hinge loss

使用方式：将本文件内容追加到 src/util.py，或在需要的地方 import。

依赖：仅依赖 src/util.py 中已有的函数：
  - canonicalize_like_query
  - parse_like_query
  - is_wild_string
"""

import numpy as np
import random
import json
import os
import torch
from copy import deepcopy


# =====================================================
#  1. 单步退化操作
# =====================================================

def _get_token_segments(query_cano):
    """
    将规范化后的 LIKE 查询拆分为 (wildcards, tokens) 交替序列。

    例如 '%John%Smith%' -> (['%', '%', '%'], ['John', 'Smith'])
         'John%Smith%'  -> (['', '%', '%'],  ['John', 'Smith'])
         '%John%Smith'  -> (['%', '%', ''],  ['John', 'Smith'])

    返回:
        segments: list of str, 按原始顺序的所有段（通配+文字交替）
        token_indices: list of int, segments 中文字段的下标
    """
    segments = []
    token_indices = []
    current = ""
    # Only '%' is a segment separator. '_' stays within tokens.
    is_pct = (query_cano[0] == "%")

    for ch in query_cano:
        if ch == "%":
            if not is_pct and current:
                # We were in a token, save it
                token_indices.append(len(segments))
                segments.append(current)
                current = ""
                is_pct = True
            current += ch
        else:
            # ch is a literal or '_', both belong to tokens
            if is_pct and current:
                segments.append(current)
                current = ""
                is_pct = False
            current += ch

    if current:
        if is_pct:
            segments.append(current)
        else:
            token_indices.append(len(segments))
            segments.append(current)

    return segments, token_indices


def _rebuild_query(segments):
    """从 segments 重建查询字符串，合并相邻的 % 和清理。"""
    raw = "".join(segments)
    # 合并连续 %
    while "%%" in raw:
        raw = raw.replace("%%", "%")
    return raw


def degrade_like_pattern(pattern, rng=None, degrade_char_prob=0.4,
                         degrade_shrink_prob=0.5,
                         degrade_under_to_pct_prob=0.0,
                         degrade_insert_pct_prob=0.1,
                         max_wildcard_ratio=0.2):
    """
    对一个 CLIQUE 格式的 LIKE 查询做一步退化，返回更宽松的模式。

    退化操作（按概率选择）：
      1. 字符→'_' 替换 (degrade_char_prob)
         选一个文字字符替换为 '_'
      2. token 边缘收缩 (degrade_shrink_prob)
         从某个 token 的边缘删除一个字符（使相邻的 '%' 吞掉它）
      3. token 内部插入 '%' (degrade_insert_pct_prob)
         将一个 literal token 切成左右两段，放松连续匹配要求
      4. '_' 升级为 '%' (degrade_under_to_pct_prob，默认关闭)
         保留兼容旧实验参数，一般不建议启用

    参数:
        pattern: str, 如 '%John%Smith%'
        rng: random.Random 实例（可选）
        degrade_*_prob: 各操作的相对概率
        max_wildcard_ratio: 通配符('%' + '_')数量 / 查询长度 的上限；<=0 表示不限制

    返回:
        str: 退化后的模式，如果无法退化则返回 '%'
        bool: 是否成功退化（False 表示已经到达 '%'）
    """
    if rng is None:
        rng = random.Random()

    if pattern == "%":
        return "%", False

    segments, token_indices = _get_token_segments(pattern)

    if not token_indices:
        return "%", False

    def wildcard_ratio(query):
        return sum(1 for ch in query if ch in "%_") / max(len(query), 1)

    ratio_limited = (
        max_wildcard_ratio > 0
        and wildcard_ratio(pattern) >= max_wildcard_ratio
    )

    # 收集可用操作
    ops = []

    # Op1: 字符 → '_' 替换
    replaceable = []  # (seg_idx, char_pos_in_seg)
    if not ratio_limited:
        for ti in token_indices:
            token = segments[ti]
            for ci, ch in enumerate(token):
                if ch not in "%_":
                    candidate_segments = [s for s in segments]
                    candidate_token = list(candidate_segments[ti])
                    candidate_token[ci] = "_"
                    candidate_segments[ti] = "".join(candidate_token)
                    candidate = _rebuild_query(candidate_segments)
                    if max_wildcard_ratio <= 0 or wildcard_ratio(candidate) <= max_wildcard_ratio:
                        replaceable.append((ti, ci))
    if replaceable and degrade_char_prob > 0 and not ratio_limited:
        ops.append(("replace", degrade_char_prob))

    # Op2: token 边缘收缩（删一个边缘字符，让相邻 '%' 吞掉它）
    # 关键：只能从靠近 '%' 的方向收缩，否则会破坏蕴含关系
    # - 如果 token 左边有 '%'，可以从左边删（相当于 %abc -> %bc，'%' 吞了 'a'）
    # - 如果 token 右边有 '%'，可以从右边删（相当于 abc% -> ab%，'%' 吞了 'c'）
    # - 如果 token 左边没有 '%'（前缀首 token），不能从左边删
    # - 如果 token 右边没有 '%'（后缀尾 token），不能从右边删
    shrinkable = []  # (seg_idx, 'left'|'right', char_pos)
    for ti in token_indices:
        token = segments[ti]
        literal_positions = [i for i, ch in enumerate(token) if ch not in "%_"]
        if not literal_positions:
            continue

        # 检查 token 左边是否有 '%' 段
        has_pct_left = (ti > 0 and "%" in segments[ti - 1])
        # 检查 token 右边是否有 '%' 段
        has_pct_right = (ti < len(segments) - 1 and "%" in segments[ti + 1])

        if has_pct_left:
            # 可以从左边删（让左边的 % 吞掉第一个字符）
            shrinkable.append((ti, "left", literal_positions[0]))
        if has_pct_right:
            # 可以从右边删（让右边的 % 吞掉最后一个字符）
            shrinkable.append((ti, "right", literal_positions[-1]))
    if shrinkable and degrade_shrink_prob > 0:
        ops.append(("shrink", degrade_shrink_prob))

    # Op3: token 内部插入 '%'
    # 不贴着已有 '%' 插入；碎片化程度由 wildcard / pattern length 比例控制。
    insertable = []  # (seg_idx, insert_pos)
    if degrade_insert_pct_prob > 0 and not ratio_limited:
        for ti in token_indices:
            token = segments[ti]
            for pos in range(1, len(token)):
                # Since '%' is a segment separator, this ensures the inserted
                # wildcard is not adjacent to an existing '%'.
                candidate = _rebuild_query(
                    segments[:ti] + [token[:pos] + "%" + token[pos:]] + segments[ti + 1:]
                )
                if max_wildcard_ratio <= 0 or wildcard_ratio(candidate) <= max_wildcard_ratio:
                    insertable.append((ti, pos))
    if insertable:
        ops.append(("insert_pct", degrade_insert_pct_prob))

    # Op4: '_' → '%' 升级（默认关闭，仅用于兼容旧实验）
    upgradeable = []  # (seg_idx_in_segments, char_pos)
    if degrade_under_to_pct_prob > 0 and not ratio_limited:
        for si, seg in enumerate(segments):
            for ci, ch in enumerate(seg):
                if ch == "_":
                    upgradeable.append((si, ci))
    if upgradeable and degrade_under_to_pct_prob > 0:
        ops.append(("upgrade", degrade_under_to_pct_prob))

    if not ops:
        return "%", False

    # 按概率选择操作
    op_names = [o[0] for o in ops]
    op_weights = [o[1] for o in ops]
    chosen_op = rng.choices(op_names, weights=op_weights, k=1)[0]

    new_segments = [s for s in segments]  # shallow copy

    if chosen_op == "replace":
        seg_idx, char_pos = rng.choice(replaceable)
        token = list(new_segments[seg_idx])
        token[char_pos] = "_"
        new_segments[seg_idx] = "".join(token)

    elif chosen_op == "shrink":
        seg_idx, direction, char_pos = rng.choice(shrinkable)
        token = list(new_segments[seg_idx])
        del token[char_pos]
        new_token = "".join(token)
        if not new_token or all(ch in "%_" for ch in new_token):
            # token 退化为纯通配或空——合并为 '%'
            new_segments[seg_idx] = "%"
        else:
            new_segments[seg_idx] = new_token

    elif chosen_op == "insert_pct":
        seg_idx, insert_pos = rng.choice(insertable)
        token = new_segments[seg_idx]
        new_segments[seg_idx] = token[:insert_pos] + "%" + token[insert_pos:]

    elif chosen_op == "upgrade":
        seg_idx, char_pos = rng.choice(upgradeable)
        seg = list(new_segments[seg_idx])
        seg[char_pos] = "%"
        new_segments[seg_idx] = "".join(seg)

    result = _rebuild_query(new_segments)

    # 确保结果不和输入相同
    if result == pattern:
        return degrade_like_pattern(pattern, rng, degrade_char_prob,
                                    degrade_shrink_prob,
                                    degrade_under_to_pct_prob,
                                    degrade_insert_pct_prob,
                                    max_wildcard_ratio)

    if not result or result.replace("%", "").replace("_", "") == "":
        return "%", True

    return result, True


# =====================================================
#  2. 退化链生成
# =====================================================

def _get_literal_length(pattern):
    """计算模式中非通配符字符的数量。"""
    return sum(1 for ch in pattern if ch not in '%_')


def generate_degradation_chain(pattern, max_chain_length=20, seed=None,
                               degrade_char_prob=0.4,
                               degrade_shrink_prob=0.5,
                               degrade_under_to_pct_prob=0.0,
                               degrade_insert_pct_prob=0.1,
                               max_wildcard_ratio=0.2,
                               min_literal_len=0):
    """
    从一个 CLIQUE 格式的查询出发，生成退化链。

    链从精确（链头）到宽松（链尾），基数严格单调递增。
    当退化模式的 literal 字符数 <= min_literal_len 时停止
    （这些短模式已被 N-gram 表覆盖，无需神经网络处理）。

    参数:
        pattern: str, 链头模式，如 '%John%Smith%'
        max_chain_length: int, 链的最大长度（含链头）
        seed: int, 随机种子
        degrade_*_prob: 退化操作概率
        min_literal_len: int, 停止退化的最小 literal 长度
            设为 0 时不过滤（退化到 '%'）
            设为 5 时（=N-gram 的 N），退化到 literal<=5 就停止

    返回:
        list of str: 模式链，如 ['%John%Smith%', '%Joh%Smith%', ...]
    """
    rng = random.Random(seed)
    chain = [pattern]
    seen = {pattern}
    current = pattern

    for _ in range(max_chain_length - 1):
        if current == "%":
            break

        # 尝试退化，最多重试 20 次避免重复
        success = False
        for _retry in range(20):
            degraded, ok = degrade_like_pattern(
                current, rng,
                degrade_char_prob=degrade_char_prob,
                degrade_shrink_prob=degrade_shrink_prob,
                degrade_under_to_pct_prob=degrade_under_to_pct_prob,
                degrade_insert_pct_prob=degrade_insert_pct_prob,
                max_wildcard_ratio=max_wildcard_ratio,
            )
            if not ok:
                return chain
            if degraded not in seen:
                success = True
                break

        if not success:
            return chain

        # 检查 literal 长度：如果退化后太短就停止
        if min_literal_len > 0 and _get_literal_length(degraded) <= min_literal_len:
            return chain

        seen.add(degraded)
        chain.append(degraded)
        current = degraded

    return chain


# =====================================================
#  3. 锚点采样
# =====================================================

def sample_anchors(chain, anchor_ratio=1.0, seed=None):
    """
    在链内选择哪些模式需要查基数（锚点）。

    规则：
      - 链头（index 0）始终是锚点
      - 非 '%' 的链尾始终是锚点
      - 链尾 '%' 的基数 = 表总行数，标记为已知但不需要查
      - 中间模式按 anchor_ratio 随机采样

    参数:
        chain: list of str, 模式链
        anchor_ratio: float, 0-1，中间模式被选为锚点的比例
        seed: int

    返回:
        list of dict: [{"pattern": str, "is_anchor": bool}, ...]
    """
    rng = random.Random(seed)
    n = len(chain)
    result = []

    for i, pat in enumerate(chain):
        if i == 0 or (i == n - 1 and pat != "%"):
            # 链头、非 '%' 链尾：总是锚点
            result.append({"pattern": pat, "is_anchor": True})
        elif pat == "%":
            # 链尾 '%'：基数已知（= n_db），不需要查
            result.append({"pattern": pat, "is_anchor": False})
        else:
            # 中间模式：按概率采样
            is_anchor = (anchor_ratio >= 1.0) or (rng.random() < anchor_ratio)
            result.append({"pattern": pat, "is_anchor": is_anchor})

    # 确保至少有 2 个锚点（链头 + 至少一个中间点）
    anchors = [r for r in result if r["is_anchor"]]
    if len(anchors) < 2 and n > 2:
        # 强制选中间点
        mid = n // 2
        if not result[mid]["is_anchor"]:
            result[mid]["is_anchor"] = True

    return result


# =====================================================
#  4. 链数据 I/O
# =====================================================

def save_chain_data(chains, save_dir, query_key):
    """
    将链数据保存为 CLIQUE 格式。

    输出文件:
      - {save_dir}/train.txt: 所有锚点的 query,card（card 初始为 -1，等基数计算后回填）
      - {save_dir}/chain_index.json: 链结构索引

    参数:
        chains: list of list of dict
            每条链是 [{"pattern": str, "is_anchor": bool}, ...]
        save_dir: str, 输出目录
        query_key: str, workload 标识
    """
    os.makedirs(save_dir, exist_ok=True)

    # 收集所有需要查基数的 pattern（去重）
    patterns_needing_card = set()
    for chain in chains:
        for item in chain:
            if item["is_anchor"]:
                patterns_needing_card.add(item["pattern"])

    # 保存查询文件（给 calculate_cardinality.py 用）
    query_file = os.path.join(save_dir, "chain_query.txt")
    sorted_patterns = sorted(patterns_needing_card)
    with open(query_file, "w") as f:
        for p in sorted_patterns:
            f.write(p + "\n")

    # 保存链结构索引
    chain_index = []
    for cid, chain in enumerate(chains):
        chain_info = {
            "chain_id": cid,
            "patterns": [item["pattern"] for item in chain],
            "is_anchor": [item["is_anchor"] for item in chain],
        }
        chain_index.append(chain_info)

    chain_index_file = os.path.join(save_dir, "chain_index.json")
    with open(chain_index_file, "w", encoding="utf-8") as f:
        json.dump(chain_index, f, indent=2, ensure_ascii=False)

    print(f"Chain data saved:")
    print(f"  Chains: {len(chains)}")
    print(f"  Patterns needing cardinality: {len(sorted_patterns)}")
    print(f"  Query file: {query_file}")
    print(f"  Chain index: {chain_index_file}")

    return query_file, chain_index_file


def load_chain_index(chain_index_path):
    """加载链结构索引。"""
    with open(chain_index_path, "r", encoding="utf-8") as f:
        chain_index = json.load(f)
    return chain_index


def sample_chain_index_preserve_labeled(chain_index, train_data, ratio, seed):
    """Sample whole chains for monotonic edges without changing labeled rows.

    Selected chains retain every node and adjacent monotonic edge.  Each labeled
    occurrence from an unselected chain is retained as a singleton chain, so it
    still contributes its MSLE term but cannot create a monotonic edge.  This
    makes ``ratio`` control chain supervision rather than silently shrinking the
    exact-label training multiset.
    """
    ratio = float(ratio)
    if not (0 < ratio <= 1.0):
        raise ValueError("chain sample ratio must be in (0, 1]")

    n_chains = len(chain_index)
    keep_n = int(round(n_chains * ratio))
    if n_chains:
        keep_n = max(1, keep_n)
    rng = np.random.RandomState(seed)
    selected = set(
        rng.choice(n_chains, size=keep_n, replace=False).tolist()
        if keep_n else []
    )
    card_by_pattern = {query: card for query, card in train_data}

    effective_chains = []
    labeled_before = 0
    labeled_after = 0
    pairs_before = 0
    pairs_after = 0
    selected_pattern_occurrences = 0
    singleton_labeled_occurrences = 0

    for chain_number, chain in enumerate(chain_index):
        patterns = list(chain.get("patterns", []))
        flags = list(chain.get("is_anchor", [False] * len(patterns)))
        if len(flags) != len(patterns):
            raise ValueError(
                f"chain {chain_number} has mismatched patterns/is_anchor lengths"
            )
        labeled_flags = [
            pattern == "%" or card_by_pattern.get(pattern, -1) >= 0
            for pattern in patterns
        ]
        labeled_before += sum(labeled_flags)
        pairs_before += max(len(patterns) - 1, 0)

        if chain_number in selected:
            kept = dict(chain)
            kept["patterns"] = patterns
            kept["is_anchor"] = flags
            kept["chain_sampling_role"] = "selected_full_chain"
            effective_chains.append(kept)
            labeled_after += sum(labeled_flags)
            pairs_after += max(len(patterns) - 1, 0)
            selected_pattern_occurrences += len(patterns)
            continue

        for pattern_number, (pattern, is_labeled) in enumerate(
            zip(patterns, labeled_flags)
        ):
            if not is_labeled:
                continue
            singleton = {
                "chain_id": f"sampled-labeled-{chain_number}-{pattern_number}",
                "patterns": [pattern],
                "is_anchor": [True],
                "source_chain_id": chain.get(
                    "source_chain_id", chain.get("chain_id", chain_number)
                ),
                "chain_sampling_role": "unselected_labeled_singleton",
            }
            effective_chains.append(singleton)
            labeled_after += 1
            singleton_labeled_occurrences += 1

    if labeled_after != labeled_before:
        raise AssertionError(
            "chain sampling changed labeled occurrence count: "
            f"{labeled_before} -> {labeled_after}"
        )
    stats = {
        "policy": "whole_chain_edges_preserve_labeled_rows_v1",
        "ratio": ratio,
        "seed": int(seed),
        "source_chains": n_chains,
        "selected_full_chains": len(selected),
        "effective_chain_entries": len(effective_chains),
        "pairs_before": pairs_before,
        "active_pairs_after": pairs_after,
        "labeled_occurrences_before": labeled_before,
        "labeled_occurrences_after": labeled_after,
        "selected_pattern_occurrences": selected_pattern_occurrences,
        "unselected_labeled_singletons": singleton_labeled_occurrences,
        "active_pattern_occurrences": sum(
            len(chain.get("patterns", [])) for chain in effective_chains
        ),
        "active_unique_patterns": len({
            pattern
            for chain in effective_chains
            for pattern in chain.get("patterns", [])
        }),
    }
    return effective_chains, stats


def build_chain_train_data(chain_index, card_dict, n_db):
    """
    根据链索引和基数字典，构建训练数据。

    同一 pattern 可能出现在多条链或同一条链的多个位置。其锚点状态按全局
    逻辑 OR 聚合：只要任意一次出现被采为锚点，该 pattern 就保留真实标签。

    参数:
        chain_index: list of dict, load_chain_index() 的输出
        card_dict: dict, {pattern: cardinality}，从基数计算结果加载
        n_db: int, 数据集总行数（'%' 的基数）

    返回:
        all_data: list of (pattern, card)  所有样本，card=-1 表示无标签
            有标签样本参与 MSLE loss，无标签样本仅参与单调性 loss
        chain_pairs: list of list of str  每条链的完整模式序列（用于单调性 loss）
    """
    all_data = []
    chain_pairs = []
    seen = set()
    anchor_by_pattern = {}
    first_anchor_by_pattern = {}

    for chain_info in chain_index:
        for pat, anchor in zip(chain_info["patterns"], chain_info["is_anchor"]):
            is_anchor = bool(anchor)
            first_anchor_by_pattern.setdefault(pat, is_anchor)
            anchor_by_pattern[pat] = anchor_by_pattern.get(pat, False) or is_anchor

    for chain_info in chain_index:
        patterns = chain_info["patterns"]
        chain_pairs.append(patterns)

        for pat in patterns:
            if pat in seen:
                continue
            seen.add(pat)

            if pat == "%":
                # '%' 的基数 = 表总行数，始终已知
                all_data.append((pat, n_db))
            elif anchor_by_pattern.get(pat, False) and pat in card_dict:
                # 任意一次出现是锚点：有真实标签
                all_data.append((pat, card_dict[pat]))
            else:
                # 非锚点或锚点但查不到基数：标记为无标签
                all_data.append((pat, -1))

    n_promoted = sum(
        1
        for pat, is_anchor in anchor_by_pattern.items()
        if pat != "%" and is_anchor and not first_anchor_by_pattern[pat]
    )
    n_labeled = sum(1 for _, c in all_data if c >= 0)
    n_unlabeled = sum(1 for _, c in all_data if c < 0)
    print(f"  Chain train data: {len(all_data)} total, "
          f"{n_labeled} labeled, {n_unlabeled} unlabeled")
    if n_promoted:
        print(f"  Duplicate anchor promotion: {n_promoted} patterns")

    return all_data, chain_pairs


# =====================================================
#  5. 单调性 Loss
# =====================================================

def hinge_monotonicity_loss(preds_tight, preds_loose, margin=0.0, tolerance=0.0):
    """
    链式单调性 hinge loss。

    对于每一对 (tight, loose)，tight 是更精确的模式，loose 是更宽松的模式，
    应满足 pred(tight) <= pred(loose)。

    L = mean(max(0, log(pred_tight + 1) - log(pred_loose + 1)
                 + margin - log(1 + tolerance)))

    tolerance 是 dead-zone，允许小幅非单调波动不被惩罚。
    例如 tolerance=0.1 时，约等于允许 10% 的相对抖动。

    参数:
        preds_tight: Tensor, 精确模式的预测基数
        preds_loose: Tensor, 宽松模式的预测基数
        margin: float, 可选的 margin（默认 0，即只要不违反即可）
        tolerance: float, 可选容忍度（默认 0，不容忍下降）

    返回:
        Tensor: scalar loss
    """
    log_tight = torch.log(preds_tight + 1)
    log_loose = torch.log(preds_loose + 1)
    tau = math.log1p(max(float(tolerance), 0.0))
    violations = torch.clamp(log_tight - log_loose + margin - tau, min=0.0)
    return violations.mean()


# =====================================================
#  6. MonoM Score 评估
# =====================================================

def compute_mono_score(model, test_queries, test_preds, degrade_steps=3, seed=42,
                       tolerance=0.0):
    """
    计算 MonoM score：模型对可比较模式对的单调性遵守率。

    对每个测试查询生成若干退化版本，检查 pred(original) <= pred(degraded)。

    参数:
        model: CLIQUEestimator 实例（已加载，eval 模式）
        test_queries: list of str
        test_preds: list of float, 对应的模型预测值
        degrade_steps: int, 每个查询生成多少个退化版本
        seed: int

    返回:
        float: MonoM score (0-1)
        int: 满足单调性的对数
        int: 总对数
    """
    rng = random.Random(seed)

    satisfied = 0
    total = 0
    skipped = 0

    for orig_idx, query in enumerate(test_queries):
        for step in range(degrade_steps):
            degraded, ok = degrade_like_pattern(query, rng)
            if not ok or degraded == "%" or degraded == query:
                continue

            # 逐个估计，跳过 N-gram 表查不到的模式
            try:
                deg_pred = model.estimate([degraded], to_cuda=False, is_tqdm=False)
                deg_pred_val = deg_pred[0]
            except (KeyError, Exception):
                skipped += 1
                continue

            total += 1
            if tolerance > 0:
                ok_mono = (
                    math.log(test_preds[orig_idx] + 1)
                    <= math.log(deg_pred_val + 1) + math.log1p(tolerance) + 1e-6
                )
            else:
                ok_mono = test_preds[orig_idx] <= deg_pred_val + 1e-6
            if ok_mono:
                satisfied += 1

    score = satisfied / total if total > 0 else 1.0
    print(f"  MonoM eval: {total} pairs evaluated, {skipped} skipped (N-gram KeyError)")
    return score, satisfied, total


def compute_chain_mono_score(model, test_queries=None, chain_index=None,
                             max_chain_length=20, compress_to=10, seed=42,
                             degrade_char_prob=0.4,
                             degrade_shrink_prob=0.5,
                             degrade_under_to_pct_prob=0.0,
                             degrade_insert_pct_prob=0.1,
                             max_wildcard_ratio=0.2,
                             min_literal_len=0,
                             tolerance=0.0):
    """
    Compute chain-based monotonicity on test chains.

    Reports two complementary scores:
      1. pair_score: fraction of adjacent chain pairs satisfying
         pred(pattern_i) <= pred(pattern_{i+1})
      2. chain_score: fraction of whole chains whose every adjacent pair
         satisfies monotonicity.

    If chain_index is provided, it must contain dicts with a "patterns" list.
    Otherwise chains are generated from test_queries only for evaluation.
    """
    if chain_index is None:
        if test_queries is None:
            raise ValueError("Either test_queries or chain_index must be provided")

        rng = random.Random(seed)
        chains = []
        for query in test_queries:
            chain = generate_degradation_chain(
                query,
                max_chain_length=max_chain_length,
                seed=rng.randint(0, 2**31),
                degrade_char_prob=degrade_char_prob,
                degrade_shrink_prob=degrade_shrink_prob,
                degrade_under_to_pct_prob=degrade_under_to_pct_prob,
                degrade_insert_pct_prob=degrade_insert_pct_prob,
                max_wildcard_ratio=max_wildcard_ratio,
                min_literal_len=min_literal_len,
            )
            if compress_to is not None and compress_to > 0:
                chain = chain[:1] if len(chain) == 1 else chain
                if len(chain) > compress_to:
                    # Keep the same coarse behavior as gen_query_chain.py.
                    idxs = np.linspace(0, len(chain) - 1, compress_to).round().astype(int)
                    chain = [chain[i] for i in sorted(set(idxs))]
            chains.append(chain)
    else:
        chains = [c["patterns"] for c in chain_index if len(c.get("patterns", [])) > 0]

    chains = [chain for chain in chains if len(chain) >= 2]
    if not chains:
        return 1.0, 0, 0, 1.0, 0, 0

    unique_patterns = sorted({pat for chain in chains for pat in chain})
    pred_map = {}
    skipped_patterns = set()

    for pat in unique_patterns:
        try:
            pred = model.estimate([pat], to_cuda=False, is_tqdm=False)[0]
            pred_map[pat] = float(pred)
        except Exception:
            skipped_patterns.add(pat)

    pair_satisfied = 0
    pair_total = 0
    chain_satisfied = 0
    chain_total = 0
    skipped_pairs = 0

    for chain in chains:
        chain_ok = True
        chain_has_pair = False

        for tight, loose in zip(chain[:-1], chain[1:]):
            if tight not in pred_map or loose not in pred_map:
                skipped_pairs += 1
                chain_ok = False
                continue

            chain_has_pair = True
            pair_total += 1
            if tolerance > 0:
                ok_mono = (
                    math.log(pred_map[tight] + 1)
                    <= math.log(pred_map[loose] + 1) + math.log1p(tolerance) + 1e-6
                )
            else:
                ok_mono = pred_map[tight] <= pred_map[loose] + 1e-6
            if ok_mono:
                pair_satisfied += 1
            else:
                chain_ok = False

        if chain_has_pair:
            chain_total += 1
            if chain_ok:
                chain_satisfied += 1

    pair_score = pair_satisfied / pair_total if pair_total > 0 else 1.0
    chain_score = chain_satisfied / chain_total if chain_total > 0 else 1.0

    print(
        f"  Chain MonoM eval: {pair_total} pairs, {chain_total} chains, "
        f"{len(skipped_patterns)} patterns skipped, {skipped_pairs} pairs skipped"
    )
    return (
        pair_score, pair_satisfied, pair_total,
        chain_score, chain_satisfied, chain_total,
    )


EXPERIMENT_SUMMARY_FIELDS = [
    "timestamp",
    "summary_schema_version",
    "dataset",
    "workload",
    "method",
    "tag",
    "seed",
    "experiment_protocol",
    "protocol_contract_sha256",
    "formal_run_id",
    "anchor_ratio",
    "anchor_seed",
    "source_workload",
    "data_id",
    "code_commit",
    "code_dirty",
    "train_sha256",
    "valid_sha256",
    "test_sha256",
    "full_chain_index_sha256",
    "full_split_info_sha256",
    "run_command",
    "lambda_mono",
    "lb_min_substr_len",
    "lb_max_substr_len",
    "lb_top_k_strings",
    "lb_top_k_substrings",
    "lb_min_freq_ratio",
    "lb_library_exact",
    "lb_library_prefix",
    "lb_library_suffix",
    "lb_library_infix",
    "lb_library_total",
    "lb_library_capacity",
    "lb_library_fill_rate",
    "train_lb_positive",
    "train_lb_coverage",
    "valid_lb_positive",
    "valid_lb_coverage",
    "test_lb_positive",
    "test_lb_coverage",
    "train_total",
    "train_labeled",
    "train_unlabeled",
    "valid_size",
    "test_size",
    "q_avg",
    "q50",
    "q75",
    "q90",
    "q95",
    "q99",
    "q100",
    "mono_score",
    "mono_satisfied",
    "mono_total",
    "chain_pair_mono_score",
    "chain_pair_mono_satisfied",
    "chain_pair_mono_total",
    "chain_all_mono_score",
    "chain_all_mono_satisfied",
    "chain_all_mono_total",
    "t_mono_score",
    "t_mono_satisfied",
    "t_mono_total",
    "chain_pair_mono_score_t",
    "chain_pair_mono_satisfied_t",
    "chain_pair_mono_total_t",
    "chain_all_mono_score_t",
    "chain_all_mono_satisfied_t",
    "chain_all_mono_total_t",
    "build_time",
    "est_time",
    "model_size",
    "count_size",
    "neural_size",
    "stat_path",
    "est_path",
]


def _count_labeled_unlabeled(data):
    labeled = 0
    unlabeled = 0
    for item in data:
        if len(item) < 2:
            continue
        card = item[1]
        if isinstance(card, (list, tuple, np.ndarray)):
            labeled += 1
        elif card is not None and card >= 0:
            labeled += 1
        else:
            unlabeled += 1
    return labeled, unlabeled


def _load_sample_info(data_name, workload):
    sample_info_path = f"data/{data_name}/training/{workload}/sample_info.json"
    if not os.path.exists(sample_info_path):
        return {}
    try:
        with open(sample_info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"WARNING: failed to load sample_info.json: {exc}")
        return {}


def _sha256_file(path):
    if not os.path.isfile(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_contract_sha256(protocol_id):
    """Hash only the selected protocol contract, not the mutable full config."""
    if not protocol_id:
        return ""
    config_path = os.path.join("manifests", "canonical_manifest_config.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        protocol = config.get("formal_protocols", {}).get(protocol_id)
        if protocol is None:
            raise KeyError(protocol_id)
        payload = json.dumps(
            protocol,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"Cannot resolve experiment protocol contract {protocol_id!r}: {exc}"
        ) from exc


def _experiment_provenance(data_name, workload):
    """Capture immutable data and code identifiers when a result is written."""
    workload_dir = os.path.join("data", data_name, "training", workload)
    sample_info = _load_sample_info(data_name, workload)
    source_workload = sample_info.get("source_workload") or "FULL"
    full_dir = os.path.join("data", data_name, "training", source_workload)
    chain_index_path = os.path.join(full_dir, "chain_index.json")
    split_info_path = os.path.join(full_dir, "split_info.json")
    chain_hash = _sha256_file(chain_index_path)
    split_hash = _sha256_file(split_info_path)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip())
    except OSError:
        commit = ""
        dirty = ""

    return {
        "data_id": (
            f"{data_name}-{chain_hash[:12]}-{split_hash[:12]}"
            if chain_hash and split_hash else ""
        ),
        "code_commit": commit,
        "code_dirty": dirty,
        "train_sha256": _sha256_file(os.path.join(workload_dir, "train.txt")),
        "valid_sha256": _sha256_file(os.path.join(workload_dir, "valid.txt")),
        "test_sha256": _sha256_file(os.path.join(workload_dir, "test.txt")),
        "full_chain_index_sha256": chain_hash,
        "full_split_info_sha256": split_hash,
    }


def append_experiment_summary(
    data_name,
    workload,
    method,
    tag,
    seed,
    train_data,
    valid_data,
    test_data,
    q_errs,
    build_time,
    est_time,
    model_size,
    count_size,
    neural_size,
    stat_path,
    est_path,
    lambda_mono="",
    lb_min_substr_len="",
    lb_max_substr_len="",
    lb_top_k_strings="",
    lb_top_k_substrings="",
    lb_min_freq_ratio="",
    lower_bound_summary=None,
    mono_score="",
    mono_satisfied="",
    mono_total="",
    chain_pair_mono_score="",
    chain_pair_mono_satisfied="",
    chain_pair_mono_total="",
    chain_all_mono_score="",
    chain_all_mono_satisfied="",
    chain_all_mono_total="",
    t_mono_score="",
    t_mono_satisfied="",
    t_mono_total="",
    chain_pair_mono_score_t="",
    chain_pair_mono_satisfied_t="",
    chain_pair_mono_total_t="",
    chain_all_mono_score_t="",
    chain_all_mono_satisfied_t="",
    chain_all_mono_total_t="",
):
    """Append one experiment result row to res/<data>/summary/experiment_summary.csv."""
    summary_dir = f"res/{data_name}/summary"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "experiment_summary.csv")

    q_errs = np.asarray(q_errs, dtype=float)
    train_labeled, train_unlabeled = _count_labeled_unlabeled(train_data)
    sample_info = _load_sample_info(data_name, workload)
    provenance = _experiment_provenance(data_name, workload)
    experiment_protocol = os.environ.get("EXPERIMENT_PROTOCOL_ID", "")
    formal_run_id = os.environ.get("FORMAL_RUN_ID", "")
    protocol_contract_sha256 = _protocol_contract_sha256(experiment_protocol)
    lower_bound_summary = lower_bound_summary or {}
    library_counts = lower_bound_summary.get("library_counts", {})
    coverage_stats = lower_bound_summary.get("coverage_stats", {})

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_schema_version": 3,
        "dataset": data_name,
        "workload": workload,
        "method": method,
        "tag": tag,
        "seed": seed,
        "experiment_protocol": experiment_protocol,
        "protocol_contract_sha256": protocol_contract_sha256,
        "formal_run_id": formal_run_id,
        "anchor_ratio": sample_info.get("anchor_ratio", ""),
        "anchor_seed": sample_info.get("anchor_seed", ""),
        "source_workload": sample_info.get("source_workload", ""),
        "data_id": provenance["data_id"],
        "code_commit": provenance["code_commit"],
        "code_dirty": provenance["code_dirty"],
        "train_sha256": provenance["train_sha256"],
        "valid_sha256": provenance["valid_sha256"],
        "test_sha256": provenance["test_sha256"],
        "full_chain_index_sha256": provenance["full_chain_index_sha256"],
        "full_split_info_sha256": provenance["full_split_info_sha256"],
        "run_command": " ".join(sys.argv),
        "lambda_mono": lambda_mono,
        "lb_min_substr_len": lb_min_substr_len,
        "lb_max_substr_len": lb_max_substr_len,
        "lb_top_k_strings": lb_top_k_strings,
        "lb_top_k_substrings": lb_top_k_substrings,
        "lb_min_freq_ratio": lb_min_freq_ratio,
        "lb_library_exact": library_counts.get("exact", ""),
        "lb_library_prefix": library_counts.get("prefix", ""),
        "lb_library_suffix": library_counts.get("suffix", ""),
        "lb_library_infix": library_counts.get("infix", ""),
        "lb_library_total": library_counts.get("total", ""),
        "lb_library_capacity": library_counts.get("capacity", ""),
        "lb_library_fill_rate": library_counts.get("fill_rate", ""),
        "train_lb_positive": coverage_stats.get("train", {}).get("positive", ""),
        "train_lb_coverage": coverage_stats.get("train", {}).get("coverage", ""),
        "valid_lb_positive": coverage_stats.get("valid", {}).get("positive", ""),
        "valid_lb_coverage": coverage_stats.get("valid", {}).get("coverage", ""),
        "test_lb_positive": coverage_stats.get("test", {}).get("positive", ""),
        "test_lb_coverage": coverage_stats.get("test", {}).get("coverage", ""),
        "train_total": len(train_data),
        "train_labeled": train_labeled,
        "train_unlabeled": train_unlabeled,
        "valid_size": len(valid_data),
        "test_size": len(test_data),
        "q_avg": float(np.mean(q_errs)),
        "q50": float(np.percentile(q_errs, 50)),
        "q75": float(np.percentile(q_errs, 75)),
        "q90": float(np.percentile(q_errs, 90)),
        "q95": float(np.percentile(q_errs, 95)),
        "q99": float(np.percentile(q_errs, 99)),
        "q100": float(np.max(q_errs)),
        "mono_score": mono_score,
        "mono_satisfied": mono_satisfied,
        "mono_total": mono_total,
        "chain_pair_mono_score": chain_pair_mono_score,
        "chain_pair_mono_satisfied": chain_pair_mono_satisfied,
        "chain_pair_mono_total": chain_pair_mono_total,
        "chain_all_mono_score": chain_all_mono_score,
        "chain_all_mono_satisfied": chain_all_mono_satisfied,
        "chain_all_mono_total": chain_all_mono_total,
        "t_mono_score": t_mono_score,
        "t_mono_satisfied": t_mono_satisfied,
        "t_mono_total": t_mono_total,
        "chain_pair_mono_score_t": chain_pair_mono_score_t,
        "chain_pair_mono_satisfied_t": chain_pair_mono_satisfied_t,
        "chain_pair_mono_total_t": chain_pair_mono_total_t,
        "chain_all_mono_score_t": chain_all_mono_score_t,
        "chain_all_mono_satisfied_t": chain_all_mono_satisfied_t,
        "chain_all_mono_total_t": chain_all_mono_total_t,
        "build_time": build_time,
        "est_time": est_time,
        "model_size": model_size,
        "count_size": count_size,
        "neural_size": neural_size,
        "stat_path": stat_path,
        "est_path": est_path,
    }

    # A separate advisory lock serializes schema migration and append.  The
    # lock covers the complete read/rewrite/append transaction, so two formal
    # lanes cannot lose or interleave summary rows.
    lock_path = summary_path + ".lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            file_exists = os.path.exists(summary_path)
            if file_exists:
                with open(summary_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    old_rows = list(reader)
                    old_fields = reader.fieldnames or []
                if old_fields != EXPERIMENT_SUMMARY_FIELDS:
                    with open(summary_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_SUMMARY_FIELDS)
                        writer.writeheader()
                        for old_row in old_rows:
                            writer.writerow({
                                k: old_row.get(k, "")
                                for k in EXPERIMENT_SUMMARY_FIELDS
                            })
                        f.flush()
                        os.fsync(f.fileno())

            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=EXPERIMENT_SUMMARY_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({k: row.get(k, "") for k in EXPERIMENT_SUMMARY_FIELDS})
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    print(f"Experiment summary appended to {summary_path}")
