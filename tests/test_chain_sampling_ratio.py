import pytest

from src.CLIQUE_bound_mono import _append_final_chain_batch
from src.util import sample_chain_index_preserve_labeled


def _batch(queries, pair_tight_idx=None, pair_loose_idx=None):
    size = len(queries)
    return {
        "queries": list(queries),
        "cards": list(range(size)),
        "upper_bounds": [100.0] * size,
        "lower_bounds": [0.0] * size,
        "pair_tight_idx": list(pair_tight_idx or []),
        "pair_loose_idx": list(pair_loose_idx or []),
    }


def test_singleton_tail_is_merged_without_dropping_rows_or_edges():
    batches = [_batch([f"q{i}" for i in range(64)], [0], [1])]
    tail = _batch(["retained-label"])

    _append_final_chain_batch(batches, tail)

    assert len(batches) == 1
    assert len(batches[0]["queries"]) == 65
    assert batches[0]["queries"][-1] == "retained-label"
    assert batches[0]["cards"][-1] == 0
    assert batches[0]["pair_tight_idx"] == [0]
    assert batches[0]["pair_loose_idx"] == [1]


def test_non_singleton_tail_remains_a_separate_batch():
    batches = [_batch([f"q{i}" for i in range(64)])]
    tail = _batch(["tail-a", "tail-b"], [0], [1])

    _append_final_chain_batch(batches, tail)

    assert [len(batch["queries"]) for batch in batches] == [64, 2]
    assert batches[-1]["pair_tight_idx"] == [0]
    assert batches[-1]["pair_loose_idx"] == [1]


def test_chain_sampling_preserves_labeled_occurrences_and_samples_only_edges():
    chains = [
        {
            "chain_id": 0,
            "patterns": ["a", "a%", "%"],
            "is_anchor": [True, False, True],
        },
        {
            "chain_id": 1,
            "patterns": ["b", "b%", "%b%"],
            "is_anchor": [True, False, True],
        },
    ]
    train_data = [
        ("a", 1), ("a%", -1), ("%", 100),
        ("b", 2), ("b%", -1), ("%b%", 4),
    ]

    sampled, stats = sample_chain_index_preserve_labeled(
        chains, train_data, ratio=0.5, seed=0
    )

    assert stats["source_chains"] == 2
    assert stats["selected_full_chains"] == 1
    assert stats["pairs_before"] == 4
    assert stats["active_pairs_after"] == 2
    assert stats["labeled_occurrences_before"] == 4
    assert stats["labeled_occurrences_after"] == 4
    assert stats["unselected_labeled_singletons"] == 2
    assert sum(max(len(chain["patterns"]) - 1, 0) for chain in sampled) == 2
    assert all(
        len(chain["patterns"]) == 1
        for chain in sampled
        if chain["chain_sampling_role"] == "unselected_labeled_singleton"
    )


def test_chain_sampling_ratio_one_is_full_chain_equivalent():
    chains = [
        {"chain_id": 0, "patterns": ["a", "a%"], "is_anchor": [True, False]},
        {"chain_id": 1, "patterns": ["b", "b%"], "is_anchor": [True, False]},
    ]
    train_data = [("a", 1), ("a%", -1), ("b", 2), ("b%", -1)]

    sampled, stats = sample_chain_index_preserve_labeled(
        chains, train_data, ratio=1.0, seed=7
    )

    assert [chain["patterns"] for chain in sampled] == [
        chain["patterns"] for chain in chains
    ]
    assert stats["active_pairs_after"] == stats["pairs_before"] == 2
    assert stats["unselected_labeled_singletons"] == 0


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1])
def test_chain_sampling_rejects_unsupported_ratio(ratio):
    with pytest.raises(ValueError):
        sample_chain_index_preserve_labeled([], [], ratio=ratio, seed=0)
