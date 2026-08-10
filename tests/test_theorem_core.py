from __future__ import annotations

import numpy as np

from shared.validation import effective_topk
from theorem_search_database.core import (
    build_slogan_groups,
    expand_slogan_groups,
    rank_candidate_rows,
)


def sample_rows() -> list[dict[str, str]]:
    return [
        {"id": "0", "contents": "a0", "slogan": "A"},
        {"id": "1", "contents": "b0", "slogan": "B"},
        {"id": "2", "contents": "a1", "slogan": "A"},
        {"id": "3", "contents": "a2", "slogan": "A"},
        {"id": "4", "contents": "c0", "slogan": "C"},
        {"id": "5", "contents": "b1", "slogan": "B"},
    ]


def test_duplicate_slogans_use_one_group_and_expand_all_contents() -> None:
    groups = build_slogan_groups(sample_rows())
    assert groups.unique_slogans == ["A", "B", "C"]
    np.testing.assert_array_equal(groups.offsets, [0, 3, 5, 6])
    np.testing.assert_array_equal(groups.content_row_ids, [0, 2, 3, 1, 5, 4])

    expanded = expand_slogan_groups(
        [1, 0],
        groups.offsets,
        groups.content_row_ids,
    )
    np.testing.assert_array_equal(expanded, [1, 5, 0, 2, 3])


def test_first_stage_uses_all_unique_slogans_when_fewer_than_twenty() -> None:
    groups = build_slogan_groups(sample_rows())
    assert effective_topk(20, len(groups.unique_slogans)) == 3


def test_contents_are_reranked_and_topk_is_clamped_to_candidates() -> None:
    embeddings = np.asarray(
        [
            [0.5, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    candidates = np.asarray([0, 1, 2], dtype=np.int64)
    row_ids, scores = rank_candidate_rows(
        np.asarray([1.0, 0.0], dtype=np.float32),
        embeddings,
        candidates,
        topk=2,
    )
    np.testing.assert_array_equal(row_ids, [1, 2])
    np.testing.assert_allclose(scores, [1.0, 1.0])

    all_row_ids, _ = rank_candidate_rows(
        np.asarray([1.0, 0.0], dtype=np.float32),
        embeddings,
        candidates,
        topk=20,
    )
    np.testing.assert_array_equal(all_row_ids, [1, 2, 0])


def test_negative_faiss_ids_are_ignored_during_expansion() -> None:
    groups = build_slogan_groups(sample_rows())
    expanded = expand_slogan_groups(
        [-1, 2],
        groups.offsets,
        groups.content_row_ids,
    )
    np.testing.assert_array_equal(expanded, [4])
