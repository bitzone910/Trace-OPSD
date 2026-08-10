"""Dependency-light grouping and reranking logic for theorem search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.validation import effective_topk


@dataclass(frozen=True)
class SloganGroups:
    unique_slogans: list[str]
    offsets: np.ndarray
    content_row_ids: np.ndarray


def build_slogan_groups(rows: Sequence[Mapping[str, Any]]) -> SloganGroups:
    """Group exact slogan strings in first-occurrence order."""

    row_count = len(rows)
    slogan_to_group: dict[str, int] = {}
    unique_slogans: list[str] = []
    group_counts: list[int] = []
    row_group_ids = np.empty(row_count, dtype=np.int64)

    for row_index in range(row_count):
        slogan = rows[row_index]["slogan"]
        group_id = slogan_to_group.get(slogan)
        if group_id is None:
            group_id = len(unique_slogans)
            slogan_to_group[slogan] = group_id
            unique_slogans.append(slogan)
            group_counts.append(0)
        group_counts[group_id] += 1
        row_group_ids[row_index] = group_id

    offsets = np.zeros(len(unique_slogans) + 1, dtype=np.int64)
    if group_counts:
        np.cumsum(np.asarray(group_counts, dtype=np.int64), out=offsets[1:])

    content_row_ids = np.empty(row_count, dtype=np.int64)
    cursors = offsets[:-1].copy()
    for row_index, group_id in enumerate(row_group_ids):
        cursor = cursors[group_id]
        content_row_ids[cursor] = row_index
        cursors[group_id] += 1

    return SloganGroups(unique_slogans, offsets, content_row_ids)


def expand_slogan_groups(
    group_ids: Sequence[int] | np.ndarray,
    offsets: np.ndarray,
    content_row_ids: np.ndarray,
) -> np.ndarray:
    """Expand retrieved unique-slogan ids to all corresponding corpus rows."""

    slices: list[np.ndarray] = []
    for raw_group_id in group_ids:
        group_id = int(raw_group_id)
        if group_id < 0:
            continue
        if group_id + 1 >= len(offsets):
            raise ValueError(f"Slogan group id is out of range: {group_id}")
        start = int(offsets[group_id])
        stop = int(offsets[group_id + 1])
        slices.append(np.asarray(content_row_ids[start:stop], dtype=np.int64))
    if not slices:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(slices)


def rank_candidate_rows(
    query_embedding: np.ndarray,
    content_embeddings: np.ndarray,
    candidate_row_ids: np.ndarray,
    *,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank a restricted candidate set by normalized-vector inner product."""

    candidate_row_ids = np.asarray(candidate_row_ids, dtype=np.int64)
    result_count = effective_topk(topk, int(candidate_row_ids.size))
    if result_count == 0:
        return candidate_row_ids, np.empty(0, dtype=np.float32)

    query_vector = np.asarray(query_embedding, dtype=np.float32)
    if query_vector.ndim == 2 and query_vector.shape[0] == 1:
        query_vector = query_vector[0]
    if query_vector.ndim != 1:
        raise ValueError("query_embedding must be a 1D vector")
    if content_embeddings.ndim != 2:
        raise ValueError("content_embeddings must be a 2D matrix")
    if content_embeddings.shape[1] != query_vector.shape[0]:
        raise ValueError("Query/content embedding dimension mismatch")
    if np.any(candidate_row_ids < 0) or np.any(
        candidate_row_ids >= content_embeddings.shape[0]
    ):
        raise ValueError("Candidate content row id is out of range")

    candidate_vectors = np.asarray(
        content_embeddings[candidate_row_ids],
        dtype=np.float32,
    )
    scores = np.asarray(candidate_vectors @ query_vector, dtype=np.float32)
    # Deterministic tie-breaking follows original corpus row order.
    order = np.lexsort((candidate_row_ids, -scores))[:result_count]
    return candidate_row_ids[order], scores[order]
