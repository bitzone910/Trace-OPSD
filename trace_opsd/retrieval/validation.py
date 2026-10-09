"""Small validation helpers shared by both retrievers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def validate_query(query: str) -> None:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must not be empty")


def effective_topk(topk: int, available: int) -> int:
    if isinstance(topk, bool) or not isinstance(topk, int):
        raise TypeError("topk must be an integer")
    if topk <= 0:
        raise ValueError("topk must be greater than zero")
    if available < 0:
        raise ValueError("available must not be negative")
    return min(topk, available)


def validate_embedding_dimension(
    index_dimension: int,
    encoder_dimension: int,
    model_path: str,
) -> None:
    if index_dimension != encoder_dimension:
        raise ValueError(
            "Model/index dimension mismatch: "
            f"index dimension is {index_dimension}, but model "
            f"'{model_path}' outputs {encoder_dimension}."
        )


def validate_corpus_row(
    row: Mapping[str, Any],
    row_index: int,
    *,
    require_slogan: bool,
) -> None:
    required_fields = {"id", "contents"}
    if require_slogan:
        required_fields.add("slogan")
    missing_fields = required_fields - set(row)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Corpus row {row_index + 1} is missing required field(s): {missing}"
        )
    if row["id"] is None:
        raise ValueError(f"Corpus row {row_index + 1} has a null 'id'")
    if not isinstance(row["contents"], str):
        raise ValueError(f"Corpus row {row_index + 1} has a non-string 'contents'")
    if require_slogan and not isinstance(row["slogan"], str):
        raise ValueError(f"Corpus row {row_index + 1} has a non-string 'slogan'")


def omit_scores(
    results: Sequence[Sequence[Mapping[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """Return API results containing only the stable id/contents fields."""

    return [
        [{"id": item["id"], "contents": item["contents"]} for item in query_results]
        for query_results in results
    ]
