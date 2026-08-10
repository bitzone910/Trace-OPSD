from __future__ import annotations

import pytest

from shared.validation import (
    effective_topk,
    omit_scores,
    validate_corpus_row,
    validate_embedding_dimension,
    validate_query,
)


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [(2, 5, 2), (5, 5, 5), (9, 5, 5)],
)
def test_effective_topk_returns_every_available_record(
    requested: int,
    available: int,
    expected: int,
) -> None:
    assert effective_topk(requested, available) == expected


@pytest.mark.parametrize("topk", [0, -1, True, 1.5, "3"])
def test_effective_topk_rejects_invalid_values(topk: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        effective_topk(topk, 5)  # type: ignore[arg-type]


def test_query_validation() -> None:
    validate_query("flat modules")
    with pytest.raises(ValueError, match="must not be empty"):
        validate_query("   ")
    with pytest.raises(TypeError, match="must be a string"):
        validate_query(12)  # type: ignore[arg-type]


def test_general_and_theorem_corpus_fields() -> None:
    validate_corpus_row({"id": "1", "contents": "text"}, 0, require_slogan=False)
    validate_corpus_row(
        {"id": "1", "contents": "text", "slogan": "summary"},
        0,
        require_slogan=True,
    )
    with pytest.raises(ValueError, match="slogan"):
        validate_corpus_row(
            {"id": "1", "contents": "text"},
            0,
            require_slogan=True,
        )
    with pytest.raises(ValueError, match="non-string 'contents'"):
        validate_corpus_row(
            {"id": "1", "contents": None},
            0,
            require_slogan=False,
        )


def test_omit_scores_preserves_compatible_shape() -> None:
    results = [[{"id": "a", "contents": "A", "score": 0.9}]]
    assert omit_scores(results) == [[{"id": "a", "contents": "A"}]]


def test_model_dimension_mismatch_is_rejected() -> None:
    validate_embedding_dimension(768, 768, "model")
    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_embedding_dimension(768, 1024, "other-model")
