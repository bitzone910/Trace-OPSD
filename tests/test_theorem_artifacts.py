from __future__ import annotations

import json

import numpy as np
import pytest

from theorem_search_database.artifacts import (
    FORMAT_VERSION,
    MANIFEST_FILE,
    read_manifest,
    validate_bundle_arrays,
)


def valid_manifest() -> dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "database_type": "theorem_search",
        "corpus_rows": 3,
        "unique_slogans": 2,
        "embedding_dimension": 2,
        "faiss_type": "Flat",
        "model_path": "models/e5-base-v2",
    }


def valid_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([0, 2, 3], dtype=np.int64),
        np.asarray([0, 2, 1], dtype=np.int64),
        np.zeros((3, 2), dtype=np.float32),
    )


def test_read_manifest_and_reject_missing_fields(tmp_path) -> None:
    (tmp_path / MANIFEST_FILE).write_text(
        json.dumps(valid_manifest()),
        encoding="utf-8",
    )
    assert read_manifest(tmp_path)["unique_slogans"] == 2

    (tmp_path / MANIFEST_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing field"):
        read_manifest(tmp_path)


def test_valid_bundle_layout() -> None:
    offsets, row_ids, embeddings = valid_arrays()
    validate_bundle_arrays(
        valid_manifest(),
        index_total=2,
        index_dimension=2,
        corpus_rows=3,
        slogan_offsets=offsets,
        content_row_ids=row_ids,
        content_embeddings=embeddings,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("corpus_rows", 4, "size mismatch"),
        ("index_total", 3, "count mismatch"),
        ("index_dimension", 3, "dimension mismatch"),
    ],
)
def test_bundle_count_mismatches_are_rejected(
    field: str,
    value: int,
    message: str,
) -> None:
    offsets, row_ids, embeddings = valid_arrays()
    kwargs = {
        "index_total": 2,
        "index_dimension": 2,
        "corpus_rows": 3,
        "slogan_offsets": offsets,
        "content_row_ids": row_ids,
        "content_embeddings": embeddings,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        validate_bundle_arrays(valid_manifest(), **kwargs)


def test_duplicate_or_out_of_range_row_mapping_is_rejected() -> None:
    offsets, _, embeddings = valid_arrays()
    with pytest.raises(ValueError, match="permutation"):
        validate_bundle_arrays(
            valid_manifest(),
            index_total=2,
            index_dimension=2,
            corpus_rows=3,
            slogan_offsets=offsets,
            content_row_ids=np.asarray([0, 0, 1], dtype=np.int64),
            content_embeddings=embeddings,
        )
