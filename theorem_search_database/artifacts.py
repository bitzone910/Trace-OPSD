"""Artifact bundle names and validation for theorem indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"
SLOGAN_INDEX_FILE = "slogan.index"
CONTENT_EMBEDDINGS_FILE = "content_embeddings.npy"
SLOGAN_OFFSETS_FILE = "slogan_offsets.npy"
CONTENT_ROW_IDS_FILE = "content_row_ids.npy"


def read_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    path = Path(bundle_dir).expanduser().resolve() / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Theorem index manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read theorem index manifest: {path}") from exc

    required = {
        "format_version",
        "database_type",
        "corpus_rows",
        "unique_slogans",
        "embedding_dimension",
        "faiss_type",
        "model_path",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(
            "Theorem index manifest is missing field(s): "
            + ", ".join(sorted(missing))
        )
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError(
            "Unsupported theorem index format_version: "
            f"{manifest['format_version']}"
        )
    if manifest["database_type"] != "theorem_search":
        raise ValueError("Index bundle is not a theorem search index")
    for field in ("corpus_rows", "unique_slogans", "embedding_dimension"):
        if isinstance(manifest[field], bool) or not isinstance(manifest[field], int):
            raise ValueError(f"Manifest field '{field}' must be an integer")
        if manifest[field] <= 0:
            raise ValueError(f"Manifest field '{field}' must be greater than zero")
    return manifest


def load_array(path: Path, *, mmap: bool = True) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Theorem index artifact does not exist: {path}")
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def validate_bundle_arrays(
    manifest: dict[str, Any],
    *,
    index_total: int,
    index_dimension: int,
    corpus_rows: int,
    slogan_offsets: np.ndarray,
    content_row_ids: np.ndarray,
    content_embeddings: np.ndarray,
) -> None:
    expected_rows = int(manifest["corpus_rows"])
    unique_slogans = int(manifest["unique_slogans"])
    dimension = int(manifest["embedding_dimension"])
    if corpus_rows != expected_rows:
        raise ValueError(
            "Index/corpus size mismatch: "
            f"manifest has {expected_rows} rows but corpus has {corpus_rows} rows."
        )
    if index_total != unique_slogans:
        raise ValueError("Slogan index/manifest count mismatch")
    if index_dimension != dimension:
        raise ValueError("Slogan index/manifest dimension mismatch")
    if slogan_offsets.dtype != np.int64 or slogan_offsets.shape != (
        unique_slogans + 1,
    ):
        raise ValueError("Invalid slogan_offsets artifact")
    if int(slogan_offsets[0]) != 0 or int(slogan_offsets[-1]) != expected_rows:
        raise ValueError("slogan_offsets does not cover the corpus exactly")
    if np.any(np.diff(slogan_offsets) <= 0):
        raise ValueError("Every slogan group must contain at least one row")
    if content_row_ids.dtype != np.int64 or content_row_ids.shape != (
        expected_rows,
    ):
        raise ValueError("Invalid content_row_ids artifact")
    if np.any(content_row_ids < 0) or np.any(content_row_ids >= expected_rows):
        raise ValueError("content_row_ids contains an out-of-range row id")
    counts = np.bincount(content_row_ids, minlength=expected_rows)
    if len(counts) != expected_rows or np.any(counts != 1):
        raise ValueError("content_row_ids must be a permutation of corpus rows")
    if content_embeddings.dtype != np.float32 or content_embeddings.shape != (
        expected_rows,
        dimension,
    ):
        raise ValueError("Invalid content_embeddings artifact")
