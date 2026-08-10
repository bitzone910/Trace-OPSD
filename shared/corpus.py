"""JSONL corpus loading shared by the two databases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .validation import validate_corpus_row


def load_corpus(
    corpus_path: str | Path,
    *,
    num_proc: int = 4,
    require_slogan: bool = False,
) -> Any:
    """Load a JSONL corpus while preserving its positional row ids."""

    if num_proc <= 0:
        raise ValueError("num_proc must be greater than zero")

    path = Path(corpus_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file does not exist: {path}")

    from datasets import load_dataset

    corpus = load_dataset(
        "json",
        data_files=str(path),
        split="train",
        num_proc=num_proc,
    )
    if len(corpus) == 0:
        raise ValueError(f"Corpus is empty: {path}")

    required_columns = {"id", "contents"}
    if require_slogan:
        required_columns.add("slogan")
    missing_columns = required_columns - set(corpus.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Corpus is missing required field(s): {missing}")

    for row_index, row in enumerate(corpus):
        validate_corpus_row(
            row,
            row_index,
            require_slogan=require_slogan,
        )
    return corpus
