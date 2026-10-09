"""Unmodified Alpaca data and LlamaFactory-compatible truncation diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from trace_opsd.compat import read_json

DATASET_NAME = "sft_math_tir"
EXPECTED_SAMPLES = 30_438


class DatasetValidationError(ValueError):
    """Invalid published SFT data; no row is silently dropped or rewritten."""


@dataclass(frozen=True)
class DatasetSummary:
    file: str
    samples: int
    sha256: str

    @property
    def digest(self) -> str:
        identity = f"alpaca.full_output.v1:{self.samples}:{self.sha256}"
        return hashlib.sha256(identity.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: str | Path) -> list[Mapping[str, str]]:
    source = Path(path)
    try:
        rows = read_json(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"cannot read Alpaca JSON: {source}") from exc
    if not isinstance(rows, list) or not rows:
        raise DatasetValidationError("SFT JSON must contain a nonempty row array")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DatasetValidationError(f"row {index} must be an object")
        for field in ("instruction", "input", "output"):
            if not isinstance(row.get(field), str):
                raise DatasetValidationError(f"row {index}.{field} must be a string")
        if not row["instruction"].strip() or not row["output"].strip():
            raise DatasetValidationError(f"row {index} has empty instruction/output")
    return rows


def validate_dataset_file(
    path: str | Path, *, expected_samples: int | None = EXPECTED_SAMPLES
) -> DatasetSummary:
    source = Path(path).expanduser().resolve()
    rows = load_rows(source)
    if expected_samples is not None and len(rows) != expected_samples:
        raise DatasetValidationError(
            f"expected all {expected_samples} published SFT rows; found {len(rows)}"
        )
    return DatasetSummary(str(source), len(rows), sha256_file(source))


def write_dataset_registry(dataset_file: Path, directory: Path) -> Path:
    """Register the original absolute file without copying/converting its rows."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "dataset_info.json"
    value = {
        DATASET_NAME: {
            "file_name": str(dataset_file.expanduser().resolve()),
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def alpaca_messages(row: Mapping[str, str]) -> list[dict[str, str]]:
    """Match AlpacaDatasetConverter, including its newline and user role."""
    query = [row[field] for field in ("instruction", "input") if row[field]]
    return [
        {"role": "user", "content": "\n".join(query)},
        {"role": "assistant", "content": row["output"]},
    ]


def audit_token_lengths(
    rows: Iterable[Mapping[str, str]],
    *,
    tokenizer: Any,
    template: Any,
    infer_seqlen: Callable[[int, int, int], tuple[int, int]],
    cutoff_len: int,
) -> dict[str, Any]:
    """Use the vendored template and infer_seqlen used by the SFT processor.

    With one Alpaca pair, no multimodal inputs, no packing/history masking, and
    train_on_prompt=false, this is exactly the native supervised length policy.
    Tool observations inside output remain supervised assistant text.
    """
    if cutoff_len <= 0:
        raise ValueError("cutoff_len must be positive")
    lengths: list[int] = []
    train_lengths: list[int] = []
    supervised_lengths: list[int] = []
    truncated: list[int] = []
    source_truncated: list[int] = []
    target_truncated: list[int] = []
    eos = int(template.efficient_eos)
    for index, row in enumerate(rows):
        pairs = list(template.encode_multiturn(tokenizer, alpaca_messages(row), "", ""))
        if len(pairs) != 1:
            raise DatasetValidationError(f"Alpaca row {index} must encode one pair")
        source, target = pairs[0]
        source_len, target_len = infer_seqlen(
            len(source), len(target), cutoff_len - eos
        )
        raw_length = len(source) + len(target) + eos
        train_length = source_len + target_len + eos
        if target_len + eos <= 0 or train_length > cutoff_len:
            raise DatasetValidationError(f"row {index} has invalid native encoding")
        lengths.append(raw_length)
        train_lengths.append(train_length)
        supervised_lengths.append(target_len + eos)
        if raw_length > train_length:
            truncated.append(index)
        if source_len < len(source):
            source_truncated.append(index)
        if target_len < len(target):
            target_truncated.append(index)
    if not lengths:
        raise DatasetValidationError("cannot audit an empty SFT dataset")
    return {
        "schema_version": "trace_opsd.sft.truncation.v1",
        "policy": "llamafactory.infer_seqlen",
        "cutoff_len": cutoff_len,
        "samples": len(lengths),
        "truncated_samples": len(truncated),
        "source_truncated_samples": len(source_truncated),
        "output_truncated_samples": len(target_truncated),
        "truncated_row_indices": truncated,
        "source_truncated_row_indices": source_truncated,
        "output_truncated_row_indices": target_truncated,
        "raw_tokens_max": max(lengths),
        "raw_tokens_mean": sum(lengths) / len(lengths),
        "training_tokens_max": max(train_lengths),
        "training_tokens_mean": sum(train_lengths) / len(train_lengths),
        "supervised_tokens_min": min(supervised_lengths),
        "all_output_text_supervised": True,
        "validation_samples": 0,
    }
