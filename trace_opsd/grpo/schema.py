"""Validation for the versioned Trace-OPSD GRPO hindsight dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from trace_opsd.compat import namespace_matches, read_json

DATASET_SCHEMA_VERSION = "trace_opsd.grpo.dataset.v2"
PROTOCOL_VERSION = "trace_opsd.xml-tools.v3"
DATA_SOURCE = "trace_opsd_math"
SPLIT_FILES = {"train": "train.parquet", "validation": "validation.parquet"}


class DatasetValidationError(ValueError):
    """Raised when GRPO input data violates the public contract."""


@dataclass(frozen=True)
class SplitSummary:
    file: str
    samples: int
    sha256: str


@dataclass(frozen=True)
class DatasetSummary:
    train: SplitSummary
    validation: SplitSummary
    manifest_sha256: str

    @property
    def digest(self) -> str:
        value = f"{self.train.sha256}:{self.validation.sha256}:{self.manifest_sha256}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_parquet_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise DatasetValidationError(
            "pyarrow is required to read GRPO parquet data"
        ) from exc
    try:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    except Exception as exc:
        raise DatasetValidationError(f"cannot read parquet file {path}: {exc}") from exc


def _nonempty_string(value: object, *, field: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{source}: {field} must be a nonempty string")
    return value


def validate_row(row: Mapping[str, Any], *, split: str, source: str) -> str:
    if not isinstance(row, Mapping):
        raise DatasetValidationError(f"{source}: row must be an object")
    if not namespace_matches(row.get('data_source'), DATA_SOURCE):
        raise DatasetValidationError(f"{source}: data_source must be {DATA_SOURCE!r}")
    if row.get("ability") != "math-tool-reasoning":
        raise DatasetValidationError(f"{source}: ability must be 'math-tool-reasoning'")

    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 1:
        raise DatasetValidationError(
            f"{source}: prompt must contain exactly one user message"
        )
    message = prompt[0]
    if not isinstance(message, Mapping) or message.get("role") != "user":
        raise DatasetValidationError(
            f"{source}: prompt may contain only one role='user' message"
        )
    content = _nonempty_string(
        message.get("content"), field="prompt content", source=source
    )
    required_phrases = (
        "<python>",
        "<search>",
        "<answer>",
    )
    if any(value not in content for value in required_phrases):
        raise DatasetValidationError(
            f"{source}: prompt is not a complete Trace-OPSD tool prompt"
        )

    reward_model = row.get("reward_model")
    if not isinstance(reward_model, Mapping) or reward_model.get("style") != "rule":
        raise DatasetValidationError(f"{source}: reward_model.style must be 'rule'")
    ground_truth = reward_model.get("ground_truth")
    targets = ground_truth.get("target") if isinstance(ground_truth, Mapping) else None
    if not isinstance(targets, list) or not targets:
        raise DatasetValidationError(
            f"{source}: reward_model.ground_truth.target must be a nonempty list"
        )
    for index, target in enumerate(targets):
        _nonempty_string(target, field=f"target[{index}]", source=source)

    _nonempty_string(row.get("problem"), field="problem", source=source)
    _nonempty_string(row.get("hindsight"), field="hindsight", source=source)

    extra = row.get("extra_info")
    if not isinstance(extra, Mapping):
        raise DatasetValidationError(f"{source}: extra_info must be an object")
    source_id = _nonempty_string(
        extra.get("source_id"), field="source_id", source=source
    )
    if extra.get("split") != split:
        raise DatasetValidationError(
            f"{source}: extra_info.split={extra.get('split')!r}, expected {split!r}"
        )
    index = extra.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise DatasetValidationError(
            f"{source}: extra_info.index must be a non-negative integer"
        )
    return source_id


def validate_token_lengths(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_prompt_length: int = 2048,
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
) -> None:
    template_kwargs = dict(apply_chat_template_kwargs or {})
    for row_index, row in enumerate(rows, start=1):
        prompt = row["prompt"]
        try:
            token_ids = tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
                **template_kwargs,
            )
        except Exception as exc:
            raise DatasetValidationError(
                f"row {row_index}: Qwen2.5 chat-template rendering failed: {exc}"
            ) from exc
        if len(token_ids) > max_prompt_length:
            raise DatasetValidationError(
                f"row {row_index}: tokenized prompt length {len(token_ids)} exceeds "
                f"{max_prompt_length}; GRPO input is never truncated"
            )


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        value = read_json(path)
    except FileNotFoundError as exc:
        raise DatasetValidationError(f"missing GRPO manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"invalid GRPO manifest JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise DatasetValidationError("GRPO manifest must be a JSON object")
    return value


def validate_dataset_directory(
    dataset_dir: Path,
    *,
    tokenizer: Any | None = None,
    max_prompt_length: int = 2048,
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
) -> DatasetSummary:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise DatasetValidationError(f"GRPO dataset directory not found: {dataset_dir}")
    manifest_path = dataset_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if not namespace_matches(manifest.get('schema_version'), DATASET_SCHEMA_VERSION):
        raise DatasetValidationError(
            f"manifest schema_version must be {DATASET_SCHEMA_VERSION}"
        )
    if not namespace_matches(manifest.get('protocol_version'), PROTOCOL_VERSION):
        raise DatasetValidationError(
            f"manifest protocol_version must be {PROTOCOL_VERSION}"
        )
    split_entries = manifest.get("splits")
    if not isinstance(split_entries, Mapping):
        raise DatasetValidationError("manifest.splits must be an object")

    summaries: dict[str, SplitSummary] = {}
    source_ids: set[str] = set()
    for split, expected_filename in SPLIT_FILES.items():
        entry = split_entries.get(split)
        if not isinstance(entry, Mapping):
            raise DatasetValidationError(f"manifest is missing split {split!r}")
        if entry.get("file") != expected_filename:
            raise DatasetValidationError(
                f"manifest split {split!r} must use {expected_filename!r}"
            )
        path = dataset_dir / expected_filename
        if not path.is_file():
            raise DatasetValidationError(f"missing GRPO split file: {path}")
        actual_sha = sha256_file(path)

        rows = list(iter_parquet_rows(path))
        declared_samples = entry.get("samples")
        if isinstance(declared_samples, bool) or declared_samples != len(rows):
            raise DatasetValidationError(
                f"sample count mismatch for {expected_filename}: "
                f"manifest={declared_samples!r}, "
                f"actual={len(rows)}"
            )
        if not rows:
            raise DatasetValidationError(f"GRPO split is empty: {expected_filename}")
        for row_number, row in enumerate(rows, start=1):
            source = f"{expected_filename}:{row_number}"
            source_id = validate_row(row, split=split, source=source)
            if source_id in source_ids:
                raise DatasetValidationError(
                    f"{source}: duplicate source_id {source_id!r}"
                )
            source_ids.add(source_id)
        if tokenizer is not None:
            validate_token_lengths(
                rows,
                tokenizer=tokenizer,
                max_prompt_length=max_prompt_length,
                apply_chat_template_kwargs=apply_chat_template_kwargs,
            )
        summaries[split] = SplitSummary(
            file=expected_filename,
            samples=len(rows),
            sha256=actual_sha,
        )

    return DatasetSummary(
        train=summaries["train"],
        validation=summaries["validation"],
        manifest_sha256=sha256_file(manifest_path),
    )


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "DatasetSummary",
    "DatasetValidationError",
    "SplitSummary",
    "iter_parquet_rows",
    "sha256_file",
    "validate_dataset_directory",
    "validate_row",
    "validate_token_lengths",
]
