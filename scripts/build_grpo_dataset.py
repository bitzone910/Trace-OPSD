#!/usr/bin/env python3
"""Build a Trace-OPSD GRPO dataset with privileged solution hindsight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trace_opsd.grpo.schema import (  # noqa: E402, I001
    DATASET_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    validate_dataset_directory,
)
from trace_opsd.rollout.prompts import build_tool_prompt  # noqa: E402
from trace_opsd.assets import asset_path  # noqa: E402


DEFAULT_INPUT = asset_path("datasets", "numinamath", "rl_trace_opsd.parquet")
DEFAULT_OUTPUT = asset_path("datasets", "grpo")
REQUIRED_COLUMNS = ("id", "problem", "ground_truth", "hindsight")

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string(), nullable=False),
        pa.field(
            "prompt",
            pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string(), nullable=False),
                        pa.field("content", pa.string(), nullable=False),
                    ]
                )
            ),
            nullable=False,
        ),
        pa.field("ability", pa.string(), nullable=False),
        pa.field(
            "reward_model",
            pa.struct(
                [
                    pa.field("style", pa.string(), nullable=False),
                    pa.field(
                        "ground_truth",
                        pa.struct(
                            [
                                pa.field(
                                    "target",
                                    pa.list_(pa.string()),
                                    nullable=False,
                                )
                            ]
                        ),
                        nullable=False,
                    ),
                ]
            ),
            nullable=False,
        ),
        pa.field(
            "extra_info",
            pa.struct(
                [
                    pa.field("index", pa.int64(), nullable=False),
                    pa.field("source_id", pa.string(), nullable=False),
                    pa.field("split", pa.string(), nullable=False),
                ]
            ),
            nullable=False,
        ),
        pa.field("problem", pa.string(), nullable=False),
        pa.field("hindsight", pa.string(), nullable=False),
    ]
)


class DatasetBuildError(ValueError):
    """Raised when cleaned NuminaMath cannot be converted safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source(
    path: Path, *, require_difficulty: bool = False
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetBuildError(f"input parquet does not exist: {path}")
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise DatasetBuildError(f"cannot read input parquet {path}: {exc}") from exc

    requested_columns = REQUIRED_COLUMNS + (
        ("difficulty",) if require_difficulty else ()
    )
    missing = set(requested_columns) - set(parquet.schema_arrow.names)
    if missing:
        raise DatasetBuildError(
            f"input parquet is missing required columns: {sorted(missing)}"
        )

    rows: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    row_number = 0
    try:
        batches = parquet.iter_batches(batch_size=2048, columns=requested_columns)
        for batch in batches:
            for raw in batch.to_pylist():
                row_number += 1
                source_id = raw.get("id")
                if isinstance(source_id, bool) or not isinstance(source_id, (int, str)):
                    raise DatasetBuildError(
                        f"input row {row_number}: id must be a string or integer"
                    )
                normalized_source_id = str(source_id).strip()
                if not normalized_source_id:
                    raise DatasetBuildError(
                        f"input row {row_number}: id must be nonempty"
                    )
                if normalized_source_id in source_ids:
                    raise DatasetBuildError(
                        f"input row {row_number}: duplicate id {normalized_source_id}"
                    )
                source_ids.add(normalized_source_id)

                problem = raw.get("problem")
                if not isinstance(problem, str) or not problem.strip():
                    raise DatasetBuildError(
                        f"input row {row_number}: problem must be a nonempty string"
                    )
                ground_truth = raw.get("ground_truth")
                if not isinstance(ground_truth, str) or not ground_truth.strip():
                    raise DatasetBuildError(
                        f"input row {row_number}: ground_truth must be a "
                        "nonempty string"
                    )
                hindsight = raw.get("hindsight")
                if not isinstance(hindsight, str) or not hindsight.strip():
                    raise DatasetBuildError(
                        f"input row {row_number}: hindsight must be a nonempty string"
                    )
                row = {
                        "id": normalized_source_id,
                        "problem": problem.strip(),
                        "ground_truth": ground_truth.strip(),
                        "hindsight": hindsight.strip(),
                    }
                if require_difficulty:
                    difficulty = raw.get("difficulty")
                    if not isinstance(difficulty, str) or not difficulty.strip():
                        raise DatasetBuildError(
                            f"input row {row_number}: difficulty must be a "
                            "nonempty string"
                        )
                    row["difficulty"] = difficulty.strip().lower()
                rows.append(row)
    except DatasetBuildError:
        raise
    except Exception as exc:
        raise DatasetBuildError(f"cannot stream input parquet {path}: {exc}") from exc
    return rows


def _output_row(source: Mapping[str, Any], *, split: str, index: int) -> dict[str, Any]:
    problem = str(source["problem"])
    return {
        "data_source": "trace_opsd_math",
        "prompt": [{"role": "user", "content": build_tool_prompt(problem)}],
        "ability": "math-tool-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": [str(source["ground_truth"])]},
        },
        "extra_info": {
            "index": index,
            "source_id": f"numinamath-{source['id']}",
            "split": split,
        },
        "problem": problem,
        "hindsight": str(source["hindsight"]),
    }


def _write_split(path: Path, sources: Sequence[Mapping[str, Any]], split: str) -> None:
    rows = [
        _output_row(source, split=split, index=index)
        for index, source in enumerate(sources)
    ]
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    pq.write_table(table, path, compression="zstd")


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _publish_directory(staged: Path, output: Path) -> None:
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
    moved_existing = False
    try:
        if output.exists() or output.is_symlink():
            os.replace(output, backup)
            moved_existing = True
        os.replace(staged, output)
    except Exception:
        if moved_existing and not output.exists() and backup.exists():
            os.replace(backup, output)
        raise
    else:
        if moved_existing:
            _remove_path(backup)


def build_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    validation_size: int = 256,
    seed: int = 42,
    validation_difficulty_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Convert cleaned NuminaMath and atomically replace ``output_dir``."""

    source_path = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if isinstance(validation_size, bool) or validation_size <= 0:
        raise DatasetBuildError("validation_size must be a positive integer")

    difficulty_counts = (
        {str(key).lower(): value for key, value in validation_difficulty_counts.items()}
        if validation_difficulty_counts is not None
        else None
    )
    if difficulty_counts is not None:
        if not difficulty_counts:
            raise DatasetBuildError(
                "validation_difficulty_counts must not be empty"
            )
        for difficulty, count in difficulty_counts.items():
            if not difficulty:
                raise DatasetBuildError("validation difficulty must be nonempty")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise DatasetBuildError(
                    "validation difficulty counts must be non-negative integers"
                )
        if sum(difficulty_counts.values()) != validation_size:
            raise DatasetBuildError(
                "validation difficulty counts must sum to validation_size"
            )

    sources = _load_source(
        source_path, require_difficulty=difficulty_counts is not None
    )
    if len(sources) <= validation_size:
        raise DatasetBuildError(
            f"input needs more than {validation_size} rows so both splits "
            "are nonempty; "
            f"found {len(sources)}"
        )

    rng = random.Random(seed)
    if difficulty_counts is None:
        shuffled = list(sources)
        rng.shuffle(shuffled)
        validation = shuffled[:validation_size]
    else:
        validation = []
        for difficulty in sorted(difficulty_counts):
            candidates = [
                row for row in sources if row["difficulty"] == difficulty
            ]
            requested = difficulty_counts[difficulty]
            if len(candidates) < requested:
                raise DatasetBuildError(
                    f"difficulty {difficulty!r} needs {requested} validation rows; "
                    f"found {len(candidates)}"
                )
            rng.shuffle(candidates)
            validation.extend(candidates[:requested])
    validation_ids = {str(row["id"]) for row in validation}
    train = [row for row in sources if str(row["id"]) not in validation_ids]

    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        split_sources = {"train": train, "validation": validation}
        split_manifest: dict[str, dict[str, Any]] = {}
        for split, rows in split_sources.items():
            filename = f"{split}.parquet"
            path = staged / filename
            _write_split(path, rows, split)
            split_manifest[split] = {
                "file": filename,
                "samples": len(rows),
                "sha256": _sha256(path),
            }

        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "source": {
                "file": source_path.name,
                "samples": len(sources),
                "sha256": _sha256(source_path),
            },
            "split_seed": seed,
            "validation_size": validation_size,
            "splits": split_manifest,
        }
        if difficulty_counts is not None:
            manifest["validation_difficulty_counts"] = difficulty_counts
        manifest_path = staged / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        summary = validate_dataset_directory(staged)
        result = {
            "input": str(source_path),
            "output": str(output),
            "input_samples": len(sources),
            "train_samples": summary.train.samples,
            "validation_samples": summary.validation.samples,
            "seed": seed,
            "train_sha256": summary.train.sha256,
            "validation_sha256": summary.validation.sha256,
            "manifest_sha256": summary.manifest_sha256,
            "dataset_digest": summary.digest,
        }
        if difficulty_counts is not None:
            result["validation_difficulty_counts"] = difficulty_counts
        _publish_directory(staged, output)
        return result
    finally:
        if staged.exists():
            _remove_path(staged)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--validation-mid-size", type=int)
    parser.add_argument("--validation-high-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validation_difficulty_counts = None
    if args.validation_mid_size is not None or args.validation_high_size is not None:
        if args.validation_mid_size is None or args.validation_high_size is None:
            print(
                "Error: --validation-mid-size and --validation-high-size must be "
                "provided together",
                file=sys.stderr,
            )
            return 1
        validation_difficulty_counts = {
            "mid": args.validation_mid_size,
            "high": args.validation_high_size,
        }
    try:
        result = build_dataset(
            args.input,
            args.output,
            validation_size=args.validation_size,
            seed=args.seed,
            validation_difficulty_counts=validation_difficulty_counts,
        )
    except (DatasetBuildError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
