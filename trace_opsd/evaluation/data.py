"""Download and seal official mathematics benchmark snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from trace_opsd.compat import namespace_matches, read_json


@dataclass(frozen=True)
class Benchmark:
    name: str
    repo: str
    source_file: str
    count: int


BENCHMARKS = (
    Benchmark(
        "aime24", "HuggingFaceH4/aime_2024", "data/train-00000-of-00001.parquet", 30
    ),
    Benchmark("aime25", "MathArena/aime_2025", "data/train-00000-of-00001.parquet", 30),
    Benchmark("aime26", "MathArena/aime_2026", "data/train-00000-of-00001.parquet", 30),
    Benchmark("math500", "HuggingFaceH4/MATH-500", "test.jsonl", 500),
    Benchmark("gsm8k", "openai/gsm8k", "main/test-00000-of-00001.parquet", 1319),
)
BY_NAME = {item.name: item for item in BENCHMARKS}
SCHEMA = "trace_opsd.evaluation.dataset.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_problem(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def _source_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def normalize_rows(
    benchmark: Benchmark, rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(rows, 1):
        problem = source.get("problem", source.get("question"))
        answer = source.get("answer")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"{benchmark.name} row {index}: missing problem")
        if benchmark.name == "gsm8k":
            if not isinstance(answer, str) or "####" not in answer:
                raise ValueError(f"gsm8k row {index}: missing #### final answer")
            answer = answer.rsplit("####", 1)[-1].strip()
        if answer is None or not str(answer).strip():
            raise ValueError(f"{benchmark.name} row {index}: missing answer")
        if benchmark.name.startswith("aime"):
            value = str(answer).strip()
            if not re.fullmatch(r"\d{1,3}", value) or not 0 <= int(value) <= 999:
                raise ValueError(f"{benchmark.name} row {index}: invalid AIME answer")
            answer = str(int(value))
        else:
            answer = str(answer).strip()
        key = normalize_problem(problem)
        if key in seen:
            raise ValueError(f"{benchmark.name} row {index}: duplicate problem")
        seen.add(key)
        metadata = {
            key: value
            for key, value in source.items()
            if key not in {"problem", "question", "answer", "solution"}
        }
        normalized.append(
            {
                "id": f"{benchmark.name}-{index:04d}",
                "problem": problem.strip(),
                "answer": answer,
                "metadata": metadata,
            }
        )
    if len(normalized) != benchmark.count:
        raise ValueError(
            f"{benchmark.name}: expected {benchmark.count}, got {len(normalized)} rows"
        )
    return normalized


def _training_questions(paths: Iterable[Path]) -> set[str]:
    questions: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(path, columns=["problem"])
            values = table.column("problem").to_pylist()
        else:
            values = [row["problem"] for row in _source_rows(path)]
        questions.update(normalize_problem(str(value)) for value in values)
    return questions


def prepare(
    output_dir: Path,
    *,
    names: Iterable[str] | None = None,
    train_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Resolve Hub revisions once and atomically publish an immutable snapshot."""
    from huggingface_hub import HfApi, hf_hub_download

    selected = tuple(names or BY_NAME)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - BY_NAME.keys()
    ):
        raise ValueError("benchmark names must be distinct supported names")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"dataset snapshot already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.{uuid4().hex}.incomplete"
    staging.mkdir()
    api = HfApi()
    train_paths = tuple(train_paths)
    training = _training_questions(train_paths)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "training_sources": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in train_paths
        ],
        "benchmarks": {},
    }
    try:
        for name in selected:
            spec = BY_NAME[name]
            revision = api.dataset_info(spec.repo).sha
            if not revision:
                raise RuntimeError(f"cannot resolve revision of {spec.repo}")
            source = Path(
                hf_hub_download(
                    repo_id=spec.repo,
                    repo_type="dataset",
                    filename=spec.source_file,
                    revision=revision,
                )
            )
            rows = normalize_rows(spec, _source_rows(source))
            collisions = [
                row["id"]
                for row in rows
                if normalize_problem(row["problem"]) in training
            ]
            target = staging / f"{name}.jsonl"
            with target.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
            manifest["benchmarks"][name] = {
                "repo": spec.repo,
                "revision": revision,
                "source_file": spec.source_file,
                "source_sha256": sha256_file(source),
                "normalized_file": target.name,
                "normalized_sha256": sha256_file(target),
                "count": len(rows),
                "training_overlap_count": len(collisions),
                "training_overlap_ids": collisions,
            }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def load_snapshot(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    directory = directory.resolve()
    manifest = read_json(directory / 'manifest.json')
    if not namespace_matches(manifest.get('schema'), SCHEMA):
        raise ValueError("unsupported evaluation dataset manifest")
    output: dict[str, list[dict[str, Any]]] = {}
    for name, info in manifest["benchmarks"].items():
        if name not in BY_NAME or info["count"] != BY_NAME[name].count:
            raise ValueError(f"invalid benchmark metadata: {name}")
        path = directory / info["normalized_file"]
        if sha256_file(path) != info["normalized_sha256"]:
            raise ValueError(f"dataset checksum changed: {path}")
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        if len(rows) != info["count"] or len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"invalid dataset rows: {name}")
        output[name] = rows
    return manifest, output
