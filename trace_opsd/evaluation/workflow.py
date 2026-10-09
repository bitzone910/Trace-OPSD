"""Resumable evaluation runs, offline rescoring, and comparable reports."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from trace_opsd.compat import metadata_path, namespace_matches, read_json
from trace_opsd.inference.vllm.health import check_vllm_health
from trace_opsd.rollout.clients import ToolHTTPBatchClient, VLLMOpenAICompletionsClient
from trace_opsd.rollout.config import ModelConfig, RuntimeConfig, ToolsConfig
from trace_opsd.rollout.prompts import (
    INVALID_ACTION_FEEDBACK,
    build_direct_prompt,
    build_final_answer_instruction,
    build_tool_prompt,
)

from .data import BY_NAME, load_snapshot, sha256_file, sha256_json
from .runner import EvalConfig, evaluate_one
from .scoring import evaluation_verifier, score_record, summarize

RUN_SCHEMA = "trace_opsd.evaluation.run.v1"
EXECUTION_SCHEMA = "trace_opsd.evaluation.execution.v1"


def _execution_identity(
    metadata: dict[str, Any], *, concurrency: int | None = None
) -> dict[str, Any]:
    """Validate serving facts, excluding model identity and log evidence paths."""
    if not namespace_matches(metadata.get("schema"), EXECUTION_SCHEMA):
        raise ValueError("unsupported evaluation execution metadata")
    inference = metadata.get("inference")
    backend = metadata.get("backend")
    verification = metadata.get("verification")
    if not isinstance(inference, dict) or not isinstance(backend, dict):
        raise ValueError("execution metadata requires inference and backend objects")
    if (
        inference.get("dtype") != "bfloat16"
        or "quantization" not in inference
        or inference["quantization"] is not None
    ):
        raise ValueError("evaluation requires bfloat16 inference without quantization")
    if (
        not isinstance(verification, dict)
        or verification.get("actual_dtype") != "bfloat16"
    ):
        raise ValueError(
            "execution metadata must verify the actual bfloat16 serving dtype"
        )
    for key in ("tensor_parallel_size", "concurrency"):
        value = inference.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"invalid execution inference {key}")
    if concurrency is not None and inference["concurrency"] != concurrency:
        raise ValueError("execution concurrency differs from evaluation concurrency")
    for key in ("vllm", "torch"):
        if not isinstance(backend.get(key), str) or not backend[key]:
            raise ValueError(f"execution metadata requires backend {key} version")
    if "cuda" not in backend or not (
        backend["cuda"] is None or isinstance(backend["cuda"], str)
    ):
        raise ValueError("execution metadata requires backend cuda version")
    return {
        "inference": {
            key: inference[key]
            for key in ("dtype", "tensor_parallel_size", "quantization", "concurrency")
        },
        "backend": {key: backend[key] for key in ("vllm", "torch", "cuda")},
        "scorer_revision": sha256_json(
            {
                "scoring": sha256_file(Path(__file__).with_name("scoring.py")),
                "verifier": sha256_file(
                    Path(__file__).parents[1] / "rollout" / "verifier.py"
                ),
                "math_verify": _package_version("math-verify"),
            }
        ),
    }


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _resume_manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    # Fresh serving logs contain timestamps/PIDs. Serving facts must still match,
    # but evidence text is retained for audit rather than blocking safe resume.
    identity = json.loads(json.dumps(manifest))
    execution = identity.get("execution_metadata")
    if isinstance(execution, dict) and isinstance(execution.get("verification"), dict):
        execution["verification"].pop("evidence", None)
    return identity


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _model_identity(model_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {model_path}")
    names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "artifact-manifest.json",
        "trace_opsd_protocol.json",
        "model.safetensors.index.json",
    )
    files = {
        metadata_path(model_path / name).name: sha256_file(
            metadata_path(model_path / name)
        )
        for name in names
        if metadata_path(model_path / name).is_file()
    }
    weights = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(
            [*model_path.glob("*.safetensors"), *model_path.glob("pytorch_model*.bin")]
        )
    ]
    if not weights:
        raise ValueError(f"no Hugging Face model weights found: {model_path}")
    return {"path": str(model_path), "files": files, "weights": weights}


def _directory_identity(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"retriever model directory not found: {directory}")
    files = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise ValueError(f"empty retriever model directory: {directory}")
    return {"path": str(directory), "files": files}


def resolve_thinking(model_path: Path, requested: str) -> str:
    if requested not in {"auto", "on", "off"}:
        raise ValueError("thinking mode must be auto, on, or off")
    model_type = read_json(model_path / "config.json").get("model_type")
    if requested == "on" and model_type != "qwen3":
        raise ValueError("thinking mode can only be enabled for Qwen3")
    if requested == "auto":
        return "on" if model_type == "qwen3" else "off"
    return requested if model_type == "qwen3" else "none"


def make_manifest(
    *,
    snapshot: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    model_path: Path,
    mode: str,
    thinking: str,
    thinking_policy: str = "auto",
    config: EvalConfig,
    served_model: str,
    model_url: str,
    tool_url: str | None,
    index_path: Path | None,
    corpus_path: Path | None,
    tool_service: dict[str, Any] | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in {"tool", "direct"}:
        raise ValueError("mode must be tool or direct")
    if mode == "tool" and (
        index_path is None
        or corpus_path is None
        or tool_url is None
        or tool_service is None
    ):
        raise ValueError("tool mode requires tool URL, index path, and corpus path")
    for name in rows:
        if name not in snapshot["benchmarks"]:
            raise ValueError(f"benchmark is not in snapshot: {name}")
    service_config = tool_service.get("evaluation_config") if tool_service else None
    if mode == "tool":
        if not isinstance(service_config, dict) or not service_config.get("model_path"):
            raise ValueError("tool health did not include evaluation_config")
        if Path(service_config["general_index_path"]).resolve() != index_path.resolve():
            raise ValueError("tool service uses a different search index")
        if (
            Path(service_config["general_corpus_path"]).resolve()
            != corpus_path.resolve()
        ):
            raise ValueError("tool service uses a different search corpus")
    tool_identity = (
        {
            "url": tool_url,
            "index_sha256": sha256_file(index_path),
            "corpus_sha256": sha256_file(corpus_path),
            "retriever": _directory_identity(Path(service_config["model_path"])),
            "service_version": tool_service.get("service_version"),
            "service_config": service_config,
        }
        if mode == "tool" and index_path is not None and corpus_path is not None
        else None
    )
    prompt_identity = sha256_json(
        {
            "tool": build_tool_prompt("PROBLEM"),
            "direct": build_direct_prompt("PROBLEM"),
            "invalid_action_feedback": INVALID_ACTION_FEEDBACK,
            "final_answer_instruction": build_final_answer_instruction(),
            "protocol": config.protocol,
        }
    )
    evaluation = {
        "datasets": {
            name: {
                "sha256": snapshot["benchmarks"][name]["normalized_sha256"],
                "ids": [row["id"] for row in selected],
                "full_count": BY_NAME[name].count,
            }
            for name, selected in rows.items()
        },
        "mode": mode,
        "thinking_policy": thinking_policy,
        "config": config.to_dict(),
        "prompt_sha256": prompt_identity,
        "tool": (
            {key: value for key, value in tool_identity.items() if key != "url"}
            if tool_identity is not None
            else None
        ),
    }
    model = _model_identity(model_path) if execution_metadata is None else None
    if execution_metadata is not None:
        from .inventory import inspect_hf_model

        inspected = inspect_hf_model(model_path)
        model = {
            **inspected["model_identity"],
            "storage_dtype_counts": inspected["storage_dtype_counts"],
        }
        evaluation["execution"] = _execution_identity(execution_metadata)
    manifest = {
        "schema": RUN_SCHEMA,
        "evaluation": evaluation,
        "comparison_fingerprint": sha256_json(evaluation),
        "model": model,
        "thinking": thinking,
        "tool_service": tool_identity,
        "served_model": served_model,
        "model_url": model_url,
    }
    if execution_metadata is not None:
        manifest["execution_metadata"] = execution_metadata
    return manifest


def _record_path(run_dir: Path, benchmark: str, row_id: str) -> Path:
    if row_id != Path(row_id).name or not row_id.startswith(f"{benchmark}-"):
        raise ValueError(f"invalid benchmark row id: {row_id}")
    return run_dir / "records" / benchmark / f"{row_id}.json"


async def run_evaluation(
    *,
    run_dir: Path,
    dataset_dir: Path,
    benchmarks: list[str],
    model_path: Path,
    served_model: str,
    model_url: str,
    tool_url: str | None,
    index_path: Path | None,
    corpus_path: Path | None,
    mode: str = "tool",
    thinking_mode: str = "auto",
    limit_per_benchmark: int | None = None,
    concurrency: int = 16,
    config: EvalConfig | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if concurrency < 1 or (limit_per_benchmark is not None and limit_per_benchmark < 1):
        raise ValueError("concurrency and sample limit must be positive")
    config = config or EvalConfig()
    if execution_metadata is not None:
        _execution_identity(execution_metadata, concurrency=concurrency)
    snapshot, all_rows = load_snapshot(dataset_dir)
    if len(set(benchmarks)) != len(benchmarks) or not benchmarks:
        raise ValueError("select distinct benchmarks")
    rows = {name: all_rows[name][:limit_per_benchmark] for name in benchmarks}
    model_path = model_path.resolve()
    thinking = resolve_thinking(model_path, thinking_mode)
    tool_service = None
    if mode == "tool":
        if tool_url is None:
            raise ValueError("tool mode requires a tool URL")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{tool_url.rstrip('/')}/health")
            response.raise_for_status()
            tool_service = response.json()
    manifest = make_manifest(
        snapshot=snapshot,
        rows=rows,
        model_path=model_path,
        mode=mode,
        thinking=thinking,
        thinking_policy=thinking_mode,
        config=config,
        served_model=served_model,
        model_url=model_url,
        tool_url=tool_url,
        index_path=index_path,
        corpus_path=corpus_path,
        tool_service=tool_service,
        execution_metadata=execution_metadata,
    )
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if _resume_manifest_identity(existing) != _resume_manifest_identity(manifest):
            raise ValueError(
                "existing run manifest differs; choose a new run directory"
            )
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"nonempty run directory has no manifest: {run_dir}")
        _write_json(manifest_path, manifest)
    pending: list[tuple[str, dict[str, Any]]] = []
    for name, selected in rows.items():
        for row in selected:
            record_path = _record_path(run_dir, name, row["id"])
            if record_path.exists():
                saved = read_json(record_path)
                if saved.get("benchmark") != name or saved.get("id") != row["id"]:
                    raise ValueError(f"invalid saved record: {record_path}")
            else:
                pending.append((name, row))
    selected_count = sum(len(selected) for selected in rows.values())
    saved_count = selected_count - len(pending)
    print(
        f"Evaluation progress: {saved_count}/{selected_count} saved; "
        f"{len(pending)} questions pending",
        file=sys.stderr,
        flush=True,
    )
    if pending:
        health = await check_vllm_health(model_url, expected_model=served_model)
        if not health.healthy:
            raise RuntimeError(f"vLLM is not ready: {health.error}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        runtime = RuntimeConfig(
            request_concurrency=concurrency, fallback_concurrency=min(8, concurrency)
        )
        model_cfg = ModelConfig(
            name_or_path=str(model_path),
            served_model_name=served_model,
            base_url=model_url,
        )
        tools_cfg = ToolsConfig(base_url=tool_url) if mode == "tool" else None
        semaphore = asyncio.Semaphore(concurrency)
        async with VLLMOpenAICompletionsClient(model_cfg, runtime) as generation:
            tool_client = ToolHTTPBatchClient(tools_cfg, runtime) if tools_cfg else None
            if tool_client is not None:
                await tool_client.__aenter__()
            try:

                async def worker(
                    name: str, row: dict[str, Any]
                ) -> tuple[str, str, dict[str, Any]]:
                    async with semaphore:
                        result = await evaluate_one(
                            benchmark=name,
                            row=row,
                            mode=mode,
                            model_name=served_model,
                            tokenizer=tokenizer,
                            generation=generation,
                            tools=tool_client,
                            thinking=thinking,
                            config=config,
                        )
                        return name, row["id"], result

                tasks = [
                    asyncio.create_task(worker(name, row)) for name, row in pending
                ]
                try:
                    for task in asyncio.as_completed(tasks):
                        name, row_id, result = await task
                        _write_json(_record_path(run_dir, name, row_id), result)
                        saved_count += 1
                        if saved_count % 25 == 0 or saved_count == selected_count:
                            print(
                                f"Evaluation progress: {saved_count}/{selected_count} "
                                f"saved; last={name}/{row_id}",
                                file=sys.stderr,
                                flush=True,
                            )
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                if tool_client is not None:
                    await tool_client.aclose()
    print("Scoring saved trajectories on CPU...", file=sys.stderr, flush=True)
    return score_run(run_dir, dataset_dir)


def score_run(run_dir: Path, dataset_dir: Path) -> dict[str, Any]:
    if importlib.util.find_spec("math_verify") is None:
        raise RuntimeError("math-verify is required for official mathematics scoring")
    run_dir = run_dir.resolve()
    manifest = read_json(run_dir / "manifest.json")
    if not namespace_matches(manifest.get("schema"), RUN_SCHEMA):
        raise ValueError("unsupported evaluation run manifest")
    if "execution_metadata" in manifest:
        current_execution = _execution_identity(manifest["execution_metadata"])
        if current_execution != manifest["evaluation"].get("execution"):
            raise ValueError(
                "scorer revision differs from the recorded evaluation protocol; "
                "use a new evaluation run directory"
            )
    snapshot, all_rows = load_snapshot(dataset_dir)
    scored: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    verifier = evaluation_verifier()
    for name, selected in manifest["evaluation"]["datasets"].items():
        if snapshot["benchmarks"][name]["normalized_sha256"] != selected["sha256"]:
            raise ValueError(f"dataset changed since run: {name}")
        by_id = {row["id"]: row for row in all_rows[name]}
        group = []
        for row_id in selected["ids"]:
            path = _record_path(run_dir, name, row_id)
            if not path.is_file():
                raise RuntimeError(f"evaluation is incomplete: {path}")
            record = read_json(path)
            if record.get("id") != row_id or record.get("benchmark") != name:
                raise ValueError(f"record identity mismatch: {path}")
            group.append(score_record(name, by_id[row_id], record, verifier=verifier))
        scored.extend(group)
        summaries[name] = summarize(group)
        overlaps = set(snapshot["benchmarks"][name].get("training_overlap_ids", []))
        summaries[name]["training_overlap_count"] = len(
            overlaps.intersection(selected["ids"])
        )
    predictions = run_dir / "predictions.jsonl"
    temporary = predictions.with_name(f".{predictions.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in scored:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, predictions)
    finally:
        temporary.unlink(missing_ok=True)
    metrics = {
        "schema": "trace_opsd.evaluation.metrics.v1",
        "comparison_fingerprint": manifest["comparison_fingerprint"],
        "complete_benchmarks": all(
            len(info["ids"]) == info["full_count"]
            for info in manifest["evaluation"]["datasets"].values()
        ),
        "benchmarks": summaries,
        "predictions_sha256": sha256_file(predictions),
    }
    _write_json(run_dir / "metrics.json", metrics)
    return metrics


def report(
    run_dirs: list[Path], output_dir: Path, *, baseline_run_dir: Path | None = None
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one evaluation run is required")
    entries = []
    fingerprint: str | None = None
    for path in run_dirs:
        manifest = read_json(path / "manifest.json")
        metrics = read_json(path / "metrics.json")
        if metrics["comparison_fingerprint"] != manifest["comparison_fingerprint"]:
            raise ValueError(f"metrics and manifest fingerprints differ: {path}")
        if sha256_file(path / "predictions.jsonl") != metrics["predictions_sha256"]:
            raise ValueError(f"predictions checksum differs from metrics: {path}")
        if not metrics["complete_benchmarks"]:
            raise ValueError(f"partial benchmark cannot enter main report: {path}")
        if any(
            stats.get("verification_errors", 0)
            for stats in metrics["benchmarks"].values()
        ):
            raise ValueError(
                f"mathematical scoring errors cannot enter official report: {path}"
            )
        current = manifest["comparison_fingerprint"]
        if fingerprint is not None and current != fingerprint:
            raise ValueError("runs use different datasets or evaluation protocols")
        fingerprint = current
        entries.append(
            {
                "label": path.name,
                "run_dir": str(path.resolve()),
                "model": manifest["model"],
                "metrics": metrics["benchmarks"],
            }
        )
    baseline = None
    if baseline_run_dir is not None:
        baseline = next(
            (
                entry
                for entry in entries
                if Path(entry["run_dir"]) == baseline_run_dir.resolve()
            ),
            None,
        )
        if baseline is None:
            raise ValueError("baseline run must be included in report runs")
    else:
        candidates = [entry for entry in entries if entry["label"] == "baseline"]
        if len(candidates) > 1:
            raise ValueError("multiple baseline runs; specify baseline_run_dir")
        if candidates:
            baseline = candidates[0]
    benchmarks = list(entries[0]["metrics"])
    for entry in entries:
        if set(entry["metrics"]) != set(benchmarks):
            raise ValueError("runs contain different benchmark summaries")
        entry["baseline_delta_pp"] = {
            name: (
                100
                * (
                    entry["metrics"][name]["pass_at_1_greedy"]
                    - baseline["metrics"][name]["pass_at_1_greedy"]
                )
                if baseline
                else None
            )
            for name in benchmarks
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "comparison_fingerprint": fingerprint,
        "baseline_run_dir": baseline["run_dir"] if baseline else None,
        "runs": entries,
    }
    _write_json(output_dir / "report.json", result)
    headings = ["Model", *(f"{name} pass@1 / Δpp" for name in benchmarks)]
    csv_headings = ["label", "model"]
    for name in benchmarks:
        csv_headings.extend(
            [
                f"{name}_correct",
                f"{name}_total",
                f"{name}_pass@1",
                f"{name}_baseline_delta_pp",
            ]
        )
    with (output_dir / "report.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(csv_headings)
        for entry in entries:
            values = [entry["label"], entry["model"]["path"]]
            for name in benchmarks:
                metric = entry["metrics"][name]
                values.extend(
                    [
                        metric["correct"],
                        metric["total"],
                        metric["pass_at_1_greedy"],
                        entry["baseline_delta_pp"][name],
                    ]
                )
            writer.writerow(values)
    lines = [
        "# Trace-OPSD greedy pass@1",
        "",
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
    ]
    for entry in entries:
        lines.append(
            "| "
            + " | ".join(
                [entry["label"]]
                + [
                    "{}/{} ({:.2%}){}".format(
                        entry["metrics"][name]["correct"],
                        entry["metrics"][name]["total"],
                        entry["metrics"][name]["pass_at_1_greedy"],
                        (
                            f" / {entry['baseline_delta_pp'][name]:+.2f}"
                            if baseline
                            else ""
                        ),
                    )
                    for name in benchmarks
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "Δpp is the accuracy difference from baseline in percentage points. "
            "Unavailable means the historical record did not store this diagnostic.",
            "",
            "| Model | Benchmark | Python/q | Search/q | Tool errors "
            "(user / infrastructure / unknown) | No answer | Truncated | "
            "Generation errors | Verification errors | Training overlaps |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for entry in entries:
        for name in benchmarks:
            metric = entry["metrics"][name]
            lines.append(
                "| "
                + " | ".join(
                    [
                        entry["label"],
                        name,
                        _report_number(metric.get("python_calls_per_problem")),
                        _report_number(metric.get("search_calls_per_problem")),
                        "{} ({} / {} / {})".format(
                            metric["tool_errors"],
                            metric.get("user_code_tool_errors", "unavailable"),
                            metric.get("infrastructure_tool_errors", "unavailable"),
                            metric.get("unknown_tool_errors", "unavailable"),
                        ),
                        str(metric.get("missing_answers", "unavailable")),
                        str(metric["truncated"]),
                        str(metric["generation_errors"]),
                        str(metric.get("verification_errors", "unavailable")),
                        str(metric.get("training_overlap_count", "unavailable")),
                    ]
                )
                + " |"
            )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    diagnostic_fields = [
        "python_calls",
        "search_calls",
        "python_calls_per_problem",
        "search_calls_per_problem",
        "tool_errors",
        "user_code_tool_errors",
        "infrastructure_tool_errors",
        "unknown_tool_errors",
        "missing_answers",
        "missing_answer_rate",
        "truncated",
        "generation_errors",
        "verification_errors",
        "answer_present_rate",
        "format_compliant_rate",
        "training_overlap_count",
    ]
    with (output_dir / "diagnostics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "benchmark", *diagnostic_fields])
        for entry in entries:
            for name in benchmarks:
                writer.writerow(
                    [
                        entry["label"],
                        name,
                        *(
                            entry["metrics"][name].get(field, "unavailable")
                            for field in diagnostic_fields
                        ),
                    ]
                )
    return result


def _report_number(value: Any) -> str:
    return "unavailable" if value is None else f"{value:.3f}"
