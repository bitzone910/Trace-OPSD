"""Independent, resumable sampled mathematics evaluation and comparisons.

Sampling never rewrites greedy runs. Each sample is an independent production
agent trajectory, and symbolic scoring/voting happens on the main thread after
generation has finished.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from trace_opsd.compat import (
    HISTORICAL_SAMPLING_METRICS_SHA256,
    is_historical_identity,
    namespace_matches,
    read_json,
)
from trace_opsd.inference.vllm.health import check_vllm_health
from trace_opsd.rollout.clients import ToolHTTPBatchClient, VLLMOpenAICompletionsClient
from trace_opsd.rollout.config import ModelConfig, RuntimeConfig, ToolsConfig

from .data import BY_NAME, load_snapshot, sha256_file, sha256_json
from .runner import EvalConfig, evaluate_one
from .sampling_metrics import VOTING_POLICY, sampling_verifier, summarize_sampling
from .scoring import score_record
from .workflow import (
    _execution_identity,
    _resume_manifest_identity,
    _write_json,
    make_manifest,
    resolve_thinking,
)

RUN_SCHEMA = "trace_opsd.evaluation.sampled-run.v1"
METRICS_SCHEMA = "trace_opsd.evaluation.sampled-metrics.v1"
SEED_POLICY = "sha256-json-v1-first64bit-positive-int63"
SAMPLED_PROTOCOL = "trace_opsd.eval.sampling.v1"


def sample_seed(base_seed: int, benchmark: str, row_id: str, sample_index: int) -> int:
    """Same question/sample seed across models, unaffected by scheduling/resume."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base seed must be a nonnegative integer")
    if (
        isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or sample_index < 0
    ):
        raise ValueError("sample index must be a nonnegative integer")
    encoded = json.dumps(
        [SEED_POLICY, base_seed, benchmark, row_id, sample_index],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def _sampling_identity(n_samples: int, pass_ks: tuple[int, ...]) -> dict[str, Any]:
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least 2")
    if (
        not pass_ks
        or len(set(pass_ks)) != len(pass_ks)
        or any(
            isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= n_samples
            for k in pass_ks
        )
    ):
        raise ValueError("pass_ks must contain distinct integers from 1 to n_samples")
    return {
        "n_samples": n_samples,
        "pass_ks": sorted({*pass_ks, n_samples}),
        "seed_policy": SEED_POLICY,
        "voting_policy": VOTING_POLICY,
        "sampling_metrics_sha256": sha256_file(
            Path(__file__).with_name("sampling_metrics.py")
        ),
        "denominator_policy": "all-selected-questions-all-samples-including-failures",
    }


def _record_path(run_dir: Path, benchmark: str, row_id: str, sample_index: int) -> Path:
    if row_id != Path(row_id).name or not row_id.startswith(f"{benchmark}-"):
        raise ValueError(f"invalid benchmark row id: {row_id}")
    return run_dir / "records" / benchmark / row_id / f"sample-{sample_index:03d}.json"


def _validate_record(
    record: dict[str, Any],
    *,
    benchmark: str,
    row_id: str,
    sample_index: int,
    base_seed: int,
    mode: str,
    path: Path,
) -> None:
    if (
        record.get("benchmark") != benchmark
        or record.get("id") != row_id
        or record.get("sample_index") != sample_index
        or type(record.get("sample_index")) is not int
        or record.get("sample_seed")
        != sample_seed(base_seed, benchmark, row_id, sample_index)
        or type(record.get("sample_seed")) is not int
        or record.get("mode") != mode
    ):
        raise ValueError(f"sample record identity or seed mismatch: {path}")


def _validate_manifest(
    manifest: dict[str, Any], *, saved_report: bool = False
) -> dict[str, Any]:
    if not namespace_matches(manifest.get("schema"), RUN_SCHEMA):
        raise ValueError("unsupported sampled evaluation run manifest")
    evaluation = manifest["evaluation"]
    if not namespace_matches(evaluation["config"].get("protocol"), SAMPLED_PROTOCOL):
        raise ValueError("sampled run requires the sampled decoding protocol")
    sampling = evaluation["sampling"]
    expected_sampling = _sampling_identity(
        sampling["n_samples"], tuple(sampling["pass_ks"])
    )
    historical_report = saved_report and is_historical_identity(manifest.get("schema"))
    if historical_report:
        expected_sampling["sampling_metrics_sha256"] = (
            HISTORICAL_SAMPLING_METRICS_SHA256
        )
    if sampling != expected_sampling:
        raise ValueError(
            "sampling scorer or voting revision differs; use a new run directory"
        )
    if manifest["comparison_fingerprint"] != sha256_json(evaluation):
        raise ValueError("sampled manifest comparison fingerprint is invalid")
    if "execution_metadata" in manifest:
        execution = _execution_identity(manifest["execution_metadata"])
        if historical_report:
            recorded_revision = evaluation.get("execution", {}).get("scorer_revision")
            if (
                not isinstance(recorded_revision, str)
                or len(recorded_revision) != 64
                or any(c not in "0123456789abcdef" for c in recorded_revision)
            ):
                raise ValueError("historical scorer revision is invalid")
            # Reporting saved scores does not rerun or replace their scorer.
            execution["scorer_revision"] = recorded_revision
        if execution != evaluation.get("execution"):
            raise ValueError("scorer revision differs from recorded sampled protocol")
    return sampling


async def run_sampled_evaluation(
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
    n_samples: int = 8,
    pass_ks: tuple[int, ...] = (1, 2, 4, 8),
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if concurrency < 1 or (limit_per_benchmark is not None and limit_per_benchmark < 1):
        raise ValueError("concurrency and question limit must be positive")
    config = config or EvalConfig(
        protocol=SAMPLED_PROTOCOL, temperature=0.7, top_p=0.95, top_k=-1
    )
    if config.protocol != SAMPLED_PROTOCOL or config.temperature <= 0:
        raise ValueError("sampled evaluation requires stochastic sampled protocol")
    sampling = _sampling_identity(n_samples, pass_ks)
    if execution_metadata is not None:
        _execution_identity(execution_metadata, concurrency=concurrency)
    snapshot, all_rows = load_snapshot(dataset_dir)
    if (
        not benchmarks
        or len(set(benchmarks)) != len(benchmarks)
        or set(benchmarks) - all_rows.keys()
    ):
        raise ValueError("select distinct benchmarks present in the snapshot")
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
    manifest["schema"] = RUN_SCHEMA
    manifest["evaluation"]["sampling"] = sampling
    manifest["comparison_fingerprint"] = sha256_json(manifest["evaluation"])
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if _resume_manifest_identity(existing) != _resume_manifest_identity(manifest):
            raise ValueError(
                "existing sampled run manifest differs; choose a new run directory"
            )
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"nonempty run directory has no sampled manifest: {run_dir}"
            )
        _write_json(manifest_path, manifest)
    _validate_manifest(manifest)
    pending = []
    expected_paths = set()
    for name, selected in rows.items():
        for row in selected:
            for index in range(n_samples):
                path = _record_path(run_dir, name, row["id"], index)
                expected_paths.add(path)
                if path.exists():
                    _validate_record(
                        read_json(path),
                        benchmark=name,
                        row_id=row["id"],
                        sample_index=index,
                        base_seed=config.seed,
                        mode=mode,
                        path=path,
                    )
                else:
                    pending.append((name, row, index))
    extras = set((run_dir / "records").rglob("*.json")) - expected_paths
    if extras:
        raise ValueError(
            f"unexpected sampled record files: {sorted(map(str, extras))[:3]}"
        )
    selected_count = sum(map(len, rows.values())) * n_samples
    saved_count = selected_count - len(pending)
    print(
        f"Sampled evaluation: {saved_count}/{selected_count} saved; "
        f"{len(pending)} pending",
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

                async def worker(name: str, row: dict[str, Any], index: int):
                    async with semaphore:
                        seed = sample_seed(config.seed, name, row["id"], index)
                        result = await evaluate_one(
                            benchmark=name,
                            row=row,
                            mode=mode,
                            model_name=served_model,
                            tokenizer=tokenizer,
                            generation=generation,
                            tools=tool_client,
                            thinking=thinking,
                            config=replace(config, seed=seed),
                            sample_index=index,
                        )
                        result.update(sample_index=index, sample_seed=seed)
                        return name, row["id"], index, result

                tasks = [asyncio.create_task(worker(*item)) for item in pending]
                try:
                    for task in asyncio.as_completed(tasks):
                        name, row_id, index, result = await task
                        path = _record_path(run_dir, name, row_id, index)
                        _validate_record(
                            result,
                            benchmark=name,
                            row_id=row_id,
                            sample_index=index,
                            base_seed=config.seed,
                            mode=mode,
                            path=path,
                        )
                        _write_json(path, result)
                        saved_count += 1
                        if saved_count % 25 == 0 or saved_count == selected_count:
                            print(
                                f"Sampled evaluation: {saved_count}/{selected_count} "
                                "saved; "
                                f"last={name}/{row_id}/sample-{index:03d}",
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
    print(
        "Scoring and voting saved samples on the main CPU thread...",
        file=sys.stderr,
        flush=True,
    )
    return score_sampled_run(run_dir, dataset_dir)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for value in values:
                stream.write(
                    json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def score_sampled_run(run_dir: Path, dataset_dir: Path) -> dict[str, Any]:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("sampled scoring and voting must run on the main thread")
    if importlib.util.find_spec("math_verify") is None:
        raise RuntimeError("math-verify is required for sampled mathematics scoring")
    run_dir = run_dir.resolve()
    manifest = read_json(run_dir / "manifest.json")
    sampling = _validate_manifest(manifest)
    snapshot, all_rows = load_snapshot(dataset_dir)
    config = manifest["evaluation"]["config"]
    n_samples = sampling["n_samples"]
    verifier = sampling_verifier()
    scored = []
    expected_paths = set()
    for name, selected in manifest["evaluation"]["datasets"].items():
        if snapshot["benchmarks"][name]["normalized_sha256"] != selected["sha256"]:
            raise ValueError(f"dataset changed since sampled run: {name}")
        by_id = {row["id"]: row for row in all_rows[name]}
        if (
            len(set(selected["ids"])) != len(selected["ids"])
            or set(selected["ids"]) - by_id.keys()
        ):
            raise ValueError(f"invalid sampled manifest question IDs: {name}")
        for row_id in selected["ids"]:
            for index in range(n_samples):
                path = _record_path(run_dir, name, row_id, index)
                expected_paths.add(path)
                if not path.is_file():
                    raise RuntimeError(f"sampled evaluation is incomplete: {path}")
                record = read_json(path)
                _validate_record(
                    record,
                    benchmark=name,
                    row_id=row_id,
                    sample_index=index,
                    base_seed=config["seed"],
                    mode=manifest["evaluation"]["mode"],
                    path=path,
                )
                scored.append(
                    score_record(name, by_id[row_id], record, verifier=verifier)
                )
    if set((run_dir / "records").rglob("*.json")) != expected_paths:
        raise ValueError("unexpected sampled record files")
    result = summarize_sampling(
        scored,
        n_samples=n_samples,
        pass_ks=tuple(sampling["pass_ks"]),
        verifier=verifier,
    )
    for name, summary in result["benchmarks"].items():
        overlap = set(snapshot["benchmarks"][name].get("training_overlap_ids", []))
        summary["training_overlap_count"] = len(
            overlap.intersection(manifest["evaluation"]["datasets"][name]["ids"])
        )
    _write_jsonl(run_dir / "predictions.jsonl", scored)
    _write_jsonl(run_dir / "questions.jsonl", result.pop("questions"))
    metrics = {
        **result,
        "schema": METRICS_SCHEMA,
        "total_questions": sum(
            metric["questions"] for metric in result["benchmarks"].values()
        ),
        "total_samples": len(scored),
        "comparison_fingerprint": manifest["comparison_fingerprint"],
        "complete_benchmarks": all(
            len(info["ids"]) == info["full_count"]
            for info in manifest["evaluation"]["datasets"].values()
        ),
        "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        "questions_sha256": sha256_file(run_dir / "questions.jsonl"),
    }
    _write_json(run_dir / "metrics.json", metrics)
    return metrics


def _validate_comparison_counts(
    path: Path, manifest: dict[str, Any], metrics: dict[str, Any]
) -> None:
    """Do not accept a completeness flag without complete distinct sample rows."""
    sampling = manifest["evaluation"]["sampling"]
    n_samples = sampling["n_samples"]
    if (
        metrics.get("n_samples") != n_samples
        or metrics.get("pass_ks") != sampling["pass_ks"]
    ):
        raise ValueError(f"sampled metric configuration differs: {path}")
    datasets = manifest["evaluation"]["datasets"]
    expected = set()
    for name, info in datasets.items():
        if (
            name not in BY_NAME
            or info["full_count"] != BY_NAME[name].count
            or len(info["ids"]) != BY_NAME[name].count
            or len(set(info["ids"])) != len(info["ids"])
        ):
            raise ValueError(
                f"sampled official dataset count is invalid: {path}/{name}"
            )
        expected.update(
            (name, row_id, index)
            for row_id in info["ids"]
            for index in range(n_samples)
        )
    predictions = {}
    with (path / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["benchmark"], row["id"], row["sample_index"])
            if (
                key not in expected
                or key in predictions
                or type(row.get("correct")) is not bool
            ):
                raise ValueError(f"duplicate or invalid sampled prediction: {path}")
            _validate_record(
                row,
                benchmark=key[0],
                row_id=key[1],
                sample_index=key[2],
                base_seed=manifest["evaluation"]["config"]["seed"],
                mode=manifest["evaluation"]["mode"],
                path=path / "predictions.jsonl",
            )
            if row["verification"].get("status") == "error" or row["verification"].get(
                "error"
            ):
                raise ValueError(f"scoring-invalid sampled prediction: {path}")
            predictions[key] = row
    if predictions.keys() != expected:
        raise ValueError(f"sampled prediction coverage is incomplete: {path}")
    questions = {}
    expected_questions = {(name, row_id) for name, row_id, _ in expected}
    with (path / "questions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["benchmark"], row["id"])
            if (
                key not in expected_questions
                or key in questions
                or row.get("n") != n_samples
                or row.get("official_metrics_available") is not True
                or row.get("vote", {}).get("available") is not True
                or type(row.get("majority_correct")) is not bool
            ):
                raise ValueError(f"duplicate or invalid sampled question: {path}")
            questions[key] = row
    if questions.keys() != expected_questions:
        raise ValueError(f"sampled question coverage is incomplete: {path}")
    if (
        metrics.get("total_questions") != len(questions)
        or metrics.get("total_samples") != len(predictions)
        or set(metrics["benchmarks"]) != set(datasets)
    ):
        raise ValueError(f"sampled total counts differ from saved records: {path}")
    for name, info in datasets.items():
        summary = metrics["benchmarks"][name]
        count = len(info["ids"])
        if (
            summary.get("questions") != count
            or summary.get("samples") != count * n_samples
        ):
            raise ValueError(f"sampled benchmark counts differ: {path}/{name}")
        correct = {
            row_id: sum(
                predictions[(name, row_id, index)]["correct"]
                for index in range(n_samples)
            )
            for row_id in info["ids"]
        }
        expected_metrics = {
            f"pass_at_{n_samples}": sum(value > 0 for value in correct.values())
            / count,
            f"avg_at_{n_samples}": sum(correct.values()) / (count * n_samples),
            f"majority_at_{n_samples}": sum(
                questions[(name, row_id)]["majority_correct"] for row_id in info["ids"]
            )
            / count,
        }
        for key, expected_value in expected_metrics.items():
            value = summary.get(key)
            if (
                type(value) not in {float, int}
                or not math.isfinite(value)
                or abs(value - expected_value) > 1e-12
            ):
                raise ValueError(
                    f"sampled {key} differs from saved records: {path}/{name}"
                )


def report_sampled_runs(
    run_dirs: list[Path], output_dir: Path, *, baseline_run_dir: Path | None = None
) -> dict[str, Any]:
    if not run_dirs or len({path.resolve() for path in run_dirs}) != len(run_dirs):
        raise ValueError("select at least one distinct sampled run directory")
    entries = []
    fingerprint = None
    for path in run_dirs:
        manifest = read_json(path / "manifest.json")
        sampling = _validate_manifest(manifest, saved_report=True)
        metrics = read_json(path / "metrics.json")
        if (
            not namespace_matches(metrics.get("schema"), METRICS_SCHEMA)
            or metrics.get("comparison_fingerprint")
            != manifest["comparison_fingerprint"]
        ):
            raise ValueError(f"sampled metrics and manifest differ: {path}")
        for name in ("predictions", "questions"):
            if sha256_file(path / f"{name}.jsonl") != metrics.get(f"{name}_sha256"):
                raise ValueError(f"sampled {name} checksum differs: {path}")
        if (
            not metrics.get("complete_benchmarks")
            or not metrics.get("complete")
            or not metrics.get("official_metrics_available")
        ):
            raise ValueError(
                "partial or scoring-invalid sampled run cannot enter "
                f"official report: {path}"
            )
        if "execution_metadata" not in manifest:
            raise ValueError(
                "official sampled comparison requires verified BF16 "
                f"execution metadata: {path}"
            )
        _validate_comparison_counts(path, manifest, metrics)
        if (
            fingerprint is not None
            and fingerprint != manifest["comparison_fingerprint"]
        ):
            raise ValueError(
                "sampled runs use different datasets or evaluation protocols"
            )
        fingerprint = manifest["comparison_fingerprint"]
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
            raise ValueError("sampled baseline must be included in report runs")
    benchmarks = list(entries[0]["metrics"])
    n_samples = sampling["n_samples"]
    metric_names = [
        f"pass_at_{n_samples}",
        f"avg_at_{n_samples}",
        f"majority_at_{n_samples}",
    ]
    for entry in entries:
        if set(entry["metrics"]) != set(benchmarks):
            raise ValueError("sampled runs contain different benchmark summaries")
        entry["baseline_delta_pp"] = {
            name: {
                metric: 100
                * (entry["metrics"][name][metric] - baseline["metrics"][name][metric])
                if baseline
                else None
                for metric in metric_names
            }
            for name in benchmarks
        }
    result = {
        "schema": "trace_opsd.evaluation.sampled-report.v1",
        "comparison_fingerprint": fingerprint,
        "n_samples": n_samples,
        "baseline_run_dir": baseline["run_dir"] if baseline else None,
        "runs": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "report.json", result)
    headings = [
        "label",
        "benchmark",
        "questions",
        "samples",
        *metric_names,
        *(f"{metric}_baseline_delta_pp" for metric in metric_names),
    ]
    with (output_dir / "report.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headings)
        for entry in entries:
            for name in benchmarks:
                metric = entry["metrics"][name]
                writer.writerow(
                    [
                        entry["label"],
                        name,
                        metric["questions"],
                        metric["samples"],
                        *(metric[key] for key in metric_names),
                        *(
                            entry["baseline_delta_pp"][name][key]
                            for key in metric_names
                        ),
                    ]
                )
    lines = [
        f"# Trace-OPSD sampled evaluation (n={n_samples})",
        "",
        "Pass@n measures whether any sample solves a question. "
        "Avg@n is mean sample accuracy. Majority@n uses answer-only "
        "equivalence voting, without ground-truth answer selection.",
        "",
        "All selected samples, including generation failures, missing answers "
        "and truncation, remain in the denominator.",
        "",
        "| Model | Benchmark | Questions | Samples | Pass@n | Avg@n | Majority@n |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in entries:
        for name in benchmarks:
            metric = entry["metrics"][name]
            lines.append(
                f"| {entry['label']} | {name} | {metric['questions']} | "
                f"{metric['samples']} | "
                + " | ".join(f"{metric[key]:.2%}" for key in metric_names)
                + " |"
            )
    diagnostic_fields = [
        "tool_calls_per_sample",
        "python_calls_per_sample",
        "search_calls_per_sample",
        "tool_errors",
        "tool_error_rate_per_call",
        "user_code_tool_errors",
        "infrastructure_tool_errors",
        "unknown_tool_errors",
        "missing_answers",
        "missing_answer_rate",
        "truncated",
        "truncation_rate",
        "generation_errors",
        "verification_errors",
        "voting_errors",
        "tie_questions",
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
                        *(entry["metrics"][name].get(key) for key in diagnostic_fields),
                    ]
                )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "Calls/sample counts each independent trajectory; these values are not "
            "calls per eight-sample question. Tool error rates use actual call counts.",
            "",
            "| Model | Benchmark | Python/sample | Search/sample | Tool errors | "
            "Missing answers | Truncated samples | Generation errors | Vote ties |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                        f"{metric['python_calls_per_sample']:.3f}",
                        f"{metric['search_calls_per_sample']:.3f}",
                        *(
                            str(metric[key])
                            for key in (
                                "tool_errors",
                                "missing_answers",
                                "truncated",
                                "generation_errors",
                                "tie_questions",
                            )
                        ),
                    ]
                )
                + " |"
            )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
