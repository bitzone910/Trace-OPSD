"""Sampling settings and completeness validation shared by the generic suite."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from trace_opsd.compat import namespace_matches, read_json

from .batch import SuiteError
from .data import BY_NAME, sha256_file, sha256_json
from .runner import EvalConfig
from .sampling_workflow import METRICS_SCHEMA, RUN_SCHEMA

BENCHMARKS = tuple(BY_NAME)


def sampling_settings(
    *,
    n_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    pass_ks: Sequence[int] | None,
) -> tuple[EvalConfig, tuple[int, ...]]:
    if type(n_samples) is not int or n_samples < 2:
        raise ValueError("--samples-per-question must be an integer of at least 2")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("--seed must be an integer in [0, 2**32 - 1]")
    ks = (
        tuple(pass_ks)
        if pass_ks is not None
        else tuple(k for k in (1, 2, 4, 8) if k <= n_samples)
    )
    if (
        not ks
        or len(set(ks)) != len(ks)
        or any(type(k) is not int or not 1 <= k <= n_samples for k in ks)
    ):
        raise ValueError(
            "--pass-ks must be distinct integers from 1 to samples-per-question"
        )
    # Always include Pass@n beside Avg@n/Majority@n, even with a custom curve.
    ks = tuple(sorted({*ks, n_samples}))
    config = EvalConfig(
        protocol="trace_opsd.eval.sampling.v1",
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    return config, tuple(sorted(ks))


def validate_result(
    run_dir: Path,
    *,
    limit_per_benchmark: int | None = None,
    n_samples: int,
    pass_ks: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, Any]:
    """Check every question/sample slot, including a limited diagnostic run."""
    try:
        manifest = read_json(run_dir / 'manifest.json')
        metrics = read_json(run_dir / 'metrics.json')
        if (
            not namespace_matches(manifest.get('schema'), RUN_SCHEMA)
            or not namespace_matches(metrics.get('schema'), METRICS_SCHEMA)
        ):
            raise ValueError("result does not use the sampled schemas")
        if manifest["comparison_fingerprint"] != sha256_json(manifest["evaluation"]):
            raise ValueError("manifest comparison fingerprint is invalid")
        if metrics["comparison_fingerprint"] != manifest["comparison_fingerprint"]:
            raise ValueError("manifest and metrics fingerprints differ")
        sampling = manifest["evaluation"]["sampling"]
        if sampling["n_samples"] != n_samples or sampling["pass_ks"] != list(pass_ks):
            raise ValueError("manifest sampling settings changed")
        if metrics["n_samples"] != n_samples or metrics["pass_ks"] != list(pass_ks):
            raise ValueError("metric sampling settings changed")
        if not metrics["complete"] or not metrics["official_metrics_available"]:
            raise ValueError("scoring/voting failed; official metrics unavailable")
        full = limit_per_benchmark is None or all(
            limit_per_benchmark >= spec.count for spec in BY_NAME.values()
        )
        if bool(metrics["complete_benchmarks"]) != full:
            raise ValueError("limited/full completeness flag is inconsistent")
        if set(metrics["benchmarks"]) != set(BENCHMARKS):
            raise ValueError("result does not contain all five benchmarks")
        for filename, hash_key in (
            ("predictions.jsonl", "predictions_sha256"),
            ("questions.jsonl", "questions_sha256"),
        ):
            if metrics[hash_key] != sha256_file(run_dir / filename):
                raise ValueError(f"{filename} checksum changed")
        expected: dict[str, set[str]] = {}
        for name in BENCHMARKS:
            count = min(limit_per_benchmark or BY_NAME[name].count, BY_NAME[name].count)
            ids = manifest["evaluation"]["datasets"][name]["ids"]
            if len(ids) != count or len(set(ids)) != count:
                raise ValueError(
                    f"{name}: question IDs/count differ from limited/full target"
                )
            expected[name] = set(ids)
            stats = metrics["benchmarks"][name]
            if stats["questions"] != count or stats["samples"] != count * n_samples:
                raise ValueError(f"{name}: question/sample count is incomplete")
            if not stats["official_metrics_available"]:
                raise ValueError(f"{name}: official metrics unavailable")
            for key in (
                f"pass_at_{n_samples}",
                f"avg_at_{n_samples}",
                f"majority_at_{n_samples}",
            ):
                value = stats[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError(f"{name}: invalid {key}")
            if limit_per_benchmark and stats.get("generation_error_samples", 0):
                raise ValueError(f"{name}: generation errors in limited diagnostic run")
            if stats.get("verification_errors", 0) or stats.get("voting_errors", 0):
                raise ValueError(f"{name}: scoring/voting errors")
        prediction_slots: Counter[tuple[str, str, int]] = Counter()
        for line in (
            (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            row = json.loads(line)
            name, identifier, index = row["benchmark"], row["id"], row["sample_index"]
            if (
                name not in expected
                or identifier not in expected[name]
                or type(index) is not int
                or not 0 <= index < n_samples
            ):
                raise ValueError("prediction has an unexpected question/sample slot")
            prediction_slots[name, identifier, index] += 1
        wanted = {
            (name, identifier, index)
            for name, ids in expected.items()
            for identifier in ids
            for index in range(n_samples)
        }
        if set(prediction_slots) != wanted or any(
            count != 1 for count in prediction_slots.values()
        ):
            raise ValueError("predictions contain missing/duplicate sample slots")
        question_slots: Counter[tuple[str, str]] = Counter()
        for line in (
            (run_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            row = json.loads(line)
            question_slots[row["benchmark"], row["id"]] += 1
        if set(question_slots) != {
            (name, identifier) for name, ids in expected.items() for identifier in ids
        } or any(count != 1 for count in question_slots.values()):
            raise ValueError("questions contain missing/duplicate question slots")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SuiteError(f"invalid sampled result at {run_dir}: {exc}") from exc
    return metrics
