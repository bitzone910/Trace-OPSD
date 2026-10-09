"""Question-level sampling metrics and answer-only mathematical voting.

Pass@k estimates the chance that at least one of k samples is correct. Avg@n
is the mean correctness of all n samples (also sampled Pass@1), not the
accuracy of a separate greedy run. Majority@n chooses the largest equivalent
answer class without consulting the reference answer or sample correctness.
"""

from __future__ import annotations

import math
import os
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from trace_opsd.rollout.verifier import AnswerVerifier, normalize_math_answer

SAMPLING_METRICS_SCHEMA = "trace_opsd.evaluation.sampling-metrics.v1"
VOTING_POLICY = "answer-only-complete-link-bidirectional-earliest-sample.v1"


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Return 1 - C(n-c, k) / C(n, k), with an unchanged n denominator."""
    if any(type(value) is not int for value in (n, c, k)):
        raise ValueError("n, c and k must be integers")
    if n < 1 or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("require n >= 1, 0 <= c <= n and 1 <= k <= n")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _main_thread() -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("sampling scoring and voting must run on the main thread")


def _undefined_rational(value: str) -> bool:
    """Recognize explicitly undefined rational literals without a symbolic call.

    A zero denominator is a malformed model answer, not an unavailable grader.
    Detect only a complete numeric ratio/fraction; do not reject valid infinite
    limits, sets, or a symbolic expression merely containing a division.
    """
    number = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    zero = r"[+-]?(?:0+(?:\.0*)?|\.0+)(?:[eE][+-]?[0-9]+)?"
    ratio = re.fullmatch(rf"(?:{number})/({number})", value)
    fraction = re.fullmatch(
        rf"[+-]?\\frac\{{(?:{number})\}}\{{({number})\}}", value
    )
    match = ratio or fraction
    return bool(match and re.fullmatch(zero, match.group(1)))


def _prediction(benchmark: str, value: Any) -> tuple[str | None, str | None]:
    if value is None or isinstance(value, str) and not value.strip():
        return None, "missing_answer"
    if not isinstance(value, str):
        return None, "invalid_answer_type"
    if benchmark.startswith("aime"):
        text = value.strip()
        if re.fullmatch(r"\+?[0-9]+", text):
            # Bound the conversion too: an adversarial huge integer must not
            # abort scoring or Python's integer-string conversion limit.
            digits = text.lstrip("+").lstrip("0") or "0"
            if len(digits) <= 3 and int(digits) <= 999:
                return str(int(digits)), None
        return None, "invalid_aime_integer"
    normalized = normalize_math_answer(value)
    if normalized is None:
        return None, "invalid_math_answer"
    if _undefined_rational(normalized):
        return None, "invalid_zero_denominator"
    return normalized, None


def _verdict_error(result: Any) -> str | None:
    status = getattr(result, "status", None)
    error = getattr(result, "error", None)
    if status == "error" or error:
        return str(error or "answer verifier returned error status")
    return None


def _rational(value: str) -> Fraction | None:
    """An exact fast path for explicit rational literals, never loose parsing."""
    number = r"[+-]?[0-9]+(?:\.[0-9]+)?"
    if re.fullmatch(number, value):
        return Fraction(value)
    match = re.fullmatch(rf"([+-]?)\\frac\{{({number})\}}\{{({number})\}}", value)
    if match:
        denominator = Fraction(match.group(3))
        if denominator:
            result = Fraction(match.group(2)) / denominator
            return -result if match.group(1) == "-" else result
    return None


def sampling_verifier() -> AnswerVerifier:
    """Reuse conservative project rules with observable symbolic exceptions."""
    from math_verify import LatexExtractionConfig, parse, verify

    timeout = None if os.name == "nt" else 5

    def parse_answer(value: str) -> Any:
        return parse(
            f"${value}$",
            extraction_config=(LatexExtractionConfig(),),
            fallback_mode="no_fallback",
            parsing_timeout=timeout,
            raise_on_error=True,
        )

    def verify_answer(reference: Any, prediction: Any) -> bool:
        return bool(
            verify(reference, prediction, timeout_seconds=timeout, raise_on_error=True)
        )

    return AnswerVerifier(symbolic_parse=parse_answer, symbolic_verify=verify_answer)


def majority_vote(
    benchmark: str,
    predictions: Sequence[tuple[int, str | None]],
    *,
    verifier: AnswerVerifier | None = None,
    _cache: dict[tuple[str, str], tuple[bool, str | None]] | None = None,
) -> dict[str, Any]:
    """Select an answer using predictions only; never inspect ground truth.

    Classes are built in sample-index order. A candidate joins a class only if
    it is equivalent to *every* member in both verifier directions. This avoids
    order-dependent transitive bridges from an imperfect equivalence checker.
    A tie is resolved by the earliest sample index in the tied classes.
    Missing/invalid answers abstain. Verification failures invalidate voting,
    are retained explicitly, and are never treated as ordinary disagreement.
    """
    _main_thread()
    ordered = sorted(predictions, key=lambda item: item[0])
    indices = [item[0] for item in ordered]
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("vote sample indices must be nonnegative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate vote sample index")
    classes: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    cache = _cache if _cache is not None else {}
    active_verifier = verifier

    def equivalent(left: str, right: str, index: int, other: int) -> bool:
        nonlocal active_verifier
        if left == right:
            return True
        if benchmark.startswith("aime"):
            return False
        left_number, right_number = _rational(left), _rational(right)
        if left_number is not None and right_number is not None:
            return left_number == right_number
        for prediction, reference in ((left, right), (right, left)):
            key = (prediction, reference)
            if key not in cache:
                try:
                    if active_verifier is None:
                        active_verifier = sampling_verifier()
                    result = active_verifier.verify(prediction, reference)
                    cache[key] = (bool(result.correct), _verdict_error(result))
                except Exception as exc:
                    cache[key] = (False, f"{type(exc).__name__}: {exc}")
            correct, error = cache[key]
            if error:
                errors.append(
                    {
                        "sample_index": index,
                        "compared_sample_index": other,
                        "prediction": prediction,
                        "reference_prediction": reference,
                        "error": error,
                    }
                )
                return False
            if not correct:
                return False
        return True

    for index, raw in ordered:
        normalized, reason = _prediction(benchmark, raw)
        if normalized is None:
            abstentions.append({"sample_index": index, "reason": reason})
            continue
        for answer_class in classes:
            if all(
                equivalent(
                    normalized,
                    member["normalized_prediction"],
                    index,
                    member["sample_index"],
                )
                for member in answer_class["members"]
            ):
                answer_class["members"].append(
                    {
                        "sample_index": index,
                        "prediction": raw,
                        "normalized_prediction": normalized,
                    }
                )
                break
        else:
            classes.append(
                {
                    "members": [
                        {
                            "sample_index": index,
                            "prediction": raw,
                            "normalized_prediction": normalized,
                        }
                    ],
                }
            )

    class_diagnostics = [
        {
            "sample_indices": [member["sample_index"] for member in group["members"]],
            "count": len(group["members"]),
            "representative_sample_index": group["members"][0]["sample_index"],
            "representative_prediction": group["members"][0]["prediction"],
            "normalized_representative": group["members"][0]["normalized_prediction"],
        }
        for group in classes
    ]
    largest = max((group["count"] for group in class_diagnostics), default=0)
    tied = [group for group in class_diagnostics if group["count"] == largest]
    selected = (
        min(tied, key=lambda group: group["representative_sample_index"])
        if tied
        else None
    )
    available = not errors
    return {
        "policy": VOTING_POLICY,
        "available": available,
        "selected_sample_index": selected["representative_sample_index"]
        if selected and available
        else None,
        "selected_prediction": selected["representative_prediction"]
        if selected and available
        else None,
        "vote_count": largest if available else None,
        "tie": len(tied) > 1,
        "valid_answer_samples": len(ordered) - len(abstentions),
        "abstentions": abstentions,
        "classes": class_diagnostics,
        "errors": errors,
    }


def _sample_diagnostics(
    rows: list[dict[str, Any]], *, questions: int
) -> dict[str, Any]:
    """Reuse existing tool/error accounting, with explicit sample denominators."""
    from .scoring import summarize

    # Generic metric-only callers need not invent unrelated tool metadata.
    # The real runner supplies these fields and its original values are kept.
    prepared = [
        {
            "termination_reason": "unknown",
            "answer_present": isinstance(row.get("prediction"), str)
            and bool(row["prediction"].strip()),
            "format_compliant": False,
            "tool_call_count": 0,
            "python_call_count": 0,
            "search_call_count": 0,
            "tool_error_count": 0,
            "truncated": False,
            **row,
        }
        for row in rows
    ]
    diagnostics = summarize(prepared)
    diagnostics.pop("benchmark")
    diagnostics["correct_samples"] = diagnostics.pop("correct")
    diagnostics.pop("total")
    # These are stochastic samples; presenting this as greedy Pass@1 would
    # compare a different decoding experiment under the same metric name.
    diagnostics.pop("pass_at_1_greedy")
    diagnostics["sample_verification_errors"] = diagnostics.pop("verification_errors")
    tool_calls = sum(int(row["tool_call_count"]) for row in prepared)
    diagnostics["tool_calls_total"] = tool_calls
    for prefix in ("tool", "python", "search"):
        diagnostics[f"{prefix}_calls_per_sample"] = diagnostics.pop(
            f"{prefix}_calls_per_problem"
        )
        calls = tool_calls if prefix == "tool" else diagnostics[f"{prefix}_calls"]
        diagnostics[f"{prefix}_calls_per_question"] = calls / questions
    diagnostics["generation_error_rate"] = diagnostics["generation_errors"] / len(rows)
    diagnostics["truncation_rate"] = diagnostics["truncated"] / len(rows)
    diagnostics["tool_error_rate_per_call"] = (
        diagnostics["tool_errors"] / tool_calls if tool_calls else None
    )
    return diagnostics


def summarize_sampling(
    scored_samples: list[dict[str, Any]],
    *,
    n_samples: int = 8,
    pass_ks: Sequence[int] | None = None,
    verifier: AnswerVerifier | None = None,
) -> dict[str, Any]:
    """Summarize complete sample groups; a missing record is not a failed sample.

    Saved generation failures and absent answers remain in all n denominators.
    A missing sample record or duplicate means the run is incomplete, so this
    function raises. A scorer/voting exception retains diagnostics, marks the
    benchmark unofficial, and exposes provisional values separately; official
    metric fields are null and cannot be confused with an observed zero score.
    """
    _main_thread()
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    ks = (
        tuple(pass_ks)
        if pass_ks is not None
        else tuple(k for k in (1, 2, 4, 8) if k <= n_samples)
    )
    if not ks or any(type(k) is not int or not 1 <= k <= n_samples for k in ks):
        raise ValueError("pass_ks must contain integers between 1 and n_samples")
    if len(set(ks)) != len(ks):
        raise ValueError("duplicate pass_ks")
    ks = tuple(sorted(ks))
    if not scored_samples:
        raise ValueError("cannot summarize empty sampling run")
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in scored_samples:
        benchmark, identifier = row.get("benchmark"), row.get("id")
        index = row.get("sample_index")
        if (
            not isinstance(benchmark, str)
            or not benchmark
            or not isinstance(identifier, str)
            or not identifier
        ):
            raise ValueError("sampling records require nonempty benchmark and id")
        if type(index) is not int or not 0 <= index < n_samples:
            raise ValueError(
                f"invalid sample_index for {benchmark}/{identifier}: {index}"
            )
        if type(row.get("correct")) is not bool:
            raise ValueError("sampling records require a boolean correct verdict")
        if "prediction" not in row or not isinstance(row.get("verification"), dict):
            raise ValueError("sampling records require prediction and verification")
        key = (benchmark, identifier)
        if index in grouped[key]:
            raise ValueError(f"duplicate sample for {benchmark}/{identifier}: {index}")
        grouped[key][index] = row
    expected = set(range(n_samples))
    for (benchmark, identifier), group in grouped.items():
        if set(group) != expected:
            missing = sorted(expected - set(group))
            raise ValueError(f"missing samples for {benchmark}/{identifier}: {missing}")

    question_diagnostics: list[dict[str, Any]] = []
    pair_cache: dict[tuple[str, str], tuple[bool, str | None]] = {}
    benchmark_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (benchmark, identifier), group in sorted(grouped.items()):
        samples = [group[index] for index in range(n_samples)]
        scoring_errors = [
            {
                "sample_index": row["sample_index"],
                "error": row["verification"].get("error")
                or "verification error status",
            }
            for row in samples
            if row["verification"].get("status") == "error"
            or row["verification"].get("error")
        ]
        vote = majority_vote(
            benchmark,
            [(row["sample_index"], row["prediction"]) for row in samples],
            verifier=verifier,
            _cache=pair_cache,
        )
        # Only after answer-only selection may correctness be consulted.
        selected_index = vote["selected_sample_index"]
        majority_correct = (
            bool(group[selected_index]["correct"])
            if selected_index is not None
            else False
        )
        inconsistent = [
            {
                "sample_indices": answer_class["sample_indices"],
                "error": (
                    "equivalent answer class has inconsistent correctness verdicts"
                ),
            }
            for answer_class in vote["classes"]
            if len(
                {group[index]["correct"] for index in answer_class["sample_indices"]}
            )
            > 1
        ]
        invalid_correct = [
            {
                "sample_index": row["sample_index"],
                "error": "missing/invalid answer was marked correct",
            }
            for row in samples
            if row["correct"] and _prediction(benchmark, row["prediction"])[0] is None
        ]
        scoring_errors.extend(inconsistent + invalid_correct)
        c = sum(row["correct"] for row in samples)
        question = {
            "benchmark": benchmark,
            "id": identifier,
            "n": n_samples,
            "correct_samples": c,
            "avg_at_n": c / n_samples,
            "pass_at_n": float(c > 0),
            "pass_at_k": {str(k): unbiased_pass_at_k(n_samples, c, k) for k in ks},
            "majority_correct": majority_correct if vote["available"] else None,
            "official_metrics_available": not scoring_errors and vote["available"],
            "scoring_errors": scoring_errors,
            "vote": vote,
            "generation_error_samples": sum(
                row.get("termination_reason") == "generation_error" for row in samples
            ),
            "truncated_samples": sum(bool(row.get("truncated")) for row in samples),
        }
        question_diagnostics.append(question)
        benchmark_rows[benchmark].append(question)

    summaries: dict[str, Any] = {}
    for benchmark, questions in sorted(benchmark_rows.items()):
        size = len(questions)
        available = all(row["official_metrics_available"] for row in questions)
        vote_available = all(row["vote"]["available"] for row in questions)
        provisional = {
            "pass_at_k": {
                str(k): sum(row["pass_at_k"][str(k)] for row in questions) / size
                for k in ks
            },
            "pass_at_n": sum(row["pass_at_n"] for row in questions) / size,
            "avg_at_n": sum(row["avg_at_n"] for row in questions) / size,
            "majority_at_n": sum(bool(row["majority_correct"]) for row in questions)
            / size
            if vote_available
            else None,
        }
        provisional[f"pass_at_{n_samples}"] = provisional["pass_at_n"]
        provisional[f"avg_at_{n_samples}"] = provisional["avg_at_n"]
        provisional[f"majority_at_{n_samples}"] = provisional["majority_at_n"]
        provisional["sampled_pass_at_1"] = provisional["avg_at_n"]
        for k in ks:
            provisional[f"pass_at_{k}"] = provisional["pass_at_k"][str(k)]
        abstention_reasons = Counter(
            item["reason"] for row in questions for item in row["vote"]["abstentions"]
        )
        summaries[benchmark] = {
            **_sample_diagnostics(
                [
                    row
                    for (name, _), group in grouped.items()
                    if name == benchmark
                    for row in group.values()
                ],
                questions=size,
            ),
            "benchmark": benchmark,
            "questions": size,
            "samples": size * n_samples,
            "n_samples": n_samples,
            "official_metrics_available": available,
            "pass_at_k": provisional["pass_at_k"]
            if available
            else {str(k): None for k in ks},
            **{
                key: value if available else None
                for key, value in provisional.items()
                if key != "pass_at_k"
            },
            "provisional_metrics": provisional,
            "verification_errors": sum(len(row["scoring_errors"]) for row in questions),
            "voting_errors": sum(len(row["vote"]["errors"]) for row in questions),
            "unofficial_questions": sum(
                not row["official_metrics_available"] for row in questions
            ),
            "missing_answer_samples": abstention_reasons.get("missing_answer", 0),
            "invalid_answer_samples": sum(
                count
                for reason, count in abstention_reasons.items()
                if reason != "missing_answer"
            ),
            "abstention_reasons": dict(sorted(abstention_reasons.items())),
            "no_vote_questions": sum(
                row["vote"]["valid_answer_samples"] == 0 for row in questions
            ),
            "tie_questions": sum(row["vote"]["tie"] for row in questions),
            "generation_error_samples": sum(
                row["generation_error_samples"] for row in questions
            ),
            "truncated_samples": sum(row["truncated_samples"] for row in questions),
        }
    official = all(value["official_metrics_available"] for value in summaries.values())
    return {
        "schema": SAMPLING_METRICS_SCHEMA,
        "n_samples": n_samples,
        "pass_ks": list(ks),
        "voting_policy": VOTING_POLICY,
        "complete": official,
        "official_metrics_available": official,
        "benchmarks": summaries,
        "questions": question_diagnostics,
    }
