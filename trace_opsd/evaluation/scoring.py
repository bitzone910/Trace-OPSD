"""Offline answer scoring and benchmark summaries."""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any

from trace_opsd.rollout.parser import (
    extract_last_complete_boxed,
    extract_last_complete_tagged_answer,
)
from trace_opsd.rollout.verifier import AnswerVerifier


def evaluation_verifier() -> AnswerVerifier:
    """Use Trace-OPSD's verifier with math-verify parsing isolated answers as LaTeX."""
    from math_verify import parse, verify

    timeout = None if os.name == "nt" else 5

    def parse_answer(value: str) -> Any:
        expression = value.strip()
        if not expression.startswith("$") or not expression.endswith("$"):
            expression = f"${expression}$"
        return parse(expression, parsing_timeout=timeout)

    def verify_answer(reference: Any, prediction: Any) -> bool:
        return bool(verify(reference, prediction, timeout_seconds=timeout))

    return AnswerVerifier(
        symbolic_parse=parse_answer,
        symbolic_verify=verify_answer,
    )


def score_record(
    benchmark: str,
    row: dict[str, Any],
    record: dict[str, Any],
    *,
    verifier: AnswerVerifier | None = None,
) -> dict[str, Any]:
    """Score the terminal model message, never a tool result or earlier turn."""
    final = record.get("final_response")
    final = final if isinstance(final, str) else ""
    prediction = extract_last_complete_boxed(final)
    compliant = extract_last_complete_tagged_answer([final]) is not None
    if benchmark.startswith("aime"):
        valid_integer = prediction is not None and re.fullmatch(
            r"\+?\d+", prediction.strip()
        )
        correct = bool(
            valid_integer
            and 0 <= int(prediction.strip()) <= 999
            and int(prediction.strip()) == int(row["answer"])
        )
        verification = {
            "method": "aime_integer",
            "status": "ok" if valid_integer else "invalid_or_missing_integer",
            "normalized_prediction": str(int(prediction.strip()))
            if valid_integer
            else None,
            "normalized_reference": str(int(row["answer"])),
        }
    else:
        try:
            verdict = (verifier or AnswerVerifier()).verify(prediction, row["answer"])
            correct = verdict.correct
            verification = verdict.to_dict()
        except Exception as exc:
            # A broken scorer is observable and stays in the fixed denominator.
            # It must not silently turn into an ordinary wrong answer.
            correct = False
            verification = {
                "method": "verification_error",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {
        **record,
        "answer": row["answer"],
        "prediction": prediction,
        "correct": correct,
        "answer_present": prediction is not None,
        "format_compliant": compliant,
        "verification": verification,
    }


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored:
        raise ValueError("cannot summarize empty benchmark")
    names = {row["benchmark"] for row in scored}
    if len(names) != 1:
        raise ValueError("summary requires one benchmark")
    count = len(scored)
    correct = sum(bool(row["correct"]) for row in scored)
    reasons = Counter(str(row["termination_reason"]) for row in scored)
    missing_answers = sum(not bool(row["answer_present"]) for row in scored)
    verification_details = [
        {
            "id": row.get("id"),
            "method": row.get("verification", {}).get("method"),
            "status": row.get("verification", {}).get("status"),
            "error": row.get("verification", {}).get("error"),
        }
        for row in scored
        if row.get("verification", {}).get("status") == "error"
        or row.get("verification", {}).get("error")
    ]
    verification_categories = Counter(
        (str(item["error"]).split(":", 1)[0] if item["error"] else item["method"])
        or "unknown"
        for item in verification_details
    )
    tool_errors = sum(int(row["tool_error_count"]) for row in scored)
    events = [event for row in scored for event in row.get("tool_events", [])]
    failed_events = [event for event in events if event.get("error")]
    user_errors = sum(
        event.get("error_origin") in {"user_code", "execution_limit"}
        and bool(event.get("execution_result_version") or event.get("protocol_version"))
        for event in failed_events
    )
    infrastructure_errors = sum(
        event.get("error_origin") in {"queue", "bootstrap", "sandbox", "transport"}
        for event in failed_events
    )
    unknown_errors = max(0, tool_errors - user_errors - infrastructure_errors)
    tool_error_categories = Counter(
        str(event.get("error_category") or "unknown") for event in failed_events
    )
    python_calls = sum(int(row["python_call_count"]) for row in scored)
    search_calls = sum(int(row["search_call_count"]) for row in scored)
    return {
        "benchmark": scored[0]["benchmark"],
        "correct": correct,
        "total": count,
        "pass_at_1_greedy": correct / count,
        "answer_present_rate": sum(bool(row["answer_present"]) for row in scored)
        / count,
        "missing_answers": missing_answers,
        "missing_answer_rate": missing_answers / count,
        "format_compliant_rate": sum(bool(row["format_compliant"]) for row in scored)
        / count,
        "tool_calls_per_problem": sum(int(row["tool_call_count"]) for row in scored)
        / count,
        "python_calls": python_calls,
        "python_calls_per_problem": python_calls / count,
        "search_calls": search_calls,
        "search_calls_per_problem": search_calls / count,
        "generation_errors": reasons.get("generation_error", 0),
        "verification_errors": len(verification_details),
        "verification_error_rate": len(verification_details) / count,
        "verification_error_categories": dict(sorted(verification_categories.items())),
        "verification_error_details": verification_details,
        "tool_errors": tool_errors,
        "user_code_tool_errors": user_errors,
        "infrastructure_tool_errors": infrastructure_errors,
        "unknown_tool_errors": unknown_errors,
        "tool_error_metadata_complete": unknown_errors == 0,
        "tool_error_categories": dict(sorted(tool_error_categories.items())),
        "truncated": sum(bool(row["truncated"]) for row in scored),
        "termination_reasons": dict(sorted(reasons.items())),
    }
