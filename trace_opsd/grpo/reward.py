"""Conservative binary answer reward for Trace-OPSD GRPO."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Iterable

from trace_opsd.rollout.parser import extract_last_complete_tagged_answer
from trace_opsd.rollout.symbolic import VERIFIER_REVISION
from trace_opsd.rollout.verifier import AnswerVerifier

_ANSWER_VERIFIER = AnswerVerifier()


def _metric_safe_extra_info(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize optional diagnostics before verl aggregates validation metrics."""

    return {key: "" if value is None else value for key, value in result.items()}


def compute_binary_em(
    solution_str: str,
    ground_truth: Mapping[str, Any],
    *,
    assistant_outputs: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return strict 1/0 correctness plus verifier diagnostics."""

    targets = ground_truth.get("target", ground_truth.get("targets", []))
    answer = (
        extract_last_complete_tagged_answer(assistant_outputs)
        if assistant_outputs is not None
        else extract_last_complete_tagged_answer([solution_str])
    )
    if answer is None:
        missing = _ANSWER_VERIFIER.verify(None, targets)
        return {
            "score": 0.0,
            "verification_revision": VERIFIER_REVISION,
            "acc": 0.0,
            "answer_present": 0.0,
            "extracted_answer": "",
            "verification_method": "tagged_boxed_protocol",
            "verification_status": "invalid_answer_format",
            "normalized_prediction": None,
            "normalized_reference": missing.normalized_reference,
            "verification_error": (
                "a complete <answer>...</answer> block containing a nonempty "
                "\\boxed{...} is required in the assistant rollout"
            ),
        }
    verdict = _ANSWER_VERIFIER.verify(answer, targets)
    score = float(verdict.correct)
    return {
        "score": score,
        "verification_revision": VERIFIER_REVISION,
        "acc": score,
        "answer_present": float(answer is not None),
        "extracted_answer": answer or "",
        "verification_method": verdict.method,
        "verification_status": verdict.status,
        "normalized_prediction": verdict.normalized_prediction,
        "normalized_reference": verdict.normalized_reference,
        "verification_error": verdict.error,
    }


try:
    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
except (ImportError, ModuleNotFoundError):  # pragma: no cover - local CPU environment
    DataProto = Any  # type: ignore[assignment,misc]
    RewardManagerBase = object  # type: ignore[assignment,misc]


class TraceOPSDSearchR1EMRewardManager(RewardManagerBase):  # type: ignore[misc]
    """Reward only assistant-generated tokens, excluding observations and feedback."""

    def __init__(
        self,
        config: Any,
        tokenizer: Any,
        compute_score: Any,
        reward_router_address: str | None = None,
        reward_model_tokenizer: Any | None = None,
    ) -> None:
        if RewardManagerBase is object:  # pragma: no cover - production guard
            raise RuntimeError(
                "Trace-OPSD reward manager requires the vendored verl runtime"
            )
        super().__init__(config, tokenizer, compute_score)
        self.tokenizer = tokenizer

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        data_item = data[-1:][0]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        fields = data_item.non_tensor_batch.get("tool_extra_fields", {})
        if hasattr(fields, "item"):
            fields = fields.item()
        if not isinstance(fields, Mapping):
            fields = {}
        model_token_ids = fields.get("scorable_response_ids")
        if model_token_ids is None:
            raise RuntimeError(
                "Trace-OPSD AgentLoop did not provide scorable_response_ids; refusing "
                "to score observations as model output"
            )
        if hasattr(model_token_ids, "tolist"):
            model_token_ids = model_token_ids.tolist()
        token_ids = [int(value) for value in model_token_ids]
        solution_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(token_ids, skip_special_tokens=True),
        )
        trace_messages = fields.get("trace_messages", [])
        assistant_outputs = [
            str(message.get("content", ""))
            for message in trace_messages
            if isinstance(message, Mapping) and message.get("role") == "assistant"
        ]
        result = await asyncio.to_thread(
            compute_binary_em,
            solution_str,
            ground_truth,
            assistant_outputs=assistant_outputs or None,
        )
        diagnostic_keys = (
            "tool_call_counts",
            "python_call_count",
            "knowledge_call_count",
            "theory_call_count",
            "tool_error_count",
            "invalid_action_count",
            "termination_reason",
        )
        for key in diagnostic_keys:
            if key in fields:
                result[key] = fields[key]
        return {
            "reward_score": result["score"],
            "reward_extra_info": _metric_safe_extra_info(result),
        }


__all__ = ["TraceOPSDSearchR1EMRewardManager", "compute_binary_em"]
