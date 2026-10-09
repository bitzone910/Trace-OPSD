"""Python/Search multi-turn AgentLoop adapter for the pinned verl runtime."""

from __future__ import annotations

import time
from typing import Any, Mapping
from uuid import uuid4

from trace_opsd.grpo.batching import get_shared_micro_batcher
from trace_opsd.rollout.models import ActionKind, ToolCall
from trace_opsd.rollout.parser import (
    action_stop_sequences,
    extract_last_complete_tagged_answer,
    parse_first_complete_action,
)
from trace_opsd.rollout.prompts import (
    INVALID_ACTION_FEEDBACK,
    build_final_answer_instruction,
)
from trace_opsd.rollout.token_history import (
    incremental_user_turn_ids,
    trim_after_answer_stop,
)

try:  # Imported by Hydra only in the full verl environment.
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - CPU-only tests
    AgentLoopBase = object  # type: ignore[assignment,misc]
    AgentLoopMetrics = Any  # type: ignore[assignment,misc]
    AgentLoopOutput = Any  # type: ignore[assignment,misc]


def truncate_observation(tokenizer: Any, observation: str, max_tokens: int) -> str:
    ids = list(tokenizer.encode(observation, add_special_tokens=False))
    if len(ids) <= max_tokens:
        return observation
    opening = "<result>\n"
    suffix = "\n...[observation truncated]\n</result>"
    body = observation
    if "<result>" in observation and "</result>" in observation:
        body = observation.split("<result>", 1)[1].rsplit("</result>", 1)[0]
    opening_ids = list(tokenizer.encode(opening, add_special_tokens=False))
    suffix_ids = list(tokenizer.encode(suffix, add_special_tokens=False))
    body_ids = list(tokenizer.encode(body, add_special_tokens=False))
    available = max(0, max_tokens - len(opening_ids) - len(suffix_ids))
    return opening + tokenizer.decode(body_ids[:available]) + suffix


class TraceOPSDAgentLoop(AgentLoopBase):  # type: ignore[misc]
    """Trace-OPSD XML protocol loop with loss masking for environment tokens."""

    def __init__(
        self,
        *args: Any,
        tool_base_url: str = "http://127.0.0.1:8002",
        max_tool_calls: int = 5,
        max_consecutive_invalid_actions: int = 2,
        max_tokens_per_turn: int = 8192,
        max_observation_tokens: int = 2048,
        tool_micro_batch_size: int = 128,
        tool_micro_batch_wait_seconds: float = 0.010,
        tool_request_concurrency: int = 64,
        tool_timeout_seconds: float = 30.0,
        tool_max_retries: int = 2,
        tool_retry_backoff_seconds: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if AgentLoopBase is object:  # pragma: no cover - production guard
            raise RuntimeError("TraceOPSDAgentLoop requires the vendored verl runtime")
        super().__init__(*args, **kwargs)
        self.response_length = int(self.rollout_config.response_length)
        self.max_tool_calls = int(max_tool_calls)
        self.max_consecutive_invalid_actions = int(max_consecutive_invalid_actions)
        if self.max_consecutive_invalid_actions < 1:
            raise ValueError("max_consecutive_invalid_actions must be positive")
        self.max_tokens_per_turn = int(max_tokens_per_turn)
        self.max_observation_tokens = int(max_observation_tokens)
        self._batcher_args = {
            "base_url": tool_base_url,
            "max_batch_size": int(tool_micro_batch_size),
            "max_wait_seconds": float(tool_micro_batch_wait_seconds),
            "request_concurrency": int(tool_request_concurrency),
            "timeout_seconds": float(tool_timeout_seconds),
            "max_retries": int(tool_max_retries),
            "retry_backoff_seconds": float(tool_retry_backoff_seconds),
        }

    async def run(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        trace_messages = [dict(message) for message in messages]
        trace_tool_turns: list[dict[str, Any]] = []
        trace_final_turn: dict[str, int | None] | None = None
        prompt_ids = list(await self.apply_chat_template(messages))
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        scorable_response_ids: list[int] = []
        request_id = uuid4().hex
        tool_counts = {"python": 0, "search": 0}
        tool_errors = 0
        python_error_categories: dict[str, int] = {}
        invalid_actions = 0
        consecutive_invalid_actions = 0
        multiple_actions = 0
        action_diagnostics: list[dict[str, Any]] = []
        assistant_turns = 0
        user_turns = 1
        termination_reason = "missing_answer"
        generation_seconds = 0.0
        tool_seconds = 0.0
        num_preempted = 0
        extra_fields: dict[str, Any] = {}
        logprobs_enabled = bool(sampling_params.get("logprobs", False))
        ended_with_eos = False
        context_suffix_start: int | None = None

        async def generate(*, final: bool) -> tuple[str, int, int, int | None] | None:
            nonlocal generation_seconds, num_preempted, assistant_turns, ended_with_eos
            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                return None
            response_start = len(response_ids)
            # The observation appended after this generation belongs to the next
            # turn, so retain the current user suffix boundary now.
            turn_context_suffix_start = context_suffix_start
            params = dict(sampling_params)
            params.update(
                {
                    "max_tokens": min(self.max_tokens_per_turn, remaining),
                    "stop": ["</answer>"]
                    if final
                    else [*action_stop_sequences(), "</answer>"],
                    "include_stop_str_in_output": True,
                }
            )
            started = time.perf_counter()
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids + response_ids,
                sampling_params=params,
            )
            generation_seconds += time.perf_counter() - started
            assistant_turns += 1
            generated_ids = trim_after_answer_stop(
                self.tokenizer, list(output.token_ids)[:remaining]
            )
            ended_with_eos = bool(
                generated_ids
                and generated_ids[-1] == getattr(self.tokenizer, "eos_token_id", None)
            )
            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))
            scorable_response_ids.extend(generated_ids)
            if output.log_probs:
                response_logprobs.extend(list(output.log_probs)[: len(generated_ids)])
            elif logprobs_enabled:
                response_logprobs.extend([0.0] * len(generated_ids))
            num_preempted += int(output.num_preempted or 0)
            for key, value in output.extra_fields.items():
                if key in {"global_steps", "min_global_steps", "max_global_steps"}:
                    if key in extra_fields and extra_fields[key] != value:
                        raise RuntimeError(
                            "Trace-OPSD trajectory spans multiple policy snapshots: "
                            f"{key} changed from {extra_fields[key]} to {value}"
                        )
                    extra_fields[key] = value
                elif key in {
                    "spec_num_draft_tokens",
                    "spec_num_accepted_tokens",
                    "spec_num_verify_steps",
                }:
                    extra_fields[key] = extra_fields.get(key, 0) + value
                else:
                    extra_fields[key] = value
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            return text, response_start, len(response_ids), turn_context_suffix_start

        def append_environment(content: str) -> bool:
            nonlocal user_turns, context_suffix_start
            ids = incremental_user_turn_ids(
                self.tokenizer,
                content,
                getattr(self, "apply_chat_template_kwargs", {}),
            )
            remaining = self.response_length - len(response_ids)
            if len(ids) > remaining:
                return False
            context_suffix_start = len(response_ids)
            response_ids.extend(ids)
            response_mask.extend([0] * len(ids))
            if response_logprobs:
                response_logprobs.extend([0.0] * len(ids))
            user_turns += 1
            return True

        generation_turn = 0
        while sum(tool_counts.values()) < self.max_tool_calls:
            generated = await generate(final=False)
            if generated is None:
                termination_reason = "missing_boxed_answer"
                break
            text, response_start, response_end, turn_context_suffix_start = generated
            assistant_message_index = len(trace_messages)
            trace_messages.append({"role": "assistant", "content": text})
            action = parse_first_complete_action(text, require_answer_tag=True)
            if action.multiple_actions:
                multiple_actions += 1
            action_diagnostics.append(
                {
                    "response_start": response_start,
                    "response_end": response_end,
                    "multiple_actions": action.multiple_actions,
                    "is_action_valid": action.is_action_valid,
                    "action_kind": action.kind.value,
                }
            )
            generation_turn += 1
            if action.kind is ActionKind.ANSWER:
                if response_end > response_start:
                    trace_final_turn = {
                        "response_start": response_start,
                        "response_end": response_end,
                        "assistant_message_index": assistant_message_index,
                        "context_suffix_start": turn_context_suffix_start,
                    }
                termination_reason = (
                    "answer"
                    if (
                        extract_last_complete_tagged_answer([action.raw_text])
                        is not None
                    )
                    else "invalid_answer_format"
                )
                break
            if action.kind is ActionKind.INVALID or action.tag is None:
                invalid_actions += 1
                consecutive_invalid_actions += 1
                if consecutive_invalid_actions >= self.max_consecutive_invalid_actions:
                    termination_reason = "invalid_action_limit"
                    break
                if not append_environment(INVALID_ACTION_FEEDBACK):
                    termination_reason = "missing_boxed_answer"
                    break
                trace_messages.append(
                    {"role": "user", "content": INVALID_ACTION_FEEDBACK}
                )
                continue

            consecutive_invalid_actions = 0
            tool_counts[action.tag] += 1
            extra_info = kwargs.get("extra_info", {})
            if not isinstance(extra_info, Mapping):
                extra_info = {}
            call = ToolCall(
                request_id=f"{request_id}:tool:{generation_turn - 1}",
                source_id=str(extra_info.get("source_id", request_id)),
                turn=generation_turn - 1,
                tool_name=action.tag,
                content=action.content,
                raw_text=action.raw_text,
            )
            started = time.perf_counter()
            result = await get_shared_micro_batcher(**self._batcher_args).submit(call)
            tool_seconds += time.perf_counter() - started
            if result.error is not None or result.status == "error":
                tool_errors += 1
            error_category = result.error_category
            if error_category is not None:
                python_error_categories[error_category] = (
                    python_error_categories.get(error_category, 0) + 1
                )
            observation = truncate_observation(
                self.tokenizer, result.observation, self.max_observation_tokens
            )
            observation_message_index: int | None = None
            appended_observation = append_environment(observation)
            if appended_observation:
                observation_message_index = len(trace_messages)
                trace_messages.append({"role": "user", "content": observation})
            trace_tool_turns.append(
                {
                    "response_start": response_start,
                    "response_end": response_end,
                    "tool_type": action.tag,
                    "assistant_message_index": assistant_message_index,
                    "context_suffix_start": turn_context_suffix_start,
                    "observation_message_index": observation_message_index,
                    "observation": observation,
                    "status": result.status,
                    "error_category": result.error_category,
                    "error_origin": result.error_origin,
                    "execution_result_version": result.execution_result_version,
                    "transport_error": result.transport_error,
                }
            )
            if not appended_observation:
                termination_reason = "missing_boxed_answer"
                break
        if (
            termination_reason == "missing_answer"
            and sum(tool_counts.values()) >= self.max_tool_calls
        ):
            final_instruction = build_final_answer_instruction()
            if append_environment(final_instruction):
                trace_messages.append({"role": "user", "content": final_instruction})
                generated = await generate(final=True)
                if generated is None:
                    termination_reason = "missing_boxed_answer"
                else:
                    (
                        text,
                        response_start,
                        response_end,
                        turn_context_suffix_start,
                    ) = generated
                    assistant_message_index = len(trace_messages)
                    trace_messages.append({"role": "assistant", "content": text})
                    if response_end > response_start:
                        trace_final_turn = {
                            "response_start": response_start,
                            "response_end": response_end,
                            "assistant_message_index": assistant_message_index,
                            "context_suffix_start": turn_context_suffix_start,
                        }
                if generated is not None and (
                    extract_last_complete_tagged_answer([text]) is not None
                ):
                    termination_reason = "final_answer"
                elif generated is not None:
                    termination_reason = "missing_boxed_answer"
            else:
                termination_reason = "missing_boxed_answer"

        diagnostics = {
            "scorable_response_ids": scorable_response_ids,
            "tool_call_counts": sum(tool_counts.values()),
            "python_call_count": tool_counts["python"],
            "search_call_count": tool_counts["search"],
            "tool_error_count": tool_errors,
            "python_error_categories": python_error_categories,
            "invalid_action_count": invalid_actions,
            "ended_with_eos": ended_with_eos,
            "multiple_action_count": multiple_actions,
            "action_diagnostics": action_diagnostics,
            "termination_reason": termination_reason,
            "turn_scores": [],
            "tool_rewards": [],
            "trace_messages": trace_messages,
            "trace_tool_turns": trace_tool_turns,
            "trace_final_turn": trace_final_turn,
        }
        extra_fields.update(diagnostics)
        metrics = AgentLoopMetrics(
            generate_sequences=generation_seconds,
            tool_calls=tool_seconds,
            num_preempted=num_preempted,
        )
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else None,
            num_turns=user_turns + assistant_turns,
            metrics=metrics,
            extra_fields=extra_fields,
        )


__all__ = [
    "TraceOPSDAgentLoop",
    "incremental_user_turn_ids",
    "truncate_observation",
]
