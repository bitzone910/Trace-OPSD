"""Standalone inference using Trace-OPSD's production protocol pieces."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from trace_opsd.compat import namespace_matches
from trace_opsd.rollout.models import ActionKind, GenerationRequest, ToolCall
from trace_opsd.rollout.parser import action_stop_sequences, parse_first_complete_action
from trace_opsd.rollout.prompts import (
    INVALID_ACTION_FEEDBACK,
    build_direct_messages,
    build_final_answer_instruction,
    build_tool_messages,
)
from trace_opsd.rollout.token_history import (
    exact_token_ids,
    incremental_user_turn_ids,
    trim_after_answer_stop,
)


class GenerationBackend(Protocol):
    async def generate_batch(self, requests: list[GenerationRequest]) -> list[Any]: ...


class ToolBackend(Protocol):
    async def execute_batch(self, calls: list[ToolCall]) -> list[Any]: ...


@dataclass(frozen=True)
class EvalConfig:
    context_tokens: int = 8192
    total_response_tokens: int = 4096
    max_tokens_per_turn: int = 1024
    max_observation_tokens: int = 512
    max_tool_calls: int = 5
    max_consecutive_invalid_actions: int = 2
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    seed: int = 42
    protocol: str = "trace_opsd.eval.greedy.v2"

    def __post_init__(self) -> None:
        if (
            min(
                self.context_tokens,
                self.total_response_tokens,
                self.max_tokens_per_turn,
                self.max_observation_tokens,
                self.max_tool_calls,
                self.max_consecutive_invalid_actions,
            )
            <= 0
        ):
            raise ValueError("evaluation token and action limits must be positive")
        if namespace_matches(self.protocol, 'trace_opsd.eval.greedy.v2'):
            if self.temperature != 0 or self.top_p != 1 or self.top_k != -1:
                raise ValueError(
                    "the main evaluation protocol requires greedy decoding"
                )
        elif namespace_matches(self.protocol, 'trace_opsd.eval.sampling.v1'):
            if not math.isfinite(self.temperature) or self.temperature <= 0:
                raise ValueError(
                    "sampled evaluation requires a positive finite temperature"
                )
            if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
                raise ValueError("sampled evaluation requires 0 < top_p <= 1")
            if (
                isinstance(self.top_k, bool)
                or not isinstance(self.top_k, int)
                or (self.top_k != -1 and self.top_k <= 0)
            ):
                raise ValueError(
                    "sampled evaluation top_k must be -1 or a positive integer"
                )
        else:
            raise ValueError(f"unsupported evaluation protocol: {self.protocol}")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError(
                "evaluation repetition_penalty must be positive and finite"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("evaluation seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _render(tokenizer: Any, messages: list[dict[str, str]], thinking: str) -> str:
    kwargs = {"enable_thinking": thinking == "on"} if thinking in {"on", "off"} else {}
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )


def _count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_observation(tokenizer: Any, observation: str, limit: int) -> str:
    if _count(tokenizer, observation) <= limit:
        return observation
    opening = "<result>\n"
    ending = "\n...[observation truncated]\n</result>"
    if "<result>" in observation and "</result>" in observation:
        body = observation.split("<result>", 1)[1].rsplit("</result>", 1)[0]
    else:
        body = observation
    budget = max(0, limit - _count(tokenizer, opening) - _count(tokenizer, ending))
    ids = tokenizer.encode(body, add_special_tokens=False)[:budget]
    return opening + tokenizer.decode(ids, skip_special_tokens=False) + ending


async def evaluate_one(
    *,
    benchmark: str,
    row: dict[str, Any],
    mode: str,
    model_name: str,
    tokenizer: Any,
    generation: GenerationBackend,
    tools: ToolBackend | None,
    thinking: str,
    config: EvalConfig,
    sample_index: int | None = None,
) -> dict[str, Any]:
    if mode not in {"tool", "direct"}:
        raise ValueError(f"unsupported evaluation mode: {mode}")
    if sample_index is not None and (
        isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or sample_index < 0
    ):
        raise ValueError("sample_index must be a nonnegative integer")
    # Preserve the original greedy IDs and record schema for existing resumes.
    request_prefix = f"{benchmark}:{row['id']}"
    if sample_index is not None:
        request_prefix += f":sample:{sample_index}"
    messages = (
        build_tool_messages(row["problem"])
        if mode == "tool"
        else build_direct_messages(row["problem"])
    )
    initial_prompt = _render(tokenizer, messages, thinking)
    prompt_ids = list(tokenizer.encode(initial_prompt, add_special_tokens=False))
    response_ids: list[int] = []
    record: dict[str, Any] = {
        "benchmark": benchmark,
        "id": row["id"],
        "mode": mode,
        "prompt": initial_prompt,
        "assistant_messages": [],
        "tool_events": [],
        "final_response": None,
        "termination_reason": "missing_answer",
        "errors": [],
        "response_tokens_used": 0,
        "tool_call_count": 0,
        "python_call_count": 0,
        "search_call_count": 0,
        "tool_error_count": 0,
        "truncated": False,
    }
    if sample_index is not None:
        record.update(sample_index=sample_index, sample_seed=config.seed)
    invalid_streak = 0

    async def append_environment(content: str) -> bool:
        template_kwargs = (
            {"enable_thinking": thinking == "on"} if thinking in {"on", "off"} else {}
        )
        ids = incremental_user_turn_ids(tokenizer, content, template_kwargs)
        if len(response_ids) + len(ids) > config.total_response_tokens:
            record["truncated"] = True
            return False
        response_ids.extend(ids)
        record["response_tokens_used"] = len(response_ids)
        messages.append({"role": "user", "content": content})
        return True

    async def generate(turn: int, *, final: bool) -> str | None:
        history_ids = prompt_ids + response_ids
        prompt = tokenizer.decode(history_ids, skip_special_tokens=False)
        remaining = config.total_response_tokens - len(response_ids)
        if remaining <= 0:
            record["termination_reason"] = "response_budget_exhausted"
            record["truncated"] = True
            return None
        prompt_tokens = len(history_ids)
        allowed = min(
            config.max_tokens_per_turn, remaining, config.context_tokens - prompt_tokens
        )
        if allowed <= 0:
            record["termination_reason"] = "context_exhausted"
            record["truncated"] = True
            return None
        request = GenerationRequest(
            request_id=f"{request_prefix}:{mode}:turn:{turn}",
            prompt=prompt,
            model=model_name,
            max_tokens=allowed,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            seed=config.seed,
            stop=("</answer>",) if final else (*action_stop_sequences(), "</answer>"),
            prompt_token_ids=tuple(history_ids),
        )
        try:
            results = await generation.generate_batch([request])
            result = results[0]
        except Exception as exc:
            record["errors"].append(f"generation: {type(exc).__name__}: {exc}")
            record["termination_reason"] = "generation_error"
            return None
        if result.error is not None:
            record["errors"].append(f"generation: {result.error}")
            record["termination_reason"] = "generation_error"
            return None
        try:
            if result.token_ids is None:
                raise ValueError("evaluation backend did not return sampled token_ids")
            generated_ids = trim_after_answer_stop(
                tokenizer,
                exact_token_ids(result.token_ids, name="evaluation output")[:remaining],
            )
            text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        except (TypeError, ValueError) as exc:
            record["errors"].append(f"generation: {exc}")
            record["termination_reason"] = "generation_error"
            return None
        response_ids.extend(generated_ids)
        record["assistant_messages"].append(text)
        record["response_tokens_used"] = len(response_ids)
        if result.finish_reason == "length":
            record["truncated"] = True
        return text

    if mode == "direct":
        text = await generate(0, final=True)
        if text is not None:
            record["final_response"] = text
            record["termination_reason"] = "answer"
        return record

    turn = 0
    while True:
        if record["tool_call_count"] >= config.max_tool_calls:
            break
        text = await generate(turn, final=False)
        if text is None:
            return record
        action = parse_first_complete_action(text, require_answer_tag=True)
        messages.append(
            {
                "role": "assistant",
                "content": text,
            }
        )
        if action.kind is ActionKind.ANSWER:
            record["final_response"] = text
            record["termination_reason"] = "answer"
            return record
        if action.kind is ActionKind.INVALID or action.tag is None:
            invalid_streak += 1
            if invalid_streak >= config.max_consecutive_invalid_actions:
                record["final_response"] = text
                record["termination_reason"] = "invalid_action_limit"
                return record
            if not await append_environment(INVALID_ACTION_FEEDBACK):
                record["final_response"] = text
                record["termination_reason"] = "response_budget_exhausted"
                return record
            turn += 1
            continue
        invalid_streak = 0
        call = ToolCall(
            request_id=f"{request_prefix}:tool:{turn}",
            source_id=row["id"],
            turn=turn,
            tool_name=action.tag,
            content=action.content,
            raw_text=action.raw_text,
        )
        record["tool_call_count"] += 1
        record[f"{action.tag}_call_count"] += 1
        if tools is None:
            record["errors"].append("tool service is not configured")
            record["termination_reason"] = "tool_service_error"
            return record
        try:
            response = (await tools.execute_batch([call]))[0]
            observation = response.observation
            tool_error = response.error is not None or response.status == "error"
            execution_metadata = {
                "status": response.status,
                "error_category": getattr(response, "error_category", None),
                "error_origin": getattr(response, "error_origin", None),
                "execution_result_version": getattr(
                    response, "execution_result_version", None
                ),
                "protocol_version": getattr(response, "execution_result_version", None),
                "transport_error": getattr(response, "transport_error", False),
            }
        except Exception as exc:
            observation = f"<result>\nError: {type(exc).__name__}: {exc}\n</result>"
            tool_error = True
            execution_metadata = {
                "status": "error",
                "error_category": None,
                "error_origin": "transport",
                "execution_result_version": None,
                "protocol_version": None,
                "transport_error": True,
            }
        if tool_error:
            record["tool_error_count"] += 1
        record["tool_events"].append(
            {
                "turn": turn,
                "tool": action.tag,
                "call": action.raw_text,
                "observation": observation,
                "error": tool_error,
                **execution_metadata,
            }
        )
        if not await append_environment(
            truncate_observation(tokenizer, observation, config.max_observation_tokens)
        ):
            record["termination_reason"] = "response_budget_exhausted"
            return record
        turn += 1

    if not await append_environment(build_final_answer_instruction()):
        record["termination_reason"] = "response_budget_exhausted"
        return record
    text = await generate(len(record["assistant_messages"]), final=True)
    if text is not None:
        record["final_response"] = text
        record["termination_reason"] = "final_answer"
    return record
