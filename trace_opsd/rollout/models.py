"""Transport-neutral models for rollout generation and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class RolloutMode(str, Enum):
    DIRECT = "direct"
    TOOL = "tool"


class ActionKind(str, Enum):
    TOOL_CALL = "tool_call"
    ANSWER = "answer"
    INVALID = "invalid"


class EventKind(str, Enum):
    PROMPT = "prompt"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FEEDBACK = "feedback"
    FINAL_ANSWER = "final_answer"
    GENERATION_ERROR = "generation_error"


class TerminationReason(str, Enum):
    ANSWER = "answer"
    DIRECT_COMPLETE = "direct_complete"
    FINAL_ANSWER = "final_answer"
    MISSING_ANSWER = "missing_answer"
    MISSING_BOXED_ANSWER = "missing_boxed_answer"
    INVALID_ANSWER_FORMAT = "invalid_answer_format"
    INVALID_ACTION_LIMIT = "invalid_action_limit"
    GENERATION_ERROR = "generation_error"


def _safe_text(value: object) -> str:
    try:
        return value if isinstance(value, str) else str(value)
    except Exception:
        return f"<{type(value).__name__}: unprintable>"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {_safe_text(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return _safe_text(value)


@dataclass(frozen=True)
class RolloutSample:
    source_id: str
    problem: str
    ground_truth: Any
    sample_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trajectory_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.problem, str) or not self.problem.strip():
            raise ValueError("problem must be a nonempty string")
        if isinstance(self.sample_index, bool) or self.sample_index < 0:
            raise ValueError("sample_index must be a non-negative integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RolloutSample":
        source = value.get("source")
        source_mapping = source if isinstance(source, Mapping) else {}
        source_id = value.get(
            "source_id",
            value.get(
                "id",
                source_mapping.get("id", value.get("index", "unknown")),
            ),
        )
        problem = value.get("problem", source_mapping.get("problem"))
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("sample mapping is missing a nonempty problem")
        ground_truth = value.get(
            "ground_truth",
            value.get(
                "reference_answer",
                value.get("answer", source_mapping.get("ground_truth")),
            ),
        )
        known = {
            "source",
            "source_id",
            "id",
            "index",
            "problem",
            "ground_truth",
            "reference_answer",
            "answer",
            "sample_index",
            "trajectory_seed",
            "metadata",
        }
        metadata = dict(value.get("metadata", {}))
        metadata.update({key: item for key, item in value.items() if key not in known})
        return cls(
            source_id=_safe_text(source_id),
            problem=problem,
            ground_truth=ground_truth,
            sample_index=int(value.get("sample_index", 0)),
            metadata=metadata,
            trajectory_seed=(
                int(value["trajectory_seed"])
                if value.get("trajectory_seed") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ParsedAction:
    kind: ActionKind
    tag: str | None
    content: str
    raw_text: str
    start: int = -1
    end: int = -1
    multiple_actions: bool = False
    is_action_valid: bool = True


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt: str
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    stop: tuple[str, ...]
    repetition_penalty: float = 1.05
    ignore_eos: bool = False
    include_stop_str_in_output: bool = True
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    prompt_token_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    text: str = ""
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw_response: Any = None
    error: str | None = None
    attempts: int = 1
    latency_ms: float | None = None
    token_ids: tuple[int, ...] | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ToolCall:
    request_id: str
    source_id: str
    turn: int
    tool_name: str
    content: str
    raw_text: str


@dataclass(frozen=True)
class ToolResult:
    request_id: str
    tool_name: str
    observation: str
    status: str = "ok"
    raw_response: Any = None
    error: str | None = None
    attempts: int = 1
    latency_ms: float | None = None
    transport_error: bool = False
    error_category: str | None = None
    execution_result_version: str | None = None
    error_origin: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    correct: bool
    prediction: str | None
    reference: Any
    normalized_prediction: str | None
    normalized_reference: str | None
    method: str
    status: str = "ok"
    symbolic_checked: bool = False
    error: str | None = None

    @property
    def extracted_answer(self) -> str | None:
        return self.prediction

    def to_dict(self) -> dict[str, Any]:
        result = _jsonable(asdict(self))
        result["extracted_answer"] = self.prediction
        return result


@dataclass
class TrajectoryEvent:
    kind: EventKind
    turn: int
    raw_text: str
    tool_name: str | None = None
    content: str | None = None
    status: str = "ok"
    error: str | None = None
    latency_ms: float | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class TrajectoryRecord:
    trajectory_id: str
    sample: RolloutSample
    mode: RolloutMode
    prompt: str
    events: list[TrajectoryEvent]
    raw_text: str
    trajectory: str
    extracted_answer: str | None
    verification: VerificationResult
    termination_reason: TerminationReason
    errors: list[str] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        return self.verification.correct

    def to_dict(self) -> dict[str, Any]:
        source = {
            "id": self.sample.source_id,
            "problem": self.sample.problem,
            "ground_truth": _jsonable(self.sample.ground_truth),
            "metadata": _jsonable(self.sample.metadata),
        }
        return {
            "trajectory_id": self.trajectory_id,
            "source": source,
            "source_id": self.sample.source_id,
            "problem": self.sample.problem,
            "ground_truth": _jsonable(self.sample.ground_truth),
            "mode": self.mode.value,
            "sample_index": self.sample.sample_index,
            "prompt": self.prompt,
            "events": [event.to_dict() for event in self.events],
            "raw_text": self.raw_text,
            "trajectory": self.trajectory,
            "extracted_answer": self.extracted_answer,
            "correct": self.correct,
            "verification": self.verification.to_dict(),
            "termination_reason": self.termination_reason.value,
            "errors": list(self.errors),
            "metadata": _jsonable(self.metadata),
        }


SampleLike = RolloutSample | Mapping[str, Any]


def coerce_samples(values: Sequence[SampleLike]) -> list[RolloutSample]:
    samples: list[RolloutSample] = []
    for value in values:
        if isinstance(value, RolloutSample):
            samples.append(value)
        elif isinstance(value, Mapping):
            samples.append(RolloutSample.from_mapping(value))
        else:
            raise TypeError("samples must be RolloutSample objects or mappings")
    return samples


__all__ = [
    "ActionKind",
    "EventKind",
    "GenerationRequest",
    "GenerationResult",
    "ParsedAction",
    "RolloutMode",
    "RolloutSample",
    "SampleLike",
    "TerminationReason",
    "ToolCall",
    "ToolResult",
    "TrajectoryEvent",
    "TrajectoryRecord",
    "VerificationResult",
    "coerce_samples",
]
