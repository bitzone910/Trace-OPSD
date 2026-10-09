"""TRACE-OPSD token evidence, turn-mean credit and shared advantage weights."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

from trace_opsd.rollout.token_history import incremental_user_turn_ids

TRACE_VIEWS = ("causal", "hindsight")
TRACE_TOOL_TYPES = ("search", "python")
TRACE_TURN_TYPES = (*TRACE_TOOL_TYPES, "final")
TRACE_TEACHER_MODE = "current_policy_solution_hindsight.turn_mean.v4"
TRACE_CREDIT_AGGREGATION = "turn_mean"
LEGACY_TOKEN_TEACHER_MODE = "current_policy_solution_hindsight.token_level.v3"
PYTHON_EXECUTION_RESULT_VERSION = "trace_opsd.python-execution-result.v1"
PYTHON_CODE_ERROR_CATEGORIES = frozenset(
    {"syntax_error", "name_error", "attribute_error", "runtime_error"}
)


class TraceOPSDConfigurationError(ValueError):
    """Raised when TRACE-OPSD configuration is unsafe or inconsistent."""


class TraceTrajectoryError(ValueError):
    """Raised when rollout metadata cannot reproduce the on-policy trajectory."""


@dataclass(frozen=True)
class TraceOPSDConfig:
    """Validated TRACE-OPSD hyperparameters."""

    enabled: bool = True
    credit_scale: float = 0.5
    final_credit_scale: float = 0.25
    evidence_scale: float = 1.0
    score_chunk_size: int = 64
    max_view_length: int | None = None
    python_error_penalty_enabled: bool = False
    python_error_penalty_coef: float = 0.25

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> TraceOPSDConfig:
        raw = dict(value or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise TraceOPSDConfigurationError(
                f"unknown TRACE-OPSD configuration fields: {unknown}"
            )
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TraceOPSDConfigurationError("enabled must be boolean")
        if not isinstance(self.python_error_penalty_enabled, bool):
            raise TraceOPSDConfigurationError(
                "python_error_penalty_enabled must be boolean"
            )
        if (
            isinstance(self.python_error_penalty_coef, bool)
            or not isinstance(self.python_error_penalty_coef, Real)
            or not math.isfinite(self.python_error_penalty_coef)
            or not 0.0 <= self.python_error_penalty_coef <= 0.5
            or (
                self.python_error_penalty_enabled
                and self.python_error_penalty_coef == 0.0
            )
        ):
            raise TraceOPSDConfigurationError(
                "python_error_penalty_coef must be finite and in [0, 0.5]; "
                "enabled penalties require a positive coefficient"
            )
        if (
            isinstance(self.credit_scale, bool)
            or not isinstance(self.credit_scale, Real)
            or not math.isfinite(self.credit_scale)
            or not 0.0 <= self.credit_scale < 1.0
        ):
            raise TraceOPSDConfigurationError("credit_scale must be in [0, 1)")
        if (
            isinstance(self.final_credit_scale, bool)
            or not isinstance(self.final_credit_scale, Real)
            or not math.isfinite(self.final_credit_scale)
            or not 0.0 <= self.final_credit_scale <= self.credit_scale
        ):
            raise TraceOPSDConfigurationError(
                "final_credit_scale must be in [0, credit_scale]"
            )
        if (
            isinstance(self.evidence_scale, bool)
            or not isinstance(self.evidence_scale, Real)
            or not math.isfinite(self.evidence_scale)
            or self.evidence_scale <= 0.0
        ):
            raise TraceOPSDConfigurationError(
                "evidence_scale must be finite and positive"
            )
        if (
            isinstance(self.score_chunk_size, bool)
            or not isinstance(self.score_chunk_size, Integral)
            or self.score_chunk_size <= 0
        ):
            raise TraceOPSDConfigurationError(
                "score_chunk_size must be a positive integer"
            )
        if self.max_view_length is not None and (
            isinstance(self.max_view_length, bool)
            or not isinstance(self.max_view_length, Integral)
            or self.max_view_length <= 0
        ):
            raise TraceOPSDConfigurationError(
                "max_view_length must be a positive integer"
            )


@dataclass(frozen=True)
class TraceView:
    """One teacher-forced scoring view for an on-policy assistant turn."""

    trajectory_index: int
    turn_index: int
    view: str
    uid: str
    turn_type: str
    response_start: int
    response_end: int
    prefix_ids: tuple[int, ...]
    action_ids: tuple[int, ...]


@dataclass(frozen=True)
class TokenEvidence:
    """Aligned token likelihood lifts for one assistant generation span."""

    trajectory_index: int
    turn_index: int
    uid: str
    turn_type: str
    response_start: int
    response_end: int
    evidence: tuple[float, ...]


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TraceTrajectoryError(f"{name} must be an integer")
    return int(value)


def _exact_token_ids(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    result = tuple(_integer(value, name=f"{name} token ID") for value in values)
    if any(value < 0 for value in result):
        raise TraceTrajectoryError(f"{name} contains a negative token ID")
    return result


def _token_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    ids = tokenizer.apply_chat_template(
        list(messages),
        add_generation_prompt=True,
        tokenize=True,
        **dict(apply_chat_template_kwargs or {}),
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise TraceTrajectoryError("chat template returned an unexpected batch")
        ids = ids[0]
    return list(_exact_token_ids(ids, name="chat template"))


def _history_with_evidence(
    history: Sequence[Mapping[str, Any]], evidence: str
) -> list[dict[str, Any]]:
    if not history or history[-1].get("role") != "user":
        raise TraceTrajectoryError(
            "an assistant action must be preceded by a user/environment message"
        )
    result = [dict(message) for message in history]
    content = result[-1].get("content")
    if not isinstance(content, str):
        raise TraceTrajectoryError("TRACE-OPSD supports text-only Trace-OPSD messages")
    result[-1]["content"] = f"{content}\n\n{evidence}"
    return result


def _solution_hindsight_evidence(hindsight: str) -> str:
    return (
        '<trace_training_hindsight kind="solution_strategy">\n'
        "A privileged solution strategy summarized from the reference solution is:\n"
        f"{hindsight}\n"
        "</trace_training_hindsight>"
    )


def build_turn_views(
    *,
    tokenizer: Any,
    trajectory_index: int,
    turn_index: int,
    uid: str,
    messages: Sequence[Mapping[str, Any]],
    turn: Mapping[str, Any],
    hindsight: str,
    action_ids: Sequence[int],
    expected_causal_prefix_ids: Sequence[int],
    max_view_length: int,
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[TraceView], str | None]:
    """Build causal and solution-hindsight views or safely skip the turn."""

    turn_type = str(turn.get("turn_type", turn.get("tool_type", ""))).lower()
    if turn_type not in TRACE_TURN_TYPES:
        raise TraceTrajectoryError(f"unsupported TRACE turn type: {turn_type!r}")
    assistant_index = turn.get("assistant_message_index")
    assistant_index = _integer(assistant_index, name="assistant_message_index")
    if not 0 <= assistant_index < len(messages):
        raise TraceTrajectoryError("assistant_message_index is outside trace_messages")
    assistant = messages[assistant_index]
    if assistant.get("role") != "assistant":
        raise TraceTrajectoryError("TRACE turn does not point to an assistant message")
    history = list(messages[:assistant_index])
    # The causal view must use the exact prefix seen during rollout.  Reapplying
    # the chat template to decoded messages is not token-lossless, especially
    # after environment turns have been appended as incremental template
    # suffixes.  The caller derives this prefix directly from the rollout batch.
    causal_prefix = list(
        _exact_token_ids(expected_causal_prefix_ids, name="on-policy causal prefix")
    )
    if not causal_prefix:
        raise TraceTrajectoryError("on-policy causal prefix contains no tokens")
    exact_action_ids = _exact_token_ids(action_ids, name="TRACE action")
    if not exact_action_ids:
        raise TraceTrajectoryError("TRACE assistant action contains no tokens")
    response_start = _integer(turn.get("response_start"), name="response_start")
    response_end = _integer(turn.get("response_end"), name="response_end")
    if response_start < 0 or response_end - response_start != len(exact_action_ids):
        raise TraceTrajectoryError("TRACE action token count does not match its span")
    if not isinstance(hindsight, str) or not hindsight.strip():
        return [], "missing_solution_hindsight"
    if "context_suffix_start" not in turn:
        raise TraceTrajectoryError("TRACE turn is missing context_suffix_start")
    evidence = _solution_hindsight_evidence(hindsight.strip())
    evidence_history = _history_with_evidence(history, evidence)
    suffix_start = turn["context_suffix_start"]
    if suffix_start is None:
        if response_start != 0:
            raise TraceTrajectoryError(
                "only the initial turn may omit a context suffix"
            )
        if _token_ids(tokenizer, history, apply_chat_template_kwargs) != causal_prefix:
            raise TraceTrajectoryError(
                "initial TRACE prompt differs from its rollout tokens"
            )
        hindsight_prefix = _token_ids(
            tokenizer, evidence_history, apply_chat_template_kwargs
        )
    else:
        suffix_start = _integer(suffix_start, name="context_suffix_start")
        prompt_length = len(causal_prefix) - response_start
        if prompt_length <= 0 or not 0 <= suffix_start < response_start:
            raise TraceTrajectoryError(
                "TRACE context suffix is outside its causal prefix"
            )
        splice_index = prompt_length + suffix_start
        original_suffix = incremental_user_turn_ids(
            tokenizer, history[-1]["content"], apply_chat_template_kwargs
        )
        if original_suffix != causal_prefix[splice_index:]:
            raise TraceTrajectoryError(
                "TRACE context suffix differs from its rollout tokens"
            )
        # Full chat-template replay can strip earlier Qwen3 reasoning and omit
        # the empty assistant seam before forced-final feedback.
        hindsight_prefix = causal_prefix[:splice_index] + incremental_user_turn_ids(
            tokenizer, evidence_history[-1]["content"], apply_chat_template_kwargs
        )
    prefixes = {"causal": causal_prefix, "hindsight": hindsight_prefix}
    if any(not prefix for prefix in prefixes.values()):
        raise TraceTrajectoryError("TRACE scoring prefix contains no tokens")
    if any(
        len(prefix) + len(exact_action_ids) > max_view_length
        for prefix in prefixes.values()
    ):
        return [], "view_too_long"
    views = [
        TraceView(
            trajectory_index=trajectory_index,
            turn_index=turn_index,
            view=view,
            uid=str(uid),
            turn_type=turn_type,
            response_start=response_start,
            response_end=response_end,
            prefix_ids=tuple(prefixes[view]),
            action_ids=exact_action_ids,
        )
        for view in TRACE_VIEWS
    ]
    return views, None


def _token_values(
    values: Sequence[float], expected_length: int, *, name: str
) -> tuple[float, ...]:
    """Validate a finite vector without averaging away its token positions."""

    try:
        result = tuple(
            float(value.detach() if hasattr(value, "detach") else value)
            for value in values
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise TraceTrajectoryError(f"{name} must be a token score vector") from exc
    if expected_length <= 0 or len(result) != expected_length:
        raise TraceTrajectoryError(f"{name} token count does not match its action span")
    if not all(math.isfinite(value) for value in result):
        raise TraceTrajectoryError(f"{name} contains non-finite token values")
    return result


def _validate_view(view: TraceView) -> None:
    for field in ("trajectory_index", "turn_index", "response_start", "response_end"):
        _integer(getattr(view, field), name=field)
    _exact_token_ids(view.action_ids, name="TRACE action")
    _exact_token_ids(view.prefix_ids, name="TRACE prefix")
    if view.view not in TRACE_VIEWS:
        raise TraceTrajectoryError(f"unknown TRACE view: {view.view}")
    if view.turn_type not in TRACE_TURN_TYPES:
        raise TraceTrajectoryError(f"unknown TRACE turn type: {view.turn_type}")
    if (
        view.response_start < 0
        or (view.response_end - view.response_start != len(view.action_ids))
        or not view.action_ids
    ):
        raise TraceTrajectoryError("TRACE action token count does not match its span")


def evidence_from_view_scores(
    views: Sequence[TraceView], scores: Sequence[Sequence[float]]
) -> list[TokenEvidence]:
    """Subtract aligned causal token scores from hindsight token scores."""

    if len(views) != len(scores):
        raise TraceTrajectoryError("TRACE views and scores must have equal length")
    grouped: dict[tuple[int, int], dict[str, tuple[TraceView, tuple[float, ...]]]] = (
        defaultdict(dict)
    )
    for view, score in zip(views, scores, strict=True):
        _validate_view(view)
        key = (view.trajectory_index, view.turn_index)
        if view.view in grouped[key]:
            raise TraceTrajectoryError(
                f"duplicate TRACE {view.view} view for turn {key}"
            )
        grouped[key][view.view] = (
            view,
            _token_values(
                score, len(view.action_ids), name=f"TRACE {view.view} scores"
            ),
        )
    result = []
    for key in sorted(grouped):
        records = grouped[key]
        if set(records) != set(TRACE_VIEWS):
            raise TraceTrajectoryError(
                f"TRACE turn {key} does not contain exactly two views"
            )
        causal_view, causal = records["causal"]
        hindsight_view, hindsight = records["hindsight"]
        signature = ("uid", "turn_type", "response_start", "response_end", "action_ids")
        if any(
            getattr(causal_view, field) != getattr(hindsight_view, field)
            for field in signature
        ):
            raise TraceTrajectoryError(
                f"TRACE view action metadata differs for turn {key}"
            )
        gaps = _token_values(
            [h - c for h, c in zip(hindsight, causal, strict=True)],
            len(causal_view.action_ids),
            name="TRACE token evidence",
        )
        result.append(
            TokenEvidence(
                trajectory_index=causal_view.trajectory_index,
                turn_index=causal_view.turn_index,
                uid=causal_view.uid,
                turn_type=causal_view.turn_type,
                response_start=causal_view.response_start,
                response_end=causal_view.response_end,
                evidence=gaps,
            )
        )
    return result


def causal_scores_from_old_log_probs(
    views: Sequence[TraceView],
    old_log_probs: Any,
    responses: Any,
    response_mask: Any,
) -> dict[tuple[int, int], tuple[float, ...]]:
    """Read detached current-policy token scores for the exact sampled spans."""

    import torch

    if old_log_probs.shape != responses.shape or responses.shape != response_mask.shape:
        raise TraceTrajectoryError(
            "old_log_probs, responses, and response_mask must have identical shapes"
        )
    if old_log_probs.ndim != 2:
        raise TraceTrajectoryError("TRACE response tensors must be two-dimensional")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise TraceTrajectoryError("TRACE response_mask must contain only zero or one")
    scores: dict[tuple[int, int], tuple[float, ...]] = {}
    for view in views:
        _validate_view(view)
        if view.view != "causal":
            continue
        row = view.trajectory_index
        start, end = view.response_start, view.response_end
        if not 0 <= row < old_log_probs.shape[0] or not (
            0 <= start < end <= old_log_probs.shape[1]
        ):
            raise TraceTrajectoryError(
                "TRACE causal span is outside current-policy old_log_probs"
            )
        sampled_ids = _exact_token_ids(
            responses[row, start:end].tolist(), name="sampled action"
        )
        if sampled_ids != view.action_ids:
            raise TraceTrajectoryError(
                "TRACE causal span no longer matches the sampled action tokens"
            )
        if not torch.all(response_mask[row, start:end].to(torch.bool)):
            raise TraceTrajectoryError(
                "TRACE causal span includes observation or padding tokens"
            )
        key = (view.trajectory_index, view.turn_index)
        if key in scores:
            raise TraceTrajectoryError(f"duplicate TRACE causal view for turn {key}")
        scores[key] = _token_values(
            old_log_probs[row, start:end].detach().float().cpu().tolist(),
            len(view.action_ids),
            name="current-policy causal scoring",
        )
    return scores


def _mean(values: Sequence[float], *, default: float = 0.0) -> float:
    """Arithmetic mean without overflowing the sum of finite likelihood gaps."""
    if not values:
        return default
    try:
        return math.fsum(values) / len(values)
    except OverflowError:
        # Divide before accumulation if an otherwise finite mean overflows
        # its sum. Preserve small residuals after cancellation when possible.
        try:
            return math.fsum(value / len(values) for value in values)
        except OverflowError:
            # At float64's maximum, division rounding can still overflow.
            pass
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    return scale * (math.fsum(value / scale for value in values) / len(values))


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    """Interpolate a percentile from pre-sorted values."""

    if not ordered:
        return 0.0
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return (1.0 - fraction) * ordered[lower] + fraction * ordered[upper]


def summarize_trace_evidence(
    views: Sequence[TraceView],
    scores: Sequence[Sequence[float]],
    evidence: Sequence[TokenEvidence],
) -> dict[str, float]:
    """Keep raw token diagnostics and separately report equal-weight turn means."""

    if len(views) != len(scores):
        raise TraceTrajectoryError("TRACE views and scores must have equal length")
    score_groups: dict[str, list[float]] = {view: [] for view in TRACE_VIEWS}
    for view, score in zip(views, scores, strict=True):
        _validate_view(view)
        score_groups[view.view].extend(
            _token_values(score, len(view.action_ids), name=f"TRACE {view.view} scores")
        )
    by_turn: dict[str, list[float]] = {kind: [] for kind in TRACE_TURN_TYPES}
    turn_means: dict[str, list[float]] = {kind: [] for kind in TRACE_TURN_TYPES}
    turn_counts = {kind: 0 for kind in TRACE_TURN_TYPES}
    for item in evidence:
        if item.turn_type not in TRACE_TURN_TYPES:
            raise TraceTrajectoryError(f"unknown TRACE turn type: {item.turn_type}")
        values = _token_values(
            item.evidence,
            item.response_end - item.response_start,
            name="TRACE evidence",
        )
        by_turn[item.turn_type].extend(values)
        turn_means[item.turn_type].append(_mean(values))
        turn_counts[item.turn_type] += 1
    values = [value for group in by_turn.values() for value in group]
    means = [value for group in turn_means.values() for value in group]
    metrics = {
        "trace/view_log_prob/causal_mean": _mean(score_groups["causal"]),
        "trace/view_log_prob/hindsight_mean": _mean(score_groups["hindsight"]),
        "trace/evidence/mean": _mean(values),
        "trace/evidence/abs_mean": _mean([abs(value) for value in values]),
        "trace/turns": float(sum(turn_counts.values())),
        "trace/tokens": float(len(values)),
        "trace/token_credit_coverage": _mean([float(value > 0.0) for value in values]),
        "trace/turn_credit/evidence_mean": _mean(means),
        "trace/turn_credit/positive_evidence_ratio": _mean(
            [float(value > 0.0) for value in means]
        ),
    }
    for kind, gaps in by_turn.items():
        prefix = f"trace/turn/{kind}"
        ordered = sorted(gaps)
        metrics.update(
            {
                f"{prefix}/turns": float(turn_counts[kind]),
                f"{prefix}/tokens": float(len(gaps)),
                f"{prefix}/positive_evidence_ratio": _mean(
                    [float(value > 0.0) for value in gaps]
                ),
                f"{prefix}/evidence_mean": _mean(gaps),
                f"{prefix}/evidence_abs_mean": _mean([abs(value) for value in gaps]),
                f"{prefix}/evidence_p50": _percentile(ordered, 0.50),
                f"{prefix}/evidence_p90": _percentile(ordered, 0.90),
                f"{prefix}/credit_evidence_mean": _mean(turn_means[kind]),
                f"{prefix}/credit_positive_ratio": _mean(
                    [float(value > 0.0) for value in turn_means[kind]]
                ),
            }
        )
    return metrics


def shape_token_advantages(
    advantages: Any,
    response_mask: Any,
    evidence: Sequence[TokenEvidence],
    *,
    credit_scale: float,
    final_credit_scale: float,
    evidence_scale: float,
) -> tuple[Any, dict[str, float]]:
    """Average raw gaps first, then broadcast one detached weight per turn."""

    import torch

    TraceOPSDConfig(
        credit_scale=credit_scale,
        final_credit_scale=final_credit_scale,
        evidence_scale=evidence_scale,
    ).validate()
    if advantages.shape != response_mask.shape or advantages.ndim != 2:
        raise TraceTrajectoryError(
            "advantages and response_mask must have identical 2D shapes"
        )
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise TraceTrajectoryError("TRACE response_mask must contain only zero or one")
    shaped = advantages.detach().clone()
    weights_by_turn: dict[str, list[float]] = {kind: [] for kind in TRACE_TURN_TYPES}
    shared_weights: dict[str, list[float]] = {kind: [] for kind in TRACE_TURN_TYPES}
    delta_by_turn: dict[str, list[float]] = {kind: [] for kind in TRACE_TURN_TYPES}
    shaped_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    for item in evidence:
        if item.turn_type not in TRACE_TURN_TYPES:
            raise TraceTrajectoryError(f"unknown TRACE turn type: {item.turn_type}")
        row = _integer(item.trajectory_index, name="trajectory_index")
        start = _integer(item.response_start, name="response_start")
        end = _integer(item.response_end, name="response_end")
        if not 0 <= row < shaped.shape[0] or not 0 <= start < end <= shaped.shape[1]:
            raise TraceTrajectoryError(
                "TRACE response span is outside the advantage tensor"
            )
        if not torch.all(response_mask[row, start:end].to(torch.bool)):
            raise TraceTrajectoryError(
                "TRACE action span includes observation/padding tokens"
            )
        if torch.any(shaped_mask[row, start:end]):
            raise TraceTrajectoryError("TRACE action spans overlap")
        gaps = _token_values(item.evidence, end - start, name="TRACE evidence")
        base = advantages[row, start:end].detach()
        if not torch.isfinite(base).all():
            raise TraceTrajectoryError(
                "TRACE base advantages contain non-finite values"
            )
        if not torch.all(base == base[0]):
            raise TraceTrajectoryError(
                "TRACE turn requires identical original GRPO advantages"
            )
        gap_tensor = torch.tensor(_mean(gaps), dtype=torch.float64, device=base.device)
        # float64 scaling keeps every finite raw gap/positive finite scale usable;
        # tanh saturates safely even when their ratio overflows.
        credit = torch.tanh(gap_tensor / evidence_scale).clamp_min(0.0)
        scale = final_credit_scale if item.turn_type == "final" else credit_scale
        shared_weight = (
            (1.0 + scale * torch.sign(base[0]) * credit).to(base.dtype).detach()
        )
        weights = shared_weight.expand_as(base)
        if not torch.isfinite(weights).all() or not torch.all(weights > 0.0):
            raise TraceTrajectoryError(
                "TRACE advantage weights must be finite and positive"
            )
        shaped[row, start:end] = base * weights
        if not torch.isfinite(shaped[row, start:end]).all():
            raise TraceTrajectoryError(
                "TRACE shaped advantages contain non-finite values"
            )
        shaped_mask[row, start:end] = True
        weights_by_turn[item.turn_type].extend(weights.float().cpu().tolist())
        shared_weights[item.turn_type].append(float(shared_weight.double().item()))
        delta_by_turn[item.turn_type].extend(
            (shaped[row, start:end] - base).abs().double().cpu().tolist()
        )
    all_weights = [value for group in weights_by_turn.values() for value in group]
    per_turn_weights = [value for group in shared_weights.values() for value in group]
    raw_values = advantages.detach()[shaped_mask]
    shaped_values = shaped[shaped_mask]
    valid_tokens = int(response_mask.to(torch.bool).sum().item())

    def abs_mean(values: Any) -> float:
        return _mean(values.detach().abs().double().cpu().tolist())

    metrics = {
        "trace/weight_mean": _mean(all_weights, default=1.0),
        "trace/weight_min": min(all_weights, default=1.0),
        "trace/weight_max": max(all_weights, default=1.0),
        "trace/weight_abs_delta_mean": _mean([abs(w - 1.0) for w in all_weights]),
        "trace/adv/token_ratio": float(len(all_weights) / valid_tokens)
        if valid_tokens
        else 0.0,
        "trace/adv/raw_abs_mean": abs_mean(raw_values),
        "trace/adv/shaped_abs_mean": abs_mean(shaped_values),
        "trace/adv/abs_delta_mean": abs_mean(shaped_values - raw_values),
        "trace/turn_credit/weight_mean": _mean(per_turn_weights, default=1.0),
        "trace/turn_credit/weight_min": min(per_turn_weights, default=1.0),
        "trace/turn_credit/weight_max": max(per_turn_weights, default=1.0),
    }
    for kind, weights in weights_by_turn.items():
        prefix = f"trace/turn/{kind}"
        metrics.update(
            {
                f"{prefix}/weight_mean": _mean(weights, default=1.0),
                f"{prefix}/weight_min": min(weights, default=1.0),
                f"{prefix}/weight_max": max(weights, default=1.0),
                f"{prefix}/weight_abs_delta_mean": _mean(
                    [abs(w - 1.0) for w in weights]
                ),
                f"{prefix}/adv_abs_delta_mean": _mean(delta_by_turn[kind]),
                f"{prefix}/shared_weight_mean": _mean(
                    shared_weights[kind], default=1.0
                ),
                f"{prefix}/shared_weight_min": min(shared_weights[kind], default=1.0),
                f"{prefix}/shared_weight_max": max(shared_weights[kind], default=1.0),
            }
        )
    return shaped, metrics


def apply_python_error_penalty(
    base_advantages: Any,
    opsd_advantages: Any,
    response_mask: Any,
    turns: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
    coef: float,
) -> tuple[Any, dict[str, float], list[dict[str, Any]]]:
    """Apply local process credit after OPSD using trusted execution outcomes.

    A failing assistant turn receives ``min(A_opsd, 0) - coef * abs(A_base)``
    independently at every action token. Observations and later turns retain
    their original advantages. Tool output text is deliberately never parsed.
    """

    import torch

    TraceOPSDConfig(
        python_error_penalty_enabled=enabled,
        python_error_penalty_coef=coef,
    ).validate()
    prefix = "trace/python_error_penalty"
    metrics = {
        f"{prefix}/enabled": float(enabled),
        f"{prefix}/coef": float(coef),
        **{
            f"{prefix}/{name}": 0.0
            for name in (
                "observed_turns",
                "eligible_turns",
                "penalized_turns",
                "penalized_tokens",
                "token_ratio",
                "positive_to_negative_tokens",
                "positive_to_negative_ratio",
                "penalty_abs_mean",
                "penalty_max",
                "base_abs_mean",
                "opsd_abs_mean",
                "final_abs_mean",
                "final_min",
                "final_max",
                "code_success_turns",
                "code_success_ratio",
                "error_turns",
                "origin_user_code_turns",
                "origin_execution_limit_turns",
                "origin_queue_turns",
                "origin_transport_turns",
                "origin_bootstrap_turns",
                "origin_sandbox_turns",
                "origin_unknown_turns",
                "origin_none_turns",
            )
        },
    }
    shaped = opsd_advantages.detach().clone()
    if not enabled:
        # Existing runs need neither new outcome metadata nor different values.
        return shaped, metrics, []
    if (
        base_advantages.ndim != 2
        or base_advantages.shape != shaped.shape
        or shaped.shape != response_mask.shape
    ):
        raise TraceTrajectoryError(
            "Python penalty advantages and response_mask must have identical 2D shapes"
        )
    if not base_advantages.is_floating_point() or not shaped.is_floating_point():
        raise TraceTrajectoryError("Python penalty advantages must be floating point")
    if base_advantages.dtype != shaped.dtype:
        raise TraceTrajectoryError("Python penalty advantages must share a dtype")
    if base_advantages.device != shaped.device or shaped.device != response_mask.device:
        raise TraceTrajectoryError("Python penalty tensors must share a device")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise TraceTrajectoryError(
            "Python penalty response_mask must contain zero or one"
        )
    if not torch.isfinite(base_advantages).all() or not torch.isfinite(shaped).all():
        raise TraceTrajectoryError(
            "Python penalty advantages contain non-finite values"
        )

    valid_mask = response_mask.to(torch.bool)
    seen_mask = torch.zeros_like(valid_mask)
    penalty_mask = torch.zeros_like(valid_mask)
    seen_turns: set[tuple[int, int]] = set()
    diagnostics: list[dict[str, Any]] = []
    penalties: list[Any] = []
    selected_base: list[Any] = []
    selected_opsd: list[Any] = []
    positive_tokens = 0
    for turn in turns:
        row = _integer(turn.get("trajectory_index"), name="trajectory_index")
        turn_index = _integer(turn.get("turn_index"), name="turn_index")
        start = _integer(turn.get("response_start"), name="response_start")
        end = _integer(turn.get("response_end"), name="response_end")
        if not 0 <= row < shaped.shape[0] or not 0 <= start < end <= shaped.shape[1]:
            raise TraceTrajectoryError("Python penalty response span is outside tensor")
        if turn_index < 0:
            raise TraceTrajectoryError("Python penalty turn_index must be nonnegative")
        key = (row, turn_index)
        if key in seen_turns:
            raise TraceTrajectoryError("duplicate Python penalty turn")
        if torch.any(seen_mask[row, start:end]):
            raise TraceTrajectoryError("Python penalty action spans overlap")
        if not torch.all(valid_mask[row, start:end]):
            raise TraceTrajectoryError(
                "Python penalty action span includes observation/padding tokens"
            )
        seen_turns.add(key)
        seen_mask[row, start:end] = True
        turn_type = str(turn.get("turn_type", turn.get("tool_type", ""))).lower()
        if turn_type not in TRACE_TURN_TYPES:
            raise TraceTrajectoryError(
                f"unknown Python penalty turn type: {turn_type!r}"
            )
        if turn_type != "python":
            continue

        metrics[f"{prefix}/observed_turns"] += 1.0
        status = turn.get("status")
        category = turn.get("error_category")
        origin = turn.get("error_origin")
        transport = turn.get("transport_error", False)
        if not isinstance(transport, bool):
            raise TraceTrajectoryError("Python penalty transport_error must be boolean")
        version = turn.get("execution_result_version")
        if not transport and version != PYTHON_EXECUTION_RESULT_VERSION:
            raise TraceTrajectoryError(
                "Python penalty requires trusted execution_result_version "
                f"{PYTHON_EXECUTION_RESULT_VERSION!r}"
            )
        if not transport and status not in {"ok", "error"}:
            raise TraceTrajectoryError(
                "Python penalty execution status must be ok or error"
            )
        origin_key = "transport" if transport else origin
        if origin_key is None:
            origin_key = "none"
        elif origin_key not in {
            "user_code",
            "execution_limit",
            "queue",
            "transport",
            "bootstrap",
            "sandbox",
        }:
            origin_key = "unknown"
        metrics[f"{prefix}/origin_{origin_key}_turns"] += 1.0
        is_error = not transport and status == "error"
        if is_error:
            metrics[f"{prefix}/error_turns"] += 1.0
        elif not transport:
            metrics[f"{prefix}/code_success_turns"] += 1.0
        eligible = is_error and (
            (origin == "user_code" and category in PYTHON_CODE_ERROR_CATEGORIES)
            or (origin == "execution_limit" and category == "timeout")
        )
        base = base_advantages[row, start:end].detach()
        before = opsd_advantages[row, start:end].detach()
        if eligible:
            after = (
                torch.minimum(before, torch.zeros_like(before)) - coef * base.abs()
            ).to(dtype=before.dtype)
            if not torch.isfinite(after).all():
                raise TraceTrajectoryError(
                    "Python penalty produced non-finite advantages"
                )
            shaped[row, start:end] = after
            penalty = before.double() - after.double()
            if not torch.isfinite(penalty).all():
                raise TraceTrajectoryError("Python penalty subtraction is non-finite")
            changed = penalty > 0
            metrics[f"{prefix}/eligible_turns"] += 1.0
            metrics[f"{prefix}/penalized_turns"] += float(torch.any(changed).item())
            penalty_mask[row, start:end] = changed
            positive_tokens += int((before > 0).sum().item())
            metrics[f"{prefix}/positive_to_negative_tokens"] += float(
                ((before > 0) & (after < 0)).sum().item()
            )
            penalties.append(penalty)
            selected_base.append(base)
            selected_opsd.append(before)
        else:
            after = before
            penalty = torch.zeros_like(before)
            changed = torch.zeros_like(before, dtype=torch.bool)
        record = {
            "uid": str(turn.get("uid", "")),
            "trajectory_index": row,
            "turn_index": turn_index,
            "response_start": start,
            "response_end": end,
            "tokens": end - start,
            "status": status,
            "error_category": category,
            "error_origin": origin,
            "execution_result_version": version,
            "transport_error": transport,
            "eligible": bool(eligible),
            "penalized": bool(torch.any(changed).item()),
            "skip_reason": (
                None
                if eligible
                else "transport_error"
                if transport
                else "success"
                if not is_error
                else "error_not_eligible"
            ),
            "positive_to_negative_tokens": int(
                ((before > 0) & (after < 0)).sum().item()
            ),
        }
        for name, values in (
            ("base", base),
            ("opsd", before),
            ("final", after),
            ("penalty", penalty),
        ):
            values = values.double()
            record.update(
                {
                    f"{name}_mean": float(values.mean().item()),
                    f"{name}_min": float(values.min().item()),
                    f"{name}_max": float(values.max().item()),
                }
            )
        diagnostics.append(record)

    valid_tokens = int(valid_mask.sum().item())
    attempted = (
        metrics[f"{prefix}/code_success_turns"] + metrics[f"{prefix}/eligible_turns"]
    )
    metrics[f"{prefix}/code_success_ratio"] = (
        metrics[f"{prefix}/code_success_turns"] / attempted if attempted else 0.0
    )
    changed_tokens = int(penalty_mask.sum().item())
    metrics[f"{prefix}/penalized_tokens"] = float(changed_tokens)
    metrics[f"{prefix}/token_ratio"] = (
        changed_tokens / valid_tokens if valid_tokens else 0.0
    )
    metrics[f"{prefix}/positive_to_negative_ratio"] = (
        metrics[f"{prefix}/positive_to_negative_tokens"] / positive_tokens
        if positive_tokens
        else 0.0
    )
    if penalties:
        all_penalties = torch.cat(penalties).double()
        metrics[f"{prefix}/penalty_abs_mean"] = float(all_penalties.abs().mean().item())
        metrics[f"{prefix}/penalty_max"] = float(all_penalties.max().item())
        metrics[f"{prefix}/base_abs_mean"] = float(
            torch.cat(selected_base).abs().double().mean().item()
        )
        metrics[f"{prefix}/opsd_abs_mean"] = float(
            torch.cat(selected_opsd).abs().double().mean().item()
        )
    final_values = shaped[valid_mask]
    if final_values.numel():
        metrics[f"{prefix}/final_abs_mean"] = float(
            final_values.abs().double().mean().item()
        )
        metrics[f"{prefix}/final_min"] = float(final_values.min().item())
        metrics[f"{prefix}/final_max"] = float(final_values.max().item())
    if not all(math.isfinite(value) for value in metrics.values()) or any(
        not math.isfinite(value)
        for record in diagnostics
        for key, value in record.items()
        if key.endswith(("_mean", "_min", "_max"))
    ):
        raise TraceTrajectoryError(
            "Python penalty diagnostics contain non-finite values"
        )
    return shaped, metrics, diagnostics


__all__ = [
    "TRACE_TOOL_TYPES",
    "TRACE_TURN_TYPES",
    "TRACE_TEACHER_MODE",
    "TRACE_CREDIT_AGGREGATION",
    "LEGACY_TOKEN_TEACHER_MODE",
    "TRACE_VIEWS",
    "PYTHON_EXECUTION_RESULT_VERSION",
    "TraceOPSDConfig",
    "TraceOPSDConfigurationError",
    "TraceTrajectoryError",
    "TraceView",
    "TokenEvidence",
    "build_turn_views",
    "causal_scores_from_old_log_probs",
    "evidence_from_view_scores",
    "shape_token_advantages",
    "apply_python_error_penalty",
    "summarize_trace_evidence",
]
