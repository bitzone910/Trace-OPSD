"""Parsing helpers for the tag-based Trace-OPSD rollout protocol."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from .models import ActionKind, EventKind, ParsedAction, TrajectoryEvent

DEFAULT_TOOL_TAGS: tuple[str, ...] = (
    "python",
    "search",
)
ANSWER_TAG = "answer"


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _balanced_brace_span(text: str, opening: int) -> tuple[str, int] | None:
    depth = 0
    for index in range(opening, len(text)):
        character = text[index]
        if character == "{" and not _is_escaped(text, index):
            depth += 1
        elif character == "}" and not _is_escaped(text, index):
            depth -= 1
            if depth == 0:
                content = text[opening + 1 : index].strip()
                return (content, index + 1) if content else None
    return None


def extract_last_complete_boxed(text: object) -> str | None:
    """Return the last nonempty, balanced ``\\boxed{...}`` payload."""

    if not isinstance(text, str):
        return None
    final: str | None = None
    for match in re.finditer(r"\\boxed\s*\{", text):
        opening = text.find("{", match.start())
        span = _balanced_brace_span(text, opening)
        if span is not None:
            final = span[0]
    return final


def extract_boxed_answer(text: object) -> str | None:
    """Extract the last complete boxed payload, independent of XML wrappers."""

    return extract_last_complete_boxed(text)


def action_stop_sequences(
    tool_tags: Sequence[str] = DEFAULT_TOOL_TAGS,
) -> tuple[str, ...]:
    return tuple(f"</{tag}>" for tag in tool_tags)


def parse_first_complete_action(
    text: object,
    *,
    tool_tags: Sequence[str] = DEFAULT_TOOL_TAGS,
    require_answer_tag: bool = False,
) -> ParsedAction:
    """Return the earliest complete tool call or valid final answer."""

    if not isinstance(text, str):
        return ParsedAction(
            ActionKind.INVALID, None, "", "", is_action_valid=False
        )
    allowed_tools = tuple(dict.fromkeys(tool_tags))
    structures: list[tuple[int, int, ActionKind, str | None, str, str]] = []
    alternatives = "|".join(re.escape(tag) for tag in allowed_tools)
    if alternatives:
        pattern = re.compile(
            rf"<(?P<tag>{alternatives})>(?P<content>.*?)</(?P=tag)>",
            flags=re.DOTALL,
        )
        for match in pattern.finditer(text):
            structures.append(
                (
                    match.start(),
                    match.end(),
                    ActionKind.TOOL_CALL,
                    match.group("tag"),
                    match.group("content").strip(),
                    match.group(0),
                )
            )
    if require_answer_tag:
        for match in re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL):
            content = extract_boxed_answer(match.group(1))
            if content is not None:
                structures.append(
                    (match.start(), match.end(), ActionKind.ANSWER, ANSWER_TAG,
                     content, match.group(0))
                )
    else:
        for match in re.finditer(r"\\boxed\s*\{", text):
            opening = text.find("{", match.start())
            span = _balanced_brace_span(text, opening)
            if span is not None:
                content, end = span
                structures.append(
                    (
                        match.start(), end, ActionKind.ANSWER, ANSWER_TAG,
                        content, text[match.start() : end],
                    )
                )
    if not structures:
        return ParsedAction(
            ActionKind.INVALID, None, "", text, is_action_valid=False
        )
    structures.sort(key=lambda item: (item[0], item[1]))
    start, end, kind, tag, content, raw_text = structures[0]
    multiple_actions = len(structures) > 1
    return ParsedAction(
        kind=kind,
        tag=tag,
        content=content,
        raw_text=raw_text,
        start=start,
        end=end,
        multiple_actions=multiple_actions,
        is_action_valid=not multiple_actions,
    )


def complete_answers(text: object) -> list[str]:
    if not isinstance(text, str):
        return []
    return [
        match.group("content").strip()
        for match in re.finditer(
            r"<answer>(?P<content>.*?)</answer>", text, flags=re.DOTALL
        )
    ]


def extract_last_complete_answer(
    assistant_events: Iterable[TrajectoryEvent | str],
) -> str | None:
    """Extract only from assistant events, never prompts or tool results."""

    final: str | None = None
    for event in assistant_events:
        if isinstance(event, TrajectoryEvent):
            if event.kind is not EventKind.ASSISTANT:
                continue
            text = event.raw_text
        elif isinstance(event, str):
            text = event
        else:
            continue
        answer = extract_boxed_answer(text)
        if answer is not None:
            final = answer
    return final


def extract_last_complete_tagged_answer(
    assistant_events: Iterable[TrajectoryEvent | str],
) -> str | None:
    """Extract the last boxed answer inside a complete assistant answer block."""

    final: str | None = None
    for event in assistant_events:
        if isinstance(event, TrajectoryEvent):
            if event.kind is not EventKind.ASSISTANT:
                continue
            text = event.raw_text
        elif isinstance(event, str):
            text = event
        else:
            continue
        for block in complete_answers(text):
            final = extract_boxed_answer(block)
    return final


__all__ = [
    "ANSWER_TAG",
    "DEFAULT_TOOL_TAGS",
    "action_stop_sequences",
    "complete_answers",
    "extract_boxed_answer",
    "extract_last_complete_boxed",
    "extract_last_complete_answer",
    "extract_last_complete_tagged_answer",
    "parse_first_complete_action",
]
