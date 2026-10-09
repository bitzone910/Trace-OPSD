"""Token-preserving chat history helpers shared by training and inference."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping, Sequence

_SENTINEL = "TRACE_OPSD_INCREMENTAL_TEMPLATE_SENTINEL_7F3A91"


def exact_token_ids(values: Sequence[int], *, name: str) -> list[int]:
    """Validate IDs without decoding or retokenizing sampled text."""

    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must contain non-negative integer token IDs")
        result.append(int(value))
    return result


def incremental_user_turn_ids(
    tokenizer: Any,
    content: str,
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    """Render only the existing assistant's closing suffix and next user turn."""

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "assistant", "content": _SENTINEL},
            {"role": "user", "content": content},
        ],
        add_generation_prompt=True,
        tokenize=False,
        **dict(apply_chat_template_kwargs or {}),
    )
    if not isinstance(rendered, str):
        rendered = tokenizer.decode(rendered, skip_special_tokens=False)
    start = rendered.rfind(_SENTINEL)
    if start < 0:
        raise RuntimeError("cannot derive incremental Qwen chat-template suffix")
    suffix = rendered[start + len(_SENTINEL) :]
    if not suffix:
        raise RuntimeError("Qwen incremental chat-template suffix is empty")
    return list(tokenizer.encode(suffix, add_special_tokens=False))


def trim_after_answer_stop(tokenizer: Any, token_ids: list[int]) -> list[int]:
    """Drop whole sampled tokens after the first answer closing tag."""

    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    end = text.find("</answer>")
    if end < 0 or end + len("</answer>") == len(text):
        return token_ids
    # Never re-encode the decoded output: tokenization is not invertible.
    left, right = 1, len(token_ids)
    while left < right:
        middle = (left + right) // 2
        prefix = tokenizer.decode(token_ids[:middle], skip_special_tokens=False)
        if "</answer>" in prefix:
            right = middle
        else:
            left = middle + 1
    return token_ids[:left]
