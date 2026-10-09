"""Conservative, structured verification for mathematical final answers."""

from __future__ import annotations

import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

from .models import VerificationResult
from .symbolic import compare_math_expressions

_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_MATRIX_RE = re.compile(r"\\begin\{matrix\}(.*?)\\end\{matrix\}", re.DOTALL)
_PLAIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def _safe_string(value: object) -> str | None:
    if value is None:
        return None
    try:
        return value if isinstance(value, str) else str(value)
    except Exception:
        return None


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _extract_balanced_braces(text: str, opening: int) -> tuple[str, int] | None:
    depth = 0
    for index in range(opening, len(text)):
        character = text[index]
        if character == "{" and not _is_escaped(text, index):
            depth += 1
        elif character == "}" and not _is_escaped(text, index):
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index], index + 1
    return None


def _extract_last_boxed(text: str) -> str | None:
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        opening = text.find("{", start)
        balanced = _extract_balanced_braces(text, opening)
        if balanced is not None:
            return balanced[0]
    return None


def _canonical_decimal(value: Decimal) -> str:
    canonical = format(value.normalize(), "f")
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    return "0" if canonical in {"-0", "+0", ""} else canonical


def normalize_math_answer(value: object) -> str | None:
    """Canonicalize only formatting that cannot change mathematical meaning."""

    text = _safe_string(value)
    if text is None:
        return None
    try:
        answer_matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
        if answer_matches:
            text = answer_matches[-1]
        if "####" in text:
            text = text.rsplit("####", 1)[-1]
        boxed = _extract_last_boxed(text)
        if boxed is not None:
            text = boxed
        text = text.strip()
        wrappers = (("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"))
        changed = True
        while changed:
            changed = False
            for left, right in wrappers:
                if text.startswith(left) and text.endswith(right):
                    text = text[len(left) : -len(right)].strip()
                    changed = True

        text = text.replace("\u2212", "-").replace("\ufe63", "-").replace("\uff0d", "-")
        text = text.replace("\\left", "").replace("\\right", "")
        text = re.sub(r"\\[dt]frac", r"\\frac", text)
        text = re.sub(
            r"\\(begin|end)\{(?:p|b|B|v|V)?matrix\}",
            lambda match: rf"\{match.group(1)}{{matrix}}",
            text,
        )
        text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
        text = re.sub(r"\s+", "", text)
        text = text.rstrip(".;")
        if not text:
            return None

        if _NUMBER_RE.fullmatch(text):
            try:
                number = Decimal(text)
                if number.is_finite():
                    text = _canonical_decimal(number)
            except InvalidOperation:
                pass
        return text
    except Exception:
        return None


def _reference_candidates(reference: Any) -> list[Any]:
    if isinstance(reference, (list, tuple, set, frozenset)):
        return list(reference)
    return [reference]


def _choice_label(text: str | None) -> str | None:
    if text is None:
        return None
    patterns = (
        r"\(?([A-Ea-e])\)?",
        r"\\text\{\(?([A-Ea-e])\)?\}",
        r"\\text(?:bf|rm)\{\(?([A-Ea-e])\)?\}",
        r"\\mathrm\{\(?([A-Ea-e])\)?\}",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def _numeric_value(text: str | None) -> tuple[Decimal, bool] | None:
    if text is None:
        return None
    percent = text.endswith("%") or text.endswith(r"\%")
    number_text = text[:-2] if text.endswith(r"\%") else text[:-1] if percent else text
    if not _NUMBER_RE.fullmatch(number_text):
        return None
    try:
        value = Decimal(number_text)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    return (value / Decimal(100) if percent else value, percent)


def _split_top_level(text: str, delimiter: str) -> list[str] | None:
    parts: list[str] = []
    start = 0
    braces = brackets = parentheses = 0
    index = 0
    while index < len(text):
        character = text[index]
        escaped = _is_escaped(text, index)
        if not escaped:
            if character == "{":
                braces += 1
            elif character == "}":
                braces -= 1
            elif character == "[":
                brackets += 1
            elif character == "]":
                brackets -= 1
            elif character == "(":
                parentheses += 1
            elif character == ")":
                parentheses -= 1
        if min(braces, brackets, parentheses) < 0:
            return None
        if braces == brackets == parentheses == 0 and text.startswith(delimiter, index):
            parts.append(text[start:index])
            index += len(delimiter)
            start = index
            continue
        index += 1
    if braces or brackets or parentheses:
        return None
    parts.append(text[start:])
    return parts


def _matrix_value(text: str | None) -> tuple[tuple[str, ...], ...] | None:
    if text is None:
        return None
    match = _MATRIX_RE.fullmatch(text)
    if match is None:
        return None
    rows = _split_top_level(match.group(1), r"\\")
    if not rows:
        return None
    result: list[tuple[str, ...]] = []
    width: int | None = None
    for row in rows:
        cells = _split_top_level(row, "&")
        if not cells or any(not cell for cell in cells):
            return None
        normalized = tuple(normalize_math_answer(cell) or "" for cell in cells)
        if any(not cell for cell in normalized):
            return None
        if width is None:
            width = len(normalized)
        elif width != len(normalized):
            return None
        result.append(normalized)
    return tuple(result)


def _set_value(text: str | None) -> tuple[str, ...] | None:
    if text is None:
        return None
    if text.startswith(r"\{") and text.endswith(r"\}"):
        body = text[2:-2]
    elif text.startswith("{") and text.endswith("}"):
        body = text[1:-1]
    else:
        return None
    parts = _split_top_level(body, ",")
    if parts is None:
        return None
    normalized = [normalize_math_answer(part) for part in parts]
    if any(part is None for part in normalized):
        return None
    return tuple(sorted(part for part in normalized if part is not None))


def _sequence_value(text: str | None) -> tuple[str, str, tuple[str, ...]] | None:
    if text is None or len(text) < 2:
        return None
    if text[0] not in "([" or text[-1] not in ")]":
        return None
    parts = _split_top_level(text[1:-1], ",")
    if parts is None or len(parts) < 2:
        return None
    normalized = [normalize_math_answer(part) for part in parts]
    if any(part is None for part in normalized):
        return None
    return text[0], text[-1], tuple(part for part in normalized if part is not None)


def _load_math_verify() -> tuple[Callable[..., Any], Callable[..., Any]] | None:
    try:
        from math_verify import parse, verify
    except (ImportError, ModuleNotFoundError):
        return None
    return parse, verify


def _contains_plain_words(text: str) -> bool:
    """Detect prose while ignoring LaTeX command names such as ``\\frac``."""

    without_commands = re.sub(r"\\[A-Za-z]+", "", text)
    return _PLAIN_WORD_RE.search(without_commands) is not None


class AnswerVerifier:
    """Verify with conservative exact/structural rules, then ``math-verify``."""

    def __init__(
        self,
        *,
        use_math_verify: bool = True,
        symbolic_parse: Callable[..., Any] | None = None,
        symbolic_verify: Callable[..., Any] | None = None,
    ) -> None:
        if (symbolic_parse is None) != (symbolic_verify is None):
            raise ValueError(
                "symbolic_parse and symbolic_verify must be provided together"
            )
        self.use_math_verify = use_math_verify
        self._symbolic_parse = symbolic_parse
        self._symbolic_verify = symbolic_verify
        self._symbolic_loaded = symbolic_parse is not None
        self._injected_symbolic = symbolic_parse is not None

    def _symbolic_functions(
        self,
    ) -> tuple[Callable[..., Any], Callable[..., Any]] | None:
        if not self.use_math_verify:
            return None
        if not self._symbolic_loaded:
            loaded = _load_math_verify()
            if loaded is not None:
                self._symbolic_parse, self._symbolic_verify = loaded
            self._symbolic_loaded = True
        if self._symbolic_parse is None or self._symbolic_verify is None:
            return None
        return self._symbolic_parse, self._symbolic_verify

    @staticmethod
    def _result(
        correct: bool,
        prediction: str | None,
        reference: Any,
        normalized_prediction: str | None,
        normalized_reference: str | None,
        method: str,
        **kwargs: Any,
    ) -> VerificationResult:
        return VerificationResult(
            correct=correct,
            prediction=prediction,
            reference=reference,
            normalized_prediction=normalized_prediction,
            normalized_reference=normalized_reference,
            method=method,
            **kwargs,
        )

    def _ordered_elements_equal(
        self, prediction: tuple[str, ...], reference: tuple[str, ...]
    ) -> bool:
        return len(prediction) == len(reference) and all(
            self.verify(predicted, expected).correct
            for predicted, expected in zip(prediction, reference)
        )

    def _unordered_elements_equal(
        self, prediction: tuple[str, ...], reference: tuple[str, ...]
    ) -> bool:
        if len(prediction) != len(reference):
            return False
        unmatched = list(reference)
        for predicted in prediction:
            for index, expected in enumerate(unmatched):
                if self.verify(predicted, expected).correct:
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched

    def verify(self, prediction: object, reference: Any) -> VerificationResult:
        """Return a structured verdict and never propagate verifier failures."""

        prediction_text = _safe_string(prediction)
        normalized_prediction = normalize_math_answer(prediction)
        candidates = _reference_candidates(reference)
        normalized_candidates = [normalize_math_answer(item) for item in candidates]
        display_reference = candidates[0] if candidates else reference
        display_normalized = normalized_candidates[0] if normalized_candidates else None

        try:
            for candidate, normalized_reference in zip(
                candidates, normalized_candidates
            ):
                if (
                    normalized_prediction is not None
                    and normalized_reference is not None
                    and normalized_prediction == normalized_reference
                ):
                    return self._result(
                        True,
                        prediction_text,
                        candidate,
                        normalized_prediction,
                        normalized_reference,
                        "normalized_exact",
                    )

            for candidate, normalized_reference in zip(
                candidates, normalized_candidates
            ):
                prediction_number = _numeric_value(normalized_prediction)
                reference_number = _numeric_value(normalized_reference)
                if (
                    prediction_number is not None
                    and prediction_number[1]
                    or reference_number is not None
                    and reference_number[1]
                ):
                    correct = (
                        prediction_number is not None
                        and reference_number is not None
                        and prediction_number[0] == reference_number[0]
                    )
                    return self._result(
                        correct,
                        prediction_text,
                        candidate,
                        normalized_prediction,
                        normalized_reference,
                        "percentage",
                    )

                prediction_choice = _choice_label(normalized_prediction)
                reference_choice = _choice_label(normalized_reference)
                if prediction_choice is not None or reference_choice is not None:
                    return self._result(
                        prediction_choice is not None
                        and reference_choice is not None
                        and prediction_choice == reference_choice,
                        prediction_text,
                        candidate,
                        normalized_prediction,
                        normalized_reference,
                        "choice",
                    )

                prediction_matrix = _matrix_value(normalized_prediction)
                reference_matrix = _matrix_value(normalized_reference)
                if prediction_matrix is not None or reference_matrix is not None:
                    correct = (
                        prediction_matrix is not None
                        and reference_matrix is not None
                        and len(prediction_matrix) == len(reference_matrix)
                        and all(
                            self._ordered_elements_equal(predicted_row, expected_row)
                            for predicted_row, expected_row in zip(
                                prediction_matrix, reference_matrix
                            )
                        )
                    )
                    return self._result(
                        correct,
                        prediction_text,
                        candidate,
                        normalized_prediction,
                        normalized_reference,
                        "matrix",
                    )

                prediction_set = _set_value(normalized_prediction)
                reference_set = _set_value(normalized_reference)
                if prediction_set is not None or reference_set is not None:
                    correct = (
                        prediction_set is not None
                        and reference_set is not None
                        and self._unordered_elements_equal(
                            prediction_set, reference_set
                        )
                    )
                    return self._result(
                        correct,
                        prediction_text,
                        candidate,
                        normalized_prediction,
                        normalized_reference,
                        "set",
                    )

                prediction_sequence = _sequence_value(normalized_prediction)
                reference_sequence = _sequence_value(normalized_reference)
                if prediction_sequence is not None or reference_sequence is not None:
                    correct = (
                        prediction_sequence is not None
                        and reference_sequence is not None
                        and prediction_sequence[:2] == reference_sequence[:2]
                        and self._ordered_elements_equal(
                            prediction_sequence[2], reference_sequence[2]
                        )
                    )
                    return self._result(
                        correct,
                        prediction_text,
                        candidate,
                        normalized_prediction,
                        normalized_reference,
                        "sequence",
                    )

            symbolic = self._symbolic_functions()
            if symbolic is None or prediction_text is None:
                return self._result(
                    False,
                    prediction_text,
                    display_reference,
                    normalized_prediction,
                    display_normalized,
                    "normalized_exact",
                )

            parse, verify = symbolic
            symbolic_errors: list[str] = []
            attempted = 0
            for candidate in candidates:
                candidate_text = _safe_string(candidate)
                if candidate_text is None:
                    continue
                # math-verify drops the base subscript in e.g. 2242_5 and
                # otherwise incorrectly matches the decimal integer 2242.
                # Preserve exact matching for base notation until it has an
                # explicit, semantics-preserving conversion implementation.
                base_notation = r"[+-]?\d+_(?:\d+|\{\d+\})"
                if re.fullmatch(
                    base_notation, normalized_prediction or ""
                ) or re.fullmatch(
                    base_notation, normalize_math_answer(candidate_text) or ""
                ):
                    continue
                # math-verify may extract a number from surrounding prose (for
                # example ``The 42!``) and incorrectly declare it equivalent to
                # ``42``. Prose answers are intentionally exact-match only.
                if _contains_plain_words(prediction_text) or _contains_plain_words(
                    candidate_text
                ):
                    continue
                attempted += 1
                try:
                    if self._injected_symbolic:
                        reference_expression = parse(candidate_text)
                        prediction_expression = parse(prediction_text)
                        correct = bool(
                            verify(reference_expression, prediction_expression)
                        )
                    else:
                        correct = compare_math_expressions(
                            normalize_math_answer(candidate_text) or candidate_text,
                            normalized_prediction or prediction_text,
                        )
                    if correct:
                        return self._result(
                            True,
                            prediction_text,
                            candidate,
                            normalized_prediction,
                            normalize_math_answer(candidate),
                            "math_verify",
                            symbolic_checked=True,
                        )
                except Exception as exc:
                    symbolic_errors.append(f"{type(exc).__name__}: {exc}")

            if attempted and len(symbolic_errors) == attempted:
                return self._result(
                    False,
                    prediction_text,
                    display_reference,
                    normalized_prediction,
                    display_normalized,
                    "math_verify",
                    status="error",
                    symbolic_checked=True,
                    error="; ".join(symbolic_errors),
                )
            if attempted == 0:
                return self._result(
                    False,
                    prediction_text,
                    display_reference,
                    normalized_prediction,
                    display_normalized,
                    "normalized_exact",
                )
            return self._result(
                False,
                prediction_text,
                display_reference,
                normalized_prediction,
                display_normalized,
                "math_verify",
                symbolic_checked=True,
                error="; ".join(symbolic_errors) or None,
            )
        except Exception as exc:
            return self._result(
                False,
                prediction_text,
                display_reference,
                normalized_prediction,
                display_normalized,
                "verification_error",
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )


__all__ = ["AnswerVerifier", "normalize_math_answer"]
