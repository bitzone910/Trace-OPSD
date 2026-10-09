"""Read historical asset identities without rewriting their bytes or hashes.

Only persisted formats are compatible. Old package imports, commands and
environment variables are deliberately not supported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LEGACY_NAMESPACE = "toolelite"
CURRENT_NAMESPACE = "trace_opsd"
# Exact pre-rename scorer source: accepted only when reporting saved results.
HISTORICAL_SAMPLING_METRICS_SHA256 = (
    "c2a5370c3ab8f0063d13c89b534180d5e075ef275c062903d04372d25b36f966"
)
METADATA_FILES = frozenset(
    {
        "trace_opsd_grpo_run.json",
        "trace_opsd_grpo_manifest.json",
        "trace_opsd_protocol.json",
    }
)
IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "protocol_version",
        "tool_protocol",
        "version",
        "data_source",
        "trace_credit_version",
        "metric_key",
        "project",
    }
)


def is_historical_identity(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(LEGACY_NAMESPACE + ".")


def identity_for_source(current: str, source_identity: Any) -> str:
    """Label inferred historical provenance with its source namespace."""
    if is_historical_identity(source_identity) and current.startswith(
        CURRENT_NAMESPACE + "."
    ):
        return LEGACY_NAMESPACE + current[len(CURRENT_NAMESPACE) :]
    return current


def canonical_identity(value: Any) -> Any:
    if isinstance(value, str):
        if value == "ToolElite":
            return "Trace-OPSD"
        value = value.replace("/toolelite_math/", "/trace_opsd_math/")
        for separator in (".", "_"):
            prefix = LEGACY_NAMESPACE + separator
            if value.startswith(prefix):
                return CURRENT_NAMESPACE + value[len(LEGACY_NAMESPACE) :]
    return value


def namespace_matches(actual: Any, expected: Any) -> bool:
    """Compare a known persisted identity, never normalize data for hashing."""
    return canonical_identity(actual) == canonical_identity(expected)


def metadata_equal(actual: Any, expected: Any) -> bool:
    """Allow namespace-only differences in identity fields; reject other edits."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            namespace_matches(actual[key], expected[key])
            if key in IDENTITY_FIELDS
            else metadata_equal(actual[key], expected[key])
            for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            metadata_equal(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected


def metadata_path(path: str | Path) -> Path:
    """Resolve an existing historical filename; never choose a write target."""
    path = Path(path)
    if path.name not in METADATA_FILES:
        return path
    legacy = path.with_name(path.name.replace(CURRENT_NAMESPACE, LEGACY_NAMESPACE, 1))
    if path.is_file() and legacy.is_file():
        current_data = json.loads(path.read_text(encoding="utf-8-sig"))
        legacy_data = json.loads(legacy.read_text(encoding="utf-8-sig"))
        if not metadata_equal(current_data, legacy_data):
            raise ValueError(
                f"Conflicting current and historical metadata: {path}, {legacy}"
            )
    return path if path.is_file() or not legacy.is_file() else legacy


def read_json(path: str | Path, **kwargs: Any) -> Any:
    """Return the original JSON values, including historical version labels."""
    return json.loads(metadata_path(path).read_text(encoding="utf-8-sig"), **kwargs)


def canonical_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Name newly computed metrics consistently; old on-disk reports stay raw."""
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        name = key.replace("/" + LEGACY_NAMESPACE + "_math/", "/trace_opsd_math/")
        if name in result and result[name] != value:
            raise ValueError(f"Conflicting metric aliases: {name}")
        result[name] = value
    return result
