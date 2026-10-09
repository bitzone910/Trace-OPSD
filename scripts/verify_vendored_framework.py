#!/usr/bin/env python3
"""Verify a vendored framework snapshot without relying on nested Git metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

METADATA_NAME = "TRACE_OPSD_VENDOR.json"
DIGEST_ALGORITHM = "trace_opsd-vendor-sha256-v1"

# Editable installs and local inspection can create these paths. They are not part
# of the upstream snapshot and must not invalidate its source digest.
IGNORED_DIRECTORY_NAMES = {
    ".cache",
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "hf_cache",
    "llamaboard_cache",
    "ms_cache",
    "om_cache",
    "output",
    "outputs",
    "saves",
    "swanlog",
    "venv",
    "wandb",
}
IGNORED_FILE_SUFFIXES = (".egg-info", ".pyc", ".pyo")


class VendorVerificationError(RuntimeError):
    """Raised when a vendor snapshot does not match its declared metadata."""


def _is_ignored(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if relative.as_posix() == METADATA_NAME:
        return True
    if any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts[:-1]):
        return True
    return relative.name.endswith(IGNORED_FILE_SUFFIXES)


def _canonical_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    # Git may check text out as CRLF on Windows and LF on Linux. Normalize only
    # valid UTF-8 files so the same vendored source verifies on both platforms.
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def calculate_snapshot(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise VendorVerificationError(f"vendor directory does not exist: {root}")

    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or _is_ignored(path, root):
            continue
        relative = path.relative_to(root).as_posix()
        content = _canonical_bytes(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
        file_count += 1

    return {
        "algorithm": DIGEST_ALGORITHM,
        "file_count": file_count,
        "sha256": digest.hexdigest(),
    }


def load_metadata(root: Path) -> dict[str, Any]:
    metadata_path = root / METADATA_NAME
    if not metadata_path.is_file():
        raise VendorVerificationError(f"missing vendor metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VendorVerificationError(
            f"cannot read vendor metadata {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise VendorVerificationError("vendor metadata must be a JSON object")
    return metadata


def verify_snapshot(
    root: Path,
    *,
    expected_name: str | None = None,
    expected_revision: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (root / ".git").exists():
        raise VendorVerificationError(
            f"nested Git metadata is not allowed in a vendor snapshot: {root / '.git'}"
        )

    metadata = load_metadata(root)
    if metadata.get("schema_version") != 1:
        raise VendorVerificationError(
            f"unsupported vendor schema_version: {metadata.get('schema_version')!r}"
        )
    if expected_name is not None and metadata.get("name") != expected_name:
        raise VendorVerificationError(
            f"vendor name mismatch: expected {expected_name!r}, "
            f"found {metadata.get('name')!r}"
        )

    upstream = metadata.get("upstream")
    if not isinstance(upstream, dict):
        raise VendorVerificationError("vendor metadata is missing upstream information")
    revision = upstream.get("revision")
    if expected_revision is not None and revision != expected_revision:
        raise VendorVerificationError(
            f"vendor revision mismatch: expected {expected_revision}, found {revision}"
        )

    declared = metadata.get("snapshot")
    if not isinstance(declared, dict):
        raise VendorVerificationError("vendor metadata is missing snapshot information")
    if declared.get("algorithm") != DIGEST_ALGORITHM:
        raise VendorVerificationError(
            f"unsupported snapshot algorithm: {declared.get('algorithm')!r}"
        )

    calculated = calculate_snapshot(root)
    if declared.get("file_count") != calculated["file_count"]:
        raise VendorVerificationError(
            "vendor file count mismatch: "
            f"expected {declared.get('file_count')}, found {calculated['file_count']}"
        )
    if declared.get("sha256") != calculated["sha256"]:
        raise VendorVerificationError(
            "vendor source digest mismatch: "
            f"expected {declared.get('sha256')}, found {calculated['sha256']}"
        )
    return metadata, calculated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="vendored framework directory")
    parser.add_argument("--expected-name")
    parser.add_argument("--expected-revision")
    parser.add_argument(
        "--calculate-only",
        action="store_true",
        help="calculate canonical snapshot fields without reading metadata",
    )
    parser.add_argument(
        "--print-digest",
        action="store_true",
        help="print only the verified SHA-256 digest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.path.resolve()
    try:
        if args.calculate_only:
            print(json.dumps(calculate_snapshot(root), sort_keys=True))
            return 0
        metadata, calculated = verify_snapshot(
            root,
            expected_name=args.expected_name,
            expected_revision=args.expected_revision,
        )
    except VendorVerificationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.print_digest:
        print(calculated["sha256"])
    else:
        revision = metadata["upstream"]["revision"]
        print(
            f"Verified {metadata['name']} vendor snapshot at {revision} "
            f"({calculated['file_count']} files, sha256={calculated['sha256']})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
