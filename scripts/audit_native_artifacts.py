#!/usr/bin/env python3
"""Reject framework-owned native artifacts in source or installed wheels."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path

NATIVE_SUFFIXES = {
    ".a",
    ".cubin",
    ".dll",
    ".dylib",
    ".fatbin",
    ".lib",
    ".o",
    ".obj",
    ".ptx",
    ".pyd",
    ".so",
}


def _is_native(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in NATIVE_SUFFIXES)


def audit_source(path: Path) -> list[str]:
    return [
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and _is_native(item)
    ]


def audit_distribution(name: str) -> list[str]:
    distribution = metadata.distribution(name)
    return [
        str(item)
        for item in distribution.files or ()
        if _is_native(Path(str(item)))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("--distribution", action="append", default=[])
    args = parser.parse_args()

    failures: list[str] = []
    for source in args.source:
        resolved = source.resolve()
        if not resolved.is_dir():
            failures.append(f"source directory does not exist: {resolved}")
            continue
        failures.extend(f"{resolved}: {item}" for item in audit_source(resolved))
    for distribution in args.distribution:
        try:
            failures.extend(
                f"distribution {distribution}: {item}"
                for item in audit_distribution(distribution)
            )
        except metadata.PackageNotFoundError:
            failures.append(f"distribution is not installed: {distribution}")

    if failures:
        print("Native artifacts are not allowed in vendored Python frameworks:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Vendored framework native-artifact audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
