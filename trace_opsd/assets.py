"""Resolve external inputs and runtime outputs for the public training pipeline."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _check_environment_names() -> None:
    legacy = sorted(name for name in os.environ if name.startswith("TOOLELITE_"))
    if legacy:
        raise ValueError(
            "Old project environment variables are unsupported; rename to TRACE_OPSD_: "
            + ", ".join(legacy)
        )


def resolve_assets_root(value: str | Path | None = None) -> Path:
    """Explicit path, environment, then the repository's sibling assets folder."""
    _check_environment_names()
    selected = value or os.environ.get("TRACE_OPSD_ASSET_ROOT")
    return Path(selected or REPO_ROOT.parent / "assets").expanduser().resolve()


def resolve_work_root(
    value: str | Path | None = None, *, assets_root: str | Path | None = None
) -> Path:
    _check_environment_names()
    selected = value or os.environ.get("TRACE_OPSD_WORK_ROOT")
    return (
        Path(selected or resolve_assets_root(assets_root) / "runs")
        .expanduser()
        .resolve()
    )


def asset_path(*parts: str, assets_root: str | Path | None = None) -> Path:
    return resolve_assets_root(assets_root).joinpath(*parts)


def configure_asset_environment(
    assets_root: str | Path | None = None, work_root: str | Path | None = None
) -> tuple[Path, Path]:
    """Make child processes use the same explicitly resolved roots."""
    assets = resolve_assets_root(assets_root)
    work = resolve_work_root(work_root, assets_root=assets)
    os.environ["TRACE_OPSD_ASSET_ROOT"] = str(assets)
    os.environ["TRACE_OPSD_WORK_ROOT"] = str(work)
    return assets, work
