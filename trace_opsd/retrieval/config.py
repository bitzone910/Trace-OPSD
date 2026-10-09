"""Project-wide paths and defaults."""

from __future__ import annotations

from trace_opsd.assets import REPO_ROOT, asset_path

PROJECT_ROOT = REPO_ROOT
DEFAULT_MODEL_PATH = asset_path("models", "e5-base-v2")
DEFAULT_MODEL = str(DEFAULT_MODEL_PATH)
