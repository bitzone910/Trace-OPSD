"""Project-wide paths and defaults."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "e5-base-v2"
DEFAULT_MODEL = str(DEFAULT_MODEL_PATH)
