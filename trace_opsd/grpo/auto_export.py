"""Publish evaluation models after committed TRACE checkpoints."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from trace_opsd.compat import namespace_matches, read_json

from .checkpoint_retention import verify_committed_checkpoint
from .export import export_checkpoint


def publish_trace_export(
    *, run_dir: Path, export_dir: Path, step: int, total_steps: int
) -> Path | None:
    """Export periodic and final steps before the next checkpoint can prune them."""
    if step % 100 != 0 and step != total_steps:
        return None

    run_dir = run_dir.resolve()
    export_dir = export_dir.resolve()
    committed = run_dir / f"global_step_{step}" / "manifest.json"
    if not committed.is_file():
        raise RuntimeError(f"TRACE checkpoint is not committed: {committed}")
    verify_committed_checkpoint(committed.parent)

    target = export_dir / f"step-{step}"
    if target.exists():
        manifest_path = target / "artifact-manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if (
                namespace_matches(
                    manifest.get("schema_version"), "trace_opsd.grpo.hf-export.v1"
                )
                and manifest.get("source_checkpoint_step") == step
                and manifest.get("source_checkpoint") == str(committed.parent / "actor")
                and (target / "config.json").is_file()
                and any(target.glob("*.safetensors"))
            ):
                return target
        raise RuntimeError(f"existing TRACE export is incomplete: {target}")

    export_dir.mkdir(parents=True, exist_ok=True)
    attempt = export_dir / f".step-{step}.{uuid.uuid4().hex}.incomplete"
    try:
        export_checkpoint(run_dir=run_dir, checkpoint=str(step), target_dir=attempt)
        os.replace(attempt, target)
    except Exception:
        if attempt.is_dir():
            shutil.rmtree(attempt)
        raise
    return target


def recover_trace_exports(
    *, run_dir: Path, export_dir: Path, total_steps: int
) -> list[Path]:
    """Repair due exports while their complete checkpoint sources still exist."""
    published = []
    for checkpoint in sorted(run_dir.glob("global_step_*")):
        suffix = checkpoint.name.removeprefix("global_step_")
        if not suffix.isdigit() or not checkpoint.is_dir():
            continue
        target = publish_trace_export(
            run_dir=run_dir,
            export_dir=export_dir,
            step=int(suffix),
            total_steps=total_steps,
        )
        if target is not None:
            published.append(target)
    return published
