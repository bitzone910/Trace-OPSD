"""Merge a local verl FSDP checkpoint into a Trace-OPSD Hugging Face export."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from trace_opsd.compat import (
    canonical_identity,
    metadata_path,
    namespace_matches,
    read_json,
)

from .checkpoint_retention import read_best_checkpoint, verify_committed_checkpoint
from .credit_revision import (
    checkpoint_credit_state,
    resolve_trace_credit_config,
    trace_credit_aggregation,
    validate_trace_credit_manifest,
    verify_checkpoint_credit_state,
)
from .preflight import EXPECTED_VERL_COMMIT, PreflightError
from .trace_opsd import LEGACY_TOKEN_TEACHER_MODE, TRACE_TEACHER_MODE

LEGACY_TRACE_TEACHER_MODE = "current_policy_solution_hindsight.v2"


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"missing or invalid run manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise PreflightError(f"run manifest must be an object: {path}")
    return value


def resolve_actor_checkpoint(run_dir: Path, checkpoint: str) -> tuple[int, Path]:
    manifest_path = metadata_path(run_dir / "trace_opsd_grpo_run.json")
    run_manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    trace_required = run_manifest.get("algorithm") == "trace_opsd"
    if trace_required and run_manifest.get("trace_teacher_mode") not in {
        TRACE_TEACHER_MODE,
        LEGACY_TOKEN_TEACHER_MODE,
        LEGACY_TRACE_TEACHER_MODE,
    }:
        raise PreflightError(
            "TRACE run uses incompatible hindsight semantics and is not "
            "compatible with this exporter"
        )
    if (
        trace_required
        and run_manifest.get("trace_teacher_mode") != LEGACY_TRACE_TEACHER_MODE
    ):
        validate_trace_credit_manifest(run_manifest)

    def complete(path: Path) -> bool:
        global_manifest = path.parent / "manifest.json"
        if global_manifest.is_file():
            try:
                metadata = _read_json(global_manifest)
                if namespace_matches(
                    metadata.get("schema_version"),
                    "trace_opsd.grpo.global-checkpoint.v1",
                ):
                    verify_committed_checkpoint(path.parent)
            except (PreflightError, RuntimeError, ValueError, OSError):
                return False
            if namespace_matches(
                metadata.get("schema_version"), "trace_opsd.grpo.global-checkpoint.v1"
            ):
                if (
                    trace_required
                    and run_manifest.get("trace_teacher_mode")
                    != LEGACY_TRACE_TEACHER_MODE
                ):
                    # Credit identity is a metadata error, not an interrupted
                    # weight commit. Never silently fall back to an older step.
                    verify_checkpoint_credit_state(
                        run_manifest,
                        _read_json(path.parent / "trainer_state.json"),
                        step=metadata["global_step"],
                    )
                return True
            # Modern actor manifests must never bypass global integrity checks.
            if (path / "manifest.json").is_file():
                return False
        actor_complete = (path / "manifest.json").is_file() or any(
            path.glob("model_world_size_*_rank_*.pt")
        )
        if not actor_complete:
            return False
        return True

    if checkpoint == "best":
        best = read_best_checkpoint(run_dir)
        step = int(best["step"])
        candidate = run_dir / f"global_step_{step}" / "actor"
        if candidate.is_dir() and complete(candidate):
            return step, candidate
        raise PreflightError(f"best actor checkpoint not found: {candidate}")
    if checkpoint == "latest":
        marker = run_dir / "latest_checkpointed_iteration.txt"
        if marker.is_file():
            value = marker.read_text(encoding="utf-8").strip()
            match = re.search(r"(\d+)$", value)
            if match:
                step = int(match.group(1))
                candidate = run_dir / f"global_step_{step}" / "actor"
                if candidate.is_dir() and complete(candidate):
                    return step, candidate
        candidates = []
        for path in run_dir.glob("global_step_*/actor"):
            match = re.fullmatch(r"global_step_(\d+)", path.parent.name)
            if match and complete(path):
                candidates.append((int(match.group(1)), path))
        if not candidates:
            raise PreflightError(f"no actor checkpoint found under {run_dir}")
        return max(candidates, key=lambda item: item[0])
    if checkpoint.isdigit():
        step = int(checkpoint)
        path = run_dir / f"global_step_{step}" / "actor"
    else:
        path = Path(checkpoint).resolve()
        match = re.search(r"global_step_(\d+)", path.as_posix())
        step = int(match.group(1)) if match else -1
    if not path.is_dir() or not complete(path):
        raise PreflightError(f"actor checkpoint not found: {path}")
    return step, path


def export_checkpoint(
    *,
    run_dir: Path,
    checkpoint: str,
    target_dir: Path,
) -> Path:
    run_dir = run_dir.resolve()
    manifest = _read_json(run_dir / "trace_opsd_grpo_run.json")
    step, actor_dir = resolve_actor_checkpoint(run_dir, checkpoint)
    trace_config = manifest.get("trace_opsd")
    credit_provenance = {}
    if (
        manifest.get("algorithm") == "trace_opsd"
        and manifest.get("trace_teacher_mode") != LEGACY_TRACE_TEACHER_MODE
    ):
        if step < 0:
            raise PreflightError("TRACE export requires a checkpoint with a known step")
        trace_config = resolve_trace_credit_config(run_dir, step)
        credit_provenance = {
            "trace_credit_aggregation": trace_credit_aggregation(manifest),
            **checkpoint_credit_state(run_dir, step),
        }
    target_dir = target_dir.resolve()
    if target_dir.exists() and any(target_dir.iterdir()):
        raise PreflightError(
            f"export target already exists and is not empty: {target_dir}"
        )
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    local_hf = actor_dir / "huggingface"
    has_local_weights = local_hf.is_dir() and any(local_hf.glob("*.safetensors"))
    if has_local_weights:
        shutil.copytree(local_hf, target_dir, dirs_exist_ok=True)
    else:
        # Read-only compatibility for checkpoints written before elastic DCP.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(actor_dir),
                "--target_dir",
                str(target_dir),
            ],
            check=True,
        )
    source_model = Path(str(manifest["model_path"])).resolve()
    source_protocol = metadata_path(source_model / "trace_opsd_protocol.json")
    target_protocol = target_dir / "trace_opsd_protocol.json"
    if source_protocol.is_file():
        protocol = dict(read_json(source_protocol))
        for key in ("schema_version", "tool_protocol"):
            if key in protocol:
                protocol[key] = canonical_identity(protocol[key])
        target_protocol.write_text(
            json.dumps(protocol, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    else:
        protocol = {
            "schema_version": "trace_opsd.model-protocol.v1",
            "chat_template": "qwen",
            "tool_protocol": "trace_opsd.xml-tools.v3",
            "model_type": manifest.get("model_type"),
            "thinking_mode": manifest.get("thinking_mode", "off"),
        }
        target_protocol.write_text(
            json.dumps(protocol, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    provenance = {
        "schema_version": "trace_opsd.grpo.hf-export.v1",
        "ready_for_verl": True,
        "source_checkpoint_step": step,
        "source_checkpoint": str(actor_dir),
        "source_model": str(source_model),
        "model_type": manifest.get("model_type"),
        "model_architecture": manifest.get("model_architecture"),
        "thinking_mode": manifest.get("thinking_mode", "off"),
        "source_dataset_digest": manifest["dataset_digest"],
        "verl_commit": EXPECTED_VERL_COMMIT,
        "algorithm": manifest.get("algorithm", "grpo"),
        "distillation_enabled": False,
        "trace_opsd_enabled": manifest.get("algorithm") == "trace_opsd",
        "trace_opsd": trace_config,
        "trace_teacher_mode": manifest.get("trace_teacher_mode"),
        **credit_provenance,
    }
    (target_dir / "artifact-manifest.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (target_dir / "trace_opsd_grpo_manifest.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        from transformers import AutoConfig, AutoTokenizer

        AutoConfig.from_pretrained(
            target_dir, local_files_only=True, trust_remote_code=False
        )
        AutoTokenizer.from_pretrained(
            target_dir, local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise PreflightError(
            f"merged checkpoint is not a loadable local Hugging Face export: {exc}"
        ) from exc
    return target_dir


__all__ = ["export_checkpoint", "resolve_actor_checkpoint"]
