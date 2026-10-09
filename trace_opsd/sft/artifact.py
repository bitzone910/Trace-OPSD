"""Project-local, atomic checkpoint manifests for Trace-OPSD SFT."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from trace_opsd.compat import namespace_matches, read_json


class ArtifactStoreError(RuntimeError):
    """Raised when a local checkpoint cannot be committed or verified."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class ArtifactObject:
    """One immutable file recorded relative to a checkpoint directory."""

    key: str
    size: int
    sha256: str


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: str
    run_id: str
    step: int
    attempt_id: str
    world_size: int
    training_fingerprint: str
    created_at: str
    objects: tuple[ArtifactObject, ...]
    metrics: Mapping[str, float]
    consumed_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["objects"] = [asdict(item) for item in self.objects]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointManifest":
        try:
            objects = tuple(ArtifactObject(**item) for item in value["objects"])
            return cls(
                schema_version=str(value["schema_version"]),
                run_id=str(value["run_id"]),
                step=int(value["step"]),
                attempt_id=str(value["attempt_id"]),
                world_size=int(value["world_size"]),
                training_fingerprint=str(value["training_fingerprint"]),
                created_at=str(value["created_at"]),
                objects=objects,
                metrics={str(k): float(v) for k, v in value.get("metrics", {}).items()},
                consumed_samples=int(value.get("consumed_samples", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactStoreError("invalid checkpoint manifest") from exc


class CheckpointRepository:
    """Atomic checkpoint directories and pointers under ``checkpoints/sft``."""

    SCHEMA_VERSION = "trace_opsd.sft.checkpoint.v2"
    _CHECKPOINT_RE = re.compile(r"checkpoint-(\d{9})$")

    def __init__(
        self,
        root: Path,
        run_id: str,
        *,
        keep_last: int = 2,
        keep_best: bool = False,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
            raise ValueError("run_id must be a nonempty path-safe value")
        if keep_last < 1:
            raise ValueError("keep_last must be positive")
        self.root = Path(root).resolve()
        self.run_id = run_id
        self.run_dir = self.root / run_id
        self.keep_last = keep_last
        self.keep_best = keep_best

    @property
    def run_manifest_path(self) -> Path:
        return self.run_dir / "run.json"

    def attempt_path(self, step: int, attempt_id: str) -> Path:
        return self.run_dir / ".incomplete" / f"checkpoint-{step:09d}-{attempt_id}"

    def checkpoint_path(self, step: int) -> Path:
        return self.run_dir / f"checkpoint-{step:09d}"

    def pointer_path(self, selector: str) -> Path:
        if selector not in {"latest", "best"}:
            raise ValueError("checkpoint selector must be latest or best")
        return self.run_dir / f"{selector}.json"

    def write_run_manifest(self, value: Mapping[str, Any]) -> None:
        serialized = json.dumps(value, sort_keys=True, indent=2) + "\n"
        if self.run_manifest_path.is_file():
            existing = self.run_manifest_path.read_text(encoding="utf-8")
            if existing != serialized:
                raise ArtifactStoreError(
                    f"run metadata changed for existing run {self.run_id!r}"
                )
            return
        _atomic_write_json(self.run_manifest_path, value)

    def commit(
        self,
        *,
        step: int,
        attempt_id: str,
        world_size: int,
        training_fingerprint: str,
        metrics: Mapping[str, float],
        best: bool,
        consumed_samples: int = 0,
    ) -> CheckpointManifest:
        temporary = self.attempt_path(step, attempt_id)
        final = self.checkpoint_path(step)
        objects = tuple(self._collect_objects(temporary))
        if not objects:
            raise ArtifactStoreError(f"checkpoint contains no objects: {temporary}")
        manifest = CheckpointManifest(
            schema_version=self.SCHEMA_VERSION,
            run_id=self.run_id,
            step=step,
            attempt_id=attempt_id,
            world_size=world_size,
            training_fingerprint=training_fingerprint,
            created_at=_utc_now(),
            objects=objects,
            metrics=dict(metrics),
            consumed_samples=max(0, int(consumed_samples)),
        )
        _atomic_write_json(temporary / "manifest.json", manifest.to_dict())
        if final.exists():
            raise ArtifactStoreError(f"checkpoint already exists: {final}")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final)
        self._write_pointer("latest", manifest)
        if best:
            self._write_pointer("best", manifest)
        self._prune()
        return manifest

    def _write_pointer(self, selector: str, manifest: CheckpointManifest) -> None:
        _atomic_write_json(
            self.pointer_path(selector),
            {
                "schema_version": self.SCHEMA_VERSION,
                "run_id": self.run_id,
                "step": manifest.step,
                "checkpoint": self.checkpoint_path(manifest.step).name,
            },
        )

    def resolve(self, selector: str | Path) -> tuple[Path, CheckpointManifest]:
        raw = str(selector)
        if raw in {"latest", "best"}:
            pointer_path = self.pointer_path(raw)
            try:
                pointer = read_json(pointer_path)
                checkpoint = self.run_dir / str(pointer["checkpoint"])
            except (OSError, KeyError, json.JSONDecodeError) as exc:
                raise ArtifactStoreError(
                    f"missing or invalid {raw} checkpoint pointer: {pointer_path}"
                ) from exc
        else:
            checkpoint = Path(selector).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = self.run_dir / checkpoint
        checkpoint = checkpoint.resolve()
        try:
            checkpoint.relative_to(self.run_dir.resolve())
        except ValueError as exc:
            raise ArtifactStoreError(
                f"checkpoint path escapes run directory: {checkpoint}"
            ) from exc
        manifest = self._read_manifest(checkpoint)
        if checkpoint.name != self.checkpoint_path(manifest.step).name:
            raise ArtifactStoreError(
                f"checkpoint path and manifest step disagree: {checkpoint}"
            )
        return checkpoint, manifest

    def try_resolve(
        self, selector: str = "latest"
    ) -> tuple[Path, CheckpointManifest] | None:
        try:
            return self.resolve(selector)
        except ArtifactStoreError:
            return None

    def verify(self, manifest: CheckpointManifest, *, deep: bool = False) -> None:
        checkpoint = self.checkpoint_path(manifest.step)
        current = self._read_manifest(checkpoint)
        if current != manifest:
            raise ArtifactStoreError("checkpoint manifest changed after commit")
        for item in manifest.objects:
            path = checkpoint / item.key
            if not path.is_file():
                raise ArtifactStoreError(f"checkpoint object is missing: {path}")
            if path.stat().st_size != item.size:
                raise ArtifactStoreError(f"checkpoint size mismatch: {path}")
            if deep and _sha256_file(path) != item.sha256:
                raise ArtifactStoreError(f"checkpoint SHA-256 mismatch: {path}")

    def _read_manifest(self, checkpoint: Path) -> CheckpointManifest:
        path = checkpoint / "manifest.json"
        try:
            value = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactStoreError(
                f"missing or invalid checkpoint manifest: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise ArtifactStoreError(f"checkpoint manifest must be an object: {path}")
        manifest = CheckpointManifest.from_dict(value)
        if (
            not namespace_matches(manifest.schema_version, self.SCHEMA_VERSION)
            or manifest.run_id != self.run_id
        ):
            raise ArtifactStoreError(f"checkpoint manifest identity mismatch: {path}")
        return manifest

    def _collect_objects(self, root: Path) -> Iterable[ArtifactObject]:
        if not root.is_dir():
            return
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file() and path.name != "manifest.json":
                yield ArtifactObject(
                    key=path.relative_to(root).as_posix(),
                    size=path.stat().st_size,
                    sha256=_sha256_file(path),
                )

    def _prune(self) -> None:
        checkpoints: list[tuple[int, Path]] = []
        for path in self.run_dir.glob("checkpoint-*"):
            match = self._CHECKPOINT_RE.fullmatch(path.name)
            if match and (path / "manifest.json").is_file():
                checkpoints.append((int(match.group(1)), path))
        checkpoints.sort()
        protected = {path for _, path in checkpoints[-self.keep_last :]}
        if self.keep_best:
            best = self.try_resolve("best")
            if best is not None:
                protected.add(best[0])
        for _, path in checkpoints:
            if path not in protected:
                shutil.rmtree(path)


__all__ = [
    "ArtifactObject",
    "ArtifactStoreError",
    "CheckpointManifest",
    "CheckpointRepository",
]
