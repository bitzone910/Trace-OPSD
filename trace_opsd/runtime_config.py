"""A800 training configuration with external runtime output paths."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trace_opsd.assets import configure_asset_environment, resolve_work_root

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "configs" / "runtime" / "a800.yaml"


class RuntimeConfigurationError(ValueError):
    """Raised when the shared training runtime configuration is invalid."""


@dataclass(frozen=True)
class RuntimeConfig:
    path: Path
    sft_gpu_count: int
    grpo_gpu_count: int
    retrieval_mode: str
    rollout_tensor_parallel_size: int
    checkpoint_root: Path
    sft_keep_last: int
    sft_keep_best: bool
    grpo_keep_last: int

    @property
    def sft_checkpoint_root(self) -> Path:
        return self.checkpoint_root / "sft"

    @property
    def grpo_checkpoint_root(self) -> Path:
        return self.checkpoint_root / "grpo"

    @property
    def export_root(self) -> Path:
        return self.checkpoint_root / "exports"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {
            key: str(item) if isinstance(item, Path) else item
            for key, item in value.items()
        }


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeConfigurationError(f"{name} must be a mapping")
    return value


def _positive_int(value: object, name: str, *, maximum: int = 8) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigurationError(f"{name} must be an integer") from exc
    if not 1 <= parsed <= maximum:
        raise RuntimeConfigurationError(f"{name} must be between 1 and {maximum}")
    return parsed


def _resolve_project_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigurationError(f"{name} must be a nonempty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = resolve_work_root() / path
    return path.resolve()


def load_runtime_config(path: str | Path = DEFAULT_RUNTIME_CONFIG) -> RuntimeConfig:
    """Load and validate the shared SFT/GRPO runtime configuration."""

    config_path = Path(path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - the locked runtime has PyYAML
        raise RuntimeError("PyYAML is required to read runtime configuration") from exc
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeConfigurationError(
            f"cannot read runtime config: {config_path}"
        ) from exc
    root = _mapping(value, "runtime configuration")
    sft = _mapping(root.get("sft"), "sft")
    grpo = _mapping(root.get("grpo"), "grpo")
    checkpoint = _mapping(root.get("checkpoint"), "checkpoint")

    retrieval_mode = str(grpo.get("retrieval_mode", ""))
    if retrieval_mode not in {"local_dedicated", "local_cpu"}:
        raise RuntimeConfigurationError(
            "grpo.retrieval_mode must be 'local_dedicated' or 'local_cpu'"
        )
    tensor_parallel = _positive_int(
        grpo.get("rollout_tensor_parallel_size"),
        "grpo.rollout_tensor_parallel_size",
    )
    if tensor_parallel != 1:
        raise RuntimeConfigurationError(
            "grpo.rollout_tensor_parallel_size must be 1 for the supported "
            "local retrieval modes"
        )

    sft_keep_best = checkpoint.get("sft_keep_best", False)
    if not isinstance(sft_keep_best, bool):
        raise RuntimeConfigurationError("checkpoint.sft_keep_best must be boolean")

    return RuntimeConfig(
        path=config_path,
        sft_gpu_count=_positive_int(sft.get("gpu_count"), "sft.gpu_count"),
        grpo_gpu_count=_positive_int(
            grpo.get("gpu_count"),
            "grpo.gpu_count",
            maximum=7 if retrieval_mode == "local_dedicated" else 8,
        ),
        retrieval_mode=retrieval_mode,
        rollout_tensor_parallel_size=tensor_parallel,
        checkpoint_root=_resolve_project_path(
            checkpoint.get("root"), "checkpoint.root"
        ),
        sft_keep_last=_positive_int(
            checkpoint.get("sft_keep_last"), "checkpoint.sft_keep_last", maximum=100
        ),
        sft_keep_best=sft_keep_best,
        grpo_keep_last=_positive_int(
            checkpoint.get("grpo_keep_last"), "checkpoint.grpo_keep_last", maximum=100
        ),
    )


def disk_summary(path: Path) -> dict[str, Any]:
    """Return non-blocking filesystem capacity information for launch logs."""

    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "path": str(path),
        "filesystem_probe": str(probe),
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "free_gib": round(usage.free / 1024**3, 2),
    }


def _lookup(config: RuntimeConfig, key: str) -> object:
    values = {
        "repo_root": REPO_ROOT,
        "sft.gpu_count": config.sft_gpu_count,
        "grpo.gpu_count": config.grpo_gpu_count,
        "grpo.retrieval_mode": config.retrieval_mode,
        "grpo.rollout_tensor_parallel_size": config.rollout_tensor_parallel_size,
        "checkpoint.root": config.checkpoint_root,
        "checkpoint.sft_root": config.sft_checkpoint_root,
        "checkpoint.grpo_root": config.grpo_checkpoint_root,
        "checkpoint.export_root": config.export_root,
        "checkpoint.sft_keep_last": config.sft_keep_last,
        "checkpoint.sft_keep_best": str(config.sft_keep_best).lower(),
        "checkpoint.grpo_keep_last": config.grpo_keep_last,
    }
    try:
        return values[key]
    except KeyError as exc:
        raise RuntimeConfigurationError(f"unknown runtime key: {key}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trace_opsd.runtime_config")
    parser.add_argument("--config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("get")
    get.add_argument("key")
    commands.add_parser("show")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_asset_environment(args.assets_root, args.work_root)
    config = load_runtime_config(args.config)
    if args.command == "get":
        print(_lookup(config, args.key))
    else:
        print(
            json.dumps(
                {
                    "repo_root": str(REPO_ROOT),
                    **config.to_dict(),
                    "disk": disk_summary(config.checkpoint_root),
                },
                sort_keys=True,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
