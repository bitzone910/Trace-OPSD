"""Filesystem adapters for PyTorch Distributed Checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def make_local_storage_writer(path: Path) -> Any:
    from torch.distributed.checkpoint import FileSystemWriter

    path.mkdir(parents=True, exist_ok=True)
    return FileSystemWriter(str(path))


def make_local_storage_reader(path: Path) -> Any:
    from torch.distributed.checkpoint import FileSystemReader

    if not (path / ".metadata").is_file():
        raise RuntimeError(f"incomplete distributed checkpoint: {path}")
    return FileSystemReader(str(path))


__all__ = ["make_local_storage_reader", "make_local_storage_writer"]
