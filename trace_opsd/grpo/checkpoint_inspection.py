"""Read-only optimizer progress checks, independent of historical run scripts."""

from __future__ import annotations

from pathlib import Path


def optimizer_steps(actor: Path) -> list[float]:
    """Load Adam's scalar steps from DCP without loading model tensors."""
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = dcp.FileSystemReader(str(actor / "dcp"))
    metadata = reader.read_metadata()
    scalars = {
        key: torch.empty(item.size, dtype=item.properties.dtype)
        for key, item in metadata.state_dict_metadata.items()
        if ".optimizer.state." in key
        and key.endswith(".step")
        and isinstance(item, TensorStorageMetadata)
    }
    if not scalars:
        raise RuntimeError("checkpoint has no Adam optimizer step tensors")
    dcp.load(scalars, storage_reader=reader, no_dist=True)
    steps = sorted({float(value.item()) for value in scalars.values()})
    if not bool(torch.isfinite(torch.tensor(steps)).all()):
        raise RuntimeError("checkpoint optimizer step is non-finite")
    return steps
