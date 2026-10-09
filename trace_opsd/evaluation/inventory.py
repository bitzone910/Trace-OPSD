"""Read-only inventory of explicitly selected Hugging Face models.

The on-disk floating-point dtype is inspected, never inferred from config.json.
No model weights are loaded into RAM, converted, or written by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from trace_opsd.compat import (
    metadata_equal,
    metadata_path,
    namespace_matches,
    read_json,
)

INVENTORY_SCHEMA = "trace_opsd.evaluation.inventory.v1"
MODEL_LIST_SCHEMA = "trace_opsd.evaluation.models.v1"
_HASH_CACHE: dict[tuple[str, int, int, int], str] = {}
_INTEGER_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
}
_DTYPE_BYTES = {
    **_INTEGER_BYTES,
    "BF16": 2,
    "F32": 4,
    "F16": 2,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}
_BUFFER_SUFFIXES = ("position_ids", "token_type_ids", "num_batches_tracked")


class InventoryError(ValueError):
    """A model cannot be identified and served with the agreed BF16 protocol."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InventoryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InventoryError(f"missing or invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InventoryError(f"JSON must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    before = path.stat()
    key = (str(path.resolve()), before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise InventoryError(f"model file changed during inventory: {path}")
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def _local_file(directory: Path, filename: str) -> Path:
    """Only flat, contained HF filenames are supported by this evaluator."""
    if (
        not filename
        or "\\" in filename
        or "/" in filename
        or PurePosixPath(filename).is_absolute()
        or PureWindowsPath(filename).drive
        or filename in {".", ".."}
    ):
        raise InventoryError(f"unsafe model shard path: {filename!r}")
    path = directory / filename
    if not path.resolve().is_relative_to(directory.resolve()):
        raise InventoryError(f"model shard escapes model directory: {path}")
    if not path.is_file():
        raise InventoryError(f"model shard is missing: {path}")
    return path


def _safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    size = path.stat().st_size
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise InventoryError(f"truncated safetensors header: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= min(100_000_000, size - 8):
                raise InventoryError(f"invalid safetensors header length: {path}")
            header = json.loads(
                handle.read(header_size), object_pairs_hook=_unique_object
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InventoryError(f"invalid safetensors header: {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise InventoryError(f"safetensors header must be an object: {path}")
    payload_size = size - 8 - header_size
    tensors: dict[str, dict[str, Any]] = {}
    spans = []
    for name, value in header.items():
        if name == "__metadata__":
            if not isinstance(value, dict) or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in value.items()
            ):
                raise InventoryError(f"invalid safetensors metadata: {path}")
            continue
        if not isinstance(value, dict):
            raise InventoryError(f"invalid tensor metadata for {name}: {path}")
        dtype, shape, offsets = (
            value.get("dtype"),
            value.get("shape"),
            value.get("data_offsets"),
        )
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            raise InventoryError(
                f"unsupported tensor dtype {dtype!r} for {name}: {path}"
            )
        if dtype not in {"BF16", "F32", *_INTEGER_BYTES}:
            raise InventoryError(
                "BF16 evaluation requires BF16/F32 source weights; "
                f"found {dtype} for {name}: {path}"
            )
        if dtype in _INTEGER_BYTES and not name.endswith(_BUFFER_SUFFIXES):
            raise InventoryError(
                f"integer/quantized model weight is not a benign buffer: {name}: {path}"
            )
        if (
            not isinstance(shape, list)
            or any(
                isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape
            )
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) for n in offsets)
        ):
            raise InventoryError(f"invalid tensor shape/offsets for {name}: {path}")
        start, end = offsets
        if (
            not 0 <= start <= end <= payload_size
            or end - start != math.prod(shape) * _DTYPE_BYTES[dtype]
        ):
            raise InventoryError(f"missing/truncated tensor payload for {name}: {path}")
        spans.append((start, end, name))
        tensors[name] = value
    if not tensors:
        raise InventoryError(f"safetensors file contains no model tensors: {path}")
    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor:
            raise InventoryError(
                f"overlapping or noncontiguous tensor offsets for {name}: {path}"
            )
        cursor = end
    if cursor != payload_size:
        raise InventoryError(f"unindexed safetensors payload bytes: {path}")
    return tensors


def _provenance(
    directory: Path, expected_step: int | None, expected_run: str | None
) -> dict[str, Any] | None:
    manifests = [
        metadata_path(directory / name)
        for name in ("artifact-manifest.json", "trace_opsd_grpo_manifest.json")
        if metadata_path(directory / name).is_file()
    ]
    if not manifests:
        if expected_step is not None:
            raise InventoryError(
                f"export provenance is missing for step {expected_step}: {directory}"
            )
        return None
    value = _json(manifests[0])
    if any(not metadata_equal(_json(path), value) for path in manifests[1:]):
        raise InventoryError(f"export provenance manifests disagree: {directory}")
    if expected_step is None and not namespace_matches(
        value.get("schema_version"), "trace_opsd.grpo.hf-export.v1"
    ):
        # SFT baseline artifacts may have their own provenance contract. Local
        # config/tokenizer/weights validation still applies; no GRPO step is
        # invented for those artifacts.
        return value
    step = value.get("source_checkpoint_step")
    if (
        not namespace_matches(
            value.get("schema_version"), "trace_opsd.grpo.hf-export.v1"
        )
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
    ):
        raise InventoryError(f"invalid HF export provenance: {directory}")
    source = str(value.get("source_checkpoint", "")).replace("\\", "/").rstrip("/")
    match = re.search(r"(?:^|/)([^/]+)/global_step_(\d+)/actor$", source)
    if not match or int(match[2]) != step:
        raise InventoryError(
            f"export source_checkpoint does not identify step {step}: {directory}"
        )
    if expected_step is not None and step != expected_step:
        raise InventoryError(
            f"export source step {step} does not match requested "
            f"step {expected_step}: {directory}"
        )
    if expected_run is not None and match[1] != expected_run:
        raise InventoryError(
            f"export source run {match[1]!r} does not match requested "
            f"run {expected_run!r}: {directory}"
        )
    return value


def inspect_hf_model(
    model_path: Path,
    *,
    expected_step: int | None = None,
    expected_run: str | None = None,
) -> dict[str, Any]:
    """Validate a local, complete, nonquantized Qwen2 HF model from headers."""
    directory = Path(model_path).resolve()
    config_path = _local_file(directory, "config.json")
    config = _json(config_path)
    if config.get("model_type") != "qwen2":
        raise InventoryError(f"expected a ToolStar Qwen2 model: {directory}")
    if any(
        config.get(name) not in (None, {}, False)
        for name in ("quantization_config", "compression_config", "quantization")
    ):
        raise InventoryError(
            "quantized/compressed weights are outside the "
            f"BF16 evaluation protocol: {directory}"
        )
    provenance = _provenance(directory, expected_step, expected_run)
    index_path = directory / "model.safetensors.index.json"
    index = _json(index_path) if index_path.is_file() else None
    if index is not None:
        if not isinstance(index.get("metadata", {}), dict):
            raise InventoryError(f"invalid safetensors index metadata: {index_path}")
        weight_map = index.get("weight_map")
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(
                not isinstance(name, str)
                or not isinstance(filename, str)
                or not filename.endswith(".safetensors")
                for name, filename in weight_map.items()
            )
        ):
            raise InventoryError(f"invalid safetensors shard index: {index_path}")
        paths = [
            _local_file(directory, name) for name in sorted(set(weight_map.values()))
        ]
        extras = {path.name for path in directory.glob("*.safetensors")} - {
            path.name for path in paths
        }
        if extras:
            raise InventoryError(
                f"unindexed/ambiguous weight files: {sorted(extras)}: {directory}"
            )
    else:
        paths = [
            _local_file(directory, path.name)
            for path in sorted(directory.glob("*.safetensors"))
        ]
        if len(paths) != 1 or paths[0].name != "model.safetensors":
            raise InventoryError(
                f"complete safetensors weights or shard index not found: {directory}"
            )
        weight_map = None
    if any(directory.glob("pytorch_model*.bin")):
        raise InventoryError(
            "ambiguous/pickle model weights are not supported by "
            f"the dtype preflight: {directory}"
        )
    dtypes: Counter[str] = Counter()
    names_seen: set[str] = set()
    payload_bytes = 0
    for path in paths:
        tensors = _safetensors_header(path)
        for name, tensor in tensors.items():
            if name in names_seen:
                raise InventoryError(
                    f"duplicate tensor across model shards: {name}: {directory}"
                )
            if weight_map is not None and weight_map.get(name) != path.name:
                raise InventoryError(f"tensor/shard index mismatch for {name}: {path}")
            names_seen.add(name)
            dtypes[tensor["dtype"]] += 1
            payload_bytes += tensor["data_offsets"][1] - tensor["data_offsets"][0]
    if weight_map is not None and names_seen != set(weight_map):
        raise InventoryError(
            "index refers to missing tensors: "
            f"{sorted(set(weight_map) - names_seen)}: {directory}"
        )
    if not any(dtypes[kind] for kind in ("BF16", "F32")):
        raise InventoryError(f"model contains no floating-point weights: {directory}")
    if index is not None and "total_size" in index.get("metadata", {}):
        total_size = index["metadata"]["total_size"]
        if (
            isinstance(total_size, bool)
            or not isinstance(total_size, int)
            or total_size != payload_bytes
        ):
            raise InventoryError(f"safetensors index total_size mismatch: {directory}")
    try:
        from transformers import AutoConfig, AutoTokenizer

        AutoConfig.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise InventoryError(
            f"model config/tokenizer is not locally loadable: {directory}: {exc}"
        ) from exc
    identity_names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "artifact-manifest.json",
        "trace_opsd_grpo_manifest.json",
        "trace_opsd_protocol.json",
        "model.safetensors.index.json",
    )
    weights = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
    ]
    identity: dict[str, Any] = {
        "path": str(directory),
        "files": {
            metadata_path(directory / name).name: _sha256(
                _local_file(directory, metadata_path(directory / name).name)
            )
            for name in identity_names
            if metadata_path(directory / name).is_file()
        },
        "weights": weights,
    }
    if provenance is not None:
        identity["provenance"] = provenance
    return {
        "storage_dtype_counts": dict(sorted(dtypes.items())),
        "tensor_count": len(names_seen),
        "model_identity": identity,
        "weights": weights,
        "inference_dtype": "bfloat16",
    }


def load_model_list(path: Path, *, assets_root: Path) -> dict[str, Any]:
    """Read an explicit list; relative HF paths are relative to the assets root.

    Checkpoint discovery and automatic export are intentionally absent. Export
    a specific committed checkpoint separately before selecting its HF folder.
    """
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader: Any, node: Any) -> dict[str, Any]:
        loader.flatten_mapping(node)
        return _unique_object(
            [
                (loader.construct_object(k), loader.construct_object(v))
                for k, v in node.value
            ]
        )

    UniqueLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping
    )
    try:
        value = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise InventoryError(f"missing or invalid model list: {path}: {exc}") from exc
    if not isinstance(value, dict) or not namespace_matches(
        value.get("schema"), MODEL_LIST_SCHEMA
    ):
        raise InventoryError(f"model list must use schema {MODEL_LIST_SCHEMA}")
    if set(value) - {"schema", "baseline", "models"}:
        raise InventoryError("unknown model-list field")
    entries = value.get("models")
    if not isinstance(entries, list) or not entries:
        raise InventoryError("model list requires at least one explicit model")
    models, names = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {
            "name",
            "path",
            "expected_step",
            "expected_run",
        }:
            raise InventoryError(
                "each model needs name/path and optional expected_step/expected_run"
            )
        name, location = entry.get("name"), entry.get("path")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name
        ):
            raise InventoryError(
                "model names must be safe directory labels, up to 64 characters"
            )
        if name in names:
            raise InventoryError(f"duplicate model name: {name}")
        if (
            not isinstance(location, str)
            or not location.strip()
            or any(c in location for c in "<>$")
        ):
            raise InventoryError(
                f"fill in a complete HF directory for {name}; "
                "placeholders are not accepted"
            )
        model_path = Path(location).expanduser()
        if not model_path.is_absolute():
            model_path = assets_root / model_path
        step, run = entry.get("expected_step"), entry.get("expected_run")
        if step is not None and (type(step) is not int or step <= 0):
            raise InventoryError(f"expected_step must be a positive integer: {name}")
        if run is not None and (
            not isinstance(run, str) or not run.strip() or step is None
        ):
            raise InventoryError(
                f"expected_run requires expected_step and a nonempty name: {name}"
            )
        names.add(name)
        models.append(
            {
                "label": name,
                "export_path": str(model_path.resolve()),
                "step": step,
                "source_run": run,
            }
        )
    baseline = value.get("baseline")
    if baseline is not None and (
        not isinstance(baseline, str) or baseline not in names
    ):
        raise InventoryError("baseline must name a model in this list")
    return {"models": models, "baseline": baseline}


def build_inventory(model_list: Path, *, assets_root: Path) -> dict[str, Any]:
    """Inspect every supplied model without loading weights or writing exports."""
    selected = load_model_list(model_list, assets_root=Path(assets_root).resolve())
    errors = []
    for entry in selected["models"]:
        entry.update(status="error", errors=[])
        try:
            result = inspect_hf_model(
                Path(entry["export_path"]),
                expected_step=entry["step"],
                expected_run=entry["source_run"],
            )
            entry.update(result, status="ready")
        except (InventoryError, OSError, ValueError, TypeError) as exc:
            entry["errors"].append(str(exc))
            errors.append(f"{entry['label']}: {exc}")
    return {
        "schema": INVENTORY_SCHEMA,
        "model_list": str(Path(model_list).resolve()),
        **selected,
        "errors": errors,
        "ready": not errors,
    }


__all__ = ["InventoryError", "build_inventory", "inspect_hf_model", "load_model_list"]
