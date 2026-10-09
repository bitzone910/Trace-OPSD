"""Export a committed local FSDP checkpoint as a local Hugging Face model."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from trace_opsd.compat import namespace_matches, read_json

from .artifact import CheckpointManifest, CheckpointRepository
from .dcp import make_local_storage_reader


def _check_host_memory(minimum_bytes: int = 24 * 1024**3) -> None:
    try:
        import psutil
    except ImportError:  # pragma: no cover - locked runtime contains psutil
        return
    available = int(psutil.virtual_memory().available)
    if available < minimum_bytes:
        raise RuntimeError(
            "HF export requires at least 24 GiB available host RAM; "
            f"found {available / 1024**3:.1f} GiB"
        )


def _file_record(path: Path, root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def verify_existing_export(target: Path, manifest: CheckpointManifest) -> bool:
    """Resume an identical finished export; refuse conflicting/corrupt weights."""
    if not target.exists() or not any(target.iterdir()):
        return False
    try:
        value = read_json(target / "artifact-manifest.json")
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"nonempty export target has no valid manifest: {target}"
        ) from exc
    if (
        not namespace_matches(
            value.get("schema_version"), "trace_opsd.sft.hf-export.v3"
        )
        or value.get("run_id") != manifest.run_id
        or value.get("source_checkpoint_step") != manifest.step
        or value.get("source_training_fingerprint") != manifest.training_fingerprint
        or value.get("dtype") != "bfloat16"
    ):
        raise RuntimeError(f"export target belongs to a different SFT state: {target}")
    objects = value.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f"export manifest has no objects: {target}")
    for item in objects:
        path = (target / item["path"]).resolve()
        if not path.is_relative_to(target.resolve()) or not path.is_file():
            raise RuntimeError(f"export object is missing or outside target: {path}")
        if _file_record(path, target) != item:
            raise RuntimeError(f"export object changed: {path}")
    return True


def bf16_export_state(state: dict[str, object]) -> dict[str, object]:
    """Cast before sharding so the index size describes the saved BF16 tensors."""
    import torch

    return {
        name: (
            tensor.detach().cpu().to(torch.bfloat16).contiguous().clone()
            if tensor.is_floating_point()
            else tensor.detach().cpu().contiguous().clone()
        )
        for name, tensor in state.items()
    }


def export_hf_model(
    *,
    repository: CheckpointRepository,
    model_path: Path,
    selector: str,
    target_dir: Path,
) -> Path:
    """Reshard locally, gather on rank zero, and write 2 GiB safetensor shards."""

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from accelerate import Accelerator
    from accelerate.utils import DistributedType
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict,
    )
    from torch.distributed.checkpoint.stateful import Stateful
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class _ExportModelState(Stateful):
        def __init__(self, wrapped_model: object) -> None:
            self.model = wrapped_model

        def state_dict(self) -> dict[str, object]:
            return {"model": get_model_state_dict(self.model)}

        def load_state_dict(self, state_dict: dict[str, object]) -> None:
            set_model_state_dict(self.model, state_dict["model"])

    checkpoint, manifest = repository.resolve(selector)
    accelerator = Accelerator()
    if accelerator.distributed_type != DistributedType.FSDP:
        raise RuntimeError("HF export must be launched with the FSDP config")
    verification_error: list[str | None] = [None]
    reuse = [False]
    if accelerator.is_main_process:
        try:
            repository.verify(manifest, deep=True)
            reuse[0] = verify_existing_export(
                target_dir.expanduser().resolve(), manifest
            )
        except BaseException as exc:
            verification_error[0] = f"local checkpoint verification failed: {exc}"
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(verification_error, src=0)
        dist.broadcast_object_list(reuse, src=0)
    if verification_error[0] is not None:
        raise RuntimeError(verification_error[0])
    if reuse[0]:
        return target_dir.expanduser().resolve()
    _check_host_memory()
    run = read_json(repository.run_manifest_path)
    cutoff_len = int(run["config"]["cutoff_len"])

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    model = accelerator.prepare_model(model, evaluation_mode=True)
    plugin = accelerator.state.fsdp_plugin
    if plugin is None or plugin.fsdp_version != 1:
        raise RuntimeError("HF export requires FSDP version 1")
    with FSDP.state_dict_type(
        model,
        plugin.state_dict_type,
        plugin.state_dict_config,
        plugin.optim_state_dict_config,
    ):
        dcp.load(
            {"app": _ExportModelState(model)},
            storage_reader=make_local_storage_reader(checkpoint / "dcp"),
            planner=DefaultLoadPlanner(),
        )
    accelerator.wait_for_everyone()

    full_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_config):
        full_state = model.state_dict()

    target_dir = target_dir.expanduser().resolve()
    export_error: list[str | None] = [None]
    if accelerator.is_main_process:
        temporary = target_dir.parent / f".{target_dir.name}.{uuid.uuid4().hex}.tmp"
        try:
            if target_dir.exists() and any(target_dir.iterdir()):
                raise RuntimeError(
                    f"export target already exists and is not empty: {target_dir}"
                )
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            temporary.mkdir(parents=True, exist_ok=False)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=False, local_files_only=True
            )
            try:
                from llamafactory.data import get_template_and_fix_tokenizer
                from llamafactory.hparams import DataArguments

                get_template_and_fix_tokenizer(
                    tokenizer, DataArguments(template="qwen", cutoff_len=cutoff_len)
                )
            except BaseException as exc:
                raise RuntimeError(
                    "failed to attach the LlamaFactory qwen chat template"
                ) from exc
            tokenizer.save_pretrained(temporary)
            accelerator.unwrap_model(model).config.save_pretrained(temporary)
            generation_config = accelerator.unwrap_model(model).generation_config
            if generation_config is not None:
                generation_config.save_pretrained(temporary)
            full_state = bf16_export_state(full_state)

            split = split_torch_state_dict_into_shards(
                full_state,
                filename_pattern="model{suffix}.safetensors",
                max_shard_size="2GB",
            )
            for filename, tensor_names in split.filename_to_tensors.items():
                shard = {name: full_state[name] for name in tensor_names}
                save_file(shard, temporary / filename, metadata={"format": "pt"})
                del shard
            if split.is_sharded:
                (temporary / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "metadata": split.metadata,
                            "weight_map": split.tensor_to_filename,
                        },
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            (temporary / "trace_opsd_protocol.json").write_text(
                json.dumps(
                    {
                        "schema_version": "trace_opsd.model-protocol.v1",
                        "chat_template": "qwen",
                        "tool_protocol": "trace_opsd.xml-tools.v2",
                        "cutoff_len": cutoff_len,
                        "supervision": "alpaca.full_output",
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            records = [
                _file_record(path, temporary)
                for path in sorted(temporary.iterdir())
                if path.is_file()
            ]
            (temporary / "artifact-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "trace_opsd.sft.hf-export.v3",
                        "run_id": repository.run_id,
                        "source_checkpoint_step": manifest.step,
                        "source_checkpoint": str(checkpoint),
                        "source_training_fingerprint": manifest.training_fingerprint,
                        "dtype": "bfloat16",
                        "max_shard_size": "2GB",
                        "ready_for_verl": True,
                        "objects": records,
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if target_dir.exists():
                target_dir.rmdir()
            os.replace(temporary, target_dir)
        except BaseException as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            export_error[0] = f"SFT export failed: {exc}"
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(export_error, src=0)
    if export_error[0] is not None:
        raise RuntimeError(export_error[0])
    accelerator.wait_for_everyone()
    return target_dir


__all__ = ["export_hf_model"]
