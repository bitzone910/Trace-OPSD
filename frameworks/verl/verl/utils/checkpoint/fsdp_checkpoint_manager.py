# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import logging
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, local_mkdir_safe
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.logger import log_with_rank
from verl.utils.transformers_compat import drop_tied_target_keys, get_auto_model_for_vision2seq

from .checkpoint_manager import BaseCheckpointManager

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class FSDPConfig:
    """Configuration for FSDP checkpointing.

    Args:
        FSDP_version (int): Version of FSDP being used.
        world_size (int): Number of processes in the distributed training setup.
    """

    FSDP_version: int
    world_size: int


class _ElasticFSDPState(Stateful):
    """Canonical model/optimizer state that DCP can reshard at load time."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
        return {"model": model_state, "optimizer": optimizer_state}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_objects(root: Path) -> list[dict[str, object]]:
    objects = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        objects.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return objects


def _verify_checkpoint_manifest(root: Path, manifest: dict) -> None:
    if manifest.get("schema_version") not in {"trace_opsd.grpo.fsdp-dcp.v1", "toolelite.grpo.fsdp-dcp.v1"}:
        raise RuntimeError(f"unsupported elastic checkpoint manifest: {root / 'manifest.json'}")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f"elastic checkpoint has no integrity objects: {root / 'manifest.json'}")
    for item in objects:
        relative_path = Path(str(item["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe elastic checkpoint path: {relative_path}")
        path = root / relative_path
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            raise RuntimeError(f"elastic checkpoint object is missing or truncated: {path}")
        if _sha256_file(path) != str(item["sha256"]):
            raise RuntimeError(f"elastic checkpoint checksum mismatch: {path}")


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents DictConfig: Configuration for checkpoint contents.
            - 'load': Components to load; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
            - 'save': Components to save; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
        trust_remote_code: Whether to trust_remote_code when loading the model configuration
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        if processing_class is None and "tokenizer" in kwargs:
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )
        self.trust_remote_code = trust_remote_code

    def _get_lora_train_meta(self, unwrap_model):
        peft_config = getattr(unwrap_model, "peft_config", None)
        if not peft_config:
            return None
        if isinstance(peft_config, dict):
            peft_config = peft_config.get("default") or next(iter(peft_config.values()), None)
        if peft_config is None:
            return None

        lora_rank = int(getattr(peft_config, "r", 0) or 0)
        if lora_rank <= 0:
            return None

        lora_alpha = int(getattr(peft_config, "lora_alpha", lora_rank) or 0)
        task_type = getattr(peft_config, "task_type", None) or "CAUSAL_LM"
        if hasattr(task_type, "value"):
            task_type = task_type.value

        return {"r": lora_rank, "lora_alpha": lora_alpha, "task_type": str(task_type)}

    def _save_lora_train_meta(self, local_path: str, unwrap_model):
        lora_train_meta = self._get_lora_train_meta(unwrap_model)
        if lora_train_meta is None:
            return None

        lora_meta_path = os.path.join(local_path, "lora_train_meta.json")
        with open(lora_meta_path, "w", encoding="utf-8") as f:
            json.dump(lora_train_meta, f, ensure_ascii=False, indent=4)
        log_with_rank(
            f"Saved LoRA rank/alpha metadata to {os.path.abspath(lora_meta_path)}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )
        return lora_meta_path

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """Load elastic DCP state, with same-world legacy shards as fallback."""
        if local_path is None:
            return
        if self.should_load_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )

        dcp_path = Path(local_path) / "dcp"
        manifest_path = Path(local_path) / "manifest.json"
        if (dcp_path / ".metadata").is_file() and manifest_path.is_file():
            if not (self.should_load_model and self.should_load_optimizer):
                raise RuntimeError("elastic FSDP resume requires model and optimizer contents")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verification_error = [None]
            if self.rank == 0:
                try:
                    _verify_checkpoint_manifest(Path(local_path), manifest)
                except BaseException as exc:
                    verification_error[0] = str(exc)
            torch.distributed.broadcast_object_list(verification_error, src=0)
            if verification_error[0] is not None:
                raise RuntimeError(verification_error[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dcp.load(
                    {"app": _ElasticFSDPState(self.model, self.optimizer)},
                    storage_reader=FileSystemReader(str(dcp_path)),
                    planner=DefaultLoadPlanner(),
                )
            saved_world_size = int(manifest["world_size"])
            log_with_rank(
                f"Loaded elastic DCP checkpoint saved with world size {saved_world_size}",
                rank=self.rank,
                logger=logger,
            )
            if self.should_load_extra:
                source_rank = self.rank % saved_world_size
                extra_path = Path(local_path) / "sidecars" / f"extra_state_rank_{source_rank}.pt"
                extra_state_dict = torch.load(extra_path, weights_only=False)
                if "rng" in extra_state_dict:
                    self.load_rng_state(extra_state_dict["rng"])
                scheduler_state = extra_state_dict.get("lr_scheduler")
                if scheduler_state is not None and self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(scheduler_state)
        else:
            # Read-only compatibility with verl's original rank-local format.
            state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
            optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
            extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                if self.should_load_model:
                    self.model.load_state_dict(torch.load(copy_to_local(model_path), weights_only=False))
                if self.should_load_optimizer:
                    self.optimizer.load_state_dict(torch.load(copy_to_local(optim_path), weights_only=False))
            if self.should_load_extra:
                extra_state_dict = torch.load(copy_to_local(extra_path), weights_only=False)
                if "rng" in extra_state_dict:
                    self.load_rng_state(extra_state_dict["rng"])
                if extra_state_dict.get("lr_scheduler") is not None and self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])
            log_with_rank("Loaded legacy same-world-size FSDP checkpoint", rank=self.rank, logger=logger)

        if del_local_after_load:
            log_with_rank(
                "Ignoring del_local_after_load for project-local resumable checkpoints",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )
        torch.distributed.barrier()

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        if self.rank == 0:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)
            checkpoint_dir = Path(local_path)
            if checkpoint_dir.is_dir() and not (checkpoint_dir / "manifest.json").is_file():
                # A tracker is updated only after the whole global step commits,
                # so this directory can only be residue from an interrupted save.
                import shutil

                shutil.rmtree(checkpoint_dir)

        torch.distributed.barrier()
        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )

        if not (self.should_save_model and self.should_save_optimizer):
            raise RuntimeError("elastic FSDP checkpoints require model and optimizer contents")
        dcp_path = Path(local_path) / "dcp"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dcp.save(
                {"app": _ElasticFSDPState(self.model, self.optimizer)},
                storage_writer=FileSystemWriter(str(dcp_path)),
                planner=DefaultSavePlanner(),
            )
        if self.should_save_extra:
            sidecar_path = Path(local_path) / "sidecars" / f"extra_state_rank_{self.rank}.pt"
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                    "rng": self.get_rng_state(),
                },
                sidecar_path,
            )
            log_with_rank(f"Saved extra state to {sidecar_path}", rank=self.rank, logger=logger)

        if self.rank == 0:
            # Save HF tokenizer/processor and model config on rank 0 to huggingface/ directory, no matter whether
            # huggingface model is requested to be saved or not.

            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)
            model_config = unwrap_model.config
            generation_config = None
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                try:
                    # Some model's name_or_path is empty if not initialized from pretrained,
                    # in this cases, we don't save generation config.
                    generation_config = GenerationConfig.from_pretrained(
                        model_config.name_or_path,
                        local_files_only=True,
                    )
                    generation_config.save_pretrained(hf_config_tokenizer_path)
                except Exception:
                    # if the generation config isn't available, we don't save it
                    pass

            if hasattr(model_config, "auto_map") and None in model_config.auto_map:
                model_config.auto_map = {k: v for k, v in model_config.auto_map.items() if k is not None}

            model_config.save_pretrained(hf_config_tokenizer_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
            # loaded from the Hub.
            if hasattr(model_config, "auto_map"):
                custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)
            self._save_lora_train_meta(local_path, unwrap_model)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.should_save_hf_model:
            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    auto_model_cls = get_auto_model_for_vision2seq()
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(
                        model_config, torch_dtype=torch.bfloat16, trust_remote_code=self.trust_remote_code
                    )

                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(
                            f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found "
                            f"in, using a generation config created from the model config when saving hf_model."
                        )

                drop_tied_target_keys(state_dict, save_model, model_config)

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        if self.rank == 0:
            manifest_path = Path(local_path) / "manifest.json"
            temporary_manifest = manifest_path.with_suffix(".tmp")
            temporary_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "trace_opsd.grpo.fsdp-dcp.v1",
                        "format": "torch.distributed.checkpoint",
                        "global_step": int(global_step),
                        "world_size": int(self.world_size),
                        "objects": _checkpoint_objects(Path(local_path)),
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_manifest, manifest_path)
            self.register_checkpoint(local_path, max_ckpt_to_keep)
