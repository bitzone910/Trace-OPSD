"""LlamaFactory trainer extension with atomic local elastic checkpoints."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from trace_opsd.compat import read_json

from .artifact import CheckpointRepository
from .dcp import make_local_storage_reader, make_local_storage_writer


@dataclass(frozen=True)
class TrainingArtifactContext:
    repository: CheckpointRepository
    training_fingerprint: str


def validate_resume_identity(
    *,
    checkpoint_world_size: int,
    current_world_size: int,
    checkpoint_fingerprint: str,
    current_fingerprint: str,
) -> None:
    """Validate immutable training identity while allowing FSDP resharding."""

    if checkpoint_world_size <= 0 or current_world_size <= 0:
        raise RuntimeError("checkpoint and current world sizes must be positive")
    if checkpoint_fingerprint != current_fingerprint:
        raise RuntimeError("checkpoint training fingerprint does not match this run")


def get_trace_opsd_trainer_class() -> type[Any]:
    """Construct the subclass lazily; local schema tests do not require torch."""

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    from torch.distributed.checkpoint.default_planner import (
        DefaultLoadPlanner,
        DefaultSavePlanner,
    )
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.checkpoint.stateful import Stateful
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
    from transformers.trainer import SCALER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
    from transformers.trainer_callback import ExportableState

    sampler_state_name = "sampler_state.json"

    class _TrainerAppState(Stateful):
        def __init__(self, model: Any, optimizer: Any) -> None:
            self.model = model
            self.optimizer = optimizer

        def state_dict(self) -> dict[str, Any]:
            model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
            return {"model": model_state, "optimizer": optimizer_state}

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optimizer"],
            )

    class TraceOPSDSFTTrainer(CustomSeq2SeqTrainer):
        """Custom trainer that commits DCP state to the project filesystem."""

        def __init__(
            self,
            *args: Any,
            artifact_context: TrainingArtifactContext,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.artifact_context = artifact_context
            self._resume_dcp_path: Path | None = None
            self._resume_staging: Path | None = None
            self._last_checkpoint_step = -1
            self._optimizer_loaded_from_checkpoint = False

        def _fsdp_context(self, model: Any) -> Any:
            plugin = self.accelerator.state.fsdp_plugin
            if plugin is None or plugin.fsdp_version != 1:
                raise RuntimeError("Trace-OPSD SFT requires Accelerate FSDP version 1")
            if plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT:
                raise RuntimeError("Trace-OPSD SFT requires SHARDED_STATE_DICT")
            return FSDP.state_dict_type(
                model,
                plugin.state_dict_type,
                plugin.state_dict_config,
                plugin.optim_state_dict_config,
            )

        def _shared_attempt_id(self) -> str:
            value = [uuid.uuid4().hex if self.is_world_process_zero() else ""]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(value, src=0)
            return value[0]

        def _rng_state(self) -> dict[str, Any]:
            import random

            import numpy as np

            value: dict[str, Any] = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state(),
            }
            if torch.cuda.is_available():
                value["cuda"] = torch.cuda.random.get_rng_state()
            return value

        def _latest_metrics(self) -> dict[str, float]:
            metrics: dict[str, float] = {}
            for entry in reversed(self.state.log_history):
                if (
                    int(entry.get("step", self.state.global_step))
                    != self.state.global_step
                ):
                    continue
                for key, value in entry.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[key] = float(value)
                if "eval_loss" in metrics:
                    break
            return metrics

        def _save_sidecars(self, checkpoint: Path) -> None:
            sidecars = checkpoint / "sidecars"
            sidecars.mkdir(parents=True, exist_ok=True)
            rank = self.args.process_index
            torch.save(self._rng_state(), sidecars / f"rng_state_{rank}.pth")
            if not self.is_world_process_zero():
                return
            for callback in [
                item
                for item in self.callback_handler.callbacks + [self.control]
                if isinstance(item, ExportableState)
            ]:
                name = callback.__class__.__name__
                state = callback.state()
                existing = self.state.stateful_callbacks.get(name)
                if isinstance(existing, list):
                    existing.append(state)
                else:
                    self.state.stateful_callbacks[name] = state
            (sidecars / TRAINER_STATE_NAME).write_text(
                json.dumps(asdict(self.state), indent=2, sort_keys=True, default=str)
                + "\n",
                encoding="utf-8",
            )
            (sidecars / sampler_state_name).write_text(
                json.dumps(
                    {
                        "schema_version": "trace_opsd.sft.sampler-state.v1",
                        "consumed_samples": self._consumed_samples(),
                        "dataset_size": len(self.train_dataset),
                        "data_seed": self.args.data_seed,
                        "epoch": self.state.epoch,
                        "world_size": self.args.world_size,
                        "per_device_train_batch_size": (
                            self.args.per_device_train_batch_size
                        ),
                        "gradient_accumulation_steps": (
                            self.args.gradient_accumulation_steps
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if self.lr_scheduler is not None:
                torch.save(self.lr_scheduler.state_dict(), sidecars / SCHEDULER_NAME)
            if self.accelerator.scaler is not None:
                torch.save(self.accelerator.scaler.state_dict(), sidecars / SCALER_NAME)

        def _consumed_samples(self) -> int:
            return int(
                self.state.global_step
                * self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps
                * self.args.world_size
            )

        def _save_local_checkpoint(self, model: Any) -> None:
            step = int(self.state.global_step)
            if step == self._last_checkpoint_step:
                return
            attempt_id = self._shared_attempt_id()
            temporary = self.artifact_context.repository.attempt_path(step, attempt_id)
            if self.is_world_process_zero():
                temporary.mkdir(parents=True, exist_ok=False)
            self.accelerator.wait_for_everyone()
            with self._fsdp_context(model):
                dcp.save(
                    {"app": _TrainerAppState(model, self.optimizer)},
                    storage_writer=make_local_storage_writer(temporary / "dcp"),
                    planner=DefaultSavePlanner(),
                )
            self._save_sidecars(temporary)
            self.accelerator.wait_for_everyone()
            if self.is_world_process_zero():
                is_best = (
                    self.artifact_context.repository.keep_best
                    and getattr(self.state, "best_global_step", None) == step
                )
                manifest = self.artifact_context.repository.commit(
                    step=step,
                    attempt_id=attempt_id,
                    world_size=self.args.world_size,
                    training_fingerprint=self.artifact_context.training_fingerprint,
                    metrics=self._latest_metrics(),
                    best=is_best,
                    consumed_samples=self._consumed_samples(),
                )
                if is_best:
                    self.state.best_model_checkpoint = str(
                        self.artifact_context.repository.checkpoint_path(manifest.step)
                    )
            self.accelerator.wait_for_everyone()
            self._last_checkpoint_step = step

        def _save_checkpoint(self, model: Any, trial: Any) -> None:
            del trial
            self.store_flos()
            self._save_local_checkpoint(model)

        def save_final_local_checkpoint(self, metrics: Mapping[str, float]) -> None:
            if metrics:
                entry = dict(metrics)
                entry["step"] = self.state.global_step
                self.state.log_history.append(entry)
                if "eval_loss" in metrics:
                    current_best = self.state.best_metric
                    if (
                        current_best is None
                        or float(metrics["eval_loss"]) < current_best
                    ):
                        self.state.best_metric = float(metrics["eval_loss"])
                        self.state.best_global_step = self.state.global_step
            self._save_local_checkpoint(self.model)

        def prepare_local_resume(self, selector: str) -> str:
            checkpoint, manifest = self.artifact_context.repository.resolve(selector)
            validate_resume_identity(
                checkpoint_world_size=manifest.world_size,
                current_world_size=self.args.world_size,
                checkpoint_fingerprint=manifest.training_fingerprint,
                current_fingerprint=self.artifact_context.training_fingerprint,
            )
            verification_error: list[str | None] = [None]
            if self.is_world_process_zero():
                try:
                    self.artifact_context.repository.verify(manifest, deep=True)
                except BaseException as exc:
                    verification_error[0] = (
                        f"local checkpoint verification failed: {exc}"
                    )
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(verification_error, src=0)
            if verification_error[0] is not None:
                raise RuntimeError(verification_error[0])

            staging = (
                self.artifact_context.repository.run_dir
                / ".resume"
                / f"step-{manifest.step:09d}-world-{self.args.world_size}"
            )
            staging_error: list[str | None] = [None]
            if self.is_world_process_zero():
                try:
                    if staging.exists():
                        shutil.rmtree(staging)
                    staging.mkdir(parents=True)
                    for filename in (
                        TRAINER_STATE_NAME,
                        SCHEDULER_NAME,
                        SCALER_NAME,
                        sampler_state_name,
                    ):
                        source = checkpoint / "sidecars" / filename
                        if source.is_file():
                            shutil.copy2(source, staging / filename)
                    sampler_state_path = staging / sampler_state_name
                    if not sampler_state_path.is_file():
                        raise RuntimeError(
                            f"checkpoint sampler state is missing: {sampler_state_path}"
                        )
                    sampler_state = read_json(sampler_state_path)
                    if (
                        int(sampler_state.get("consumed_samples", -1))
                        != manifest.consumed_samples
                    ):
                        raise RuntimeError(
                            "checkpoint sampler state and manifest consumed "
                            "sample count disagree"
                        )
                except BaseException as exc:
                    staging_error[0] = str(exc)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(staging_error, src=0)
            if staging_error[0] is not None:
                raise RuntimeError(staging_error[0])

            source_rank = self.args.process_index % manifest.world_size
            source_rng = checkpoint / "sidecars" / f"rng_state_{source_rank}.pth"
            rng_error = None
            try:
                if not source_rng.is_file():
                    raise RuntimeError(
                        f"checkpoint RNG sidecar is missing: {source_rng}"
                    )
                shutil.copy2(
                    source_rng, staging / f"rng_state_{self.args.process_index}.pth"
                )
            except BaseException as exc:
                rng_error = str(exc)
            rng_errors = [rng_error]
            if dist.is_available() and dist.is_initialized():
                rng_errors = [None] * self.args.world_size
                dist.all_gather_object(rng_errors, rng_error)
            if any(error is not None for error in rng_errors):
                raise RuntimeError(
                    "checkpoint RNG staging failed: "
                    + "; ".join(str(error) for error in rng_errors if error is not None)
                )
            self._resume_dcp_path = checkpoint / "dcp"
            self._resume_staging = staging
            self._last_checkpoint_step = manifest.step
            return str(staging)

        def _load_from_checkpoint(
            self, resume_from_checkpoint: str, model: Any = None
        ) -> None:
            del resume_from_checkpoint
            if self._resume_dcp_path is None:
                raise RuntimeError("local resume checkpoint was not prepared")
            model = model or self.model
            with self._fsdp_context(model):
                dcp.load(
                    {"app": _TrainerAppState(model, self.optimizer)},
                    storage_reader=make_local_storage_reader(self._resume_dcp_path),
                    planner=DefaultLoadPlanner(),
                )
            self._optimizer_loaded_from_checkpoint = True

        def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
            if checkpoint is None:
                return
            if (
                self._resume_dcp_path is None
                or not self._optimizer_loaded_from_checkpoint
            ):
                raise RuntimeError(
                    "model and optimizer were not restored from local DCP"
                )
            scheduler_path = Path(checkpoint) / SCHEDULER_NAME
            if scheduler_path.is_file() and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(
                    torch.load(scheduler_path, map_location="cpu", weights_only=True)
                )
            scaler_path = Path(checkpoint) / SCALER_NAME
            if scaler_path.is_file() and self.accelerator.scaler is not None:
                self.accelerator.scaler.load_state_dict(
                    torch.load(scaler_path, map_location="cpu", weights_only=True)
                )

        def cleanup_resume_staging(self) -> None:
            self.accelerator.wait_for_everyone()
            if self.is_world_process_zero() and self._resume_staging is not None:
                shutil.rmtree(self._resume_staging.parent, ignore_errors=True)

    return TraceOPSDSFTTrainer
