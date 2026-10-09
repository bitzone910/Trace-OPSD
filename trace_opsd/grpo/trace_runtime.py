"""Distributed verl integration for current-policy hindsight TRACE-OPSD."""

from __future__ import annotations

import json
import math
import time
import uuid
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.core_algos import kl_penalty
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.trainer.ppo.utils import need_critic
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from trace_opsd.compat import canonical_metrics, read_json

from .auto_export import publish_trace_export, recover_trace_exports
from .checkpoint_retention import (
    TraceCheckpointRetention,
    publish_latest_step,
    read_latest_step,
    reconcile_trace_checkpoints,
    restore_latest_step,
    verify_trace_checkpoint,
)
from .credit_revision import assert_trainable_trace_manifest, checkpoint_credit_state
from .curriculum import write_stage_receipt
from .dynamic_sampling import (
    DynamicBatchAccumulator,
    DynamicSamplingBudgetError,
    filter_mixed_reward_groups,
)
from .lr_control import constant_restored_learning_rate, scale_restored_learning_rate
from .metrics_io import append_metrics
from .sampling import DifficultyPoolSampler, TraceSamplingConfig
from .stability_metrics import summarize_trajectory_stability
from .trace_opsd import (
    TRACE_TURN_TYPES,
    TraceOPSDConfig,
    TraceTrajectoryError,
    TraceView,
    apply_python_error_penalty,
    build_turn_views,
    causal_scores_from_old_log_probs,
    evidence_from_view_scores,
    shape_token_advantages,
    summarize_trace_evidence,
)
from .turn_credit_io import write_turn_credit
from .update_contract import (
    validate_actor_update_metrics,
    validate_trace_actor_batch,
    validate_trace_update_config,
)


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, Mapping):
        raise TraceTrajectoryError(f"{name} must be a mapping")
    return value


def _as_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if hasattr(value, "item") and not isinstance(value, (list, tuple)):
        value = value.item()
    if not isinstance(value, (list, tuple)):
        raise TraceTrajectoryError(f"{name} must be a list")
    return value


def collect_python_error_turns(batch: DataProto) -> list[dict[str, Any]]:
    """Collect execution metadata independently of hindsight/view eligibility."""
    result = []
    padding = batch.non_tensor_batch.get("trace_opsd_is_padding")
    rows = batch.batch["responses"].shape[0]
    all_turns = batch.non_tensor_batch.get("trace_tool_turns")
    if all_turns is None or len(all_turns) != rows:
        raise TraceTrajectoryError(
            "Python error credit requires complete tool turn metadata"
        )
    for row in range(rows):
        if padding is not None and bool(padding[row]):
            continue
        turns = _as_sequence(all_turns[row], name="trace_tool_turns")
        for index, raw in enumerate(turns):
            turn = dict(_as_mapping(raw, name="trace_tool_turn"))
            if turn.get("tool_type") != "python":
                continue
            turn.update(
                trajectory_index=row,
                turn_index=index,
                uid=str(batch.non_tensor_batch["uid"][row]),
            )
            result.append(turn)
    return result


def _unpadded_prompt_ids(batch: DataProto, row: int) -> list[int]:
    prompt_width = batch.batch["prompts"].shape[1]
    mask = batch.batch["attention_mask"][row, :prompt_width].to(torch.bool)
    return [int(value) for value in batch.batch["prompts"][row][mask].tolist()]


def collect_trace_views(
    batch: DataProto,
    *,
    tokenizer: Any,
    max_view_length: int,
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[TraceView], dict[str, float]]:
    """Build exact action views for tools and the final assistant generation."""

    views: list[TraceView] = []
    skipped = {
        kind: {"view_too_long": 0, "missing_solution_hindsight": 0}
        for kind in TRACE_TURN_TYPES
    }
    rows_without_tools = 0
    padding_rows = 0
    response_ids = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    if response_ids.shape != response_mask.shape or response_ids.ndim != 2:
        raise TraceTrajectoryError(
            "TRACE response IDs and masks must have identical 2D shapes"
        )
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise TraceTrajectoryError("TRACE response_mask must contain only zero or one")
    if response_ids.dtype not in {
        torch.int32,
        torch.int64,
    } or torch.any(response_ids < 0):
        raise TraceTrajectoryError("TRACE response IDs must be non-negative integers")
    is_padding = batch.non_tensor_batch.get("trace_opsd_is_padding")
    final_turns = batch.non_tensor_batch.get("trace_final_turn")
    hindsight_rows = batch.non_tensor_batch.get("hindsight")
    for row in range(response_ids.shape[0]):
        if is_padding is not None and bool(is_padding[row]):
            padding_rows += 1
            continue
        if final_turns is None:
            raise TraceTrajectoryError(
                "TRACE rollout metadata is missing trace_final_turn"
            )
        messages = _as_sequence(
            batch.non_tensor_batch["trace_messages"][row], name="trace_messages"
        )
        tools = _as_sequence(
            batch.non_tensor_batch["trace_tool_turns"][row], name="trace_tool_turns"
        )
        turns = [
            {
                **_as_mapping(raw, name="trace_tool_turn"),
                "turn_type": _as_mapping(raw, name="trace_tool_turn").get("tool_type"),
            }
            for raw in tools
        ]
        if not tools:
            rows_without_tools += 1
        final = final_turns[row]
        if hasattr(final, "item"):
            final = final.item()
        if final is not None:
            turns.append(
                {**_as_mapping(final, name="trace_final_turn"), "turn_type": "final"}
            )
        if not turns:
            continue
        uid_value = batch.non_tensor_batch["uid"][row]
        if hasattr(uid_value, "item"):
            uid_value = uid_value.item()
        hindsight_value = None if hindsight_rows is None else hindsight_rows[row]
        if hasattr(hindsight_value, "item"):
            hindsight_value = hindsight_value.item()
        prompt_ids = _unpadded_prompt_ids(batch, row)
        previous_end = 0
        previous_assistant_index = -1
        for turn_index, turn in enumerate(turns):
            kind = str(turn.get("turn_type", "")).lower()
            if kind not in TRACE_TURN_TYPES:
                raise TraceTrajectoryError(f"unsupported TRACE turn type: {kind!r}")
            bounds = [turn.get("response_start"), turn.get("response_end")]
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in bounds
            ):
                raise TraceTrajectoryError(
                    "TRACE response span bounds must be integers"
                )
            start, end = map(int, bounds)
            if not 0 <= start < end <= response_ids.shape[1]:
                raise TraceTrajectoryError("TRACE response span is outside responses")
            if start < previous_end:
                raise TraceTrajectoryError(
                    "TRACE action spans overlap or are out of order"
                )
            previous_end = end
            if not torch.all(response_mask[row, start:end].to(torch.bool)):
                raise TraceTrajectoryError(
                    "TRACE action span includes observation or padding tokens"
                )
            assistant_index = turn.get("assistant_message_index")
            if isinstance(assistant_index, bool) or not isinstance(
                assistant_index, Integral
            ):
                raise TraceTrajectoryError("assistant_message_index must be an integer")
            if not 0 <= assistant_index < len(messages) or (
                assistant_index <= previous_assistant_index
            ):
                raise TraceTrajectoryError(
                    "TRACE assistant message indices are invalid or out of order"
                )
            previous_assistant_index = assistant_index
            assistant = messages[assistant_index]
            if assistant.get("role") != "assistant":
                raise TraceTrajectoryError(
                    "TRACE turn does not point to an assistant message"
                )
            action_ids = [int(value) for value in response_ids[row, start:end].tolist()]
            if tokenizer.decode(action_ids, skip_special_tokens=False) != assistant.get(
                "content"
            ):
                raise TraceTrajectoryError(
                    "TRACE action token IDs do not match the assistant message"
                )
            expected_prefix = prompt_ids + [
                int(value) for value in response_ids[row, :start].tolist()
            ]
            turn_views, skip_reason = build_turn_views(
                tokenizer=tokenizer,
                trajectory_index=row,
                turn_index=turn_index,
                uid=str(uid_value),
                messages=messages,
                turn=turn,
                hindsight=hindsight_value,
                action_ids=action_ids,
                expected_causal_prefix_ids=expected_prefix,
                max_view_length=max_view_length,
                apply_chat_template_kwargs=apply_chat_template_kwargs,
            )
            if skip_reason is not None:
                skipped[kind][skip_reason] += 1
                continue
            views.extend(turn_views)
    metrics = {
        "trace/skipped_view_too_long": float(
            sum(counts["view_too_long"] for counts in skipped.values())
        ),
        "trace/skipped_missing_solution_hindsight": float(
            sum(counts["missing_solution_hindsight"] for counts in skipped.values())
        ),
        "trace/direct_or_no_tool_rows": float(rows_without_tools),
        "trace/skipped_padding_rows": float(padding_rows),
    }
    for kind, counts in skipped.items():
        for reason, count in counts.items():
            metrics[f"trace/turn/{kind}/skipped_{reason}"] = float(count)
    return views, metrics


def _make_score_batch(views: Sequence[TraceView], pad_token_id: int) -> DataProto:
    if not views:
        raise ValueError("cannot score an empty TRACE view batch")
    prompt_width = max(len(view.prefix_ids) for view in views)
    response_width = max(len(view.action_ids) for view in views)
    batch_size = len(views)
    prompts = torch.full((batch_size, prompt_width), pad_token_id, dtype=torch.long)
    responses = torch.full((batch_size, response_width), pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros((batch_size, prompt_width), dtype=torch.long)
    response_attention = torch.zeros((batch_size, response_width), dtype=torch.long)
    for row, view in enumerate(views):
        prefix = torch.tensor(view.prefix_ids, dtype=torch.long)
        action = torch.tensor(view.action_ids, dtype=torch.long)
        prompts[row, -len(prefix) :] = prefix
        prompt_mask[row, -len(prefix) :] = 1
        responses[row, : len(action)] = action
        response_attention[row, : len(action)] = 1
    attention_mask = torch.cat((prompt_mask, response_attention), dim=1)
    input_ids = torch.cat((prompts, responses), dim=1)
    position_ids = compute_position_id_with_mask(attention_mask)
    tensors = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_attention,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )
    return DataProto(batch=tensors)


class TraceActorRolloutWorker(ActorRolloutRefWorker):
    """Actor/rollout worker used for current-policy hindsight scoring."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def configure_trace_learning_rate(self, target_scale: float) -> dict:
        if self.actor is None:
            raise RuntimeError(
                "TRACE learning-rate scaling requires an initialized actor"
            )
        engine = self.actor.engine
        return scale_restored_learning_rate(
            engine.optimizer, engine.lr_scheduler, target_scale
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def configure_trace_constant_learning_rate(self, learning_rate: float) -> dict:
        if self.actor is None:
            raise RuntimeError(
                "TRACE constant learning rate requires an initialized actor"
            )
        engine = self.actor.engine
        return constant_restored_learning_rate(
            engine.optimizer, engine.lr_scheduler, learning_rate
        )


class TraceRayPPOTrainer(RayPPOTrainer):
    """Inject TRACE credit estimation immediately before the normal PPO update."""

    def _training_epoch(self) -> int:
        # Candidate cycles are independent of optimizer steps and dataloader epochs.
        return 0

    def _dynamic_candidates(self, deficits, exclude_source_ids=()):
        counts = self.dynamic_sampler.allocate_counts(
            deficits,
            candidate_batch_size=self.trace_sampling_config.problems_per_step,
            exclude_source_ids=exclude_source_ids,
        )
        indices = self.dynamic_sampler.draw(counts, exclude_source_ids)
        if not indices:
            raise RuntimeError("TRACE dynamic sampling has no distinct candidates")
        batch_dict = self.dynamic_collate_fn(
            [self.train_dataset[index] for index in indices]
        )
        batch_dict["trace_source_id"] = np.asarray(
            [self.dynamic_sampler.source_ids[index] for index in indices],
            dtype=object,
        )
        batch_dict["trace_candidate_difficulty"] = np.asarray(
            [self.curriculum_difficulties[index] for index in indices], dtype=object
        )
        batch_dict["trace_candidate_rank"] = np.arange(len(indices))
        return batch_dict

    def _iter_training_batches(self):
        curriculum = self.config.trainer.trace_curriculum
        quotas = {
            "mid": int(curriculum.mid_per_step),
            "high": int(curriculum.high_per_step),
        }
        while self.global_steps <= self.total_training_steps:
            yield self._dynamic_candidates(quotas)

    def _trainer_state_extension(self) -> dict[str, Any]:
        return {
            "trace_dynamic_sampling": self.dynamic_sampler.state_dict(),
            **(
                {"trace_learning_rate": self.trace_learning_rate}
                if hasattr(self, "trace_learning_rate")
                else {}
            ),
            **checkpoint_credit_state(
                Path(self.config.trainer.default_local_dir), self.global_steps
            ),
        }

    def _prepare_training_batch(
        self, batch_dict, metrics, timing_raw, curr_step_profile
    ):
        curriculum = self.config.trainer.trace_curriculum
        quotas = {
            "mid": int(curriculum.mid_per_step),
            "high": int(curriculum.high_per_step),
        }
        rollout_n = self.trace_sampling_config.rollouts_per_problem
        temperature = self.config.actor_rollout_ref.rollout.temperature
        accumulator = DynamicBatchAccumulator(
            quotas,
            rollouts_per_problem=rollout_n,
            max_num_gen_batches=self.dynamic_max_num_gen_batches,
        )
        parts = []
        reward_keys = None
        generation_failed = True
        try:
            while not accumulator.complete:
                accumulator.ensure_can_generate()
                # HYBRID engines cannot wake without a weight transfer. Keep the
                # same snapshot resident through CPU-reward candidate refills;
                # release it once below, before actor/TRACE scoring.
                self.dynamic_sampler.record_candidate_batch()
                round_timing = {}
                batch, reward_tensor, reward_info = self._generate_and_reward_batch(
                    batch_dict,
                    metrics,
                    round_timing,
                    curr_step_profile,
                    balance_before_reward=False,
                    sleep_after_generation=False,
                )
                for key, value in round_timing.items():
                    timing_raw[key] = timing_raw.get(key, 0.0) + value
                if "acc" not in reward_info:
                    raise ValueError(
                        "TRACE dynamic sampling requires reward metric acc"
                    )
                if reward_tensor.shape != batch.batch["responses"].shape:
                    raise ValueError("dynamic sampling reward/token shape mismatch")
                if not torch.isfinite(reward_tensor).all():
                    raise ValueError("dynamic sampling reward tensor is non-finite")
                current_keys = tuple(reward_info)
                if reward_keys is None:
                    reward_keys = current_keys
                elif set(reward_keys) != set(current_keys):
                    raise ValueError("dynamic sampling reward diagnostic keys changed")
                batch.batch["rm_scores"] = reward_tensor
                for key, values in reward_info.items():
                    values = np.asarray(values)
                    if values.ndim == 0 or len(values) != len(batch):
                        raise ValueError(f"reward diagnostic {key} is not row-aligned")
                    batch.non_tensor_batch[key] = values
                padding = np.asarray(
                    batch.non_tensor_batch["trace_opsd_is_padding"], dtype=bool
                )
                ranks = batch.non_tensor_batch["trace_candidate_rank"]
                expected_ids = batch_dict["trace_source_id"]
                expected_difficulties = batch_dict["trace_candidate_difficulty"]
                uid_by_rank = {}
                for row in np.flatnonzero(~padding):
                    rank_value = ranks[row]
                    if (
                        isinstance(rank_value, (bool, np.bool_))
                        or not isinstance(rank_value, Integral)
                        or not 0 <= rank_value < len(expected_ids)
                    ):
                        raise ValueError("invalid original candidate rank")
                    rank, uid = int(rank_value), batch.non_tensor_batch["uid"][row]
                    if (
                        batch.non_tensor_batch["trace_source_id"][row]
                        != expected_ids[rank]
                        or batch.non_tensor_batch["trace_candidate_difficulty"][row]
                        != expected_difficulties[rank]
                    ):
                        raise ValueError(
                            "candidate identity differs from sampled prompt"
                        )
                    if rank in uid_by_rank and uid_by_rank[rank] != uid:
                        raise ValueError("candidate rank has multiple attempt UIDs")
                    uid_by_rank[rank] = uid
                if set(uid_by_rank) != set(range(len(expected_ids))):
                    raise ValueError("generation is missing a complete candidate group")
                groups = filter_mixed_reward_groups(
                    batch.non_tensor_batch["uid"],
                    batch.non_tensor_batch["trace_source_id"],
                    batch.non_tensor_batch["trace_candidate_difficulty"],
                    reward_info["acc"],
                    rollouts_per_problem=rollout_n,
                    is_padding=padding,
                    candidate_order=[uid_by_rank[rank] for rank in sorted(uid_by_rank)],
                )
                selection = accumulator.ingest(groups)
                if selection.row_indices:
                    part = batch.select_idxs(np.asarray(selection.row_indices))
                    # Chunk-local timing/length metadata cannot be concatenated.
                    # Scoring metadata and row diagnostics are rebuilt afterwards.
                    part.meta_info = {
                        "temperature": temperature,
                        "reward_extra_keys": list(reward_keys),
                    }
                    # Generation already received the trainer step. vLLM returns
                    # per-trajectory global_steps in non_tensor_batch; adding a
                    # scalar under the same key breaks DataProto.to_tensordict().
                    parts.append(part)
                if not accumulator.complete:
                    accumulator.ensure_can_generate()
                    batch_dict = self._dynamic_candidates(
                        accumulator.deficits, accumulator.accepted_source_ids
                    )
            generation_failed = False
        except DynamicSamplingBudgetError:
            report = {
                "global_step": int(self.global_steps),
                "reason": "max_num_gen_batches_exhausted",
                "max_num_gen_batches": self.dynamic_max_num_gen_batches,
                "deficits": accumulator.deficits,
                "metrics": accumulator.metrics(),
                "sampling": self.dynamic_sampler.metrics(),
                "resume": "latest_complete_checkpoint",
            }
            report_path = (
                Path(self.config.trainer.default_local_dir)
                / "dynamic_sampling_failure.json"
            )
            temporary = report_path.with_suffix(".tmp")
            try:
                temporary.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                temporary.replace(report_path)
            except Exception as report_error:
                print(
                    f"TRACE dynamic-sampling failure report also failed: {report_error}"
                )
            print(
                f"TRACE dynamic sampling failed: {json.dumps(report, sort_keys=True)}"
            )
            raise
        finally:
            try:
                self.checkpoint_manager.sleep_replicas()
            except Exception as cleanup_error:
                if not generation_failed:
                    raise
                print(f"TRACE rollout cleanup also failed: {cleanup_error}")
        batch = DataProto.concat(parts)
        expected_rows = self.trace_sampling_config.problems_per_step * rollout_n
        if len(batch) != expected_rows:
            raise RuntimeError("dynamic sampling assembled an incomplete actor batch")
        # Pad COMPLETE placeholder groups, not individual trajectories. This
        # preserves UID grouping for GRPO and PrefixGrouper at arbitrary DP sizes.
        actor_dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        prompt_divisor = (
            actor_dp_size
            if getattr(self, "use_prefix_grouper", False)
            else actor_dp_size // math.gcd(actor_dp_size, rollout_n)
        )
        pad_groups = (-self.trace_sampling_config.problems_per_step) % prompt_divisor
        if pad_groups:
            placeholder = batch.select_idxs(np.arange(rollout_n)).repeat(
                repeat_times=pad_groups, interleave=False
            )
            placeholder.non_tensor_batch["uid"] = np.repeat(
                [str(uuid.uuid4()) for _ in range(pad_groups)], rollout_n
            ).astype(object)
            placeholder.non_tensor_batch["trace_opsd_is_padding"] = np.ones(
                len(placeholder), dtype=bool
            )
            placeholder.batch["response_mask"].zero_()
            placeholder.batch["rm_scores"].zero_()
            batch = DataProto.concat([batch, placeholder])
        self._finalize_training_batch(batch, metrics)
        validate_trace_actor_batch(self.config, len(batch))
        # Extract AFTER balance: external arrays must follow final row order.
        reward_tensor = batch.batch["rm_scores"]
        reward_info = {key: batch.non_tensor_batch[key] for key in reward_keys}
        self._pending_trace_quotas = dict(quotas)
        metrics.update(
            {
                f"trace/dynamic/{key}": value
                for key, value in accumulator.metrics().items()
            }
        )
        metrics.update(
            {
                f"trace/sampling/{key}": value
                for key, value in self.dynamic_sampler.metrics().items()
            }
        )
        return batch, reward_tensor, reward_info

    def _on_actor_update_success(
        self, batch: DataProto, actor_metrics: dict[str, Any]
    ) -> None:
        validate_actor_update_metrics(actor_metrics)
        quotas = getattr(self, "_pending_trace_quotas", None)
        if quotas is None:
            raise RuntimeError("TRACE actor update has no pending accepted batch")
        self.dynamic_sampler.record_accepted_update(quotas)
        self._pending_trace_quotas = None
        actor_metrics.update(
            {
                f"trace/sampling/{key}": value
                for key, value in self.dynamic_sampler.metrics().items()
            }
        )

    def _recover_exports(self) -> None:
        export_dir = self.config.trainer.get("trace_auto_export_dir", None)
        if export_dir:
            recover_trace_exports(
                run_dir=Path(self.config.trainer.default_local_dir),
                export_dir=Path(export_dir),
                total_steps=int(self.config.trainer.total_training_steps),
            )

    def _cleanup_training(self, *, failed: bool) -> None:
        """Release both resources even when one cleanup fails."""
        errors = []
        manager = getattr(self, "checkpoint_manager", None)
        if manager is not None:
            try:
                manager.sleep_replicas()
            except Exception as exc:
                errors.append(exc)
        executor = getattr(self, "_dump_executor", None)
        if executor is not None:
            try:
                self._shutdown_dump_executor()
            except Exception as exc:
                errors.append(exc)
            finally:
                try:
                    executor.shutdown(wait=True)
                except Exception as exc:
                    errors.append(exc)
        if errors:
            if failed:
                for error in errors:
                    print(f"TRACE cleanup failed after training error: {error}")
            else:
                raise errors[0]

    def _log_metrics(self, logger, *, data: dict, step: int) -> None:
        if hasattr(self, "trace_lr_scale"):
            data["actor/trace_lr_scale"] = self.trace_lr_scale
        if hasattr(self, "trace_learning_rate"):
            data["actor/trace_learning_rate"] = self.trace_learning_rate
        curriculum = self.config.trainer.trace_curriculum
        append_metrics(
            Path(self.config.trainer.default_local_dir) / "metrics.jsonl",
            step=step,
            data=data,
            stage_start=int(curriculum.start_step),
            stage_end=int(curriculum.end_step),
        )
        super()._log_metrics(logger, data=data, step=step)

    def _persist_update_metrics(self, batch, metrics: dict) -> None:
        # The boundary update must survive a failure during checkpoint saving
        # or validation. The final record later adds validation/timing metrics.
        if hasattr(self, "trace_lr_scale"):
            metrics["actor/trace_lr_scale"] = self.trace_lr_scale
        if hasattr(self, "trace_learning_rate"):
            metrics["actor/trace_learning_rate"] = self.trace_learning_rate
        padding = np.asarray(
            batch.non_tensor_batch.get("trace_opsd_is_padding", []), dtype=bool
        )
        real_batch = (
            batch.select_idxs(np.flatnonzero(~padding))
            if padding.size and padding.any()
            else batch
        )
        snapshot = {
            **metrics,
            **compute_data_metrics(batch=real_batch, use_critic=self.use_critic),
            "training/global_step": self.global_steps,
        }
        curriculum = self.config.trainer.trace_curriculum
        append_metrics(
            Path(self.config.trainer.default_local_dir) / "metrics.jsonl",
            step=self.global_steps,
            data=snapshot,
            stage_start=int(curriculum.start_step),
            stage_end=int(curriculum.end_step),
            phase="pre_checkpoint",
        )

    def fit(self) -> None:
        failed = True
        try:
            result = self._fit_stage()
            failed = False
            return result
        except BaseException:
            sampler = getattr(self, "dynamic_sampler", None)
            if sampler is not None:
                self.global_steps = sampler.accepted_updates
            self._pending_trace_quotas = None
            raise
        finally:
            self._cleanup_training(failed=failed)

    def _fit_stage(self) -> None:
        curriculum = self.config.trainer.get("trace_curriculum", None)
        if curriculum is None:
            raise ValueError("TRACE training requires curriculum stage configuration")
        full_steps = int(self.config.trainer.total_training_steps)
        end_step = int(curriculum.end_step)
        if full_steps not in {2, 622} or not 0 < end_step <= full_steps:
            raise ValueError("invalid TRACE curriculum training horizon")
        frequency = 1 if full_steps == 2 else 50
        if (
            int(self.config.trainer.save_freq) != frequency
            or int(self.config.trainer.test_freq) != frequency
        ):
            raise ValueError(
                f"TRACE curriculum requires save/test frequency {frequency}"
            )
        if bool(curriculum.get("revalidate", False)):
            self.global_steps = 0
            self._load_checkpoint()
            if self.global_steps != end_step:
                raise RuntimeError(
                    "revalidation checkpoint step differs from stage end"
                )
            self.checkpoint_manager.update_weights(self.global_steps)
            metrics = self._validate()
            print(f"Recovered TRACE curriculum validation: step={end_step}, {metrics}")
            self._recover_exports()
            return
        # The optimizer schedule was built with the full training horizon in
        # _create_dataloader; only this process's stopping point is shortened.
        self.total_training_steps = end_step
        return super().fit()

    def _load_checkpoint(self) -> None:
        curriculum = self.config.trainer.get("trace_curriculum", None)
        if curriculum is None:
            raise ValueError("TRACE training requires curriculum stage configuration")
        target_lr = self.config.trainer.get("trace_learning_rate", None)
        target_scale = self.config.trainer.get("trace_lr_scale", None)
        if target_lr is not None and target_scale is not None:
            raise ValueError(
                "TRACE trace_learning_rate and trace_lr_scale are mutually exclusive"
            )
        run_dir = Path(self.config.trainer.default_local_dir).resolve()
        latest = reconcile_trace_checkpoints(
            run_dir,
            sampler=self.dynamic_sampler,
            sampling_config=self.trace_sampling_config,
        )
        if self.config.trainer.resume_mode == "disable":
            if target_scale is not None or target_lr is not None:
                raise RuntimeError(
                    "TRACE learning-rate scaling requires checkpoint resume"
                )
            if int(curriculum.start_step) != 0:
                raise RuntimeError(
                    "only the first curriculum stage can start without a checkpoint"
                )
            if latest is not None:
                raise RuntimeError("existing curriculum checkpoint must be resumed")
            return
        if self.config.trainer.resume_mode != "auto":
            raise RuntimeError("curriculum requires the latest checkpoint in this run")
        if latest is None:
            raise RuntimeError("TRACE resume requires a complete checkpoint")
        checkpoint = (run_dir / f"global_step_{latest}").resolve()
        if checkpoint.parent != run_dir:
            raise RuntimeError("curriculum resume checkpoint is outside this run")
        state = verify_trace_checkpoint(
            checkpoint,
            sampler=self.dynamic_sampler,
            sampling_config=self.trace_sampling_config,
        )
        self.global_steps = int(checkpoint.name.removeprefix("global_step_"))
        if self.global_steps < int(curriculum.start_step) or self.global_steps > int(
            curriculum.end_step
        ):
            raise RuntimeError("checkpoint is outside the active curriculum stage")
        self.actor_rollout_wg.load_checkpoint(
            str(checkpoint / "actor"),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                str(checkpoint / str(Role.Critic)),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
        if int(state.get("global_step", -1)) != self.global_steps:
            raise RuntimeError("curriculum checkpoint trainer step mismatch")
        expected_samples = (
            self.global_steps * self.trace_sampling_config.problems_per_step
        )
        if int(state.get("consumed_samples", -1)) != expected_samples:
            raise RuntimeError("curriculum checkpoint sample offset mismatch")
        if "trace_dynamic_sampling" not in state:
            raise RuntimeError("checkpoint is missing TRACE dynamic sampling state")
        self.dynamic_sampler.load_state_dict(
            state["trace_dynamic_sampling"],
            expected_global_step=self.global_steps,
            problems_per_step=self.trace_sampling_config.problems_per_step,
        )
        target_scale = self.config.trainer.get("trace_lr_scale", None)
        saved_lr = state.get("trace_learning_rate", None)
        if saved_lr is not None and target_scale is not None:
            raise ValueError(
                "TRACE constant learning rate cannot be changed with trace_lr_scale"
            )
        if target_lr is None:
            target_lr = saved_lr
        if target_lr is not None:
            states = self.actor_rollout_wg.configure_trace_constant_learning_rate(
                target_lr
            )
            if not isinstance(states, (list, tuple)) or not states:
                raise RuntimeError(
                    "TRACE constant learning-rate control returned no worker states"
                )
            if any(item != states[0] for item in states):
                raise RuntimeError(
                    "TRACE learning-rate state differs across actor ranks"
                )
            self.trace_learning_rate = states[0]["learning_rate"]
            print(
                f"TRACE restored constant learning rate: step={self.global_steps}, "
                f"{states[0]}"
            )
        if target_scale is not None:
            states = self.actor_rollout_wg.configure_trace_learning_rate(target_scale)
            if not isinstance(states, (list, tuple)) or not states:
                raise RuntimeError(
                    "TRACE learning-rate scaling returned no worker states"
                )
            if any(item != states[0] for item in states):
                raise RuntimeError(
                    "TRACE learning-rate state differs across actor ranks"
                )
            self.trace_lr_scale = states[0]["scale"]
            print(
                f"TRACE restored learning-rate scaling: step={self.global_steps}, "
                f"{states[0]}"
            )
        self._recover_exports()
        print(
            f"Curriculum resume: step={self.global_steps}, "
            f"dynamic_sampling={self.dynamic_sampler.metrics()}"
        )

    def _save_checkpoint(self) -> None:
        run_dir = Path(self.config.trainer.default_local_dir)
        previous_latest = read_latest_step(run_dir, required=False)
        try:
            super()._save_checkpoint()
            verify_trace_checkpoint(
                run_dir / f"global_step_{self.global_steps}",
                sampler=self.dynamic_sampler,
                sampling_config=self.trace_sampling_config,
            )
            publish_latest_step(run_dir, int(self.global_steps))
        except Exception:
            try:
                restore_latest_step(run_dir, previous_latest)
            except Exception as restore_error:
                print(f"TRACE latest-pointer restore also failed: {restore_error}")
            raise
        export_dir = self.config.trainer.get("trace_auto_export_dir", None)
        if export_dir:
            target = publish_trace_export(
                run_dir=Path(self.config.trainer.default_local_dir),
                export_dir=Path(export_dir),
                step=int(self.global_steps),
                total_steps=int(self.config.trainer.total_training_steps),
            )
            if target is not None:
                print(f"Published TRACE evaluation model: {target}")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        config = kwargs.get("config", args[0] if args else None)
        if config is None or config.trainer.get("trace_curriculum", None) is None:
            raise ValueError("TRACE training requires curriculum stage configuration")
        validate_trace_update_config(config, use_critic=need_critic(config))
        checkpoint_config = config.actor_rollout_ref.actor.checkpoint
        for field in ("save_contents", "load_contents"):
            if not {"model", "optimizer", "extra"}.issubset(checkpoint_config[field]):
                raise ValueError(
                    f"TRACE checkpoint.{field} must include model,optimizer,extra"
                )
        if config.trainer.del_local_ckpt_after_load:
            raise ValueError("TRACE requires del_local_ckpt_after_load=false")
        self.strict_actor_update = True
        self._pending_trace_quotas = None
        if int(config.trainer.total_epochs) != 1:
            raise ValueError("TRACE curriculum requires total_epochs=1")
        if (
            config.data.get("gen_batch_size", config.data.train_batch_size)
            != config.data.train_batch_size
        ):
            raise ValueError("TRACE gen_batch_size must equal train_batch_size")
        self.curriculum_difficulties = tuple(kwargs.pop("curriculum_difficulties"))
        self.dynamic_sampler = kwargs.pop("dynamic_sampler", None)
        if not isinstance(self.dynamic_sampler, DifficultyPoolSampler):
            raise ValueError("TRACE curriculum requires a difficulty-pool sampler")
        if kwargs.get("train_sampler") is not None:
            raise ValueError("TRACE dynamic sampling does not use a dataloader sampler")
        self.dynamic_collate_fn = kwargs.get("collate_fn")
        if not callable(self.dynamic_collate_fn):
            raise ValueError("TRACE dynamic sampling requires a collate function")
        filter_config = config.algorithm.get("filter_groups", {})
        budget = filter_config.get("max_num_gen_batches", 10)
        if (
            filter_config.get("enable") is not True
            or filter_config.get("metric") != "acc"
            or isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget <= 0
        ):
            raise ValueError(
                "TRACE requires dynamic filtering on acc with positive budget"
            )
        self.dynamic_max_num_gen_batches = budget
        super().__init__(*args, **kwargs)
        if self.use_rm:
            raise ValueError(
                "TRACE dynamic sampling requires CPU/inline binary rewards; "
                "colocated reward models are not supported"
            )
        raw = OmegaConf.to_container(
            self.config.actor_rollout_ref.get("trace_opsd", {}), resolve=True
        )
        self.trace_config = TraceOPSDConfig.from_mapping(raw)
        raw_sampling = OmegaConf.to_container(
            self.config.actor_rollout_ref.get("trace_sampling", {}), resolve=True
        )
        self.trace_sampling_config = TraceSamplingConfig.from_mapping(raw_sampling)
        if self.trace_sampling_config.rollouts_per_problem != 8:
            raise ValueError("TRACE dynamic sampling requires 8 rollouts per problem")
        self.apply_chat_template_kwargs = dict(
            OmegaConf.to_container(
                self.config.data.get("apply_chat_template_kwargs", {}), resolve=True
            )
            or {}
        )
        if not self.trace_config.enabled:
            raise ValueError("TraceRayPPOTrainer requires trace_opsd.enabled=true")
        correction = self.config.algorithm.get("rollout_correction", None)
        if correction and correction.get("bypass_mode", False):
            raise ValueError(
                "TRACE scoring requires actor-recomputed old_log_probs; "
                "rollout_correction.bypass_mode must be false"
            )
        if self.config.algorithm.use_kl_in_reward:
            raise ValueError("TRACE-OPSD does not use KL as reward")
        if self.config.distillation.enabled:
            raise ValueError("TRACE-OPSD has no independent distillation loss")
        lora_rank = int(
            self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            or self.config.actor_rollout_ref.model.get("lora_rank", 0)
            or 0
        )
        if lora_rank > 0 or self.config.actor_rollout_ref.model.get(
            "lora_adapter_path"
        ):
            raise ValueError(
                "TRACE-OPSD current-policy hindsight scoring does not support LoRA"
            )
        strategy = str(self.config.actor_rollout_ref.actor.strategy)
        if strategy not in {"fsdp", "fsdp2"}:
            raise ValueError("TRACE-OPSD currently supports FSDP/FSDP2 actors")
        configured_max = self.trace_config.max_view_length
        self.trace_max_view_length = int(
            configured_max
            if configured_max is not None
            else self.config.actor_rollout_ref.rollout.max_model_len
        )
        self.trace_dp_divisor = int(
            self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        self.trace_checkpoint_retention: TraceCheckpointRetention | None = None
        run_dir = Path(self.config.trainer.default_local_dir)
        run_manifest = read_json(run_dir / 'trace_opsd_grpo_run.json')
        assert_trainable_trace_manifest(run_manifest)
        reconcile_trace_checkpoints(
            run_dir,
            sampler=self.dynamic_sampler,
            sampling_config=self.trace_sampling_config,
        )
        retention_config = self.config.trainer.get("trace_checkpoint", None)
        if retention_config and bool(retention_config.get("enabled", False)):
            if int(self.config.trainer.save_freq) != int(self.config.trainer.test_freq):
                raise ValueError(
                    "TRACE latest/best retention requires matching "
                    "save/test frequencies"
                )
            if self.config.trainer.get("max_actor_ckpt_to_keep", None) is not None:
                raise ValueError(
                    "TRACE latest/best retention requires generic actor "
                    "pruning disabled"
                )
            val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
            if (
                int(self.config.data.get("val_max_samples", -1)) != 256
                or int(val_kwargs.n) != 1
                or float(val_kwargs.temperature) != 0.0
                or bool(val_kwargs.do_sample)
            ):
                raise ValueError(
                    "TRACE latest/best retention requires fixed 256-sample "
                    "greedy pass@1 validation"
                )
            validation = OmegaConf.to_container(
                self.config.trainer.get("trace_validation", {}), resolve=True
            )
            self.trace_checkpoint_retention = TraceCheckpointRetention(
                run_dir=Path(self.config.trainer.default_local_dir),
                metric_key=str(retention_config.metric_key),
                margin=float(retention_config.best_margin),
                validation=dict(validation or {}),
                export_dir=(
                    Path(self.config.trainer.trace_auto_export_dir)
                    if self.config.trainer.get("trace_auto_export_dir", None)
                    else None
                ),
                total_steps=(
                    int(self.config.trainer.total_training_steps)
                    if self.config.trainer.get("trace_auto_export_dir", None)
                    else None
                ),
            )
            self.trace_checkpoint_retention.recover()

    def _validate(self, merged: bool = False):
        metrics = canonical_metrics(super()._validate(merged=merged))
        if merged:
            return metrics
        if self.trace_checkpoint_retention is None:
            curriculum = self.config.trainer.trace_curriculum
            if int(self.global_steps) == int(curriculum.end_step):
                write_stage_receipt(
                    Path(self.config.trainer.default_local_dir),
                    step=int(self.global_steps),
                    score=float(metrics["val-core/trace_opsd_math/acc/mean@1"]),
                    validation=OmegaConf.to_container(
                        self.config.trainer.trace_validation, resolve=True
                    ),
                )
            return metrics
        metric_key = self.trace_checkpoint_retention.metric_key
        if metric_key not in metrics:
            raise RuntimeError(
                f"TRACE best-checkpoint metric is missing from validation: {metric_key}"
            )
        if int(self.global_steps) == 0:
            receipt = self.trace_checkpoint_retention.record_initial_validation(
                score=float(metrics[metric_key]),
                model_path=str(self.config.actor_rollout_ref.model.path),
                generation={
                    "n": int(self.config.actor_rollout_ref.rollout.val_kwargs.n),
                    "temperature": float(
                        self.config.actor_rollout_ref.rollout.val_kwargs.temperature
                    ),
                    "do_sample": bool(
                        self.config.actor_rollout_ref.rollout.val_kwargs.do_sample
                    ),
                },
            )
            print(
                "Recorded TRACE initial-model baseline: "
                f"pass@1={float(metrics[metric_key]):.6f}, receipt={receipt}"
            )
            return metrics
        promoted = self.trace_checkpoint_retention.record_validation(
            step=int(self.global_steps), score=float(metrics[metric_key])
        )
        curriculum = self.config.trainer.get("trace_curriculum", None)
        if curriculum is not None and int(self.global_steps) == int(
            curriculum.end_step
        ):
            write_stage_receipt(
                Path(self.config.trainer.default_local_dir),
                step=int(self.global_steps),
                score=float(metrics[metric_key]),
                validation=self.trace_checkpoint_retention.validation,
            )
        if promoted:
            print(
                "Published TRACE best checkpoint: "
                f"step={self.global_steps}, pass@1={float(metrics[metric_key]):.6f}"
            )
        return metrics

    def _compute_hindsight_log_probs(
        self, views: Sequence[TraceView], *, temperature: float
    ) -> list[tuple[float, ...]]:
        if any(view.view == "causal" for view in views):
            raise ValueError("causal TRACE views must use current-policy old_log_probs")
        scores: list[tuple[float, ...]] = []
        chunk_size = max(
            self.trace_dp_divisor,
            self.trace_config.score_chunk_size,
        )
        for start in range(0, len(views), chunk_size):
            chunk = list(views[start : start + chunk_size])
            data = _make_score_batch(chunk, int(self.tokenizer.pad_token_id))
            padded, pad_size = pad_dataproto_to_divisor(data, self.trace_dp_divisor)
            batch_td = left_right_2_no_padding(padded.to_tensordict())
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=False,
                compute_loss=False,
                temperature=temperature,
            )
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            log_probs = tu.get(output, "log_probs")
            log_probs = no_padding_2_padding(log_probs, batch_td)
            scored = DataProto.from_tensordict(
                tu.get_tensordict({"trace_log_probs": log_probs.float()})
            )
            scored = unpad_dataproto(scored, pad_size=pad_size)
            mask = data.batch["response_mask"].to(torch.bool)
            values = scored.batch["trace_log_probs"]
            if values.shape != mask.shape:
                raise TraceTrajectoryError(
                    "TRACE hindsight token score shape does not match its response mask"
                )
            if not torch.isfinite(values[mask]).all():
                raise TraceTrajectoryError(
                    "current-policy hindsight scoring produced non-finite "
                    "log-probabilities"
                )
            for row, view in enumerate(chunk):
                scores.append(
                    tuple(values[row, : len(view.action_ids)].detach().cpu().tolist())
                )
        return scores

    def _score_trace_views(
        self, batch: DataProto, views: Sequence[TraceView]
    ) -> list[tuple[float, ...]]:
        old_log_probs = batch.batch.get("old_log_probs")
        if old_log_probs is None:
            raise TraceTrajectoryError(
                "TRACE shaping requires current-policy old_log_probs"
            )
        expected_temperature = float(self.config.actor_rollout_ref.rollout.temperature)
        temperature = float(batch.meta_info.get("temperature", expected_temperature))
        if not math.isclose(
            temperature, expected_temperature, rel_tol=0.0, abs_tol=1e-12
        ):
            raise TraceTrajectoryError(
                "TRACE causal and hindsight scoring temperatures do not match"
            )
        causal_scores = causal_scores_from_old_log_probs(
            views,
            old_log_probs,
            batch.batch["responses"],
            batch.batch["response_mask"],
        )
        hindsight_views = [view for view in views if view.view != "causal"]
        hindsight_scores = iter(
            self._compute_hindsight_log_probs(hindsight_views, temperature=temperature)
        )
        scores: list[tuple[float, ...]] = []
        for view in views:
            if view.view != "causal":
                try:
                    scores.append(next(hindsight_scores))
                except StopIteration as exc:
                    raise TraceTrajectoryError(
                        "TRACE hindsight score count does not match its views"
                    ) from exc
                continue
            scores.append(causal_scores[(view.trajectory_index, view.turn_index)])
        try:
            next(hindsight_scores)
        except StopIteration:
            return scores
        raise TraceTrajectoryError(
            "TRACE hindsight score count does not match its views"
        )

    def _shape_trace_advantages(
        self, batch: DataProto
    ) -> tuple[DataProto, dict[str, float]]:
        view_build_started = time.perf_counter()
        views, metrics = collect_trace_views(
            batch,
            tokenizer=self.tokenizer,
            max_view_length=self.trace_max_view_length,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
        )
        metrics["timing_s/trace_view_build"] = time.perf_counter() - view_build_started
        scores = []
        if views:
            score_started = time.perf_counter()
            scores = self._score_trace_views(batch, views)
            metrics["timing_s/trace_teacher_score"] = (
                time.perf_counter() - score_started
            )
        else:
            metrics["timing_s/trace_teacher_score"] = 0.0

        credit_shape_started = time.perf_counter()
        evidence = evidence_from_view_scores(views, scores)
        metrics.update(summarize_trace_evidence(views, scores, evidence))
        base_advantages = batch.batch["advantages"]
        shaped, weight_metrics = shape_token_advantages(
            base_advantages,
            batch.batch["response_mask"],
            evidence,
            credit_scale=self.trace_config.credit_scale,
            final_credit_scale=self.trace_config.final_credit_scale,
            evidence_scale=self.trace_config.evidence_scale,
        )
        metrics.update(weight_metrics)
        enabled = self.trace_config.python_error_penalty_enabled
        turns = collect_python_error_turns(batch) if enabled else []
        final, penalty_metrics, diagnostics = apply_python_error_penalty(
            base_advantages,
            shaped,
            batch.batch["response_mask"],
            turns,
            enabled=enabled,
            coef=self.trace_config.python_error_penalty_coef,
        )
        batch.batch["advantages"] = final
        metrics.update(penalty_metrics)
        if enabled:
            write_turn_credit(
                Path(self.config.trainer.default_local_dir),
                step=self.global_steps,
                credit_state=checkpoint_credit_state(
                    Path(self.config.trainer.default_local_dir), self.global_steps
                ),
                trajectory_count=int(batch.batch["responses"].shape[0]),
                turns=diagnostics,
            )
        metrics["timing_s/trace_credit_shape"] = (
            time.perf_counter() - credit_shape_started
        )
        return batch, metrics

    def _log_rollout_data(
        self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir
    ):
        # Retain the pre-padding row index: per-turn credit summaries point at
        # these exact trajectories even after batch balancing/permutation.
        padding = batch.non_tensor_batch.get("trace_opsd_is_padding")
        indices = [
            i for i in range(len(batch)) if padding is None or not bool(padding[i])
        ]
        real_batch = batch.select_idxs(indices)
        extra = {
            key: [values[i] for i in indices] if len(values) == len(batch) else values
            for key, values in reward_extra_infos_dict.items()
        }
        extra["trace_trajectory_index"] = indices
        extra["trajectory_uid"] = [
            str(batch.non_tensor_batch["uid"][i]) for i in indices
        ]
        extra["trace_tool_turns"] = [
            batch.non_tensor_batch["trace_tool_turns"][i] for i in indices
        ]
        width = batch.batch["responses"].shape[1]
        counts = batch.batch["attention_mask"][:, -width:].sum(-1).tolist()
        extra["trace_response_token_count"] = [int(counts[i]) for i in indices]
        spans = []
        for row in batch.batch["response_mask"][indices].to(torch.int8).cpu():
            edges = torch.diff(
                torch.cat(
                    (
                        torch.zeros(1, dtype=row.dtype),
                        row,
                        torch.zeros(1, dtype=row.dtype),
                    )
                )
            )
            starts = (edges == 1).nonzero().flatten().tolist()
            ends = (edges == -1).nonzero().flatten().tolist()
            spans.append(
                [[start, end] for start, end in zip(starts, ends, strict=True)]
            )
        extra["trace_assistant_spans"] = spans
        return super()._log_rollout_data(
            real_batch, extra, timing_raw, rollout_data_dir
        )

    def _postprocess_advantages(
        self, batch: DataProto
    ) -> tuple[DataProto, dict[str, float]]:
        batch, trace_metrics = self._shape_trace_advantages(batch)
        is_padding = batch.non_tensor_batch.get("trace_opsd_is_padding")
        real_rows = [
            index
            for index in range(batch.batch["responses"].shape[0])
            if is_padding is None or not bool(is_padding[index])
        ]
        scores = batch.batch["token_level_scores"].sum(dim=-1).tolist()
        trace_metrics.update(
            summarize_trajectory_stability(
                uids=[batch.non_tensor_batch["uid"][index] for index in real_rows],
                scores=[scores[index] for index in real_rows],
                ended_with_eos=[
                    batch.non_tensor_batch["ended_with_eos"][index]
                    for index in real_rows
                ],
                termination_reasons=[
                    batch.non_tensor_batch["termination_reason"][index]
                    for index in real_rows
                ],
            )
        )
        if self.config.actor_rollout_ref.actor.use_kl_loss:
            mask = batch.batch["response_mask"][real_rows].to(torch.bool)
            if mask.any():
                per_token_kl = kl_penalty(
                    logprob=batch.batch["old_log_probs"][real_rows],
                    ref_logprob=batch.batch["ref_log_prob"][real_rows],
                    kl_penalty=self.config.actor_rollout_ref.actor.kl_loss_type,
                )
                trace_metrics["trace/reference_kl/mean"] = float(
                    per_token_kl[mask].mean().item()
                )
        sampling = self.trace_sampling_config
        trace_metrics.update(
            {
                "trace/sampling/pool_size": float(sampling.pool_size),
                "trace/sampling/problems_per_step": float(sampling.problems_per_step),
                "trace/sampling/rollouts_per_problem": float(
                    sampling.rollouts_per_problem
                ),
                "trace/sampling/trajectories_per_step": float(
                    sampling.problems_per_step * sampling.rollouts_per_problem
                ),
                "trace/sampling/planned_steps": float(sampling.total_steps),
            }
        )
        trace_metrics.update(
            {
                f"trace/sampling/{key}": value
                for key, value in self.dynamic_sampler.metrics().items()
            }
        )
        return batch, trace_metrics


__all__ = [
    "TraceActorRolloutWorker",
    "TraceRayPPOTrainer",
    "collect_trace_views",
    "collect_python_error_turns",
]
