"""Hydra-compatible local entry point for TRACE-OPSD training."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from pprint import pprint

import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.main_ppo import (
    TaskRunner,
    create_rl_dataset,
    run_ppo,
)
from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local

from trace_opsd.compat import namespace_matches

from .cli import synchronize_trace_run_manifest
from .credit_revision import (
    assert_trainable_trace_manifest,
    resolve_trace_credit_config,
)
from .curriculum import (
    LEDGER_NAME,
    CurriculumError,
    _read_json,
    _validate_ledger,
    load_difficulties,
    validate_curriculum_plan,
)
from .sampling import (
    DifficultyPoolSampler,
    TraceSamplingConfig,
)
from .schema import validate_dataset_directory
from .trace_runtime import TraceActorRolloutWorker, TraceRayPPOTrainer
from .turn_credit_io import check_python_execution_protocol
from .update_contract import validate_trace_update_config

ROOT = Path(__file__).resolve().parents[2]
VERL_CONFIG = ROOT / "frameworks" / "verl" / "verl" / "trainer" / "config"
TRACE_CONFIG = ROOT / "configs" / "grpo" / "trace_opsd.yaml"


def _resolve_sampling_config(config, pool_size: int) -> TraceSamplingConfig:
    if config.trainer.get("trace_curriculum", None) is None:
        raise CurriculumError("TRACE training requires curriculum stage configuration")
    train_max_samples = config.data.get("train_max_samples", -1)
    if train_max_samples not in (-1, None):
        raise ValueError("TRACE fixed-pool training requires data.train_max_samples=-1")
    problems_per_step = int(
        config.data.get("gen_batch_size", config.data.train_batch_size)
    )
    if problems_per_step != int(config.data.train_batch_size):
        raise CurriculumError("TRACE gen_batch_size must equal train_batch_size")
    if int(config.trainer.total_epochs) != 1:
        raise CurriculumError("TRACE curriculum requires total_epochs=1")
    total_steps = config.trainer.total_training_steps
    if total_steps is None:
        raise ValueError("TRACE fixed-pool training requires total_training_steps")
    seed = config.data.get("seed", 42)
    if seed is None:
        seed = 42
    return TraceSamplingConfig(
        mode="curriculum_dynamic_cycles",
        seed=int(seed),
        pool_size=int(pool_size),
        total_steps=int(total_steps),
        problems_per_step=problems_per_step,
        rollouts_per_problem=int(config.actor_rollout_ref.rollout.n),
    )


def _validation_descriptor(dataset) -> dict[str, object]:
    """Describe the exact, ordered validation subset used for pass@1."""

    dataframe = getattr(dataset, "dataframe", None)
    if dataframe is None or "extra_info" not in dataframe.column_names:
        raise ValueError("TRACE validation dataset must expose extra_info.source_id")
    source_ids: list[str] = []
    for extra_info in dataframe["extra_info"]:
        source_id = (
            extra_info.get("source_id") if isinstance(extra_info, dict) else None
        )
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("TRACE validation row is missing extra_info.source_id")
        source_ids.append(source_id)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("TRACE validation source IDs must be unique")
    digest = hashlib.sha256(
        json.dumps(source_ids, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "selection": "first_n",
        "samples": len(source_ids),
        "source_id_sha256": digest,
    }


class TraceTaskRunner(TaskRunner):
    """Construct the normal verl workflow with Trace-OPSD TRACE extensions."""

    def add_actor_rollout_worker(self, config):
        role = (
            Role.ActorRolloutRef if need_reference_policy(config) else Role.ActorRollout
        )
        self.role_worker_mapping[role] = ray.remote(TraceActorRolloutWorker)
        self.mapping[role] = "global_pool"
        return TraceActorRolloutWorker, RayWorkerGroup

    def run(self, config) -> None:
        run_manifest_path = (
            Path(config.trainer.default_local_dir) / "trace_opsd_grpo_run.json"
        )
        assert_trainable_trace_manifest(_read_json(run_manifest_path))
        validate_trace_update_config(config, use_critic=need_critic(config))
        if config.trainer.get("trace_curriculum", None) is None:
            raise CurriculumError(
                "TRACE training requires curriculum stage configuration"
            )
        print(f"TRACE TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True
        )
        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        sampling_config = _resolve_sampling_config(config, len(train_dataset))
        curriculum_config = config.trainer.get("trace_curriculum", None)
        run_dir = Path(config.trainer.default_local_dir)
        ledger = _read_json(run_dir / LEDGER_NAME)
        source_path = Path(str(curriculum_config.source_path))
        dataset_dir = Path(str(config.data.train_files)).parent
        dataset_summary = validate_dataset_directory(dataset_dir)
        source_ids, difficulties, source_hash = load_difficulties(
            dataset_dir, source_path
        )
        actual_ids = [row["source_id"] for row in train_dataset.dataframe["extra_info"]]
        if actual_ids != source_ids:
            raise CurriculumError(
                "loaded training rows differ from curriculum source ID order"
            )
        _validate_ledger(
            ledger,
            dataset_digest=dataset_summary.digest,
            source_hash=source_hash,
            config=sampling_config,
        )
        if (
            ledger.get("train_source_ids_sha256")
            != hashlib.sha256(
                json.dumps(source_ids, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
        ):
            raise CurriculumError("training source ID order changed")
        stages = ledger["stages"]
        if (
            not stages
            or stages[-1]["start_step"] != int(curriculum_config.start_step)
            or stages[-1]["end_step"] != int(curriculum_config.end_step)
            or stages[-1]["mid_per_step"] != int(curriculum_config.mid_per_step)
            or stages[-1]["high_per_step"] != int(curriculum_config.high_per_step)
        ):
            raise CurriculumError("active curriculum stage differs from run ledger")
        validate_curriculum_plan(sampling_config, difficulties, ledger)
        dynamic_sampler = DifficultyPoolSampler(
            source_ids,
            difficulties,
            seed=sampling_config.seed,
            dataset_digest=dataset_summary.digest,
        )
        with open_dict(config.actor_rollout_ref):
            config.actor_rollout_ref.trace_sampling = OmegaConf.create(
                sampling_config.to_dict()
            )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        validation_config = _validation_descriptor(val_dataset)
        run_dir = Path(config.trainer.default_local_dir)
        revalidate = bool(curriculum_config.get("revalidate", False))
        if revalidate:
            with open_dict(config.actor_rollout_ref):
                config.actor_rollout_ref.trace_opsd = OmegaConf.create(
                    resolve_trace_credit_config(
                        run_dir, int(curriculum_config.end_step)
                    )
                )
        if config.actor_rollout_ref.trace_opsd.python_error_penalty_enabled:
            agents = OmegaConf.load(
                config.actor_rollout_ref.rollout.agent.agent_loop_config_path
            )
            tool_url = next(
                (
                    item.get("tool_base_url")
                    for item in agents
                    if namespace_matches(item.get('name'), 'trace_opsd_agent')
                ),
                None,
            )
            if not tool_url:
                raise ValueError(
                    "Python error credit requires a configured tool_base_url"
                )
            check_python_execution_protocol(str(tool_url))
        synchronize_trace_run_manifest(
            run_dir,
            OmegaConf.to_container(config.actor_rollout_ref.trace_opsd, resolve=True),
            sampling_config.to_dict(),
            validation_config,
            dynamic_sampling_config={
                "enable": config.algorithm.filter_groups.enable,
                "metric": config.algorithm.filter_groups.metric,
                "max_num_gen_batches": config.algorithm.filter_groups.get(
                    "max_num_gen_batches"
                ),
                "gen_batch_size": config.data.get(
                    "gen_batch_size", config.data.train_batch_size
                ),
            },
            credit_transition_after_step=config.trainer.get(
                "trace_credit_transition_after_step"
            ),
            curriculum_start_step=int(curriculum_config.start_step),
            curriculum_end_step=int(curriculum_config.end_step),
            revalidate=revalidate,
        )
        with open_dict(config.trainer):
            config.trainer.trace_validation = OmegaConf.create(validation_config)
        trainer = TraceRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            dynamic_sampler=dynamic_sampler,
            curriculum_difficulties=difficulties,
        )
        trainer.init_workers()
        trainer.fit()


def _compose_config(overrides: list[str]):
    with initialize_config_dir(config_dir=str(VERL_CONFIG), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    defaults = OmegaConf.load(TRACE_CONFIG)
    supplied = config.actor_rollout_ref.get("trace_opsd", {})
    with open_dict(config.actor_rollout_ref):
        config.actor_rollout_ref.trace_opsd = OmegaConf.merge(defaults, supplied)
    return config


def main(argv: list[str] | None = None) -> int:
    config = _compose_config(list(sys.argv[1:] if argv is None else argv))
    if config.trainer.get("trace_curriculum", None) is None:
        raise CurriculumError("TRACE training requires curriculum stage configuration")
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    runner_class = ray.remote(num_cpus=1)(TraceTaskRunner)
    run_ppo(config, task_runner_class=runner_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
