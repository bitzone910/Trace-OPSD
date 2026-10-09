"""Trace-OPSD-owned SFT workflow assembled from vendored LlamaFactory components."""

from __future__ import annotations

from typing import Any

from .trainer import TrainingArtifactContext, get_trace_opsd_trainer_class


def run_sft(
    args: dict[str, Any],
    *,
    artifact_context: TrainingArtifactContext,
    resume: str | None = None,
) -> dict[str, float]:
    """Run full-parameter SFT without changing the vendored framework."""

    from llamafactory.data import (
        SFTDataCollatorWith4DAttentionMask,
        get_dataset,
        get_template_and_fix_tokenizer,
    )
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.extras.misc import calculate_tps
    from llamafactory.extras.ploting import plot_loss
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_model, load_tokenizer

    model_args, data_args, training_args, finetuning_args, generating_args = (
        get_train_args(args)
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="sft",
        **tokenizer_module,
    )
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=(
            IGNORE_INDEX
            if data_args.ignore_pad_token_for_loss
            else tokenizer.pad_token_id
        ),
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [
        tokenizer.eos_token_id
    ] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    trainer_class = get_trace_opsd_trainer_class()
    trainer = trainer_class(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        artifact_context=artifact_context,
        gen_kwargs=gen_kwargs,
        ref_model=None,
        **dataset_module,
        **tokenizer_module,
    )
    resume_dir = trainer.prepare_local_resume(resume) if resume else None
    result = trainer.train(resume_from_checkpoint=resume_dir)
    if finetuning_args.include_effective_tokens_per_second:
        result.metrics["effective_tokens_per_sec"] = calculate_tps(
            dataset_module["train_dataset"], result.metrics, stage="sft"
        )
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    # The published Alpaca data is all training data. No validation subset or
    # best-loss selection is introduced by this stage.
    trainer.save_final_local_checkpoint(result.metrics)
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        plot_loss(training_args.output_dir, keys=["loss"])
    trainer.cleanup_resume_staging()
    return {
        key: float(value)
        for key, value in result.metrics.items()
        if isinstance(value, (int, float))
    }
