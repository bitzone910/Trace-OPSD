# Staged TRACE Training

A new run starts from complete SFT HF weights and first evaluates a fixed
256-question baseline. Each invocation trains at most 50 global steps, then saves
a checkpoint, validates, packages complete metrics, and exits. The final stage
ends at 622. Each step accepts 32 problem groups with eight trajectories per
problem; mid and high quotas must sum to 32. Dynamic sampling defaults to at most
10 candidate batches and preserves actual sampling cycles/cursors and quotas.

```bash
bash scripts/run_trace_curriculum.sh --assets-root ../assets --run-name trace-math \
  --model-path ../assets/runs/exports/sft/sft-math/final \
  --after-step 0 --mid-per-step 16 --high-per-step 16 --resume none
```

The 16/16 mix is an example. For subsequent stages, explicitly specify the previous
stage's final step and the new difficulty mix, and use `--resume latest` to preserve
the optimizer, scheduler, random state, and sampling cursor.

Defaults are four A800s, CPU retrieval, LR 5e-7, 50 warmup steps, cosine minimum
ratio 0.1, and KL coefficient 0.005. Budgets are prompt 2048, context 8192, response
4096, generation per turn 1024, observation 512, and at most five tool calls.
All new training uses turn-mean v4. Python error penalties default to enabled with
coefficient 0.25. See [Algorithm](trace-opsd-method.md).

Checkpoints reside in `<work>/checkpoints/grpo/<run>`. Retention keeps the two latest
complete checkpoints and the best checkpoint. Rollouts, validation, per-turn credit,
and training metrics are written under the external work directory. Each stage
automatically produces an analysis archive. Missing metrics make the archive
explicitly incomplete; missing values are not filled with zero.

Historical token-credit runs support read-only inspection/export, without in-place
optimizer recovery into the new algorithm. Standalone export uses the internal
`scripts/run_grpo.sh export` command with an explicit run and step; see its help.
