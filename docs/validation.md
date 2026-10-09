# Validation

Local CPU regressions cover the turn-mean formula, PPO gradient direction, Python
error penalties, tool protocol, curriculum sampling, checkpoint recovery/retention,
SFT input/truncation, and greedy/multi-sample evaluation.

Earlier cleanup validation collected 1,015 cases: batch checks covered 1,014 passes
and one skip because the Windows Torch build lacked Gloo transport support. Ruff,
help for the four launchers, all Bash syntax checks, and both complete vendor
digests passed. Executor-user tests used command substitutes; they did not install
Linux packages or create actual accounts.

Rename validation passed 684 CPU regression cases and 14 new compatibility cases,
with one Windows Gloo case skipped. Working test copies and full records reside
locally in the outer `records/project-renaming/2026-10-09/validation/` directory.
Public source does not contain a root `tests` directory, so `pytest tests` is not
available there. See [Naming and migration](project-renaming.md) for scope.

```bash
python scripts/verify_vendored_framework.py frameworks/verl --expected-name verl
python scripts/verify_vendored_framework.py frameworks/LlamaFactory --expected-name LlamaFactory
```

Use `--help` for current arguments. The four public launchers can display help
without starting GPU services. Before a formal A800 run, verify assets and perform
a small process check:

```bash
export ASSETS=/root/autodl-tmp/assets
export WORK="$ASSETS/runs"
# Recheck four-GPU CUDA, FlashAttention, and NCCL after environment setup.
bash scripts/setup_env.sh --validate-only --gpu-count 4 \
  --assets-root "$ASSETS" --work-root "$WORK"
bash scripts/run_sft.sh validate --assets-root "$ASSETS" --work-root "$WORK"
bash scripts/run_grpo.sh validate --assets-root "$ASSETS" --work-root "$WORK" \
  --model-path "$WORK/exports/sft/sft-math/final"
bash scripts/run_eval_suite.sh --assets-root "$ASSETS" --work-root "$WORK" \
  --models-file /path/to/models.yaml --protocol all \
  --name process-check --preflight-only
bash scripts/run_eval_suite.sh --assets-root "$ASSETS" --work-root "$WORK" \
  --models-file /path/to/models.yaml --protocol all \
  --limit-per-benchmark 2 --output-root "$WORK/evaluation/process-check"
```

Small checks cover loading, generation, Python/search execution, saving, scoring,
and recovery. They have no accuracy threshold and must remain separate from formal
complete results. A800 training/inference was not run locally during cleanup.
GPU kernels, NCCL, actual SFT updates, and actual TRACE updates must be validated
on the target device; CPU substitutes do not establish those checks.
