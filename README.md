# Trace-OPSD

Dual-tool post-training for multi-turn mathematical reasoning: **SFT → TRACE-OPSD → evaluation**.
The default setup uses four A800 GPUs for training and one A800 for evaluation.
Python execution and General Search run on the CPU.

The model invokes tools with `<python>...</python>` or `<search>...</search>` and
returns a final answer as `<answer>...\boxed{...}</answer>`. Rewards depend on the
final mathematical answer. Tool observations are excluded from the policy loss.

## Assets and runtime directories

```text
Trace-OPSD-workspace/
├── Trace-OPSD/                 # This source repository
├── assets/
│   ├── models/                # Complete HF models, including Qwen2.5-3B-Instruct and e5-base-v2
│   ├── datasets/sft/sft_math_tir.json
│   ├── datasets/numinamath/rl_trace_opsd.parquet
│   ├── datasets/grpo/         # Fixed train/validation splits and manifest
│   ├── retrieval/general_search/{corpus,index}/
│   ├── wheels/               # Pinned FlashAttention wheel for A800
│   └── runs/                 # Default location for checkpoints, exports, metrics, evaluation, and caches
└── records/                  # Local historical experiment records, outside the source repository
```

All public launchers accept `--assets-root` and `--work-root`. These paths can also
be set through `TRACE_OPSD_ASSET_ROOT` and `TRACE_OPSD_WORK_ROOT`. Assets default to
the sibling directory `../assets`; runtime outputs default to its `runs` directory.
Environment setup does not download large models or datasets. See
[External assets](docs/assets.md) for sources, pinned revisions, and verification.

## Four launchers

Run these examples from the repository root on Linux/AutoDL. Set `ASSETS` to your
actual asset directory.

```bash
export ASSETS=/root/autodl-tmp/assets
export WORK="$ASSETS/runs"

# 1. Set up the A800 environment and an isolated Python executor user.
bash scripts/setup_env.sh --assets-root "$ASSETS" --work-root "$WORK" --gpu-count 4

# 2. Train on prepared TIR data and export a complete HF model on completion.
bash scripts/run_sft.sh train --assets-root "$ASSETS" --work-root "$WORK" \
  --run-name sft-math --model-path "$ASSETS/models/Qwen2.5-3B-Instruct" \
  --dataset-file "$ASSETS/datasets/sft/sft_math_tir.json"

# 3. Run the first TRACE stage. The 16/16 difficulty mix is an example.
bash scripts/run_trace_curriculum.sh --assets-root "$ASSETS" --work-root "$WORK" \
  --run-name trace-math --model-path "$WORK/exports/sft/sft-math/final" \
  --after-step 0 --mid-per-step 16 --high-per-step 16 --resume none

# For subsequent stages, use the same run, the previous stage's final step,
# and the difficulty mix for the new stage:
# --after-step 50 --mid-per-step 16 --high-per-step 16 --resume latest

# 4. Copy and fill in a model list, then select greedy, sampling8, or both.
mkdir -p "$WORK"
cp configs/evaluation/models.example.yaml "$WORK/evaluation-models.yaml"
# Edit model names, complete HF paths, and the optional baseline in this file.
bash scripts/run_eval_suite.sh --assets-root "$ASSETS" --work-root "$WORK" \
  --models-file "$WORK/evaluation-models.yaml" --protocol all \
  --output-root "$WORK/evaluation/math-comparison"
```

Evaluation requires an explicit model list. Unfilled placeholders fail preflight;
the launcher does not infer historical runs or substitute the latest checkpoint.
Use `--protocol greedy` or `--protocol sampling8` to run either protocol separately.
Every launcher provides `--help` and retains asset, tool-service, and checkpoint checks.

## Default training and inference settings

| Setting | SFT | TRACE / evaluation |
|---|---|---|
| Precision | Full-parameter BF16 training | BF16 inference; full-parameter TRACE training |
| Token budgets | Full training sequence truncated at 15,000 | Context 8192; total response 4096; generation per turn 1024; observation 512 |
| Tool calls | Prepared tool trajectories; no online generation budget | At most 5 calls; stop after 2 consecutive invalid actions |
| Data / sampling | All 30,438 Alpaca records used for training | TRACE: 32 accepted problem groups per step, 8 trajectories per problem |
| Optimization | Batch 1 per GPU; accumulation 4; LR 7e-6; 3 epochs; cosine; warmup 0.1 | LR 5e-7; 50 warmup steps; cosine minimum ratio 0.1; KL coefficient 0.005 |

SFT follows Tool-Star's 15,000-token truncation behavior and reports truncation
counts. It preserves tool feedback inside the original `output` without splitting
the record. TRACE first evaluates the initial model on a fixed 256-question
baseline. Every 50 steps, it saves a checkpoint, validates, packages the stage's
metrics, and exits. The final stage ends at step 622. Retention keeps the two latest
complete checkpoints and the best checkpoint.

## Algorithm and evaluation

The current TRACE algorithm is `current_policy_solution_hindsight.turn_mean.v4`.
It averages raw teacher–student log-probability gaps within each assistant turn,
computes one credit weight, and broadcasts that weight to every valid generated
token in the turn. Python error penalties are enabled by default with coefficient
0.25 and are applied after OPSD scaling.

The fixed evaluation snapshot covers GSM8K, MATH500, and AIME24/25/26, totaling
1,909 questions. Greedy evaluation reports Pass@1. Eight-sample evaluation reports
Pass@8, Avg@8, Majority@8, and diagnostic Pass@1/2/4 curves. Both retain tool calls,
execution errors, missing answers, truncation, and generation failures.
Avg@8 is accuracy under the sampling protocol; it is distinct from greedy Pass@1
at temperature 0.

Historical training and the published step400 model used the credit mechanism
active at that time. Their existing evaluation results do not validate turn-mean
v4. Historical checkpoints support read-only inspection and HF export for use as
initial weights in a new run; optimizer state is not migrated to the new mechanism.

## Documentation and validation

- [SFT](docs/sft.md), [TRACE training](docs/trace-training.md), [Algorithm](docs/trace-opsd-method.md)
- [Evaluation](docs/evaluation.md), [Eight-sample metrics](docs/evaluation-sampling.md)
- [A800 environment](docs/hardware-profiles.md), [General Search](docs/general-search.md)
- [Python error credit](docs/trace-python-error-credit.md), [Validation](docs/validation.md)
- [Project naming and AutoDL migration](docs/project-renaming.md)

Complete vendored snapshots of verl and LLaMA-Factory retain their original
licenses. Environment setup verifies their content digests. Upstream examples are
third-party source, separate from Trace-OPSD's public experiment launchers. See
`NOTICE` for attribution. CPU regression checks cover the algorithm, tool protocol,
recovery, and evaluation. The full GPU workflow after repository cleanup has not
been executed locally.
