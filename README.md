# Trace-OPSD

Dual-tool post-training for multi-turn mathematical reasoning: **SFT → TRACE-OPSD → evaluation**.

The model invokes tools with `<python>...</python>` or `<search>...</search>` and
returns a final answer as `<answer>...\boxed{...}</answer>`. 

## Assets and runtime directories

```text
Trace-OPSD-workspace/
├── Trace-OPSD/                 # This source repository
├── assets/
│   ├── models/                
│   ├── datasets/sft/sft_math_tir.json
│   ├── datasets/numinamath/rl_trace_opsd.parquet
│   ├── datasets/grpo/         
│   ├── retrieval/general_search/{corpus,index}/
│   ├── wheels/               
│   └── runs/                 
└── records/                 
```

## Four launchers

```bash
export ASSETS=/root/autodl-tmp/assets
export WORK="$ASSETS/runs"

# 1. Set up the environment.
bash scripts/setup_env.sh --assets-root "$ASSETS" --work-root "$WORK" --gpu-count 4

# 2. Train on prepared TIR data.
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

# 4. Evaluation.
mkdir -p "$WORK"
cp configs/evaluation/models.example.yaml "$WORK/evaluation-models.yaml"
# Edit model names, complete HF paths, and the optional baseline in this file.
bash scripts/run_eval_suite.sh --assets-root "$ASSETS" --work-root "$WORK" \
  --models-file "$WORK/evaluation-models.yaml" --protocol all \
  --output-root "$WORK/evaluation/math-comparison"
```

## Default training and inference settings

| Setting | SFT | TRACE / evaluation |
|---|---|---|
| Precision | Full-parameter BF16 training | BF16 inference; full-parameter TRACE training |
| Token budgets | Full training sequence truncated at 15,000 | Context 8192; total response 4096; generation per turn 1024; observation 512 |
| Tool calls | Prepared tool trajectories; no online generation budget | At most 5 calls; stop after 2 consecutive invalid actions |
| Data / sampling | All 30,438 Alpaca records used for training | TRACE: 32 accepted problem groups per step, 8 trajectories per problem |
| Optimization | Batch 1 per GPU; accumulation 4; LR 7e-6; 3 epochs; cosine; warmup 0.1 | LR 5e-7; 50 warmup steps; cosine minimum ratio 0.1; KL coefficient 0.005 |

## Algorithm and evaluation

The current TRACE algorithm averages raw teacher–student log-probability gaps within each assistant turn,
computes one credit weight, and broadcasts that weight to every valid generated
token in the turn. 

The fixed evaluation snapshot covers GSM8K, MATH500, and AIME24/25/26, totaling
1,909 questions. 

## Documentation and validation

- [SFT](docs/sft.md), [TRACE training](docs/trace-training.md), [Algorithm](docs/trace-opsd-method.md)
- [Evaluation](docs/evaluation.md)
