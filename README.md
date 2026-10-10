# Trace-OPSD

A post-training framework for multi-turn mathematical reasoning: **SFT → TRACE-OPSD → evaluation**.

![TRACE-OPSD: turn-level credit assignment for multi-turn mathematical reasoning](docs/images/trace-opsd-overview.png)


## Evaluation results

![Mathematical reasoning results for TRACE-OPSD step400, SFT, and Qwen2.5-3B-Instruct, grouped by benchmark and metric](docs/images/evaluation-comparison.png)

All scores are percentages. **Pass@1** uses greedy decoding; **Pass@8** measures whether at least one of eight sampled trajectories is correct; **Avg@8** measures their average accuracy. The sampling protocol uses temperature 0.7 and top-p 0.95. All three models use BF16 inference and the same tool and token budgets.

[Download the table values (CSV)](docs/evaluation-results.csv). The AIME26 snapshot includes one recorded exact overlap with the training source.

## Algorithm pseudocode

![Algorithm 1: TRACE-OPSD training with turn-mean credit assignment in 22 numbered steps](docs/images/trace-opsd-algorithm.png)

[Read the method description](docs/trace-opsd-method.md). The diagram describes the current turn-mean implementation; the evaluation table uses the historical step400 checkpoint.

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
## Documentation 

- [SFT](docs/sft.md),[Algorithm](docs/trace-opsd-method.md)
- [Evaluation](docs/evaluation.md)
