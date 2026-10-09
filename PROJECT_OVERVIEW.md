# Trace-OPSD Project Overview

## Objective

Train a model for multi-turn mathematical reasoning that can invoke General Search
and Python Executor, integrate observations, and produce verifiable answers.
The public workflow is **SFT → TRACE-OPSD → evaluation**.

## Training workflow

1. **SFT:** Read prepared `sft_math_tir` Alpaca records and supervise the complete
   `output`. Use all 30,438 records, a 15,000-token cutoff, four A800 GPUs, BF16,
   and an effective batch size of 16.
2. **TRACE-OPSD:** Start from a complete SFT HF model and compute GRPO advantages
   from final-answer rewards. The current policy scores identical tokens under
   causal and solution-hindsight views. Average the raw log-probability gap within
   each turn, compute one shared weight, and scale the advantages of all valid
   generated tokens in that turn. The objective remains token-wise; tool
   observations are excluded from the loss. Trusted Python error penalties follow
   this scaling.
3. **Evaluation:** Use an explicit model list and fixed mathematical benchmarks
   for greedy and/or eight-sample evaluation. Report Pass@1, Pass@8, Avg@8,
   Majority@8, and tool diagnostics.

The algorithm identity is `current_policy_solution_hindsight.turn_mean.v4`.
There is no training switch for the previous token-level credit mechanism.
Historical runs/checkpoints retain their original identities for inspection and
export; their metrics retain their historical interpretation. The historical
step400 export and evaluation are not experimental evidence for turn-mean v4.

## Runtime conventions

Training defaults to four A800 GPUs; evaluation uses one. Retrieval and Python run
on the CPU. Only the A800 environment profile is maintained. Models, training data,
indices, wheels, and runtime outputs reside in external asset/work directories.
Small fixed evaluation snapshots and sample inputs remain in source. See the
README and launcher documentation for the four public entry points.

TRACE performs baseline validation before training, then runs stages of at most
50 steps. Each stage exits after checkpointing, validation, and metric packaging;
the final stage ends at 622. Difficulty quotas are explicit. Dynamic sampling
preserves actual sampling cursors. Retention keeps the two latest complete training
checkpoints and the best checkpoint.

## Implementation status and validation scope

General Search, isolated Python execution, SFT training/export, TRACE
training/recovery/metrics, and greedy/multi-sample evaluation are implemented.
Repository cleanup moved historical teacher-data generation, difficulty filtering,
standalone GRPO experiments, step-specific repair scripts, update packaging, and
standalone smoke launchers out of public source. Shared dependencies and necessary
CPU regression records were retained.

Complete historical training and evaluation evidence is stored in the outer
`records` directory. Validation of the current code is recorded separately. The
full four-GPU training and single-GPU evaluation workflow after cleanup still
requires real A800 validation; CPU checks cannot substitute for it.

The project is named Trace-OPSD; the package, commands, and environment use
`trace_opsd` / `trace-opsd`. Rename validation passed 684 CPU regression cases and
14 historical-format compatibility cases, with one Windows Gloo case skipped.
Help checks for six commands and four Bash launchers, read-only checks of real
historical BF16 models, and vendor verification passed. Historical asset digests
were preserved. Legacy-named runs support read-only inspection/export. See
[Naming and migration](docs/project-renaming.md) for details.
