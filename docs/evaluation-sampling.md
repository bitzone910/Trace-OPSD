# Multi-Sample Mathematical Evaluation

Use the unified [evaluation launcher](evaluation.md) with an explicit model list
and `--protocol sampling8` (or `all`). The historical ten-model list and standalone
sampling experiment launcher have been moved out of public source.

Defaults are eight independent trajectories per question, temperature 0.7,
top_p=0.95, top_k=-1, and base seed 42. Each question/sample slot receives a stable,
independent seed, shared across models. Results reside in
`full/sampling8/<model>/`, separately from greedy results.

| Metric | Definition |
|---|---|
| Pass@8 | Fraction of questions with at least one correct trajectory among eight |
| Avg@8 | Mean across questions of the number of correct trajectories divided by eight |
| Majority@8 | Accuracy of the most-voted answer after clustering mathematically equivalent predictions |

Avg@8 is single-sample accuracy under the sampling protocol, distinct from greedy
Pass@1. The same trajectories provide diagnostic Pass@1/2/4 curves using the standard
estimator for sampling without replacement; no additional inference is needed.
Voting reads predictions only, never ground truth or correctness labels. Ties go
to the earliest sample. Missing answers abstain; all-abstention questions are wrong.

```bash
bash scripts/run_eval_suite.sh \
  --models-file /path/to/models.yaml \
  --protocol sampling8 \
  --name my-sampling8-comparison \
  --gpu 0 \
  --samples-per-question 8 \
  --temperature 0.7 --top-p 0.95 --top-k -1 \
  --seed 42 --pass-ks 1,2,4,8
```

Changed parameters require a new result directory. Each model produces
1,909 × 8 = 15,272 formal trajectories; the model count is user-defined.
First use a new name with `--limit-per-benchmark 2` to check two questions per
benchmark per model, with eight samples each. Use a separate name for the full run.

Per-sample predictions, per-question vote groups, seeds, tool usage and structured
errors, generation failures, truncation, and missing answers are retained. Failures
stay in the fixed denominator. Scoring or voting exceptions mark results unavailable
and block formal comparison. Only complete results with matching protocols enter
formal JSON/CSV/Markdown tables. Completeness checks reject duplicate, missing, or
unknown questions/sample slots.

The lower-level `python -m trace_opsd.evaluation sample`, `sample-score`, and
`sample-report` commands remain available for running services or offline analysis
of saved trajectories. Voting and formal scoring run in the main thread. The unified
launcher automatically creates the complete archive.
