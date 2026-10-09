# Mathematical Evaluation

The public launcher `scripts/run_eval_suite.sh` loads complete Hugging Face model
directories sequentially from an explicit model list. Export training DCP
checkpoints at the desired step through the TRACE export entry point first.
Evaluation does not search for `latest` or substitute another model.

## Model list and external resources

Copy `configs/evaluation/models.example.yaml` into the work directory. Fill in
safe, unique names and model paths, and remove all placeholders. `baseline` is
optional; when specified, it must appear in the list and is used for percentage-point
changes in reports. Relative model paths resolve from `--assets-root`; absolute
paths are also supported.

```yaml
schema: trace_opsd.evaluation.models.v1
baseline: sft
models:
  - name: sft
    path: models/SFT-Qwen2.5-3B
  - name: trace
    path: runs/exports/my-trace-model
```

This example illustrates the format; it is not a default evaluation list. Each
directory must contain complete weights, config, and a locally loadable tokenizer.
Preflight checks every shard, actual tensor dtypes, file SHA256 hashes, and optional
source step/run metadata. BF16 and FP32 weights on disk are supported; inference
always uses BF16. FP16, quantized, and incomplete models are rejected.

External resources default to the sibling `../assets` directory. Override it with
`--assets-root` or `TRACE_OPSD_ASSET_ROOT`:

- `models/e5-base-v2/`
- `retrieval/general_search/index/e5_Flat.index`
- `retrieval/general_search/corpus/search_r1_corpus.jsonl`

Output defaults to `assets/runs/evaluation/<name>`. Set a work root with
`--work-root` or `TRACE_OPSD_WORK_ROOT`, or pass `--output-root` explicitly.
Results must reside outside the source repository.

## Fixed snapshot and protocols

The `data/evaluation` snapshot contains 30 questions each from AIME24/25/26,
500 from MATH500, and 1,319 from GSM8K: 1,909 questions per model. Every run verifies
the data digest and preserves exact training-overlap flags. Existing results do
not trigger data downloads or changes to the sealed snapshot.

The common setup uses one A800, TP=1, BF16, CPU Python and General Search, and
concurrency 16. Budgets are context 8192, total response 4096, generation per turn
1024, and tool observation 512. At most five tool calls are allowed; two consecutive
invalid actions terminate the trajectory. The launcher reuses or configures a
non-root Python executor user.

Explicitly select `--protocol greedy|sampling8|all`. Greedy evaluation uses
temperature 0, top_p=1, top_k=-1, and seed 42 to report greedy Pass@1. See
[Eight-sample evaluation](evaluation-sampling.md) for sampling settings and metrics.

```bash
bash scripts/run_eval_suite.sh \
  --models-file /path/to/models.yaml \
  --protocol all \
  --name my-math-comparison \
  --gpu 0
```

Append `--preflight-only` to check assets, models, data, and dependencies. For a
small process check, use a new name and append `--limit-per-benchmark 2`. Such a run
is excluded from formal comparison tables and archived as incomplete diagnostics;
it has no accuracy threshold.

Manifests record models, data, budgets, sampling settings, and the actual execution
backend. Complete trajectories are saved atomically per question. Restarting the
same command fills only missing entries; changed fingerprints require a new output
directory. Offline scoring runs in the main thread; scoring exceptions are recorded
and block formal comparisons. Generation failures, truncation, and missing answers
remain in the fixed denominator.

Each model has separate trajectories, predictions, metrics, and tool-error provenance.
Formal results include JSON/CSV/Markdown comparison tables for each protocol and
`<name>-analysis.tar.gz`, containing manifests, trajectories, and logs without model
weights. Partial, failed, or interrupted runs produce
`<name>-analysis.incomplete.tar.gz`; `batch_status.json` records completion status.
