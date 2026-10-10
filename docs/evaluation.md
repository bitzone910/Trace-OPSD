# Mathematical Evaluation

The public launcher `scripts/run_eval_suite.sh` loads complete Hugging Face model
directories sequentially from an explicit model list. Export training DCP
checkpoints at the desired step through the TRACE export entry point first.

## Model list and external resources

Copy `configs/evaluation/models.example.yaml` into the work directory. Fill in
safe, unique names and model paths, and remove all placeholders. `baseline` is
optional; when specified, it must appear in the list and is used for percentage-point
changes in reports.

```yaml
schema: trace_opsd.evaluation.models.v1
baseline: sft
models:
  - name: sft
    path: models/SFT-Qwen2.5-3B
  - name: trace
    path: runs/exports/my-trace-model
```

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
the data digest and preserves exact training-overlap flags. 

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
