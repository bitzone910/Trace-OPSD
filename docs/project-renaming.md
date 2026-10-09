# Trace-OPSD Naming and Migration

The project is named **Trace-OPSD**, the Python package is `trace_opsd`, and the
distribution/default Conda environment is `trace-opsd`. Renaming changed engineering
identifiers and historical-format readers. It did not alter the algorithm, prompts,
tool actions, training parameters, dependency versions, or evaluation metrics.

## Local layout

```text
C:/Users/lenovo/Desktop/ToolElite/
├── Trace-OPSD/             # Renamed inner project; published source
├── assets/                 # External assets and runtime directories
├── records/                # Historical experiments, rename backups, and validation
└── .trace-engineering-venv/ # Local engineering environment
```

The outer directory, Hugging Face repository names, models, and data filenames were
not renamed. Pre-rename source backups and file inventories reside in the outer
`records/project-renaming/2026-10-09/` directory.
The GitHub repository is now `bitzone910/Trace-OPSD`.

## Current interfaces

| Component | Name |
|---|---|
| Python package/imports | `trace_opsd` |
| Distribution/Conda | `trace-opsd` |
| Environment variables | `TRACE_OPSD_` prefix |
| W&B project | `Trace-OPSD` |
| Python executor user | `trace-opsd-exec` |
| Tool service | `trace-opsd-tool-server` |
| Index builder | `trace-opsd-general-index` |
| Retrieval service | `trace-opsd-general-server` |
| SFT | `trace-opsd-sft` |
| TRACE | `trace-opsd-train` |
| Evaluation | `trace-opsd-eval` |

The four Bash filenames remain `setup_env.sh`, `run_sft.sh`,
`run_trace_curriculum.sh`, and `run_eval_suite.sh`. Dynamic imports, agent
registration, and environment configuration use the new identifiers. Old Python
packages, commands, and environment variables have no aliases. Legacy project
environment variables cause an error; remove them in a fresh terminal and set
the new variables to avoid resolving the wrong asset directories.

New artifacts use `trace_opsd_grpo_run.json`, `trace_opsd_protocol.json`, and
`trace_opsd_grpo_manifest.json`. New schema/protocol identities use `trace_opsd.*`.

## Reading historical files

`trace_opsd.compat` handles historical filenames, schemas, credit versions, and
metric names. It returns original historical JSON values without rewriting files
or recomputing provenance digests. Conflicting old/new metadata causes an explicit
error; namespace-only differences can represent the same identity.

Historical checkpoints can be verified and exported. Export creates a new directory
and protocol file while preserving the actual historical algorithm version,
credit configuration, source step, and data digest. Historical token-credit models
are not labeled turn-mean models. Legacy-named training runs are not migrated
in place. To reuse historical weights, export a complete HF model and start the
current algorithm under a new run name, without migrating the historical optimizer.

Historical evaluation snapshots/reports retain their original formats, protocol
fingerprints, and scores. Saved eight-sample results support read-only reporting
with their actual scorer version. New generation/rescoring requires the current
code fingerprint; it cannot mix into old directories or force incompatible
historical/current protocols into one comparison.

The following intentionally retain historical names:

| File | Reason |
|---|---|
| `data/evaluation/manifest.json` | Sealed evaluation schema, training-source paths, and digests |
| `configs/assets/manifest.json` | Pre-migration asset paths and recorded file hashes |
| `configs/assets/downloadable-assets.json` | Pinned HF commits and hashes for published assets |
| `trace_opsd/compat.py` | Historical metadata, offline W&B, and metric readers |
| `trace_opsd/assets.py`, `scripts/assets_env.sh` | Detection and rejection of legacy environment variables |
| Two project-customized verl files | Direct reading of historical global-checkpoint/FSDP DCP schemas; usable independently by third-party modules |

Third-party source, licenses, upstream identities, and outer historical archives
are preserved. Both frameworks' project vendor metadata files are named
`TRACE_OPSD_VENDOR.json`, with snapshot digests recalculated from actual source.

## AutoDL

Place the current source in `/root/autodl-tmp/Trace-OPSD/` and prepare a complete
external asset directory. Some recoverable local assets were previously removed
as requested; source upload alone does not restore models, training data, or indices.
See [External assets](assets.md) for pinned recovery instructions.

```bash
cd /root/autodl-tmp/Trace-OPSD
export ASSETS=/root/autodl-tmp/assets
export WORK=/root/autodl-tmp/runs/trace-opsd

bash scripts/setup_env.sh --assets-root "$ASSETS" --work-root "$WORK" --gpu-count 4
conda activate trace-opsd

bash scripts/run_sft.sh train --assets-root "$ASSETS" --work-root "$WORK" \
  --run-name sft-math --model-path "$ASSETS/models/Qwen2.5-3B-Instruct" \
  --dataset-file "$ASSETS/datasets/sft/sft_math_tir.json"

bash scripts/run_trace_curriculum.sh --assets-root "$ASSETS" --work-root "$WORK" \
  --run-name trace-math --model-path "$WORK/exports/sft/sft-math/final" \
  --after-step 0 --mid-per-step 16 --high-per-step 16 --resume none

# Copy and fill in the model list before evaluation.
mkdir -p "$WORK"
cp configs/evaluation/models.example.yaml "$WORK/evaluation-models.yaml"
bash scripts/run_eval_suite.sh --assets-root "$ASSETS" --work-root "$WORK" \
  --models-file "$WORK/evaluation-models.yaml" --protocol all \
  --output-root "$WORK/evaluation/math-comparison"
```

The 16/16 difficulty mix is an example. A complete SFT or historical HF export can
serve as initial TRACE weights. A DCP training directory is not an HF model.
Refer to the four launchers' `--help` for current arguments.

## Rename validation

- Editable installation of the new package, six command help checks, four Bash
  launcher help checks, and all Bash syntax checks passed.
- CPU regression: 684 passed, one skipped for Windows Gloo limitations; 14 new
  rename-compatibility cases passed. Working test copies are external records;
  original archives were not rewritten.
- Real local SFT and historical TRACE models passed HF config/tokenizer/shard
  checks. Each had 434 BF16 tensors; neither was loaded onto a GPU.
- All 1,587 protected files matched their original hashes, including both main
  experiment archives. The fixed evaluation snapshot contains 1,909 questions.
- Key ASTs for advantage shaping, Python error penalties, evidence statistics,
  and tool prompts matched the pre-rename versions.
- Both complete vendor snapshots passed verification. A800 training, inference,
  and actual Linux installation were not run during rename validation. That
  validation did not change HF repositories.

The source was subsequently committed and pushed to GitHub, and the repository
was renamed to [Trace-OPSD](https://github.com/bitzone910/Trace-OPSD).
