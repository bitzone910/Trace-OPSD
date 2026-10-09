# External Assets and Recovery

The default asset root is the sibling directory `../assets`. Override it with
`--assets-root` or `TRACE_OPSD_ASSET_ROOT`. Runtime outputs default to
`<assets>/runs`; override them with `--work-root` or `TRACE_OPSD_WORK_ROOT`.
Input contents are preserved, including the original GRPO split manifest.

| Relative asset path | Source |
|---|---|
| `models/Qwen2.5-3B-Instruct` | Complete official Qwen model, prepared locally |
| `models/e5-base-v2` | intfloat/e5-base-v2; verified revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` |
| `datasets/sft/sft_math_tir.json` | BitZone910/sft_math_tir; 30,438 records |
| `datasets/numinamath/rl_trace_opsd.parquet` | BitZone910/rl_trace_opsd; 20,161 records |
| `datasets/grpo/{train,validation}.parquet` and `manifest.json` | Same repository; 19,905 training and 256 validation records |
| `retrieval/general_search/corpus/search_r1_corpus.jsonl` | BitZone910/wiki_math; Search-R1 corpus format |
| `retrieval/general_search/index/e5_Flat.index` | `index/e5_Flat.index` in BitZone910/wiki_math; pinned revision `157dd03b9245158741b066ee00832759261e663c` |
| `wheels/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl` | Pinned environment artifact; SHA256 in the hardware profile and manifest |

`configs/assets/manifest.json` records asset paths, sizes, SHA256 hashes, and
verified HF sources/pinned commits. `local_preserved` means retained locally, not
uploaded. The 2.8.1 wheel and RL overlap provenance records remain in assets.
NuminaMath cleaning/quality-control records, early pilot trajectories, and old SFT
selection records were deleted at the user's request. Public launchers reference
only current resources. Setup does not download large models, data, or indices.

On 2026-10-09, published local SFT/RL data and GRPO splits, the E5 copy, corpus,
FAISS index, and unused FlashAttention 2.8.3 wheel were removed as requested.
The current `manifest.json` describes the five remaining local assets: RL overlap
provenance records and the 2.8.1 wheel. `configs/assets/downloadable-assets.json`
retains paths, SHA256 hashes, and pinned commits for 18 recoverable HF files.
Restore the required inputs above, or provide a complete external asset directory,
before running the workflow.

For example, restore the published training data at pinned revisions using the
local HF CLI. Public downloads do not require a token.

```bash
hf download BitZone910/sft_math_tir --repo-type dataset \
  --revision 11b98e9d1e442026768931aae1dd7b2a24fedbc5 --local-dir ../assets/datasets/sft
hf download BitZone910/rl_trace_opsd --repo-type dataset \
  --revision dfcaaf807f042f91cabd7f2d70eca7b72b9b0223 --local-dir ../assets/datasets/rl-download
mkdir -p ../assets/datasets/grpo ../assets/datasets/numinamath
cp ../assets/datasets/rl-download/rl_trace_opsd.parquet ../assets/datasets/numinamath/
cp ../assets/datasets/rl-download/grpo/* ../assets/datasets/grpo/
```

Follow each source repository's license. Qwen-derived model licenses and Tool-Star
attribution are retained in `docs/hf-assets/licenses/` and the corresponding HF
model/dataset cards. E5 is not a model trained by Trace-OPSD. Download the pinned
FAISS index following [General Search](general-search.md), or explicitly rebuild
it from the matching E5 model and corpus. Do not substitute another mathematical
database's index or overwrite existing files.
