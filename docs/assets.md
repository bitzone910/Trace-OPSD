# 外部资产与恢复

默认资产根为仓库同级 `../assets`，可用 `--assets-root` 或 `TRACE_OPSD_ASSET_ROOT`指定。
运行根默认 `<assets>/runs`，可用 `--work-root` 或 `TRACE_OPSD_WORK_ROOT`指定。
输入文件内容保持不变，原始GRPO划分manifest不因迁移改写。

| 相对资产路径 | 来源 |
|---|---|
| `models/Qwen2.5-3B-Instruct` | Qwen官方完整模型，本地准备 |
| `models/e5-base-v2` | intfloat/e5-base-v2；已核对固定revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` |
| `datasets/sft/sft_math_tir.json` | BitZone910/sft_math_tir，30,438条 |
| `datasets/numinamath/rl_trace_opsd.parquet` | BitZone910/rl_trace_opsd，20,161条 |
| `datasets/grpo/{train,validation}.parquet`和`manifest.json` | 同上，19,905训练+256验证 |
| `retrieval/general_search/corpus/search_r1_corpus.jsonl` | BitZone910/wiki_math，Search-R1 corpus格式 |
| `retrieval/general_search/index/e5_Flat.index` | BitZone910/wiki_math 的 `index/e5_Flat.index`，固定revision `157dd03b9245158741b066ee00832759261e663c` |
| `wheels/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl` | 环境锁定artifact，SHA256见硬件配置与清单 |

`configs/assets/manifest.json`记录当前资产路径、大小、SHA256与已验证的HF来源/固定commit。
`local_preserved`表示本地保留，不表示已上传。当前2.8.1 wheel及RL去重记录仍保留在assets；
NuminaMath清洗/质检、早期pilot与旧SFT筛选记录已按用户要求删除。
正式入口只引用当前资源。环境安装不会自动下载大型模型、数据或索引。

2026-10-09按用户要求删除了本机已发布的SFT/RL数据与GRPO划分、E5副本、
corpus、FAISS索引，以及未使用的FlashAttention 2.8.3 wheel。
当前 `manifest.json` 只记录仍在本地的5个资产：RL去重来源记录和2.8.1 wheel。
`configs/assets/downloadable-assets.json` 保留18个可从HF恢复的正式文件的路径、
SHA256和固定commit；上表仍是运行所需输入，运行前需要重新下载或指定完整的外部资产目录。

例如按固定版本恢复已发布训练数据（先安装/登录本地hf CLI，公开下载无需token）：

```bash
hf download BitZone910/sft_math_tir --repo-type dataset \
  --revision 11b98e9d1e442026768931aae1dd7b2a24fedbc5 --local-dir ../assets/datasets/sft
hf download BitZone910/rl_trace_opsd --repo-type dataset \
  --revision dfcaaf807f042f91cabd7f2d70eca7b72b9b0223 --local-dir ../assets/datasets/rl-download
mkdir -p ../assets/datasets/grpo ../assets/datasets/numinamath
cp ../assets/datasets/rl-download/rl_trace_opsd.parquet ../assets/datasets/numinamath/
cp ../assets/datasets/rl-download/grpo/* ../assets/datasets/grpo/
```

按源仓库许可证使用这些资产；Qwen衍生模型许可和Tool-Star来源说明保留在
`docs/hf-assets/licenses/`及相应HF模型/数据卡。E5不是Trace-OPSD自训模型。
FAISS可按[General Search说明](general-search.md)下载已发布的固定版本；也可从匹配的
E5和corpus显式重建。不要替换为其它数学库索引，也不要覆盖现有文件。
