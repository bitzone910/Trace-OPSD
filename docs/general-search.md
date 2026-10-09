# General Search

General Search使用Search-R1格式corpus：每行包含 `id`、`contents`；E5对query与
contents编码，通过FAISS inner-product检索。默认CPU检索，服务用于TRACE及evaluation。
保持corpus、E5模型与FAISS索引的行顺序和版本一致。

完整索引已发布在 `BitZone910/wiki_math`，共573,011个768维向量，类型IndexFlatIP。
SHA256为 `09bf053b50922ad283620b4fdf84b4ab37e824a7cfd7cfb61e310c8452d7afb0`。
下载固定版本索引及对应corpus（在仓库根目录执行）：

```bash
hf download BitZone910/wiki_math index/e5_Flat.index index/index-metadata.json \
  --repo-type dataset --revision 157dd03b9245158741b066ee00832759261e663c \
  --local-dir ../assets/retrieval/general_search
hf download BitZone910/wiki_math search_r1_corpus.jsonl \
  --repo-type dataset --revision 157dd03b9245158741b066ee00832759261e663c \
  --local-dir ../assets/retrieval/general_search/corpus
```

编码器固定为 `intfloat/e5-base-v2` 的 `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`。
核验文件哈希和corpus行顺序后再启动检索；具体元数据在下载的 `index/index-metadata.json`。
上传的是已有索引，未重新生成embedding；历史构建的max_length和device没有独立记录。

需要重新构建时显式执行，不覆盖已有索引：

```bash
python -m trace_opsd.retrieval.general.build_index \
  --corpus_path ../assets/retrieval/general_search/corpus/search_r1_corpus.jsonl \
  --save_dir ../assets/retrieval/general_search/index \
  --model_path ../assets/models/e5-base-v2 \
  --faiss_type Flat --device cpu --no-faiss_gpu
```

完整corpus与模型已移出源码。小型示例保留在
`data/general_search/corpus/sample_corpus.jsonl`，可以用于接口检查。
发布时已核验索引与本地内容一致。2026-10-09按用户要求删除本机索引、corpus和
E5副本；本机运行检索前需按固定版本恢复这些输入。
