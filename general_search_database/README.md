# General Search Database

General Search Database 使用 Search-R1 风格的单阶段 dense retrieval：query 与每条
`contents` 的 E5 向量直接进行 FAISS inner-product 检索。

从项目根目录构建完整索引：

```bash
python -m general_search_database.build_index \
  --corpus_path general_search_database/corpus/search_r1_corpus.jsonl \
  --save_dir general_search_database/index \
  --faiss_type Flat \
  --device auto \
  --use_fp16
```

启动服务：

```bash
python -m general_search_database.retrieval_server \
  --index_path general_search_database/index/e5_Flat.index \
  --corpus_path general_search_database/corpus/search_r1_corpus.jsonl \
  --device auto \
  --use_fp16 \
  --port 8000
```

CPU 小语料验证时使用 `corpus/sample_corpus.jsonl`，并传入
`--device cpu --no-faiss_gpu`。请求的 topk 大于语料数量时会返回全部记录。
