# Theorem Search Database

Theorem Search Database 对 `math_corpus.jsonl` 执行两阶段检索：

1. query 检索 top20 个唯一 slogan；
2. 展开这些 slogan 对应的所有 contents，再按 query–contents 相似度返回 topk。

完全相同的 slogan 只占一个第一阶段索引条目。最终 `score` 只表示第二阶段的
contents 相似度。

从项目根目录构建完整 bundle：

```bash
python -m theorem_search_database.build_index \
  --corpus_path theorem_search_database/corpus/math_corpus.jsonl \
  --save_dir theorem_search_database/index \
  --faiss_type Flat \
  --device auto \
  --use_fp16
```

输出目录为 `theorem_search_database/index/e5_Flat/`，其中包含 slogan FAISS
索引、content embedding mmap 文件、分组映射和 manifest。

启动服务：

```bash
python -m theorem_search_database.retrieval_server \
  --index_dir theorem_search_database/index/e5_Flat \
  --corpus_path theorem_search_database/corpus/math_corpus.jsonl \
  --device auto \
  --use_fp16 \
  --slogan_topk 20 \
  --port 8001
```

CPU 小语料验证时改用 `corpus/sample_corpus.jsonl`，并传入
`--device cpu --no-faiss_gpu`。如果候选 contents 少于请求的 topk，服务返回全部候选。
