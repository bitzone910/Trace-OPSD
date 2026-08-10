# ToolElite 双检索数据库

ToolElite 提供两个相互独立、共享同一套 E5 模型与运行环境的本地 dense retrieval
数据库：

- **General Search Database**：query 直接检索 Search-R1 corpus 的 `contents`。
- **Theorem Search Database**：query 先检索 top20 个唯一 `slogan`，展开对应的全部
  `contents` 后再进行第二次相似度排序。

两者都使用 E5 query/passage 前缀、attention-mask mean pooling、L2 归一化和
FAISS inner product。服务返回的结构统一为 `{id, contents, score}`。

## 目录结构

```text
ToolElite/
├── shared/                         # 公共 E5、FAISS、校验与配置
├── general_search_database/
│   ├── build_index.py
│   ├── retriever.py
│   ├── retrieval_server.py
│   ├── test_query.py
│   ├── corpus/
│   │   ├── sample_corpus.jsonl
│   │   └── search_r1_corpus.jsonl  # 完整语料，不纳入 Git
│   └── index/                      # 生成的 general FAISS 索引
├── theorem_search_database/
│   ├── build_index.py
│   ├── retriever.py
│   ├── retrieval_server.py
│   ├── test_query.py
│   ├── corpus/
│   │   ├── sample_corpus.jsonl
│   │   └── math_corpus.jsonl       # 完整语料，不纳入 Git
│   └── index/                      # 生成的 theorem index bundle
├── models/e5-base-v2/             # 两个数据库共同使用的默认模型
├── tests/
├── environment.yml
└── retriever_requirements.txt
```

所有 Python 入口都应从项目根目录通过 `python -m ...` 执行。旧的根目录脚本和
旧 import 路径不再保留。

## 环境

推荐环境为 AutoDL Ubuntu 22.04、Python 3.12、PyTorch 2.5.1、CUDA 12.4 和
`faiss-gpu-cuvs=1.12.0`：

```bash
conda env create -f environment.yml
conda activate retriever
```

`environment.yml` 同时服务两个数据库。默认模型为项目内的
`models/e5-base-v2`；各入口仍可用 `--model_path` 或 `--retriever_model` 覆盖。
不要额外安装 `faiss-cpu`，它会与 Conda 中的 GPU FAISS 冲突。

## General Search Database

构建索引：

```bash
python -m general_search_database.build_index \
  --corpus_path general_search_database/corpus/search_r1_corpus.jsonl \
  --save_dir general_search_database/index \
  --faiss_type Flat \
  --device auto \
  --use_fp16 \
  --batch_size 128
```

启动服务：

```bash
python -m general_search_database.retrieval_server \
  --index_path general_search_database/index/e5_Flat.index \
  --corpus_path general_search_database/corpus/search_r1_corpus.jsonl \
  --device auto \
  --use_fp16 \
  --topk 3 \
  --host 127.0.0.1 \
  --port 8000
```

## Theorem Search Database

`math_corpus.jsonl` 每行格式为：

```json
{"id":"0","contents":"theorem text","slogan":"semantic summary"}
```

构建索引 bundle：

```bash
python -m theorem_search_database.build_index \
  --corpus_path theorem_search_database/corpus/math_corpus.jsonl \
  --save_dir theorem_search_database/index \
  --faiss_type Flat \
  --device auto \
  --use_fp16 \
  --batch_size 128
```

生成的 `theorem_search_database/index/e5_Flat/` 包含：

- `slogan.index`：每个唯一 slogan 一个向量；
- `content_embeddings.npy`：按 corpus 行号对齐的 float32 contents 向量；
- `slogan_offsets.npy` 与 `content_row_ids.npy`：slogan 到所有 contents 的映射；
- `manifest.json`：行数、维度、索引类型和模型信息。

启动服务：

```bash
python -m theorem_search_database.retrieval_server \
  --index_dir theorem_search_database/index/e5_Flat \
  --corpus_path theorem_search_database/corpus/math_corpus.jsonl \
  --device auto \
  --use_fp16 \
  --slogan_topk 20 \
  --topk 3 \
  --host 127.0.0.1 \
  --port 8001
```

`--slogan_topk` 默认是 20，只能在服务启动时设置；单次 HTTP 请求不能修改。
第二阶段只对展开后的候选 contents 排名。当候选数量小于 topk 时返回全部候选。

## API

两个服务都提供：

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["What is mathematics?"],"topk":3,"return_scores":true}'
```

Theorem 服务将端口改为 8001。响应结构一致：

```json
{
  "result": [
    [
      {"id": "doc-id", "contents": "document text", "score": 0.87}
    ]
  ]
}
```

`queries` 支持批量请求。设置 `return_scores: false` 会移除 `score`。General 的
topk 大于 corpus 行数时返回全部文档；Theorem 的 topk 大于第二阶段候选数时返回
全部候选。

## CPU 小语料验证与测试

将构建命令的 corpus 改为对应目录下的 `sample_corpus.jsonl`，并使用
`--device cpu --no-faiss_gpu`（不要传 `--use_fp16`）。运行依赖较轻的核心测试：

```bash
pytest -q
```

安装完整环境后，可以额外运行真实本地模型与 CPU FAISS 的小语料集成测试：

```bash
RUN_RETRIEVAL_INTEGRATION=1 pytest -q -m integration
```

完整 corpus 会生成数 GB 的索引/embedding 文件，需要足够的磁盘、内存和 GPU
资源。HNSW64 仍受支持，但必须通过 `--faiss_type HNSW64 --no-faiss_gpu` 在 CPU
上构建和查询。
