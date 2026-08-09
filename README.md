# Search-R1 独立本地 Dense 检索组件

这个项目只保留 Search-R1 的本地 `e5 + FAISS` 检索链路：输入一条或多条 query，从自己的 JSONL 知识库返回 top-k 条 `{id, contents, score}` 记录。

它不包含 veRL、PPO/GRPO、policy/value/reward model、vLLM rollout、训练 prompt、在线搜索、BM25/Pyserini 或 wiki-18 下载代码。运行时只会加载用于检索的 E5 embedding 模型。

## 核心行为

实现保持 Search-R1 的 dense 检索逻辑：

1. 文档编码前添加 `passage: `，query 编码前添加 `query: `。
2. 对最后一层 hidden state 做 attention-mask mean pooling。
3. 对向量做 L2 归一化。
4. 使用 FAISS inner product 搜索，因此归一化后等价于 cosine similarity 排序。
5. FAISS 向量编号与 JSONL 行号严格对应；检索后按原行取回 `id` 和 `contents`。

## 项目结构

```text
.
├── README.md
├── environment.yml          # 推荐的一键 Conda 环境
├── retriever_requirements.txt
├── retriever.py
├── build_index.py
├── retrieval_server.py
├── test_query.py
├── search_r1_corpus.jsonl    # 你的真实语料（本地已有，不纳入版本控制）
├── corpus/
│   └── my_corpus.jsonl       # 小样本
└── index/                    # 生成的 FAISS 索引
```

## 数据格式

语料必须是 UTF-8 JSON Lines 文件，每行一个 JSON 对象，至少包含：

```json
{"id": "doc-001", "contents": "Document title\nDocument body"}
```

- `id` 不能是 `null`，其原始 JSON 类型会保留在结果中。
- `contents` 必须是字符串。
- 建索引后不能调整、插入、删除或重排 JSONL 行；任何语料变化都必须重新建索引。

当前目录下的 `search_r1_corpus.jsonl` 可以直接作为完整语料使用；`corpus/my_corpus.jsonl` 仅用于小规模验证。

## AutoDL 环境安装

目标基线：AutoDL Ubuntu 22.04、NVIDIA A800、Python 3.12、PyTorch 2.5.1、CUDA 12.4、`faiss-gpu-cuvs=1.12.0`。推荐使用项目提供的 Conda YAML 一键创建隔离环境：

```bash
conda env create -f environment.yml
conda activate retriever
```

如果已经存在同名环境，请不要直接混装或更新。确认旧环境不再需要后先删除并重建：

```bash
conda env remove -n retriever
conda env create -f environment.yml
conda activate retriever
```

创建完成后验证环境：

```bash
python - <<'PY'
import numpy as np
import torch
import faiss
import transformers
import datasets
import fastapi
import uvicorn

print("Python environment: OK")
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

assert torch.cuda.is_available()
assert faiss.get_num_gpus() > 0
cpu_index = faiss.IndexFlatIP(2)
index = faiss.index_cpu_to_all_gpus(cpu_index)
index.add(np.asarray([[1.0, 0.0]], dtype=np.float32))
scores, ids = index.search(np.asarray([[1.0, 0.0]], dtype=np.float32), 1)
assert ids.tolist() == [[0]]
print("FAISS:", faiss.__version__)
print("FAISS GPUs:", faiss.get_num_gpus())
print("FAISS GPU smoke test: OK", scores.tolist())
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("FastAPI:", fastapi.__version__)
print("Uvicorn:", uvicorn.__version__)
PY
```

`environment.yml` 固定使用 `pytorch=2.5.1`、`pytorch-cuda=12.4`、`cuda-version=12.4` 和 `faiss-gpu-cuvs=1.12.0`。E5 embedding 会在 `--device auto --use_fp16` 下使用 CUDA，Flat FAISS 建索引与查询也默认使用全部可见 GPU。A800 支持该 CUDA 12.4 运行栈。

YAML 不安装 Linux NVIDIA 驱动。云服务器必须预先具备兼容 CUDA 12.4 的宿主机驱动。`retriever_requirements.txt` 只安装 Python 层依赖；FAISS 必须通过 Conda 安装，不能用该文件中的 pip 命令替代。

不要在该环境中另外安装 `faiss-cpu`，它会与 `faiss-gpu-cuvs` 冲突。需要手动安装时使用：

```bash
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge \
  libnvjitlink faiss-gpu-cuvs=1.12.0 cuda-version=12.4
```


## 构建索引

### 先用小样本验证

```bash
python build_index.py \
  --corpus_path corpus/my_corpus.jsonl \
  --save_dir index \
  --faiss_type Flat \
  --model_path intfloat/e5-base-v2 \
  --device auto \
  --use_fp16
```

输出文件为 `index/e5_Flat.index`。Flat 索引默认在 GPU 上构建，保存前会转换为可移植的 CPU 索引文件。如果需要临时进行纯 CPU 调试，去掉 `--use_fp16`，使用 `--device cpu --no-faiss_gpu`。

### 使用已有完整语料

```bash
python build_index.py \
  --corpus_path search_r1_corpus.jsonl \
  --save_dir index \
  --faiss_type Flat \
  --model_path intfloat/e5-base-v2 \
  --device auto \
  --use_fp16 \
  --batch_size 128
```

已有同名索引时脚本会拒绝覆盖。确认要基于当前语料重建后，增加 `--overwrite`。

`build_index.py` 与原 Search-R1 一样，会先生成全部文档 embedding，再建立索引。完整语料需要足够的主机内存和磁盘空间；显存不足时调小 `--batch_size`。

### HNSW64 近似索引

```bash
python build_index.py \
  --corpus_path search_r1_corpus.jsonl \
  --save_dir index \
  --faiss_type HNSW64 \
  --device auto \
  --use_fp16 \
  --no-faiss_gpu
```

输出为 `index/e5_HNSW64.index`。HNSW64 使用 CPU 构建，必须传入 `--no-faiss_gpu`；当前面向 A800 的默认运行方式仍是 Flat GPU 索引。

## 启动服务

小样本 Flat 索引示例：

```bash
python retrieval_server.py \
  --index_path index/e5_Flat.index \
  --corpus_path corpus/my_corpus.jsonl \
  --retriever_model intfloat/e5-base-v2 \
  --device auto \
  --use_fp16 \
  --topk 3 \
  --host 127.0.0.1 \
  --port 8000
```

完整语料只需将 `--corpus_path` 改为 `search_r1_corpus.jsonl`，并传入由该文件生成的索引。服务默认将 Flat 索引载入全部可见 GPU；使用 `CUDA_VISIBLE_DEVICES` 可以限制所用卡。建索引和启动服务必须使用同一个 E5 模型，否则脚本会在向量维度不一致时拒绝启动；即使维度相同，换模型也会使检索结果无效。

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

响应中的 `embedding_device` 应为 `cuda`，`faiss_version` 应为 `1.12.0`，`faiss_device` 应为 `gpu`，`faiss_gpus` 是 FAISS 当前可见的 GPU 数量。

## 发起查询

### curl

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["What is mathematics?"],"topk":3,"return_scores":true}'
```

响应结构：

```json
{
  "result": [
    [
      {
        "id": "math-001",
        "contents": "\"Mathematics\"\nMathematics studies ...",
        "score": 0.87
      }
    ]
  ]
}
```

`queries` 可以包含多条 query。省略 `topk` 时使用服务启动参数的默认值；设置 `return_scores: false` 会从结果中移除 `score`。

### 测试脚本

```bash
python test_query.py --query "What is mathematics?" --topk 3
```

### Python 直接调用

```python
from retriever import DenseRetriever

retriever = DenseRetriever(
    index_path="index/e5_Flat.index",
    corpus_path="corpus/my_corpus.jsonl",
    model_path="intfloat/e5-base-v2",
    device="auto",
    use_fp16=True,
)

results = retriever.search("What is mathematics?", topk=3)
print(results)
```

CPU 调试时设置 `device="cpu", use_fp16=False, faiss_gpu=False`；A800 部署无需传 `faiss_gpu`，默认值就是 `True`。

## 常见错误

- **Index/corpus size mismatch**：索引不是由当前 JSONL 文件构建，或语料行数已变化。重新建索引。
- **Model/index dimension mismatch**：服务使用了不同维度的 embedding 模型。使用建索引时的模型。
- **CUDA requested but unavailable**：确认 A800 已分配给实例、驱动可用且 `CUDA_VISIBLE_DEVICES` 未屏蔽设备；仅做 CPU 调试时使用 `--device cpu --no-faiss_gpu` 并移除 `--use_fp16`。
- **GPU FAISS APIs are unavailable**：环境中装错了 `faiss-cpu`，删除并按上面的 Conda 命令安装 `faiss-gpu-cuvs=1.12.0`。
- **显存不足**：调小 `--batch_size` 或用 `CUDA_VISIBLE_DEVICES` 调整可见卡；必要时使用 `--no-faiss_gpu` 让 FAISS 查询回退到 CPU。

