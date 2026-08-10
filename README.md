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
├── tools/                          # 三工具函数、沙箱与统一 HTTP 服务
├── general_search_database/
│   ├── build_index.py
│   ├── retriever.py
│   ├── retrieval_server.py
│   ├── corpus/
│   │   ├── sample_corpus.jsonl
│   │   └── search_r1_corpus.jsonl  # 完整语料，不纳入 Git
│   └── index/                      # 生成的 general FAISS 索引
├── theorem_search_database/
│   ├── build_index.py
│   ├── retriever.py
│   ├── retrieval_server.py
│   ├── corpus/
│   │   ├── sample_corpus.jsonl
│   │   └── math_corpus.jsonl       # 完整语料，不纳入 Git
│   └── index/                      # 生成的 theorem index bundle
├── models/e5-base-v2/             # 两个数据库共同使用的默认模型
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

## CPU 小语料试运行

将构建命令的 corpus 改为对应目录下的 `sample_corpus.jsonl`，并使用
`--device cpu --no-faiss_gpu`（不要传 `--use_fp16`），即可用少量数据完成索引构建与
查询流程。

完整 corpus 会生成数 GB 的索引/embedding 文件，需要足够的磁盘、内存和 GPU
资源。HNSW64 仍受支持，但必须通过 `--faiss_type HNSW64 --no-faiss_gpu` 在 CPU
上构建和查询。

## 三工具后端

`tools` 包在现有两个检索数据库之上提供稳定的 rollout 工具协议，并新增隔离子进程
执行数学 Python 代码的能力。工具调用结果统一为：

```text
<result>
工具返回内容
</result>
```

支持的调用标签为 `<python>`、`<search_knowledge>` 和 `<search_theory>`。同一段生成
文本中出现多个同名标签时取最后一个完整标签。Search 默认返回 top 3，格式为
`[序号] 文档ID` 加正文，不向模型暴露相似度分数。

### 进程内调用

在 rollout worker 中初始化一次，然后复用返回的 `ToolRuntime`：

```python
from tools import ToolConfig, initialize_tools

runtime = initialize_tools(
    ToolConfig(
        general_index_path="general_search_database/index/e5_Flat.index",
        general_corpus_path=(
            "general_search_database/corpus/search_r1_corpus.jsonl"
        ),
        theorem_index_dir="theorem_search_database/index/e5_Flat",
        theorem_corpus_path="theorem_search_database/corpus/math_corpus.jsonl",
        device="cuda",
        use_fp16=True,
        faiss_gpu=True,
    )
)

observation = runtime.call_tool(
    "python",
    "<python>print(sum(range(10)))</python>",
)
```

也可以在完成 `initialize_tools()` 后直接导入 `call_tool()`、
`python_executor_tool()`、`search_knowledge_tool()` 和 `search_theory_tool()`。
初始化前调用这些模块级函数不会抛异常，而会返回带错误信息的 `<result>`。

### 统一 HTTP 服务

统一服务只加载一份 E5 模型，并由两个 Search Retriever 共享。现有两个数据库的
`/retrieve` 服务和启动方式保持不变。

```bash
python -m tools.service \
  --general_index_path general_search_database/index/e5_Flat.index \
  --general_corpus_path general_search_database/corpus/search_r1_corpus.jsonl \
  --theorem_index_dir theorem_search_database/index/e5_Flat \
  --theorem_corpus_path theorem_search_database/corpus/math_corpus.jsonl \
  --device auto \
  --use_fp16 \
  --faiss_gpu \
  --host 127.0.0.1 \
  --port 8002
```

三个 POST 路由分别为 `/tools/python`、`/tools/search_knowledge` 和
`/tools/search_theory`，请求及响应示例：

```json
{"raw_text":"<search_knowledge>What is a group?</search_knowledge>"}
```

```json
{"result":"<result>\n[1] document-id\nretrieved content\n</result>"}
```

格式正确的请求即使工具执行失败也返回 HTTP 200，并把错误放在 `result` 中；非法
JSON、缺少 `raw_text` 或字段类型错误仍返回 HTTP 422。`GET /health` 报告两个索引、
共享模型设备和 Python Executor 的隔离能力。

### Python Executor 边界

Executor 默认限制墙钟和 CPU 时间为 10 秒、虚拟内存为 512 MiB、原始输出为
64 KiB、并发子进程为 8，并将最终 result 正文截断到 2000 字符。每次调用使用独立
临时工作目录；POSIX 超时会清理整个进程组。可以通过 `ToolConfig` 或服务 CLI 参数
调整这些限制。

Linux 服务以非 root 用户运行时，子进程继承该权限。如果服务本身以 root 启动，
必须同时传入 `--executor_uid` 和 `--executor_gid`，否则 Python 工具拒绝执行。Windows
只提供独立进程、墙钟超时、临时目录、输出限制和 AST 危险操作检查，不提供 POSIX
资源限制或同等级进程组清理。

AutoDL 等以 root 运行、且 Conda 环境位于 `/root` 下的部署还需要让降权后的子进程
能够读取该环境，否则 NumPy 等包会表现为 `ModuleNotFoundError`。激活目标环境后，
以 root 幂等运行仓库提供的部署脚本：

```bash
conda activate retriever
bash scripts/setup_executor_user.sh

export EXECUTOR_UID=$(id -u toolelite-exec)
export EXECUTOR_GID=$(id -g toolelite-exec)
```

脚本创建专用的 `toolelite-exec` 系统用户，通过 ACL 只授予其穿过父目录并读取当前
Conda 环境的权限，同时验证隔离模式下的 Python 和 NumPy import。若系统没有
`setfacl`，先安装 `acl` 包。服务启动时传入：

```bash
--executor_uid "$EXECUTOR_UID" \
--executor_gid "$EXECUTOR_GID"
```

操作系统用户和 ACL 属于实例配置，不会随 FileZilla 上传代码自动创建；新实例或
未包含这些配置的镜像需要重新运行脚本。UID/GID 由实例动态读取，不应写死为某个
固定数字。

这套实现只适合可信训练流量。AST 拦截、隔离目录和资源限制属于纵深防御，并不是
面向恶意输入的安全边界；接入不可信用户输入时必须在容器或专用沙箱中运行。

### 训练缓存启动脚本

AutoDL 训练命令应通过统一启动脚本执行，避免相同 corpus 因缓存根目录不同而重复
生成 Hugging Face Datasets Arrow 缓存：

```bash
bash scripts/launch_training.sh python -m your_training_module [训练参数]
```

脚本默认设置：

```text
HF_HOME=/root/autodl-tmp/huggingface_cache
HF_DATASETS_CACHE=/root/autodl-tmp/huggingface_cache/datasets
```

非 AutoDL 环境可以通过 `TOOLELITE_DATA_ROOT` 覆盖数据盘根目录，或分别通过
`TOOLELITE_HF_HOME`、`TOOLELITE_HF_DATASETS_CACHE` 覆盖最终缓存路径。
