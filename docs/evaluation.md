# 数学评测

公开入口 `scripts/run_eval_suite.sh` 按显式模型清单串行加载完整 Hugging Face 模型目录。训练 DCP checkpoint 需要先通过 TRACE 导出入口导出指定 step；评测不查找 latest，不自动替换模型。

## 模型清单与外部资源

复制 `configs/evaluation/models.example.yaml` 到运行目录，填写安全且唯一的模型名称及路径，移除所有占位符。`baseline` 可省略；填写时必须属于清单，比较报告以其计算百分点变化。相对模型路径从 `--assets-root` 解析，绝对路径也有效。

```yaml
schema: trace_opsd.evaluation.models.v1
baseline: sft
models:
  - name: sft
    path: models/SFT-Qwen2.5-3B
  - name: trace
    path: runs/exports/my-trace-model
```

此示例只说明格式，不是默认待评测模型。每个路径须具备完整权重、config 和可本地加载的 tokenizer。预检检查所有分片、真实 tensor dtype、文件 SHA256 及可选来源 step/run。支持 BF16 和 FP32 磁盘权重，推理统一 BF16；FP16、量化及残缺模型拒绝启动。

外部资源默认在仓库同级 `../assets`，也可通过 `--assets-root` 或 `TRACE_OPSD_ASSET_ROOT` 指定：

- `models/e5-base-v2/`
- `retrieval/general_search/index/e5_Flat.index`
- `retrieval/general_search/corpus/search_r1_corpus.jsonl`

输出默认在 `assets/runs/evaluation/<name>`；`--work-root` 或 `TRACE_OPSD_WORK_ROOT` 可指定运行根目录，也可显式提供 `--output-root`。结果目录必须位于源码仓库之外。

## 固定快照与协议

仓库 `data/evaluation` 快照覆盖 AIME24/25/26 各 30 题、MATH500 500 题、GSM8K 1,319 题，总计每模型 1,909 题。每次运行校验数据摘要，保留训练精确重合标记，已有结果不会重新下载数据或改写快照。

统一单张 A800、TP=1、BF16、CPU Python 与 General Search、并发16。上下文8192、总响应4096、每轮生成1024、工具观察512、最多5次工具调用、连续2次无效动作终止。脚本复用或配置非 root 的 Python 执行用户。

必须显式指定 `--protocol greedy|sampling8|all`：贪心使用温度0、top_p=1、top_k=-1、seed42，得到贪心 Pass@1。八采样协议及指标见 [evaluation-sampling.md](evaluation-sampling.md)。

```bash
bash scripts/run_eval_suite.sh \
  --models-file /path/to/models.yaml \
  --protocol all \
  --name my-math-comparison \
  --gpu 0
```

仅检查资源、模型、数据及依赖：追加 `--preflight-only`。小样本流程检查使用新名称并追加 `--limit-per-benchmark 2`；该运行不进入正式比较表，归档明确为不完整诊断，不设正确率门槛。

模型、数据、基准预算、采样设置和实际执行后端均记录于清单。每题完整轨迹原子保存，同一命令重启只补缺失项；指纹改变必须换结果目录。判分在主线程离线执行，记录异常并阻止正式比较；生成失败、截断及无答案保留在固定分母中。

每个模型独立保存完整轨迹、预测、指标和工具错误来源。正式结果输出各协议的 JSON/CSV/Markdown 比较表及 `<name>-analysis.tar.gz`，包含清单、轨迹和日志，不含模型权重。部分运行、异常或中断生成 `<name>-analysis.incomplete.tar.gz`，完整状态以 `batch_status.json` 为准。
