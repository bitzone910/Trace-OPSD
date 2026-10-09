# Trace-OPSD 命名与迁移

项目名称统一为 **Trace-OPSD**，Python 包为 `trace_opsd`，安装包和默认 Conda
环境为 `trace-opsd`。本次仅改变工程命名及历史格式读取；算法、提示词、工具动作、
训练参数、依赖版本和评测指标没有调整。

## 当前本地目录

```text
C:/Users/lenovo/Desktop/ToolElite/
├── Trace-OPSD/             # 原内层项目，待发布源码
├── assets/                 # 保留的外部资产和运行目录
├── records/                # 历史实验、改名备份和验证结果
└── .trace-engineering-venv/ # 本地工程环境
```

外层目录未改名；Hugging Face 仓库名、模型和数据文件也未改名。
改名前源码备份和逐文件清单位于外层 `records/project-renaming/2026-10-09/`。

## 当前接口

| 对象 | 名称 |
|---|---|
| Python 包/导入 | `trace_opsd` |
| 安装包/Conda | `trace-opsd` |
| 环境变量 | `TRACE_OPSD_` 前缀 |
| W&B project | `Trace-OPSD` |
| Python 执行用户 | `trace-opsd-exec` |
| 工具服务 | `trace-opsd-tool-server` |
| 索引构建 | `trace-opsd-general-index` |
| 检索服务 | `trace-opsd-general-server` |
| SFT | `trace-opsd-sft` |
| TRACE | `trace-opsd-train` |
| Evaluation | `trace-opsd-eval` |

四个 Bash 入口文件名保留：`setup_env.sh`、`run_sft.sh`、
`run_trace_curriculum.sh`、`run_eval_suite.sh`。动态导入、agent 注册和环境配置使用
新名称。旧 Python 包、命令及环境变量不提供别名；检测到旧项目环境变量时会报错，
应在新终端移除旧变量并设置新变量，避免误用资源目录。

新产物使用 `trace_opsd_grpo_run.json`、`trace_opsd_protocol.json`、
`trace_opsd_grpo_manifest.json`；新的 schema/protocol 采用 `trace_opsd.*`。

## 历史文件读取

`trace_opsd.compat` 集中识别历史文件名、schema、信用版本和指标名。读取时返回
历史 JSON 原值，不改写原文件，不重新计算其来源摘要。新旧元数据同时存在且内容
不一致时明确报错；仅命名空间差异可视为同一身份。

旧 checkpoint 可校验及导出。导出创建新目录、新协议文件，同时保留实际历史
算法版本、信用配置、来源 step 和数据摘要。不会把历史 token 信用模型标为
turn-mean 模型。旧命名的训练 run 不会被就地迁移；需要继续使用历史权重时，
先导出完整 HF 模型，再以新 run 名启动当前算法，不迁移历史优化器。

历史评测快照与报告保持原格式、协议指纹和分数。已保存的八采样结果可以只读汇总，
保留其实际评分版本；新生成/重新计分需使用当前代码指纹，不能混入旧目录或将
历史和新协议强行合并。

以下文件刻意保留历史名称：

| 文件 | 保留原因 |
|---|---|
| `data/evaluation/manifest.json` | 已封存评测数据的 schema、训练来源路径和摘要 |
| `configs/assets/manifest.json` | 资产迁移前路径及已记录的文件哈希 |
| `configs/assets/downloadable-assets.json` | 已发布 HF 资产的固定 commit 和哈希 |
| `trace_opsd/compat.py` | 历史元数据、离线 W&B 和指标别名读取 |
| `trace_opsd/assets.py`、`scripts/assets_env.sh` | 检测并拒绝旧环境变量 |
| verl 的两个项目定制文件 | 直接读取历史 global checkpoint / FSDP DCP schema；第三方模块可独立使用 |

第三方源码、许可证、上游身份及外层历史归档不批量替换。两个框架的项目 vendor
元数据改为 `TRACE_OPSD_VENDOR.json`，按实际源码重新计算快照摘要。

## AutoDL

上传整个当前源码目录至 `/root/autodl-tmp/Trace-OPSD/`，准备完整外部资产目录。
本地已按先前要求删除部分可恢复资产；仅上传源码不能补齐模型、训练数据及索引，
固定版本恢复方法见 [资产说明](assets.md)。

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

# 复制并填写模型清单后进行评测
mkdir -p "$WORK"
cp configs/evaluation/models.example.yaml "$WORK/evaluation-models.yaml"
bash scripts/run_eval_suite.sh --assets-root "$ASSETS" --work-root "$WORK" \
  --models-file "$WORK/evaluation-models.yaml" --protocol all \
  --output-root "$WORK/evaluation/math-comparison"
```

难度配比 16/16 为示例。可指定已有完整 HF SFT/历史导出模型作为 TRACE 初始权重；
不要把 DCP 训练目录当作 HF 模型。详细参数仍以四个入口的 `--help` 为准。

## 本次验证

- 新包 editable 安装、六个命令帮助、四个 Bash 入口帮助及全部 Bash 语法通过。
- CPU 回归 684 项通过、1 项因 Windows Gloo 支持限制跳过；新增改名兼容测试
  14 项通过。测试工作副本保存在外层记录目录，原始归档未改写。
- 实际桌面 SFT 与历史 TRACE 模型均通过 HF config/tokenizer/分片检查，
  各 434 个 tensor 均为 BF16；未加载到 GPU。
- 1,587 项保护文件校验无差异（含两份实验主归档）；固定评测共 1,909 题。
- 优势重塑、Python 错误惩罚、证据统计和工具提示词的关键函数 AST 与改名前一致。
- 两个完整 vendor 快照重新校验通过。未执行 A800 训练、推理或实际 Linux 安装，
  未提交或推送 GitHub，也未修改 HF 仓库。
