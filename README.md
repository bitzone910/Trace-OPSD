# Trace-OPSD

面向多轮数学推理的双工具后训练项目：**SFT → TRACE-OPSD → evaluation**。
默认使用 4 张 A800 训练、单张 A800 评测；Python 与 General Search 在 CPU 上运行。

模型生成 `<python>...</python>` 或 `<search>...</search>` 调用工具，最终答案采用
`<answer>...\boxed{...}</answer>`。奖励按最终数学答案判定，工具观察不参与策略 loss。

## 资源与运行目录

```text
Trace-OPSD-workspace/
├── Trace-OPSD/                 # 本仓库
├── assets/
│   ├── models/                # Qwen2.5-3B-Instruct、e5-base-v2等完整HF目录
│   ├── datasets/sft/sft_math_tir.json
│   ├── datasets/numinamath/rl_trace_opsd.parquet
│   ├── datasets/grpo/         # 固定train/validation/manifest
│   ├── retrieval/general_search/{corpus,index}/
│   ├── wheels/               # A800锁定的FlashAttention wheel
│   └── runs/                 # 默认checkpoint、导出模型、指标、评测及缓存
└── records/                  # 本地保留的历史实验记录，不进入源码仓库
```

所有公开入口支持 `--assets-root`、`--work-root`，也可设置
`TRACE_OPSD_ASSET_ROOT`、`TRACE_OPSD_WORK_ROOT`。默认资源为仓库同级 `../assets`，
默认运行目录为该资源目录下的 `runs`。模型和大型数据不会由环境安装自动下载。
资源来源、固定版本和校验方法见 [资产说明](docs/assets.md)。

## 四个入口

以下示例在 Linux/AutoDL 仓库根目录执行。`ASSETS` 可换成你的实际资源目录。

```bash
export ASSETS=/root/autodl-tmp/assets
export WORK="$ASSETS/runs"

# 1. 配置完整A800环境和Python隔离执行用户
bash scripts/setup_env.sh --assets-root "$ASSETS" --work-root "$WORK" --gpu-count 4

# 2. 使用已有TIR数据做SFT，完成后自动导出完整HF模型
bash scripts/run_sft.sh train --assets-root "$ASSETS" --work-root "$WORK" \
  --run-name sft-math --model-path "$ASSETS/models/Qwen2.5-3B-Instruct" \
  --dataset-file "$ASSETS/datasets/sft/sft_math_tir.json"

# 3. 执行TRACE的第一阶段；16/16是示例配比，每阶段自行设定
bash scripts/run_trace_curriculum.sh --assets-root "$ASSETS" --work-root "$WORK" \
  --run-name trace-math --model-path "$WORK/exports/sft/sft-math/final" \
  --after-step 0 --mid-per-step 16 --high-per-step 16 --resume none

# 后续阶段使用同一run、上一阶段结束step和本阶段难度配比
# --after-step 50 --mid-per-step 16 --high-per-step 16 --resume latest

# 4. 先复制并填写模型清单，然后选择贪心、8采样或两者
mkdir -p "$WORK"
cp configs/evaluation/models.example.yaml "$WORK/evaluation-models.yaml"
# 编辑evaluation-models.yaml中的模型名称、完整HF路径和可选baseline
bash scripts/run_eval_suite.sh --assets-root "$ASSETS" --work-root "$WORK" \
  --models-file "$WORK/evaluation-models.yaml" --protocol all \
  --output-root "$WORK/evaluation/math-comparison"
```

评测模型没有默认值，占位符未填写时预检会明确失败；不会猜测历史run或替换成latest。
可用 `--protocol greedy` 或 `--protocol sampling8` 单独执行相应协议。
各入口提供 `--help`；必要的资产、工具服务和checkpoint预检仍保留。

## 默认训练与推理设置

| 项目 | SFT | TRACE / evaluation |
|---|---|---|
| 精度 | BF16全参数训练 | BF16推理；TRACE全参数训练 |
| token限制 | 完整训练序列截断15,000 | 上下文8192、总响应4096、每轮1024、观察512 |
| 工具调用 | 读取已有工具轨迹，无在线生成预算 | 最多5次工具调用，连续2次无效动作终止 |
| 数据/采样 | 30,438条Alpaca全部训练 | TRACE每step32个有效题组、每题8条轨迹 |
| 优化 | 每卡batch1、累积4、lr7e-6、3epoch、cosine、warmup0.1 | lr5e-7、warmup50步、cosine最小倍率0.1、KL系数0.005 |

SFT 沿用 Tool-Star 的15,000截断行为，报告截断数量，不拆开或删除原始output内工具反馈。
TRACE 初始模型先做256题baseline，每50步保存、验证并打包完整阶段指标后退出，
最后一个阶段止于总步数622。checkpoint保留最新两个完整版本及最佳版本。

## 算法与评测

TRACE 当前版本为 `current_policy_solution_hindsight.turn_mean.v4`：对同一assistant turn
中的原始teacher–student log-probability gap求均值，得到一个权重，广播到该turn的全部
有效生成token。Python错误惩罚默认启用，系数0.25，在OPSD缩放之后执行。

正式评测快照覆盖 GSM8K、MATH500、AIME24/25/26，共1,909题。
贪心协议报告Pass@1；8采样报告Pass@8、Avg@8、Majority@8和Pass@1/2/4诊断曲线，
同时统计工具调用、执行错误、无答案、截断和生成失败。
Avg@8是采样协议的平均正确率，不能与温度0的贪心Pass@1混称。

历史训练及已发布step400模型使用当时的信用机制，其既有评测结果不能作为turn-mean v4的验证。
旧checkpoint可以只读检查和导出HF权重，用作新run初始模型；新机制不迁移旧优化器状态。

## 文档与验证

- [SFT](docs/sft.md)、[TRACE训练](docs/trace-training.md)、[算法公式](docs/trace-opsd-method.md)
- [Evaluation](docs/evaluation.md)、[8采样指标](docs/evaluation-sampling.md)
- [A800环境](docs/hardware-profiles.md)、[General Search](docs/general-search.md)
- [Python错误信用](docs/trace-python-error-credit.md)、[验收命令](docs/validation.md)
- [项目改名与AutoDL迁移](docs/project-renaming.md)

vendored verl 与 LLaMA-Factory 保留完整快照和原许可证，环境安装检查其内容摘要。
其中上游示例属于第三方源码，不是Trace-OPSD的公开实验入口。来源见 `NOTICE`。
CPU回归覆盖算法、工具协议、恢复和评测；本轮整理后的完整GPU流程尚未在本机执行。
