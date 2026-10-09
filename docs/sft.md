# 四张A800的SFT

从已准备的完整Qwen2.5-3B-Instruct HF目录开始，读取 `instruction/input/output`
Alpaca数据。每条完整output（含多个推理、工具动作与result）保持原始监督语义；
不隐式拆成ShareGPT轮次，不把工具反馈从output中删去。

默认使用全部30,438条，不划分验证集，不按eval_loss选优。
LLaMA-Factory模板为qwen，cutoff_len=15000，使用其原生截断规则，预检记录截断数量。
cutoff包含输入与完整输出；静态SFT没有在线工具调用或rollout生成预算。

| 设置 | 默认 |
|---|---:|
| A800卡数 / 后端 / 精度 | 4 / FSDP1 / BF16 |
| 每卡batch / 梯度累积 / 有效batch | 1 / 4 / 16 |
| 学习率 / epoch | 7e-6 / 3 |
| 调度 / warmup_ratio | cosine / 0.1 |
| logging_steps / save_steps | 10 / 2000 |

这些数值参考桌面Tool-Star实际训练YAML：原8卡batch1累积2，在4卡下改累积4以保持
有效batch16。继续复用Trace-OPSD既有FSDP1和checkpoint管理，不引入第二套ZeRO后端。

```bash
bash scripts/run_sft.sh validate --assets-root ../assets
bash scripts/run_sft.sh train --assets-root ../assets --run-name sft-math
```

默认模型 `../assets/models/Qwen2.5-3B-Instruct`，数据
`../assets/datasets/sft/sft_math_tir.json`，可显式指定 `--model-path`、`--dataset-file`。
输出位于 `<work>/checkpoints/sft/<run>`，保留最近两个checkpoint。
训练结束自动在独立进程中导出完整HF模型到 `<work>/exports/sft/<run>/final`。
恢复与单独导出参数见入口 `--help`。预检仍校验数据、模型、tokenizer与环境。
