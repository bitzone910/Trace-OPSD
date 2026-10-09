# 验收

本地CPU回归覆盖turn均值公式、PPO梯度方向、Python错误处罚、工具协议、课程采样、
checkpoint恢复与保留、SFT原始数据/截断以及贪心和多采样评测。

此前仓库整理的本地验收共收集1,015项：分批复核覆盖1,014项通过，1项因Windows
Torch构建不支持Gloo传输而跳过。Ruff、四入口帮助、全部Bash语法及两套完整vendor
摘要校验通过。环境执行用户测试使用命令替身，没有实际安装Linux包或创建账户。

改名后的验收为684项CPU回归及14项新增兼容测试通过，1项Windows Gloo测试跳过。
测试工作副本和完整记录位于本地外层 `records/project-renaming/2026-10-09/validation/`；
当前公开源码未包含根目录tests，不能直接运行 `pytest tests`。
具体迁移与验证范围见[项目改名说明](project-renaming.md)。

```bash
python scripts/verify_vendored_framework.py frameworks/verl --expected-name verl
python scripts/verify_vendored_framework.py frameworks/LlamaFactory --expected-name LlamaFactory
```

具体工具参数以 `--help` 为准。四个公开入口的 `--help` 不需要先启动GPU服务。
正式A800运行前检查资产，再进行少量数据的流程验证，例如：

```bash
export ASSETS=/root/autodl-tmp/assets
export WORK="$ASSETS/runs"
# 已配置环境后复核四卡CUDA、FlashAttention及NCCL
bash scripts/setup_env.sh --validate-only --gpu-count 4 \
  --assets-root "$ASSETS" --work-root "$WORK"
bash scripts/run_sft.sh validate --assets-root "$ASSETS" --work-root "$WORK"
bash scripts/run_grpo.sh validate --assets-root "$ASSETS" --work-root "$WORK" \
  --model-path "$WORK/exports/sft/sft-math/final"
bash scripts/run_eval_suite.sh --assets-root "$ASSETS" --work-root "$WORK" \
  --models-file /path/to/models.yaml --protocol all \
  --name process-check --preflight-only
bash scripts/run_eval_suite.sh --assets-root "$ASSETS" --work-root "$WORK" \
  --models-file /path/to/models.yaml --protocol all \
  --limit-per-benchmark 2 --output-root "$WORK/evaluation/process-check"
```

小样本只检查加载、生成、Python与搜索执行、保存、判分及恢复，不设准确率门槛，
不得与完整正式结果混合。本轮没有在本机运行A800训练/推理；GPU kernel、NCCL、
SFT实际更新与TRACE实际更新需在目标设备验收，不以CPU替身测试宣称通过。
