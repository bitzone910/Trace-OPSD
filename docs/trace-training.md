# TRACE分阶段训练

新run从完整SFT HF权重开始，初始执行固定256题baseline。
每次训练最多50个global step，保存checkpoint、验证并打包完整指标后退出，
最后阶段止于622。每step接收32个有效题组，每题8条轨迹；mid+high必须合计32。
dynamic sampling默认最多10个候选batch，保持真实采样cycle/cursor和课程配额。

```bash
bash scripts/run_trace_curriculum.sh --assets-root ../assets --run-name trace-math \
  --model-path ../assets/runs/exports/sft/sft-math/final \
  --after-step 0 --mid-per-step 16 --high-per-step 16 --resume none
```

16/16仅为示例。后续阶段显式设置上一阶段结束step和新的难度配比，并使用
`--resume latest`，保留优化器、调度器、随机状态和采样游标。

默认4A800，CPU检索，lr5e-7、warmup50步、cosine最小倍率0.1、KL系数0.005。
prompt2048、上下文8192、响应4096、单轮1024、观察512、工具最多5次。
全部新训练使用turn-mean v4，Python错误惩罚默认true、系数0.25；算法见方法文档。

checkpoint在 `<work>/checkpoints/grpo/<run>`，保留最新两个完整版本及最佳版本。
rollout、validation、逐turn信用与训练指标写到外部work目录，每阶段自动生成分析包。
不能打包完整指标时明确标记incomplete；缺失值不填零。

旧token信用run只能只读检查/导出，不能用新算法原地恢复优化器。
独立导出使用内部 `scripts/run_grpo.sh export`，传入明确run与step，详见其帮助。
