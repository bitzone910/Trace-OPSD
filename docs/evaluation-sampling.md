# 数学多采样评测

使用统一 [评测入口](evaluation.md)，必填模型清单和 `--protocol sampling8`（或 `all`）。原来的历史十模型清单及单独采样实验启动器已移出公开源码。

默认每题8条独立轨迹，温度0.7、top_p=0.95、top_k=-1、基础seed42。每个题目及采样序号获得稳定独立seed，不同模型共享seed清单。结果在 `full/sampling8/<model>/`，与贪心结果分别保存。

| 指标 | 定义 |
| --- | --- |
| Pass@8 | 八条轨迹至少一条正确的题目比例 |
| Avg@8 | 每题正确轨迹数除以8，再对题目平均 |
| Majority@8 | 数学等价答案聚类投票，最多票答案的准确率 |

Avg@8是随机采样协议的单样本正确率，不与贪心 Pass@1 混称。由同批轨迹按标准无放回估计计算 Pass@1/2/4 曲线，无需额外推理。多数投票仅读取预测答案，不读取 ground truth 或正确性标签；并列选最早采样，无答案弃权，全部弃权判错。

```bash
bash scripts/run_eval_suite.sh \
  --models-file /path/to/models.yaml \
  --protocol sampling8 \
  --name my-sampling8-comparison \
  --gpu 0 \
  --samples-per-question 8 \
  --temperature 0.7 --top-p 0.95 --top-k -1 \
  --seed 42 --pass-ks 1,2,4,8
```

参数改变必须使用新结果目录。每模型1,909题×8=15,272条正式轨迹；用户自选模型数量。先用新名称加 `--limit-per-benchmark 2` 检查每模型每基准2题×8条流程，之后用独立名称正式评测。

逐样本预测、逐题投票分组、采样seed、工具使用与结构化错误、生成失败、截断及无答案全部保留。失败样本保留固定分母；判分或投票异常标记不可用并阻止正式比较。仅完整、协议一致的结果合并到正式 JSON/CSV/Markdown 表。完整性检查拒绝重复、缺失或未知题目/采样槽。

低层 `python -m trace_opsd.evaluation sample`、`sample-score`、`sample-report` 继续用于已启动服务或保存轨迹的离线分析；多数投票及正式计分在主线程完成。完整归档由统一入口自动生成。
