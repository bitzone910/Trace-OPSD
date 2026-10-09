# Python错误turn的优势惩罚

新TRACE训练默认启用，系数α=0.25。原始GRPO优势为A₀，turn共享权重缩放后的优势为B，
可信失败Python turn的最终优势为 `min(B, 0) - α * abs(A₀)`。
整个失败assistant turn（推理和代码动作）受罚，后续修复turn与final分别计算，不累计处罚。

只依据执行器结构化状态处理用户代码异常和已进入执行阶段的超时。
排队、网络、bootstrap、sandbox与来源不明的失败不处罚；成功打印错误样式文本，或
返回False/None也不处罚。工具观察与padding不受影响；关闭功能时保持原OPSD结果。

配置位于 `configs/grpo/trace_opsd.yaml`。旧run缺失字段按历史关闭状态解释，
不能用新默认值改写旧信用hash。局部处罚不修改终局奖励、dynamic sampling、KL或PPO定义。
监测错误turn/token数、扣减幅度、工具成功率、调用频率及验证准确率。
checkpoint保留最新两个完整版本和最佳版本，重合只保留一份。
