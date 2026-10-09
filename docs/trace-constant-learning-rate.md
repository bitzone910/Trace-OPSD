# TRACE学习率控制

默认是峰值5e-7、50步warmup与cosine调度，最小学习率倍率0.1。
一般阶段续训会恢复优化器与调度器，不重启warmup。

可选 `+trainer.trace_lr_scale=<倍率>` 调整恢复后的调度倍率；
可选 `+trainer.trace_learning_rate=<数值>` 改为恒定实际学习率。二者互斥。
恒定模式记录在checkpoint，后续恢复继承；各参数组保留基础学习率比例。
这些选项不改变信用机制，阶段变更需满足run的既有兼容检查。
调参前分析actor/lr、KL、熵、梯度与验证指标。本次仓库整理不改变训练超参数。
