# TRACE-OPSD：基于解题思路 Hindsight 的 Turn 均值信用

## 1. 方法目标

Trace-OPSD 的数学 verifier 能可靠判断最终答案是否正确，但无法指出一条多轮工具轨迹中
哪个推理或工具决策产生了关键影响。普通 GRPO 将同一个轨迹级 advantage 广播给
整条 response，因此一段有用的思考、一次冗余工具调用和最终答案可能得到相同强度的
更新。

TRACE-OPSD 使用数据集中的 `hindsight` 解题思路作为训练期 privileged information，
通过当前策略在因果上下文和解题思路上下文下对相同的生成 token 重评分，估计
工具动作和最终回答的 turn 级信用，并保留 token 级策略目标：

- GRPO 的终局奖励决定更新方向；
- solution hindsight 决定每个可评分 turn 应继承多少终局信用，同一 turn 的所有生成 token 共享一个权重；
- hindsight 不产生独立 actor gradient，只通过正的有界权重重塑 advantage；
- rollout 和部署 prompt 只包含 `problem`，从不包含 `hindsight` 或 `solution`。

## 2. 问题定义

对问题 $x$，当前策略生成第 $i$ 条 on-policy 轨迹：

$$
\tau_i=(h_{i,1},a_{i,1},o_{i,1},\ldots,h_{i,K_i},a_{i,K_i},o_{i,K_i},y_i),
$$

其中 $h_{i,k}$ 是第 $k$ 个工具动作之前的真实可见历史，$a_{i,k}$ 是完整工具
turn，$o_{i,k}$ 是工具 observation，$y_i$ 是最终回答。这里一个 turn 指 agent loop
的一次完整 assistant generation：它可以包含 `<think>...</think>`、工具 XML 标签及
Python 代码或 Search query，也可以是最后一次包含 `<think>` 和 `<answer>` 的回答。
工具返回的 user/environment observation 不属于 assistant turn，也不进入策略 loss。
数学 verifier 产生终局奖励
$R_i$，GRPO 在同题 rollout 组内计算：

$$
A_i=\frac{R_i-\operatorname{mean}_jR_j}
{\operatorname{std}_jR_j+\epsilon}.
$$

若同题 rollout 组没有奖励方差，则 $A_i=0$，TRACE-OPSD 不人为制造更新方向。

## 3. 当前策略 Hindsight 自教师

每个 optimizer step 在更新前，对学生实际生成的工具 turn 和最终回答进行当前策略
$\pi_{\theta_s}$ 的 teacher-forced 重评分。对于其中第 $t$ 个生成 token，因果视图
只使用 rollout 当时真实可见的历史；hindsight 视图额外加入该题的解题思路 $H_x$：

$$
\ell^c_{i,k,t}=\log\pi_{\theta_s}(a_{i,k,t}\mid h_{i,k},a_{i,k,<t}),
\qquad
\ell^h_{i,k,t}=\log\pi_{\theta_s}(a_{i,k,t}\mid h_{i,k},H_x,a_{i,k,<t}).
$$

因果 token log-probability 直接取自 optimizer 更新前重算的 `old_log_probs`；
hindsight 分支在相同策略快照、相同采样 token 和温度下重评分，不生成新动作、不执行
额外工具调用。两视图的前缀长度可以不同，但被评分的 action token 必须逐个对齐。
先取得逐 token 的原始对数概率差，再对该 turn 的全部有效生成 token 求算术均值：

$$
e_{i,k,t}=\ell^h_{i,k,t}-\ell^c_{i,k,t}.
$$

$$
\bar e_{i,k}=\frac{1}{T_{i,k}}\sum_{t=1}^{T_{i,k}}e_{i,k,t}.
$$

均值包括本次 assistant generation 的推理、动作标签、代码或最终回答，排除工具观察、
环境反馈和 padding。实现使用稳定的双精度均值计算，避免极端有限 gap 的求和溢出。
两视图的逐 token 对齐仍然是必要条件；信用聚合发生在对齐之后。
正的 $\bar e_{i,k}$ 表示参考解题思路提高了该已采样 turn 的平均 log-probability，
不能单独证明它代表数学上正确的思路。`solution` 只用于离线生成 `hindsight`，
不进入训练数据或评分上下文。

## 4. 正证据门控与有界权重

不再按题目、工具类型或 turn 轮数计算组均值、方差和 $z$-score。设共享的正尺度
$\tau=1$（配置项 `evidence_scale`），只对正向 hindsight likelihood lift 启用信用：

$$
c_{i,k}=\max\!\left(0,\tanh\frac{\bar e_{i,k}}{\tau}\right).
$$

因此 $\bar e\leq0$ 时 $c=0$，保持普通 GRPO advantage；$\bar e>0$ 时 $0<c\leq1$
（有限精度下可饱和到 1）。先平均原始 gap，再执行 $\tanh$ 和正证据门控；
不平均变换后的信用或 token 权重。例如 gap `[-1, 1]` 的均值为 0，整个 turn 的权重为 1。
`evidence_scale` 只控制原始 gap 的压缩程度，**不是** rollout 采样温度，也不改变
两视图评分时使用的采样温度。证据在本 step 计算后停止梯度，不向 hindsight
评分分支反向传播。

## 5. Verifier-Grounded Turn Advantage Shaping

令 $m\in\{\text{search},\text{python},\text{final}\}$ 表示生成 turn 的类别；默认
工具 turn 的 $\lambda_m=0.5$，最终回答的 $\lambda_{\mathrm{final}}=0.25$。每个
turn 使用方向感知、严格为正的共享权重，广播到该 turn 的所有有效生成 token：

$$
w_{i,k}=1+\lambda_m\operatorname{sign}(A_i)c_{i,k},
\qquad
\widetilde A_{i,k,t}=A_{i,k,t}w_{i,k}.
$$

当 $A_i>0$ 时，正证据提高强化强度；当 $A_i<0$ 时，正证据减轻惩罚。
同一 turn 的原始 GRPO 优势必须一致；实现会验证这一条件，防止按不同 token 的符号生成
不一致权重。权重和输出优势均停止梯度，保持原优势的 dtype/device，并校验有限性。
因为 $1-\lambda_m\leq w_{i,k}\leq1+\lambda_m$ 且 $\lambda_m<1$，每个
token 的优势方向不变。这里的“密集”指在不同生成 turn 间分配不同的**终局梯度幅度**，
并非添加独立的 token 监督：全错或全对题组若有 $A_i=0$，乘性重塑后仍是零；失败
轨迹中的有效步骤只能少受惩罚，不能被翻成正向奖励。

整个已记录的 assistant generation 都在候选范围内，包括 `<think>` 内容、工具
XML 标签、Python 代码、Search query，以及最终回答中的 `<think>` 和 `<answer>`。
工具 observation、环境反馈和 padding 不属于策略输出，不参与证据或优势重塑。
无可用 hindsight、超出评分上下文预算或没有可评分生成 token 的位置保留原始
GRPO advantage。最终回答也允许在**没有工具调用**的轨迹中接受较小幅度的重塑。
`trace_final_turn` 记录正常答案或工具预算耗尽后的实际最终生成；后者即使格式错误，
只要存在非空、可对齐的生成片段，也使用 final 强度。普通无效动作仍使用原始
GRPO，因此并非轨迹内每一段 `<think>` 都会被评分。final 强度固定为较弱系数，
不根据 prefix 长度或分布熵自动调整。

可选 Python 错误处罚继续在 OPSD 缩放之后执行：
`A_final = min(A_OPSD, 0) - python_error_penalty_coef * abs(A_GRPO)`。
它只使用可信执行状态，整个失败 Python turn 受罚，后续修复和 final 不受该错误连带处罚。

actor 优化经过重塑的 clipped GRPO objective：

$$
\mathcal L_{\mathrm{TRACE}}(\theta)=-\mathbb E_{i,k,t}
\left[\min\left(\rho_{i,k,t}\widetilde A_{i,k,t},
\operatorname{clip}(\rho_{i,k,t},1-\epsilon,1+\epsilon)
\widetilde A_{i,k,t}\right)\right].
$$

TRACE-OPSD 没有独立的 hindsight KL/JSD 蒸馏目标。训练使用 DAPO dynamic sampling
丢弃全对／全错题组并补采，解决有效终局梯度组不足的问题；它不改变 TRACE 的证据公式。
训练配置可以另外使用固定 reference policy KL 限制策略漂移；该 reference
不参与 hindsight 信用估计。

## 6. 完整训练闭环

1. 按 curriculum 的有效题缺额比例采样候选 problem；mid/high 子题库分别循环，
   每 cycle 内不重复，同次更新的有效题不重复。rollout prompt 中不含 `hindsight`。
2. 当前策略为每题生成多条 on-policy 轨迹，并执行 Python 或 General Search。
3. 数学 verifier 产生二值准确率，只接收同题 8 条轨迹中既有正确又有错误的完整组。
   不足时使用同一策略快照补采，直至精确填满 mid/high 配额；有效溢出组丢弃。
   默认每次更新最多 10 批，失败则停止并从最近完整 checkpoint 恢复，不执行部分更新。
   对完整有效训练 batch 计算 GRPO 轨迹级 advantage。
4. 保存每次有效工具动作和最终回答的 assistant generation token 边界，从
   `old_log_probs` 取得逐 token 因果分数。
5. 当前策略在注入该题 `hindsight` 后，重评分完全相同的生成 token。
6. 逐 token 计算原始 $e=\ell^h-\ell^c$，对每个完整 assistant turn 求均值 $\bar e$，
   再通过正证据门控和 $\tanh(\bar e/\tau)$ 得到该 turn 的 $c$；不计算分组 $z$-score。
7. 按工具或最终回答对应的 $\lambda_m$ 计算一个共享权重，缩放该 turn 的所有生成 token
   advantage；未评分的 token 仍使用原始优势。若启用 Python 错误处罚，随后应用处罚。
8. actor 仅通过 TRACE-GRPO objective（以及配置中的固定 reference KL）更新。
9. 将新权重同步给 rollout worker，进入下一轮 on-policy 采样。

所有视图评分和 advantage 重塑必须发生在 optimizer 更新之前，以保证因果视图与
hindsight 视图来自同一当前策略快照。候选过滤发生在 old/ref/hindsight 评分之前，
不改变奖励、温度、reference KL 或 clipped GRPO objective；只完成一次 actor
更新才推进 step。正式训练每步有效 32 题，候选每批 32 题，622 步的最后阶段为
`600→622`，而不是 622 个候选批次或题库 pass。

同一次更新的候选补采期间保持 rollout 引擎及策略快照驻留，不调用 HYBRID
后端不支持的直接 wake-up，也不重新同步权重；完整批次组装前统一 sleep，
异常退出同样释放引擎。该路径只支持本项目的 CPU/inline 二值奖励，
不支持需要与 rollout 争用显存的 colocated reward model。

## 7. 配置、指标与边界

默认配置：

```yaml
enabled: true
credit_scale: 0.5
final_credit_scale: 0.25
evidence_scale: 1.0
score_chunk_size: 64
max_view_length: null
python_error_penalty_enabled: true
python_error_penalty_coef: 0.25
```

所有 TRACE 训练入口固定使用 `current_policy_solution_hindsight.turn_mean.v4`，
run manifest 记录 `trace_credit_aggregation=turn_mean`，信用版本为
`trace_opsd.trace.credit.v3`。新信用配置哈希包含聚合方式与归一化后的 OPSD 参数，
无旧 token 信用的训练开关。系数和评分预算仍可配置。

训练侧固定启用 `algorithm.filter_groups.enable=true`、`metric=acc`，
`max_num_gen_batches=10`（可通过启动参数提高），候选 `data.gen_batch_size`
等于有效题 batch 大小。新 run 使用 v7 manifest、v3 curriculum ledger；旧运行仅可
导出模型后作为新运行初始模型，不能原地切换。

动态过滤流程来源于 [DAPO 官方实现](https://github.com/verl-project/verl-recipe/blob/e477deff97b15d067f8b4e71f75b80ae58ad64c4/dapo/dapo_ray_trainer.py)
的 UID 分组、过滤和累计缓冲，适配时要求完整题组，保留 Trace-OPSD 的工具 mask 和
TRACE 元数据；不引入 Clip-Higher、其他 loss 聚合或奖励改造。

主要指标包括：

- `trace/view_log_prob/causal_mean`、`trace/view_log_prob/hindsight_mean`；
- `trace/evidence/mean`、`trace/evidence/abs_mean`；
- `trace/token_credit_coverage`、`trace/weight_*`、`trace/adv/token_ratio`、
  `trace/adv/*` 与 `timing_s/trace_*`；
- `trace/turn/{python,search,final}/tokens`、`positive_evidence_ratio`、
  `evidence_mean`、`evidence_abs_mean`、`evidence_p50`、`evidence_p90`、
  `weight_mean`、`weight_abs_delta_mean`、`skipped_view_too_long`、
  `skipped_missing_solution_hindsight`；上述 gap 与 weight 指标保留 token 加权的历史口径。
- `trace/turn_credit/evidence_mean`、`positive_evidence_ratio`、`weight_mean`、
  `weight_min`、`weight_max`：每个已评分 turn 等权统计其原始 gap 均值和共享权重。
- `trace/turn/{python,search,final}/credit_evidence_mean`、`credit_positive_ratio`、
  `shared_weight_mean`、`shared_weight_min`、`shared_weight_max`：对应类别的 turn 等权统计。
- `trace/dynamic/*` 记录原始候选和有效组正确率、全对／全错拒绝、补采和溢出；
  `trace/sampling/{mid,high}/*` 记录独立题库 cycle 与真实消费位置。

`trace/token_credit_coverage` 和 `trace/turn/{python,search,final}/positive_evidence_ratio` 的分母是已评分
token 数，统计的是原始 $e_{i,k,t}>0$ 的比例；它们不代表新的信用门控覆盖率。
`trace/turn_credit/positive_evidence_ratio` 和各类 `credit_positive_ratio` 的分母
是已评分 turn 数，统计的是 $\bar e_{i,k}>0$ 的比例。两种口径都不代表实际非零优势改变量。
当 $A=0$ 时，
即使 gap 为正，`weight_abs_delta_mean` 和 `trace/adv/abs_delta_mean` 仍为零。

方法不使用工具结果或正确答案作为 hindsight，不使用外部教师、EMA 教师、critic、
搜索树或额外环境 rollout。`hindsight` 仅在训练期评分分支可见，推理协议保持不变。

## 8. 验收与历史解释

CPU回归与A800流程验证见[验收说明](validation.md)。
新训练默认启用Python错误处罚，系数0.25，在turn共享权重缩放后应用。
旧token信用版本仅用于历史checkpoint只读检查和导出，已有训练和evaluation结果
仍按当时版本解释。本轮整理后的GPU完整流程尚未在本机执行。
## 8. 验收与历史解释

CPU回归与A800流程验证见[验收说明](validation.md)。
新训练默认启用Python错误处罚，系数0.25，在turn共享权重缩放后应用。
旧token信用版本仅用于历史checkpoint只读检查和导出，已有训练和evaluation结果
仍按当时版本解释。本轮整理后的GPU完整流程尚未在本机执行。
## 8. 验收与历史解释

CPU回归与A800流程验证见[验收说明](validation.md)。
新训练默认启用Python错误处罚，系数0.25，在turn共享权重缩放后应用。
旧token信用版本仅用于历史checkpoint只读检查和导出，已有训练和evaluation结果
仍按当时版本解释。本轮整理后的GPU完整流程尚未在本机执行。
## 8. 验收与历史解释

CPU回归与A800流程验证见[验收说明](validation.md)。
新训练默认启用Python错误处罚，系数0.25，在turn共享权重缩放后应用。
旧token信用版本仅用于历史checkpoint只读检查和导出，已有训练和evaluation结果
仍按当时版本解释。本轮整理后的GPU完整流程尚未在本机执行。
## 8. 验收与历史解释

CPU回归与A800流程验证见[验收说明](validation.md)。
新训练默认启用Python错误处罚，系数0.25，在turn共享权重缩放后应用。
旧token信用版本仅用于历史checkpoint只读检查和导出，已有训练和evaluation结果
仍按当时版本解释。本轮整理后的GPU完整流程尚未在本机执行。
