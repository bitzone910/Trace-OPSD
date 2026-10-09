# Trace-OPSD 项目整体大纲

## 项目目标

面向多轮数学推理任务，使用 General Search 与 Python Executor 训练可调用工具、
整合观察并输出可验证答案的模型。公开流程统一为 SFT → TRACE-OPSD → evaluation。

## 训练路线

1. SFT：直接读取已准备的 `sft_math_tir` Alpaca数据，完整output作为监督目标。
   全部30,438条用于训练，序列截断15,000；4张A800、BF16、有效batch16。
2. TRACE-OPSD：从完整SFT HF模型开始，使用终局答案奖励计算GRPO优势；
   当前策略在因果视图与solution-hindsight视图对相同token重评分。先对turn内原始
   log-probability gap取均值，再计算共享权重，缩放该turn全部有效生成token优势。
   目标仍逐token计算，工具观察不参与loss。可信Python错误惩罚在缩放后执行。
3. Evaluation：用户显式提供模型清单，在固定数学快照上进行贪心和/或8采样评测；
   报告Pass@1、Pass@8、Avg@8、Majority@8及工具诊断。

TRACE算法身份固定为 `current_policy_solution_hindsight.turn_mean.v4`，不提供旧token
信用的训练开关。旧run/checkpoint仅按其历史身份检查、导出；已有历史指标保留原解释。
历史step400导出模型与历史evaluation不是新turn-mean机制的实验证据。

## 运行约定

默认4张A800训练、单张A800评测，检索和Python在CPU运行。仅维护A800环境配置。
模型、训练数据、索引、wheel与运行输出放在仓库外部assets/work目录；小型固定评测
快照和测试样例留在源码。四个公开入口及参数以README和相应文档为准。

TRACE先baseline，再分阶段训练，每阶段最多50step，完成保存、验证和指标打包后退出，
最后阶段止于622。课程配比显式设置，dynamic sampling保留真实采样游标。
训练checkpoint保留最新两个完整版本与最佳版本。

## 实现状态与验收边界

General Search、隔离Python执行、SFT训练/导出、TRACE训练/恢复/指标以及贪心/多采样
评测均有代码实现。此次仓库整理移出历史Teacher数据生成、难度筛选、独立GRPO实验、
特定step修复、更新包和独立smoke入口，保留共同依赖及必要CPU回归。

历史完整训练与评测记录保存于外层records。当前代码的本地验收结果另行记录；
整理后的4GPU完整训练和单卡评测需要在真实A800上验证，不能以CPU测试替代。

项目工程名现为 Trace-OPSD，包/入口/环境统一使用 trace_opsd / trace-opsd。
改名 CPU 回归684项及历史格式兼容14项通过，1项Windows Gloo测试跳过；六命令与
四Bash入口帮助、实际BF16历史模型只读检查及vendor校验通过。历史资产摘要未改写，
旧命名run仅只读校验/导出；迁移细节见docs/project-renaming.md。
