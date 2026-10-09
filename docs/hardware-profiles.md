# A800环境

本项目仅维护A800（sm80）环境，Python3.12、CUDA/PyTorch/vLLM组合以
`configs/hardware/a800.yaml`和`requirements/locks/a800-cu128.txt`为准。
FlashAttention使用配置内的固定wheel及SHA256；完整vendored框架的内容摘要也须通过。

```bash
bash scripts/setup_env.sh --gpu-count 4 --assets-root ../assets
```

默认训练4卡，评测GPU0单卡TP1；Python及E5/FAISS在CPU运行。
CUDA、Triton、TorchInductor等缓存和运行状态放在外部work根，不写入源码。
启动时保留实际依赖、GPU和工具协议预检，删除过的安装状态JSON不会单独阻断训练。
以root启动时入口配置隔离Python执行用户；不要让用户代码使用root身份执行。

环境安装只安装运行依赖，不自动下载大型模型或训练数据。
GPU kernel/NCCL/框架artifact检查属于必要环境验收，CPU回归不能代替这些检查。
