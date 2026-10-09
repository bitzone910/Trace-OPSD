# A800 Environment

The project maintains only the A800 (sm80) environment. Python 3.12 and the
CUDA/PyTorch/vLLM combination are specified in `configs/hardware/a800.yaml` and
`requirements/locks/a800-cu128.txt`. FlashAttention uses the pinned wheel and
SHA256 in the profile. Complete vendored framework snapshots must also pass
content-digest verification.

```bash
bash scripts/setup_env.sh --gpu-count 4 --assets-root ../assets
```

Training defaults to four GPUs; evaluation uses GPU 0 with TP=1. Python and E5/FAISS
run on the CPU. CUDA, Triton, and TorchInductor caches and runtime state reside
under the external work root, outside source.

Launchers retain actual dependency, GPU, and tool-protocol checks. A missing
installation-state JSON alone does not block training. When launched as root,
the entry point configures an isolated Python executor user; user code must not
execute as root.

Setup installs runtime dependencies without downloading large models or training
data. GPU kernel, NCCL, and framework-artifact checks require environment validation;
CPU regression checks cannot replace them.
