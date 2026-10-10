# SFT 

Start from a prepared, complete Qwen2.5-3B-Instruct HF directory. Preserve supervision over each
complete `output`, including reasoning, tool actions, and results.

| Setting | Default |
|---|---:|
| A800 GPUs / backend / precision | 4 / FSDP1 / BF16 |
| Batch per GPU / gradient accumulation / effective batch | 1 / 4 / 16 |
| Learning rate / epochs | 7e-6 / 3 |
| Scheduler / warmup_ratio | cosine / 0.1 |
| logging_steps / save_steps | 10 / 2000 |

```bash
bash scripts/run_sft.sh validate --assets-root ../assets
bash scripts/run_sft.sh train --assets-root ../assets --run-name sft-math
```

