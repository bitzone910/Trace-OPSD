# SFT Input

The full dataset is stored externally at `datasets/sft/sft_math_tir.json` under the
asset root. All 30,438 Alpaca records with `instruction/input/output` fields are
used for training with the `qwen` template and a 15,000-token sequence cutoff.
This directory does not contain full training data or generated artifacts.
See [SFT](../../docs/sft.md) for data preparation, preflight, and export.
