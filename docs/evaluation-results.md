# Mathematical reasoning accuracy

Acc (accuracy) is the fraction of questions answered correctly with one greedy trajectory per question; it is the same measurement previously labeled Pass@1. All values are percentages. Avg is the equal-weight arithmetic mean across the six displayed benchmarks, calculated from unrounded accuracies.

| Model | GSM8K | MATH500 | AIME24 | AIME25 | AIME26 | OlympiadBench | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen-2.5-3B | 46.63 | 49.40 | 3.33 | 0.00 | 3.33 | 23.26 | 20.99 |
| SFT | 78.39 | 59.60 | 10.00 | 10.00 | 6.67 | 30.37 | 32.50 |
| Trace-OPSD | 83.32 | 65.80 | 20.00 | 20.00 | 26.67 | 33.33 | 41.52 |

## Evaluation data

| Benchmark | Questions |
| --- | ---: |
| GSM8K | 1,319 |
| MATH500 | 500 |
| AIME24 | 30 |
| AIME25 | 30 |
| AIME26 | 30 |
| OlympiadBench | 675 |

Qwen-2.5-3B denotes Qwen2.5-3B-Instruct. The Trace-OPSD row uses the published historical model. The OlympiadBench comparison contains 675 English open-ended mathematics questions; it does not cover the full multilingual benchmark. The existing snapshots retain 30 recorded training-source overlaps in OlympiadBench and one in AIME26. These questions remain in the displayed denominators.

All three OlympiadBench runs have 675 unique records and zero generation or verification exceptions. Missing answers and truncated trajectories remain in the denominator. The supplementary runs use BF16 inference, greedy decoding (temperature 0), top-p 1, top-k -1, seed 42, an 8,192-token context, a 4,096-token response budget, 1,024 tokens per assistant turn, 512 observation tokens, and at most five tool calls.

[CSV](evaluation-results.csv) and [counts and provenance](evaluation-results.json) provide machine-readable values.
