# Advantage Penalties for Failed Python Turns

New TRACE runs enable the penalty by default with coefficient α=0.25. Let A₀ be
the original GRPO advantage and B the advantage after shared turn-weight scaling.
For a trusted failed Python turn, the final advantage is
`min(B, 0) - α * abs(A₀)`. The entire failed assistant turn, including reasoning
and code, is penalized. Later repair turns and the final answer are handled
independently, without accumulated penalties.

Penalties use structured executor status for user-code exceptions and timeouts
after user-code execution has begun. Queue, network, bootstrap, sandbox, and
unknown-origin failures are excluded. Successful code printing error-like text or
returning `False`/`None` is not penalized. Observations and padding are unaffected.
Disabling the feature leaves the OPSD result unchanged.

Configuration resides in `configs/grpo/trace_opsd.yaml`. Missing fields in old runs
retain the historical disabled state; new defaults must not rewrite historical
credit hashes. This local penalty does not modify terminal rewards, dynamic
sampling, KL, or PPO definitions. Monitor penalized turns/tokens, deduction size,
tool success rate, call frequency, and validation accuracy.

Retention keeps the two latest complete checkpoints and the best checkpoint,
storing only one copy when these sets overlap.
