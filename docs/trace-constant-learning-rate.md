# TRACE Learning-Rate Control

The default schedule has a peak learning rate of 5e-7, 50 warmup steps, and cosine
decay with a minimum ratio of 0.1. Normal stage continuation restores the optimizer
and scheduler without restarting warmup.

Use `+trainer.trace_lr_scale=<factor>` to scale the restored schedule, or
`+trainer.trace_learning_rate=<value>` to set a constant effective learning rate.
These options are mutually exclusive. Constant mode is recorded in checkpoints
and inherited on recovery. Relative base learning rates across parameter groups
are preserved.

These options do not change the credit mechanism. Stage changes must satisfy
existing run-compatibility checks. Review `actor/lr`, KL, entropy, gradients, and
validation metrics before tuning. Repository cleanup did not alter hyperparameters.
