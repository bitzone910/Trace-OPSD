# TRACE-OPSD: Turn-Mean Credit from Solution Hindsight

## 1. Motivation

Trace-OPSD's mathematical verifier evaluates final-answer correctness but does not
identify which reasoning or tool decision matters within a multi-turn trajectory.
Standard GRPO broadcasts a trajectory-level advantage across the response, so a
useful reasoning step, a redundant tool call, and the final answer can receive the
same update strength.

TRACE-OPSD uses the dataset's solution-oriented `hindsight` as training-only
privileged information. The current policy rescores identical generated tokens
under causal and solution-hindsight contexts to estimate credit for tool turns and
final answers while retaining a token-wise policy objective:

- GRPO terminal rewards determine the update direction.
- Solution hindsight determines how much terminal credit each scoreable turn
  receives; all generated tokens in that turn share one weight.
- Hindsight supplies no independent actor gradient; positive, bounded weights
  reshape advantages.
- Rollout and deployment prompts contain only `problem`, never `hindsight` or
  `solution`.

## 2. Problem formulation

For problem $x$, the current policy generates on-policy trajectory $i$:

$$
\tau_i=(h_{i,1},a_{i,1},o_{i,1},\ldots,h_{i,K_i},a_{i,K_i},o_{i,K_i},y_i),
$$

Here $h_{i,k}$ is the actual visible history before tool action $k$, $a_{i,k}$ is
the complete tool turn, $o_{i,k}$ is its observation, and $y_i$ is the final answer.
A turn is one complete assistant generation from the agent loop: it may contain
`<think>...</think>`, tool XML tags, and Python code or a Search query, or the last
response containing `<think>` and `<answer>`. User/environment observations returned
by tools are excluded from assistant turns and policy loss. The mathematical
verifier provides terminal reward $R_i$, and GRPO computes within the same problem's
rollout group:

$$
A_i=\frac{R_i-\operatorname{mean}_jR_j}
{\operatorname{std}_jR_j+\epsilon}.
$$

If the group has no reward variance, $A_i=0$; TRACE-OPSD does not invent an update
direction.

## 3. Current-policy hindsight self-teacher

Before each optimizer update, the current policy $\pi_{\theta_s}$ performs
teacher-forced rescoring of the student's generated tool turns and final answer.
For generated token $t$, the causal view uses only the history actually visible
during rollout; the hindsight view adds the problem's solution outline $H_x$:

$$
\ell^c_{i,k,t}=\log\pi_{\theta_s}(a_{i,k,t}\mid h_{i,k},a_{i,k,<t}),
\qquad
\ell^h_{i,k,t}=\log\pi_{\theta_s}(a_{i,k,t}\mid h_{i,k},H_x,a_{i,k,<t}).
$$

Causal token log-probabilities come directly from `old_log_probs` recomputed before
the optimizer update. The hindsight branch scores the same sampled tokens under
the same policy snapshot and temperature; it generates no new actions and makes
no extra tool calls. Prefix lengths may differ, but scored action tokens must align
exactly. Compute raw token log-probability gaps, then take the arithmetic mean over
all valid generated tokens in the turn:

$$
e_{i,k,t}=\ell^h_{i,k,t}-\ell^c_{i,k,t}.
$$

$$
\bar e_{i,k}=\frac{1}{T_{i,k}}\sum_{t=1}^{T_{i,k}}e_{i,k,t}.
$$

The mean includes reasoning, action tags, code, or final-answer content in the
assistant generation, excluding observations, environment feedback, and padding.
The implementation uses a stable double-precision mean to avoid overflow when
summing extreme finite gaps. Exact token alignment is required before credit
aggregation. Positive $\bar e_{i,k}$ means the solution outline increases the sampled
turn's average log-probability; it does not independently prove mathematical
correctness. `solution` is used only to generate `hindsight` offline and is excluded
from training data and scoring contexts.

## 4. Positive evidence and bounded weights

No group mean, variance, or $z$-score is computed by problem, tool type, or turn
index. With shared positive scale $\tau=1$ (`evidence_scale`), credit is activated
only for positive hindsight likelihood lift:

$$
c_{i,k}=\max\!\left(0,\tanh\frac{\bar e_{i,k}}{\tau}\right).
$$

For $\bar e\leq0$, $c=0$ and the ordinary GRPO advantage is preserved; for
$\bar e>0$, $0<c\leq1$ (finite precision can saturate at 1). Average raw gaps
**before** applying $\tanh$ and the positive-evidence gate. Do not average
transformed credits or token weights. For example, gaps `[-1, 1]` average to zero,
so the entire turn has weight 1. `evidence_scale` controls compression of raw gaps;
it is not the rollout sampling temperature and does not alter the temperature used
for either scoring view. Evidence is detached after computation in the current
step; gradients do not flow into the hindsight scoring branch.

## 5. Verifier-grounded turn advantage shaping

Let $m\in\{\text{search},\text{python},\text{final}\}$ denote the turn category.
Tool turns default to $\lambda_m=0.5$ and final answers to
$\lambda_{\mathrm{final}}=0.25$. Each turn receives a direction-aware, strictly
positive shared weight, broadcast to all its valid generated tokens:

$$
w_{i,k}=1+\lambda_m\operatorname{sign}(A_i)c_{i,k},
\qquad
\widetilde A_{i,k,t}=A_{i,k,t}w_{i,k}.
$$

When $A_i>0$, positive evidence strengthens reinforcement; when $A_i<0$, it reduces
the penalty. Original GRPO advantages must be identical within a turn. The
implementation checks this to prevent inconsistent weights from different token
signs. Weights and output advantages are detached, retain the original dtype/device,
and are checked for finiteness. Since
$1-\lambda_m\leq w_{i,k}\leq1+\lambda_m$ and $\lambda_m<1$, multiplicative shaping
preserves each token's advantage direction. Dense credit here means different
**terminal-gradient magnitudes** across generated turns, not independent token
supervision. All-correct/all-incorrect groups with $A_i=0$ remain zero; useful steps
in failed trajectories can be penalized less but cannot become positively rewarded
through this multiplication.

Candidate spans cover the entire recorded assistant generation: `<think>` content,
tool XML tags, Python code, Search queries, and final-response `<think>`/`<answer>`
content. Observations, environment feedback, and padding are not policy outputs
and do not enter evidence or advantage shaping. Positions without usable hindsight,
over the scoring-context budget, or without scoreable generated tokens retain
their original GRPO advantages. Final answers may receive weaker shaping even
in trajectories with no tool calls. `trace_final_turn` records the actual final
generation, including generation after tool-budget exhaustion. Even malformed
output uses final strength if its generated span is nonempty and alignable.
Ordinary invalid actions retain GRPO advantages, so not every `<think>` segment
is scored. Final strength is a fixed weaker coefficient, independent of prefix
length or distribution entropy.

The optional Python error penalty is applied after OPSD scaling:
`A_final = min(A_OPSD, 0) - python_error_penalty_coef * abs(A_GRPO)`.
It uses trusted execution status and penalizes the entire failed Python turn,
without carrying that failure's penalty into later repairs or the final answer.

The actor optimizes a clipped GRPO objective with shaped advantages:

$$
\mathcal L_{\mathrm{TRACE}}(\theta)=-\mathbb E_{i,k,t}
\left[\min\left(\rho_{i,k,t}\widetilde A_{i,k,t},
\operatorname{clip}(\rho_{i,k,t},1-\epsilon,1+\epsilon)
\widetilde A_{i,k,t}\right)\right].
$$

When Python error penalties are enabled, the final penalized advantages replace
the OPSD-scaled advantages in this objective for affected tokens.

TRACE-OPSD has no separate hindsight KL/JSD distillation objective. DAPO dynamic
sampling drops all-correct/all-incorrect groups and replenishes them to supply
groups with effective terminal gradients; it does not change the TRACE evidence
formula. A separately configured fixed reference-policy KL can limit policy drift.
That reference does not participate in hindsight credit estimation.

## 6. Training loop

1. Sample candidate problems according to the curriculum's remaining accepted
   quotas. Mid/high pools cycle independently without repetition within a cycle;
   accepted problems are unique within an update. Rollout prompts exclude hindsight.
2. Generate multiple on-policy trajectories per problem with Python or General Search.
3. The verifier returns binary correctness. Accept only complete eight-trajectory
   groups containing both correct and incorrect outcomes. Replenish under the same
   policy snapshot until exact mid/high quotas are filled; discard accepted overflow.
   The default limit is ten candidate batches per update. Failure stops training for
   recovery from the latest complete checkpoint, without a partial update. Compute
   trajectory-level GRPO advantages on the full accepted training batch.
4. Save assistant-generation token boundaries for valid tool actions and final
   answers. Obtain causal token scores from `old_log_probs`.
5. Rescore identical generated tokens with the current policy after adding hindsight.
6. Compute raw $e=\ell^h-\ell^c$ per token, average over each complete assistant turn
   to obtain $\bar e$, then apply the positive gate and $\tanh(\bar e/\tau)$ to obtain
   that turn's $c$. No group $z$-score is computed.
7. Compute one shared weight using the tool/final coefficient $\lambda_m$ and scale
   every generated token advantage in the turn. Unscored tokens retain their
   original advantages. Apply the Python error penalty afterward if enabled.
8. Update the actor only through the TRACE-GRPO objective and any configured fixed
   reference KL.
9. Synchronize updated weights to rollout workers and begin the next on-policy step.

All view scoring and advantage shaping occur before the optimizer update, keeping
causal and hindsight views on the same current policy snapshot. Candidate filtering
precedes old/reference/hindsight scoring. It does not change rewards, temperature,
reference KL, or the clipped GRPO objective. A global step advances only after a
completed actor update. Formal training uses 32 accepted problems per step and
32 candidates per batch. The last stage of 622-step training is `600→622`;
global steps do not count candidate batches or dataset passes.

During replenishment within an update, the rollout engine and policy snapshot stay
resident. The loop does not invoke a direct wake-up unsupported by the HYBRID
backend or resynchronize weights. The engine sleeps before full-batch processing
and is released on failure as well. This path supports the project's CPU/inline
binary rewards, not colocated reward models competing with rollout for GPU memory.

## 7. Configuration, metrics, and boundaries

Default configuration:

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

All TRACE training entry points use `current_policy_solution_hindsight.turn_mean.v4`.
Run manifests record `trace_credit_aggregation=turn_mean` and credit version
`trace_opsd.trace.credit.v3`. Credit hashes include aggregation and normalized OPSD
parameters. There is no old token-credit training switch; coefficients and scoring
budgets remain configurable.

Training enforces `algorithm.filter_groups.enable=true` and `metric=acc`, with
`max_num_gen_batches=10` (adjustable through launch arguments). Candidate
`data.gen_batch_size` equals the accepted problem batch size. New runs use v7
manifests and v3 curriculum ledgers. Historical runs can supply exported initial
weights for a new run but cannot switch algorithms in place.

Dynamic filtering adapts UID grouping, filtering, and accumulation from the
[official DAPO implementation](https://github.com/verl-project/verl-recipe/blob/e477deff97b15d067f8b4e71f75b80ae58ad64c4/dapo/dapo_ray_trainer.py).
It requires complete problem groups and preserves Trace-OPSD's tool masks and TRACE
metadata. It introduces no Clip-Higher, alternative loss aggregation, or reward changes.

Key metrics:

- `trace/view_log_prob/causal_mean`, `trace/view_log_prob/hindsight_mean`.
- `trace/evidence/mean`, `trace/evidence/abs_mean`.
- `trace/token_credit_coverage`, `trace/weight_*`, `trace/adv/token_ratio`,
  `trace/adv/*`, and `timing_s/trace_*`.
- `trace/turn/{python,search,final}/tokens`, `positive_evidence_ratio`,
  `evidence_mean`, `evidence_abs_mean`, `evidence_p50`, `evidence_p90`,
  `weight_mean`, `weight_abs_delta_mean`, `skipped_view_too_long`, and
  `skipped_missing_solution_hindsight`. These gap/weight metrics retain the
  historical token-weighted convention.
- `trace/turn_credit/evidence_mean`, `positive_evidence_ratio`, `weight_mean`,
  `weight_min`, and `weight_max`: equal weight per scored turn for its raw mean
  gap and shared weight.
- `trace/turn/{python,search,final}/credit_evidence_mean`, `credit_positive_ratio`,
  `shared_weight_mean`, `shared_weight_min`, and `shared_weight_max`: equal weight
  per scored turn within each category.
- `trace/dynamic/*`: candidate/accepted-group accuracy, all-correct/all-incorrect
  rejection, replenishment, and overflow. `trace/sampling/{mid,high}/*` records
  independent pool cycles and actual consumption positions.

`trace/token_credit_coverage` and
`trace/turn/{python,search,final}/positive_evidence_ratio` use scored tokens as
the denominator and measure raw $e_{i,k,t}>0$, not the new credit-gate coverage.
`trace/turn_credit/positive_evidence_ratio` and category `credit_positive_ratio`
use scored turns and measure $\bar e_{i,k}>0$. Neither convention measures actual
nonzero advantage changes. For $A=0$, positive gaps still yield zero
`weight_abs_delta_mean` and `trace/adv/abs_delta_mean`.

The method does not use tool results or ground-truth answers as hindsight, external
or EMA teachers, a critic, search trees, or additional environment rollouts.
`hindsight` is visible only in the training scoring branch; inference remains unchanged.

## 8. Validation and historical interpretation

See [Validation](validation.md) for CPU regressions and A800 process checks.
New training enables Python error penalties by default at coefficient 0.25, after
shared turn-weight scaling. Historical token-credit versions are supported for
read-only checkpoint inspection/export; their training and evaluation retain the
interpretation applicable at the time. The full GPU workflow after cleanup has
not been executed locally.
