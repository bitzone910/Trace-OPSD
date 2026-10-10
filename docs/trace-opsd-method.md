# TRACE-OPSD: Turn-Mean Credit from Solution Hindsight

## 1. Motivation

Trace-OPSD's mathematical verifier evaluates final-answer correctness but does not
identify which reasoning or tool decision matters within a multi-turn trajectory.

TRACE-OPSD uses the `hindsight` as training-only
privileged information. The current policy rescores identical generated tokens
under causal and solution-hindsight contexts to estimate credit for tool turns and
final answers while retaining a token-wise policy objective:

- GRPO terminal rewards determine the update direction.
- Solution hindsight determines how much terminal credit each scoreable turn
  receives; all generated tokens in that turn share one weight.

## 2. Problem formulation

For problem $`x`$, the current policy generates on-policy trajectory $`i`$:

```math
\tau_i=(h_{i,1},a_{i,1},o_{i,1},\ldots,h_{i,K_i},a_{i,K_i},o_{i,K_i},y_i),
```

Here $`h_{i,k}`$ is the actual visible history before tool action $`k`$, $`a_{i,k}`$ is
the complete tool turn, $`o_{i,k}`$ is its observation, and $`y_i`$ is the final answer.
A turn is one complete assistant generation from the agent loop: it may contain
`<think>...</think>`, tool XML tags, and Python code or a Search query, or the last
response containing `<think>` and `<answer>`. User/environment observations returned
by tools are excluded from assistant turns and policy loss. The mathematical
verifier provides terminal reward $`R_i`$, and GRPO computes within the same problem's
rollout group:

```math
A_i=\frac{R_i-\mathrm{mean}_jR_j}
{\mathrm{std}_jR_j+\epsilon}.
```

If the group has no reward variance, $`A_i=0`$; TRACE-OPSD does not invent an update
direction.

## 3. Current-policy hindsight self-teacher

Before each optimizer update, the current policy $`\pi_{\theta_s}`$ performs
teacher-forced rescoring of the student's generated tool turns and final answer.
For generated token $`t`$, the causal view uses only the history actually visible
during rollout; the hindsight view adds the problem's solution outline $`H_x`$:

```math
\ell^c_{i,k,t}=\log\pi_{\theta_s}(a_{i,k,t}\mid h_{i,k},a_{i,k,\lt t}),
\qquad
\ell^h_{i,k,t}=\log\pi_{\theta_s}(a_{i,k,t}\mid h_{i,k},H_x,a_{i,k,\lt t}).
```

Causal token log-probabilities come directly from `old_log_probs` recomputed before
the optimizer update. The hindsight branch scores the same sampled tokens under
the same policy snapshot and temperature; it generates no new actions and makes
no extra tool calls. Compute raw token log-probability gaps, then take the arithmetic mean over
all valid generated tokens in the turn:

```math
e_{i,k,t}=\ell^h_{i,k,t}-\ell^c_{i,k,t}.
```

```math
\bar e_{i,k}=\frac{1}{T_{i,k}}\sum_{t=1}^{T_{i,k}}e_{i,k,t}.
```

The mean includes reasoning, action tags, code, or final-answer content in the
assistant generation. Exact token alignment is required before credit
aggregation. Positive $`\bar e_{i,k}`$ means the solution outline increases the sampled
turn's average log-probability. `solution` is used only to generate `hindsight` offline and is excluded
from training data and scoring contexts.

## 4. Positive evidence and bounded weights

No group mean, variance, or $`z`$-score is computed by problem, tool type, or turn
index. With shared positive scale $`\tau=1`$ (`evidence_scale`), credit is activated
only for positive hindsight likelihood lift:

```math
c_{i,k}=\max\!\left(0,\tanh\frac{\bar e_{i,k}}{\tau}\right).
```

For $`\bar e\leq0`$, $`c=0`$ and the ordinary GRPO advantage is preserved; for
$`\bar e\gt 0`$, $`0\lt c\leq1`$ (finite precision can saturate at 1). Average raw gaps
**before** applying $`\tanh`$ and the positive-evidence gate. Do not average
transformed credits or token weights. For example, gaps `[-1, 1]` average to zero,
so the entire turn has weight 1. `evidence_scale` controls compression of raw gaps;
it is not the rollout sampling temperature and does not alter the temperature used
for either scoring view. Evidence is detached after computation in the current
step; gradients do not flow into the hindsight scoring branch.

## 5. Verifier-grounded turn advantage shaping

Let $`m\in\{\text{search},\text{python},\text{final}\}`$ denote the turn category.
Tool turns default to $`\lambda_m=0.5`$ and final answers to
$`\lambda_{\mathrm{final}}=0.25`$. Each turn receives a direction-aware, strictly
positive shared weight, broadcast to all its valid generated tokens:

```math
w_{i,k}=1+\lambda_m\mathrm{sign}(A_i)c_{i,k},
\qquad
\widetilde A_{i,k,t}=A_{i,k,t}w_{i,k}.
```

When $`A_i\gt 0`$, positive evidence strengthens reinforcement; when $`A_i\lt 0`$, it reduces
the penalty. Original GRPO advantages must be identical within a turn. The
implementation checks this to prevent inconsistent weights from different token
signs. Weights and output advantages are detached, retain the original dtype/device,
and are checked for finiteness. Since
$`1-\lambda_m\leq w_{i,k}\leq1+\lambda_m`$ and $`\lambda_m\lt 1`$, multiplicative shaping
preserves each token's advantage direction. Dense credit here means different
**terminal-gradient magnitudes** across generated turns, not independent token
supervision. All-correct/all-incorrect groups with $`A_i=0`$ remain zero; useful steps
in failed trajectories can be penalized less but cannot become positively rewarded
through this multiplication.

Candidate spans cover the entire recorded assistant generation: `<think>` content,
tool XML tags, Python code, Search queries, and final-response `<think>`/`<answer>`
content. `trace_final_turn` records the actual final
generation, including generation after tool-budget exhaustion. Even malformed
output uses final strength if its generated span is nonempty and alignable. Final strength is a fixed weaker coefficient, independent of prefix
length or distribution entropy.

The actor optimizes a clipped GRPO objective with shaped advantages:

```math
\mathcal L_{\mathrm{TRACE}}(\theta)=-\mathbb E_{i,k,t}
\left[\min\left(\rho_{i,k,t}\widetilde A_{i,k,t},
\mathrm{clip}(\rho_{i,k,t},1-\epsilon,1+\epsilon)
\widetilde A_{i,k,t}\right)\right].
```

## 6. Algorithm pseudocode

![Algorithm 1: TRACE-OPSD training with turn-mean credit assignment, in 22 numbered steps](images/trace-opsd-algorithm.png)

