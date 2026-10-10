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

---

**Algorithm 1 — TRACE-OPSD with Turn-Mean Credit Assignment**

**Require:** Initial policy $`\pi_\theta`$; fixed reference policy $`\pi_{\mathrm{ref}}`$; Python and Search tools $`\mathcal T`$; mathematical verifier $`V`$; training steps $`S`$; group size $`G`$; accepted problem count $`B`$; stage difficulty quotas and replenishment limit; credit scale $`\tau\gt0`$; turn coefficients $`\lambda_{\mathrm{tool}},\lambda_{\mathrm{final}}`$; optional Python-error penalty $`\alpha`$.

**Input:** Dataset $`D=\{(x,y^\star,H_x)\}`$, where $`x`$ is a problem, $`y^\star`$ is its verification target, and $`H_x`$ is an offline solution outline.

| Line | Procedure |
| ---: | :--- |
| 1 | **for** step $`s=1,\ldots,S`$ **do** |
| 2 | &nbsp;&nbsp;&nbsp;&nbsp;$`\pi_{\mathrm{old}}\leftarrow\pi_\theta`$; initialize an empty accepted batch |
| 3 | &nbsp;&nbsp;&nbsp;&nbsp;**repeat until** $`B`$ complete groups satisfy the stage difficulty quotas |
| 4 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample candidate problems $`x`$ from $`D`$ |
| 5 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generate $`G`$ multi-turn trajectories per problem with $`\pi_{\mathrm{old}}`$ and $`\mathcal T`$ |
| 6 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▷ Rollout sees the problem and tool observations, never $`y^\star`$ or $`H_x`$ |
| 7 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$`R_i\leftarrow V(y_i,y^\star)`$ for each trajectory $`i`$ |
| 8 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Retain complete groups containing both correct and incorrect answers |
| 9 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Abort the step if the replenishment limit is reached before completion |
| 10 | &nbsp;&nbsp;&nbsp;&nbsp;**end repeat** |
| 11 | &nbsp;&nbsp;&nbsp;&nbsp;**for each** accepted problem group **do** |
| 12 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$`A_i\leftarrow\frac{R_i-\mathrm{mean}_j(R_j)}{\mathrm{std}_j(R_j)+\epsilon}`$ |
| 13 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$`A^0_{i,t}\leftarrow A_i`$ for all valid generated tokens in trajectory $`i`$ |
| 14 | &nbsp;&nbsp;&nbsp;&nbsp;**end for** |
| 15 | &nbsp;&nbsp;&nbsp;&nbsp;$`A^{\mathrm{final}}\leftarrow A^0`$ |
| 16 | &nbsp;&nbsp;&nbsp;&nbsp;Recompute causal token log-probabilities with $`\pi_{\mathrm{old}}`$ as `old_log_probs` |
| 17 | &nbsp;&nbsp;&nbsp;&nbsp;**for each** assistant turn $`k`$ in each accepted trajectory $`i`$ **do** |
| 18 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$`I_{i,k}\leftarrow`$ exact generated-token span of the entire assistant turn |
| 19 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Assert valid, non-overlapping spans and constant $`A^0`$ within $`I_{i,k}`$ |
| 20 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $`H_x`$ exists and aligned scoring views fit the scoring budget **then** |
| 21 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$`\ell^c_{i,k,t}\leftarrow`$ `old_log_probs` for $`t\in I_{i,k}`$ |
| 22 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$`\ell^h_{i,k,t}\leftarrow\log\pi_{\mathrm{old}}(a_{i,k,t}\mid h_{i,k},H_x,a_{i,k,\lt t})`$ |
| 23 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$`\bar e_{i,k}\leftarrow\frac{1}{\lvert I_{i,k}\rvert}\sum_{t\in I_{i,k}}(\ell^h_{i,k,t}-\ell^c_{i,k,t})`$ |
| 24 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$`c_{i,k}\leftarrow\max(0,\tanh(\bar e_{i,k}/\tau))`$ |
| 25 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$`\lambda_k\leftarrow\lambda_{\mathrm{final}}`$ for final turns; otherwise $`\lambda_{\mathrm{tool}}`$ |
| 26 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$`w_{i,k}\leftarrow1+\lambda_k\mathrm{sign}(A_i)c_{i,k}`$ |
| 27 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$`A^{\mathrm{final}}_{i,t}\leftarrow A^0_{i,t}w_{i,k}`$ for every $`t\in I_{i,k}`$ |
| 28 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▷ **One shared weight for all generated tokens in the turn** |
| 29 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end if** |
| 30 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▷ Missing hindsight or unscoreable views retain the base advantage |
| 31 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** Python-error penalty is enabled and $`k`$ is a trusted failed Python turn **then** |
| 32 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$`A^{\mathrm{final}}_{i,t}\leftarrow\min(A^{\mathrm{final}}_{i,t},0)-\alpha\lvert A^0_{i,t}\rvert`$ |
| 33 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for every $`t\in I_{i,k}`$ |
| 34 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end if** |
| 35 | &nbsp;&nbsp;&nbsp;&nbsp;**end for** |
| 36 | &nbsp;&nbsp;&nbsp;&nbsp;Detach $`A^{\mathrm{final}}`$; assert finite values and preserve the original dtype/device |
| 37 | &nbsp;&nbsp;&nbsp;&nbsp;Update $`\pi_\theta`$ with the token-wise clipped GRPO/PPO objective using $`A^{\mathrm{final}}`$ |
| 38 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and the configured reference-KL loss against $`\pi_{\mathrm{ref}}`$ |
| 39 | &nbsp;&nbsp;&nbsp;&nbsp;Synchronize rollout weights; save metrics and scheduled checkpoints/validation |
| 40 | **end for** |

**Output:** Trained policy $`\pi_\theta`$; inference uses the problem and tools without hindsight.

---

Each $`I_{i,k}`$ includes the turn's reasoning, action tags, code or query, and final-answer content where applicable; it excludes tool observations and padding. Both scoring views use the **same policy snapshot, sampled tokens, and scoring temperature**; $`\tau`$ is the credit scale, not the sampling temperature. Credit computation is detached, so only the policy objective contributes update gradients.

Default settings are $`G=8`$, $`B=32`$, $`\tau=1`$, $`\lambda_{\mathrm{tool}}=0.5`$, $`\lambda_{\mathrm{final}}=0.25`$, and $`\alpha=0.25`$. The Python-error penalty is an optional implementation feature applied **after** OPSD scaling, independently of hindsight availability. It uses trusted structured execution outcomes, affects only the failed assistant turn, and leaves later repair turns separate; queue, transport, and bootstrap failures are excluded.

