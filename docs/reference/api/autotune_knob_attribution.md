# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Autotune per-knob attribution API reference

# Autotune Per-Knob Attribution

A reward report scores a candidate as a whole and breaks the score into reward
*components*. It does not say which *knob* earned the score. When a reviewer is
asked to trust a candidate before it is promoted, the question is concrete: how
much of the gain came from `alpha` versus `zeta` versus each channel weight, and
is any knob doing nothing — or actively hurting?

`attribute_knob_policy` answers that. Given a candidate, a baseline to attribute
against, and a value function that scores any candidate, it credits each knob
that differs between the two with:

- its **Shapley value** — the knob's average marginal contribution over every
  order in which the knobs could switch from the baseline to the candidate. It is
  the unique attribution that is *efficient* (the per-knob contributions sum
  exactly to the candidate-minus-baseline reward), *symmetric*, and assigns zero
  to a knob that never moves the value function. It is computed exactly over all
  coalitions for a small number of active knobs and by deterministic Monte-Carlo
  permutation sampling, with a reported standard error, above a threshold;
- its **marginal (leave-one-out)** contribution — the reward lost when that one
  knob alone is reset to baseline;
- the same Shapley breakdown **per reward component**, so a reviewer can see, for
  example, that a knob's positive coherence contribution is partly cancelled by
  its actuation cost.

The value function is supplied by the caller and treated as a black box, so the
same attribution works for a fixed-observation reward, a replay evaluator that
re-runs the candidate over recorded data, or any other scorer. The module
performs no control actuation and has no side effects.

```python
from scpn_phase_orchestrator.autotune import (
    KnobPolicyCandidate,
    RewardObservation,
    attribute_knob_policy,
    evaluate_knob_policy,
)

candidate = KnobPolicyCandidate(alpha=0.2, zeta=0.05, channel_weights=(1.0, 0.8))
baseline = KnobPolicyCandidate(alpha=0.0, zeta=0.0, channel_weights=(0.0, 0.0))

# Any candidate -> reward report scorer works; here a replay evaluator is stubbed
# by a fixed observation so the example is self-contained.
def evaluate(policy: KnobPolicyCandidate):
    observation = RewardObservation(coherence=0.82, previous_coherence=0.74)
    return evaluate_knob_policy(policy, observation)

report = attribute_knob_policy(candidate, baseline, evaluate)

# The most influential knob first; the contributions sum to the reward spread.
for item in report.attributions:
    print(item.knob, round(item.shapley_total, 4))
assert report.attributed_total == report.candidate_reward - report.baseline_reward
```

## Why a Shapley attribution rather than a sensitivity sweep

A one-knob-at-a-time sweep (the leave-one-out marginal) is cheap but
double-counts or drops interactions: if two knobs only help in combination, each
looks worthless alone. The Shapley value is the only credit assignment that
distributes interaction effects consistently and adds up to the whole, which is
what makes the resulting record defensible in a review rather than merely
suggestive. The marginal is still reported alongside it because it is the
quantity an operator's intuition expects, and the gap between the two is itself
informative.

## Where this sits in the autotune track

Attribution consumes the same candidate and reward surface as
[Reward Evaluation](autotune_reward.md) and
[Replay Policy Search](autotune_policy_search.md): a candidate is generated and
scored, the search ranks candidates, and attribution then explains the chosen
one knob by knob. It produces evidence, not actions — the explanation is part of
the review bundle a candidate must carry before any later learner is allowed to
propose production control.

::: scpn_phase_orchestrator.autotune.knob_attribution
