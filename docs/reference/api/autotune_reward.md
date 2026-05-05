# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Autotune reward API reference

# Autotune Reward Evaluation

The reward evaluator is the first safe slice of the reinforcement-learning
autotune track. It scores candidate knob policies from replay or simulation
metrics without applying control actions directly.

The default reward is coherence improvement minus target deficit, low-coherence
risk, actuation energy, unsafe rollout flags, and regime churn. The output is an
audit-ready record that can be used by later PPO/SAC or hybrid physics-RL
learners.

```python
from scpn_phase_orchestrator.autotune import (
    KnobPolicyCandidate,
    RewardObservation,
    evaluate_knob_policy,
)

candidate = KnobPolicyCandidate(K=0.2, alpha=0.0, zeta=0.05, Psi=0.1)
observation = RewardObservation(coherence=0.82, previous_coherence=0.74)
report = evaluate_knob_policy(candidate, observation)

assert report.to_audit_record()["reward"] == report.reward
```

::: scpn_phase_orchestrator.autotune.reward
