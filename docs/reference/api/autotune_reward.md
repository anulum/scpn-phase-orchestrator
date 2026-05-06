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

Replay-trained or simulation-trained searches can rank multiple candidates
without applying any control action:

```python
from scpn_phase_orchestrator.autotune import rank_replay_candidates

ranked = rank_replay_candidates(
    (
        (KnobPolicyCandidate(K=0.15), RewardObservation(coherence=0.72)),
        (KnobPolicyCandidate(K=0.30), RewardObservation(coherence=0.65)),
    ),
    top_k=1,
)

best_report = ranked[0].to_audit_record()
```

Offline search can generate deterministic coordinate candidates around a seed
policy before replay scoring:

```python
from scpn_phase_orchestrator.autotune import (
    OfflinePolicySearchConfig,
    generate_offline_policy_candidates,
)

candidates = generate_offline_policy_candidates(
    candidate,
    OfflinePolicySearchConfig(K_step=0.05, zeta_step=0.02, max_abs_knob=1.0),
)
```

Proposal records apply simple acceptance gates and remain review artefacts:

```python
from scpn_phase_orchestrator.autotune import (
    PolicyProposalConfig,
    propose_replay_policy,
)

proposal = propose_replay_policy(
    (
        (candidate, RewardObservation(coherence=0.82)),
    ),
    proposal_config=PolicyProposalConfig(min_coherence=0.75),
)

audit_record = proposal.to_audit_record()
```

For the higher-level replay-only search wrapper that generates candidates,
evaluates them through a caller-supplied replay adapter, and returns a proposal
record, see [Autotune Replay Policy Search](autotune_policy_search.md).

::: scpn_phase_orchestrator.autotune.reward
