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
risk, actuation energy, unsafe rollout flags, regime churn, positive Lyapunov
growth, negative STL robustness, and explicit safety cost. The output is an
audit-ready record that can be used by later PPO/SAC or hybrid physics-RL
learners.

```python
from scpn_phase_orchestrator.autotune import (
    KnobPolicyCandidate,
    RewardObservation,
    evaluate_knob_policy,
)

candidate = KnobPolicyCandidate(K=0.2, alpha=0.0, zeta=0.05, Psi=0.1)
observation = RewardObservation(
    coherence=0.82,
    previous_coherence=0.74,
    lyapunov_exponent=-0.015,
    stl_robustness=0.08,
    safety_cost=0.01,
)
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
policy before replay scoring. The candidate surface includes the universal
knobs, per-channel weights, and cross-channel coupling gains; the generator is
still side-effect free and only emits replay candidates.

```python
from scpn_phase_orchestrator.autotune import (
    OfflinePolicySearchConfig,
    generate_offline_policy_candidates,
)

candidates = generate_offline_policy_candidates(
    KnobPolicyCandidate(
        K=0.2,
        zeta=0.05,
        channel_weights=(1.0, 0.8),
        cross_channel_gains=(0.3, 0.5),
    ),
    OfflinePolicySearchConfig(
        K_step=0.05,
        zeta_step=0.02,
        channel_weight_step=0.1,
        cross_channel_gain_step=0.1,
        max_abs_knob=1.0,
    ),
)
```

Proposal records apply simple acceptance gates and remain review artefacts:

```python
from scpn_phase_orchestrator.autotune import (
    PolicyProposalConfig,
    SafetyConstraintConfig,
    propose_replay_policy,
)

proposal = propose_replay_policy(
    (
        (
            candidate,
            RewardObservation(
                coherence=0.82,
                lyapunov_exponent=-0.015,
                stl_robustness=0.08,
                safety_cost=0.01,
            ),
        ),
    ),
    proposal_config=PolicyProposalConfig(
        min_coherence=0.75,
        safety_constraints=SafetyConstraintConfig(
            max_lyapunov_exponent=0.0,
            min_stl_robustness=0.0,
            max_safety_cost=0.05,
            require_lyapunov=True,
            require_stl=True,
            require_safety_cost=True,
        ),
    ),
)

audit_record = proposal.to_audit_record()
```

The safety-constraint gate is intentionally conservative. If a proposal config
requires Lyapunov or STL evidence and a replay observation omits it, the
candidate is rejected even when its coherence reward is high. This keeps
safe-RL integration reviewable: the learner may optimise, but the acceptance
record must still carry explicit stability and temporal-logic evidence.

For the higher-level replay-only search wrapper that generates candidates,
evaluates them through a caller-supplied replay adapter, and returns a proposal
record, see [Autotune Replay Policy Search](autotune_policy_search.md).

## Operational overview

This module is the scoring seam between raw simulation outcomes and policy action.
Rewards are structured to preserve risk awareness while still allowing optimization
experiments.

When used in a staged autonomy lane, teams typically:

- score candidates with replay evidence first,
- enforce explicit safety gates,
- only then promote a candidate into a proposal record.

That order is important because it separates numeric optimisation from safety
admissibility. A candidate can look strong on raw coherence and still fail safe policy
constraints.

## Safe RL readiness

The shape and fields in `evaluate_knob_policy` are aligned with later RL loop
integration:

- Observations carry stability, coherence, and safety fields together.
- Proposal records contain audit-ready traces and gating decisions.
- Rejection reasons are represented in the proposal output, not only in external
  logs.

This makes the reward module usable as the first hard requirement layer for PPO/SAC
or other search learners without changing governance rules later.

## Why this function is separated from control execution

`evaluate_knob_policy` is intentionally scoped to scoring and evidence generation.
It computes reward and audit fields so a learner can rank candidates, but it does not
decide control actions by itself.

When integrating with RL stacks, keep the same sequence:

- replay or simulation produces observations,
- reward scoring evaluates candidates with coherence, stability, and safety terms,
- proposal gating enforces the hard safety thresholds,
- deployment code consumes only accepted proposals that preserve evidence boundaries.

This keeps model-driven optimisation inside a constrained review envelope rather
than directly connecting policy gradients to hardware or external actuators.

::: scpn_phase_orchestrator.autotune.reward
