# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Autotune replay policy search API reference

# Autotune Replay Policy Search

The replay policy-search API is the non-actuating bridge between deterministic
candidate generation and future learner-backed autotune loops. It generates
bounded candidates around a seed policy, delegates each candidate to a replay or
simulation evaluator, and returns a reviewable proposal record.

The evaluator is deliberately supplied by the caller. Production integrations
should connect it to replay buffers, dry-run simulation, or hardware-in-the-loop
shadow evaluation before a proposal is considered for deployment.

```python
from scpn_phase_orchestrator.autotune import (
    AdaptiveReplayPolicySearchConfig,
    KnobPolicyCandidate,
    OfflinePolicySearchConfig,
    PolicyProposalConfig,
    RewardObservation,
    SafetyConstraintConfig,
    search_replay_policy,
)

seed = KnobPolicyCandidate(
    K=0.2,
    alpha=0.0,
    zeta=0.05,
    Psi=0.1,
    channel_weights=(1.0, 0.8),
    cross_channel_gains=(0.3, 0.5),
)


def replay(candidate: KnobPolicyCandidate) -> RewardObservation:
    return RewardObservation(
        coherence=0.82,
        previous_coherence=0.74,
        lyapunov_exponent=-0.015,
        stl_robustness=0.08,
        safety_cost=0.01,
    )


result = search_replay_policy(
    seed,
    replay,
    search_config=OfflinePolicySearchConfig(
        K_step=0.05,
        channel_weight_step=0.1,
        cross_channel_gain_step=0.1,
        max_abs_knob=1.0,
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

audit_record = result.to_audit_record()
```

For bounded learner-style refinement without enabling live actuation, use the
adaptive search wrapper. It repeatedly evaluates replay candidates around the
best replay-scored candidate, decays the coordinate step sizes, and then applies
the same final proposal gates across all replay observations:

```python
from scpn_phase_orchestrator.autotune import search_adaptive_replay_policy

adaptive_result = search_adaptive_replay_policy(
    seed,
    replay,
    adaptive_config=AdaptiveReplayPolicySearchConfig(
        base_search_config=OfflinePolicySearchConfig(
            K_step=0.05,
            zeta_step=0.02,
            max_abs_knob=1.0,
        ),
        iterations=3,
        step_decay=0.5,
    ),
    proposal_config=PolicyProposalConfig(min_coherence=0.75),
)

adaptive_audit_record = adaptive_result.to_audit_record()
```

The result keeps the seed, generated candidates, and proposal together so audit
logs can prove which replay-only candidates were evaluated before a policy was
accepted or rejected.

When `SafetyConstraintConfig` is attached to `PolicyProposalConfig`, the search
will reject candidates that lack required Lyapunov or STL evidence, exceed the
configured Lyapunov exponent bound, violate the STL robustness floor, or exceed
the safety-cost ceiling. These gates run after replay scoring and before a
proposal is accepted, so the search cannot promote a high-reward candidate that
fails the explicit safety evidence contract.

## Practical narrative

This page is the control bridge between experimentation and controlled deployment:

- Generate candidates (bounded).
- Score in replay/simulation (comparative).
- Enforce safety gates (governance).
- Emit one auditable proposal record (promotion boundary).

The design is intentionally conservative because it prevents “good-scoring but
unsafe” candidates from silently entering a deployment lane. This pattern is the
same pattern used across safety-critical control stacks where model uncertainty and
environmental shift can produce false positives.

## Review and audit interpretation

Replay search is only as useful as its provenance. Keep the evaluator deterministic
for a given seed and profile, and record search configurations so repeated runs can
compare:

- candidate envelopes,
- proposal thresholds,
- safety boundary settings.

That evidence makes it possible to justify in review why a specific search depth or
step size was chosen, and why rejected candidates were blocked.

::: scpn_phase_orchestrator.autotune.policy_search
