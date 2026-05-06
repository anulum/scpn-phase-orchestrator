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
    KnobPolicyCandidate,
    OfflinePolicySearchConfig,
    PolicyProposalConfig,
    RewardObservation,
    search_replay_policy,
)

seed = KnobPolicyCandidate(K=0.2, alpha=0.0, zeta=0.05, Psi=0.1)


def replay(candidate: KnobPolicyCandidate) -> RewardObservation:
    return RewardObservation(coherence=0.82, previous_coherence=0.74)


result = search_replay_policy(
    seed,
    replay,
    search_config=OfflinePolicySearchConfig(K_step=0.05, max_abs_knob=1.0),
    proposal_config=PolicyProposalConfig(min_coherence=0.75),
)

audit_record = result.to_audit_record()
```

The result keeps the seed, generated candidates, and proposal together so audit
logs can prove which replay-only candidates were evaluated before a policy was
accepted or rejected.

::: scpn_phase_orchestrator.autotune.policy_search
