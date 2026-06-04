<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Autotune learner guide -->

# Replay-Only Autotune Learners

The learner surface provides PPO-like, SAC-like, and hybrid-physics proposal
generators behind the existing non-actuating replay gates. These are
learner-shaped proposal interfaces, not claims of trained production policies.

All learner outputs are `LearnerPolicyProposal` records:

- `actuation_permitted` is always false,
- replay search evidence is serialised for audit,
- unsafe proposals are rejected by the same reward/proposal gates used by the
  deterministic replay search,
- optional safe-RL gates can require Lyapunov exponent, STL robustness, and
  bounded safety-cost evidence before a replay proposal is accepted,
- future PPO/SAC implementations must plug into the same contract before they
  can be reviewed.

Example:

```python
from scpn_phase_orchestrator.autotune import (
    PolicyProposalConfig,
    SafetyConstraintConfig,
)
from scpn_phase_orchestrator.autotune.learners import generate_ppo_like_proposal

proposal = generate_ppo_like_proposal(
    seed,
    evaluator,
    seed_value=7,
    proposal_config=PolicyProposalConfig(
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
audit = proposal.to_audit_record()
assert audit["actuation_permitted"] is False
```

Use this surface to compare replay candidates and export review records. A
candidate whose reward is high but whose Lyapunov/STL evidence is missing or
violating remains rejected. Do not feed learner proposals directly to hardware
or live actuation.
