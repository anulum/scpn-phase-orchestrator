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
- future PPO/SAC implementations must plug into the same contract before they
  can be reviewed.

Example:

```python
from scpn_phase_orchestrator.autotune.learners import generate_ppo_like_proposal

proposal = generate_ppo_like_proposal(seed, evaluator, seed_value=7)
audit = proposal.to_audit_record()
assert audit["actuation_permitted"] is False
```

Use this surface to compare replay candidates and export review records. Do not
feed learner proposals directly to hardware or live actuation.
