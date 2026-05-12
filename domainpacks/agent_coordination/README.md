<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Agent Coordination Domainpack -->

# Agent Coordination Domainpack

Models multi-agent task coordination as a coupled oscillator system.
Agent liveness, task cadence, topic alignment, operator intent, and
derived conflict risk become phase layers that the supervisor can monitor.

The binding spec includes a `value_alignment` template for review-time task
redistribution and deadline-drive actuation guards.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|-------------|---------|---------|
| heartbeat | 4 | P | Agent and operator liveness cadence |
| task_flow | 4 | I | Task event rhythm |
| topic_coherence | 4 | S | Shared topic-state alignment |
| operator_intent | 2 | H | Human priority cadence |
| conflict_risk | 2 | Risk | Derived merge and ownership risk |

## Run

```bash
spo validate domainpacks/agent_coordination/binding_spec.yaml
spo run domainpacks/agent_coordination/binding_spec.yaml --steps 100
python domainpacks/agent_coordination/run.py
```

## Read Next

- [Binding Spec Schema](../../docs/specs/binding_spec.schema.json)
- [Policy DSL](../../docs/specs/policy_dsl.md)
- [Interactive Tools](../../docs/guide/interactive_tools.md)
