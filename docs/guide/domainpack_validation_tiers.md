<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Domainpack validation tiers

SCPN Phase Orchestrator ships dozens of domainpacks — power grid, cardiac rhythm,
financial markets, and more. **A binding is a reusable scaffold, not a validated
detector.** To keep a broad gallery from reading as a broad set of validated
solutions, every `binding_spec.yaml` declares a machine-readable
`validation_tier` recording how much external evidence the scaffold carries.

This is distinct from `safety_tier` (the deployment risk class). A pack can be
`safety_tier: production` and still be `validation_tier: scaffold` — the risk
class governs how carefully it must be operated; the validation tier states
whether its end-to-end behaviour has been checked against an independent
reference.

## The three tiers

| Tier | Token | Meaning |
| --- | --- | --- |
| **Scaffold** | `scaffold` | A reusable mapping of a domain onto the engine, with **no** external- or synthetic-reference validation record. The honest default. |
| **Partial** | `partial` | Some validation evidence, but not a full independent-reference test on real data. |
| **Externally validated** | `externally_validated` | Clears an independent-reference test on real data end to end. |

The vocabulary lives in `VALID_VALIDATION_TIERS`; the loader defaults an
undeclared spec to `DEFAULT_VALIDATION_TIER` (`scaffold`) so an unlabelled pack
is treated as unvalidated rather than silently trusted, and the validator
rejects any unknown tier. Both are defined in
[`binding.types`][scpn_phase_orchestrator.binding.types].

## Current posture: every shipped pack is a scaffold

Today **all** shipped domainpacks declare `validation_tier: scaffold`. None
carries an independent end-to-end validation trail in its own materials — each
pack's README states it is a template *for simulation, replay, and policy
review*, not a live system. The one externally-validated detection niche in the
programme, grid modal damping, lives in the
[monitor validation-status registry](monitor_validation_status.md), not in a
domainpack. A pack is promoted above `scaffold` only when a citable evidence
trail earns it.

## Filtering a gallery

A Studio Hub that lists the packs can select or group by tier so an operator sees
the scaffold status plainly:

```python
from scpn_phase_orchestrator.binding.gallery import (
    group_specs_by_validation_tier,
    select_specs_by_validation_tier,
)

# only the externally-validated packs (currently none)
select_specs_by_validation_tier(specs, "externally_validated")

# every tier, empty tiers included, for a tiered display
group_specs_by_validation_tier(specs)
```

See the [binding API reference](../reference/api/binding.md) for the full
machine-readable surface.
