# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Autotune supervisor-candidate bundle API reference

# Autotune Supervisor-Candidate Bundle

A reviewer asked to trust a supervisor policy candidate has three questions, and
the autotune track answers each on its own: is it *better* (the
[reward report](autotune_reward.md) and a comparison against the incumbent), *why*
these knobs ([per-knob attribution](autotune_knob_attribution.md)), and is it
*safe* ([the safety certificate](autotune_candidate_safety_certificate.md)).
`build_supervisor_candidate_bundle` glues those answers into one sealed,
review-only artefact so a candidate carries its whole case in a single record.

The bundle scores the candidate, attributes it against a baseline, certifies its
safety, compares it against the incumbent it would replace, and stamps the result
with its numeric provenance — which compute backend produced the numbers and the
parity tolerance they were checked to — and its safety tier. The record is the
`studio.supervisor_candidate.v1` shape, content-addressed by a canonical-JSON
SHA-256 seal, and carries the safety certificate's evidence modality. It proposes
nothing and actuates nothing.

```python
from scpn_phase_orchestrator.autotune import (
    KnobPolicyCandidate,
    NumericProvenance,
    RewardObservation,
    SafetyConstraintConfig,
    build_supervisor_candidate_bundle,
    evaluate_knob_policy,
)

def evaluate(policy: KnobPolicyCandidate):
    return evaluate_knob_policy(policy, RewardObservation(coherence=0.82))

bundle = build_supervisor_candidate_bundle(
    KnobPolicyCandidate(alpha=0.2, zeta=0.05),       # candidate
    KnobPolicyCandidate(alpha=0.0, zeta=0.0),        # baseline for attribution
    KnobPolicyCandidate(alpha=0.1, zeta=0.02),       # incumbent to compare against
    evaluate,
    observations=[RewardObservation(coherence=0.82, lyapunov_exponent=-0.02)],
    constraints=SafetyConstraintConfig(max_lyapunov_exponent=0.0),
    safety_tier="research",
    numeric_provenance=NumericProvenance("python", 1e-9),
)

record = bundle.to_audit_record()
assert record["schema"] == "studio.supervisor_candidate.v1"
assert record["digest"] == bundle.digest
# One flag a reviewer can gate on: is it safe AND an improvement?
print(bundle.safe_and_improved)
```

## What the bundle is, and is not

The bundle is *evidence*, not an action. It does not promote a candidate, drive a
supervisor, or touch hardware; it is the record a separate, human review consumes
before a candidate is promoted. Keeping the assembly side-effect free is what lets
the same bundle be produced in replay, in a notebook, or in CI without any of them
implying a control decision.

The comparison against the incumbent is deliberately lightweight — a difference of
two reward reports under the same evaluator — rather than a learned-policy
comparison, so the bundle stays free of the optional differentiable-learning
dependencies and reduces to plain arithmetic a reviewer can re-check by hand.

## Numeric provenance and the studio contract

`numeric_provenance` and `safety_tier` are first-class fields, not afterthoughts:
the same numbers can come from any of the polyglot compute backends, so the bundle
records which backend produced them and the parity tolerance they were held to,
and it carries the deployment safety tier the candidate targets. Together with the
evidence modality from the safety certificate, these are the fields the
cross-studio `studio.*.v1` evidence contract depends on.

::: scpn_phase_orchestrator.autotune.supervisor_candidate
