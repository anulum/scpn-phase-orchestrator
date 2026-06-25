# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Autotune candidate safety certificate API reference

# Autotune Candidate Safety Certificate

A reward report and a per-knob attribution say whether a candidate is *good* and
*why*. They do not say whether it is *safe*. Before a candidate goes anywhere near
a plant, a reviewer needs the safety question answered with evidence rather than a
clamp: did the system stay inside the safe set under this candidate, with what
margin, and is that margin a measured replay outcome or a proven forward-invariant
one?

`certify_candidate_safety` produces that evidence by combining three sources into
one content-addressed certificate:

- **Barrier margin** — the worst value of a control-barrier function `h(x)` over
  the states the candidate visited in replay. A non-negative worst margin means
  the system never left the safe set `{x : h(x) >= 0}`; the number of violating
  states is reported alongside it.
- **Constraint margins** — the worst Lyapunov-exponent, STL-robustness, and
  safety-cost margins over the replay observations against the bounds in a
  `SafetyConstraintConfig`, with the proposal gate's require-evidence semantics: a
  required constraint whose evidence is missing fails closed.
- **Forward invariance** — an optional `BarrierCertificate` from a
  forward-invariance verification. When it is present, verified, and the replay
  stayed within its certified shell, the certificate's evidence is reported as
  *formally proven*; otherwise it is a *measured* replay margin. The certificate
  also carries `filter_digest` and `verification_digest` fields so a runtime CBF
  caller can reject stale or mismatched neural-barrier evidence.

```python
import numpy as np

from scpn_phase_orchestrator.actuation.control_barrier import NeuralBarrier
from scpn_phase_orchestrator.autotune import (
    KnobPolicyCandidate,
    RewardObservation,
    SafetyConstraintConfig,
    certify_candidate_safety,
)

candidate = KnobPolicyCandidate(alpha=0.1, zeta=0.05)
# h(x) = x[0]; the safe set is {x : x[0] >= 0}.
barrier = NeuralBarrier(weights=(np.array([[1.0]]),), biases=(np.array([0.0]),))

certificate = certify_candidate_safety(
    candidate,
    [
        RewardObservation(coherence=0.82, lyapunov_exponent=-0.02, safety_cost=0.01),
        RewardObservation(coherence=0.85, lyapunov_exponent=-0.03, safety_cost=0.02),
    ],
    SafetyConstraintConfig(max_lyapunov_exponent=0.0, max_safety_cost=0.1),
    barrier=barrier,
    replay_states=np.array([[0.5], [0.7]]),
)

assert certificate.safe is True
assert certificate.evidence_kind == "measured"  # no forward-invariance supplied
record = certificate.to_audit_record()
assert record["digest"] == certificate.digest
```

## Why a certificate, not a clamp

A control-barrier *filter* keeps a single live action safe by projecting it onto
the admissible set. That is the right tool at run time, but it leaves no reviewable
statement about a *candidate*: a reviewer cannot tell, from a clamp, whether the
candidate was already safe or was silently corrected on every step. The
certificate is the complementary artefact — it records the margin the candidate
actually held over a replay and, where a forward-invariance proof covers that
replay, upgrades the claim from measured to formally proven. That distinction is
deliberately preserved: a proven margin and a measured one are different
*modalities* of evidence, not different grades of the same number.

## Where this sits in the autotune track

The certificate consumes the same candidate surface as
[Reward Evaluation](autotune_reward.md) and
[Per-Knob Attribution](autotune_knob_attribution.md). Together they answer a
reviewer's three questions about a candidate — is it good, why, and is it safe —
as evidence rather than action: the function actuates nothing and only emits a
sealed record.

::: scpn_phase_orchestrator.autotune.candidate_safety_certificate
