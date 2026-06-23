# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — candidate safety certificate for autotune

"""Bind an autotune candidate to a safety certificate over a replay.

A reward report and a per-knob attribution say whether a candidate is *good* and
*why*. They do not say whether it is *safe*. Before a candidate is allowed near a
plant a reviewer needs the safety question answered with evidence, not a clamp:
did the system stay inside the safe set under this candidate, with what margin,
and is that margin a measured replay outcome or a proven forward-invariant one?

:func:`certify_candidate_safety` answers that by combining three sources:

* **Barrier margin** — the worst value of a control-barrier function ``h(x)`` over
  the states the candidate visited in replay. A non-negative worst margin means
  the system never left the safe set ``{x : h(x) >= 0}``; the count of violating
  states is reported alongside it.
* **Constraint margins** — the worst Lyapunov-exponent, STL-robustness, and
  safety-cost margins over the replay observations against the bounds in a
  :class:`~scpn_phase_orchestrator.autotune.reward.SafetyConstraintConfig`, with
  the same require-evidence semantics as the proposal gate: a required constraint
  whose evidence is missing fails closed.
* **Forward invariance** — an optional
  :class:`~scpn_phase_orchestrator.actuation.control_barrier.BarrierCertificate`
  from a forward-invariance verification. When it is present, verified, and the
  replay stayed within its certified shell, the certificate's evidence is
  *formally proven*; otherwise it is a *measured* replay margin.

The certificate is content-addressed (a canonical-JSON SHA-256 seal) and the
function performs no control actuation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.control_barrier import (
    BarrierCertificate,
    NeuralBarrier,
)
from scpn_phase_orchestrator.autotune.reward import (
    KnobPolicyCandidate,
    RewardObservation,
    SafetyConstraintConfig,
)

__all__ = [
    "CandidateSafetyCertificate",
    "certify_candidate_safety",
]

FloatArray = NDArray[np.float64]

_FORMALLY_PROVEN = "formally-proven"
_MEASURED = "measured"


@dataclass(frozen=True)
class CandidateSafetyCertificate:
    """A safety certificate for one candidate over a replay window.

    Parameters
    ----------
    candidate : KnobPolicyCandidate
        The certified candidate.
    barrier_worst_margin : float | None
        The smallest barrier value ``h(x)`` over the replay states, or ``None``
        when no barrier was supplied. Non-negative means the safe set was never
        left.
    barrier_violations : int
        The number of replay states with a negative barrier value.
    lyapunov_margin : float | None
        ``max_lyapunov_exponent`` minus the worst (largest) replay Lyapunov
        exponent, or ``None`` when the constraint is not configured.
    stl_margin : float | None
        The worst (smallest) replay STL robustness minus ``min_stl_robustness``,
        or ``None`` when the constraint is not configured.
    safety_cost_margin : float | None
        ``max_safety_cost`` minus the worst (largest) replay safety cost, or
        ``None`` when the constraint is not configured.
    constraint_verdicts : Mapping[str, bool]
        Pass/fail verdict for each of ``"lyapunov"``, ``"stl"`` and
        ``"safety_cost"``; a constraint that is not configured passes.
    forward_invariance_verified : bool | None
        The verification flag of a supplied forward-invariance certificate, or
        ``None`` when none was supplied.
    evidence_kind : str
        ``"formally-proven"`` when a verified forward-invariance certificate
        covers the replay, otherwise ``"measured"``.
    safe : bool
        ``True`` when there are no barrier violations and every constraint
        verdict passes.
    digest : str
        Canonical-JSON SHA-256 content address of the certificate body.
    """

    candidate: KnobPolicyCandidate
    barrier_worst_margin: float | None
    barrier_violations: int
    lyapunov_margin: float | None
    stl_margin: float | None
    safety_cost_margin: float | None
    constraint_verdicts: Mapping[str, bool]
    forward_invariance_verified: bool | None
    evidence_kind: str
    safe: bool
    digest: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-ready, deterministic record of the certificate.

        Returns
        -------
        dict[str, object]
            A mapping with the margins, per-constraint verdicts, evidence kind,
            overall safety verdict, and the content-address digest.
        """
        body = _certificate_body(
            barrier_worst_margin=self.barrier_worst_margin,
            barrier_violations=self.barrier_violations,
            lyapunov_margin=self.lyapunov_margin,
            stl_margin=self.stl_margin,
            safety_cost_margin=self.safety_cost_margin,
            constraint_verdicts=self.constraint_verdicts,
            forward_invariance_verified=self.forward_invariance_verified,
            evidence_kind=self.evidence_kind,
            safe=self.safe,
        )
        return {**body, "digest": self.digest}


def _certificate_body(
    *,
    barrier_worst_margin: float | None,
    barrier_violations: int,
    lyapunov_margin: float | None,
    stl_margin: float | None,
    safety_cost_margin: float | None,
    constraint_verdicts: Mapping[str, bool],
    forward_invariance_verified: bool | None,
    evidence_kind: str,
    safe: bool,
) -> dict[str, object]:
    """Build the deterministic, digest-free certificate body."""
    return {
        "barrier_worst_margin": barrier_worst_margin,
        "barrier_violations": barrier_violations,
        "lyapunov_margin": lyapunov_margin,
        "stl_margin": stl_margin,
        "safety_cost_margin": safety_cost_margin,
        "constraint_verdicts": {
            name: constraint_verdicts[name] for name in sorted(constraint_verdicts)
        },
        "forward_invariance_verified": forward_invariance_verified,
        "evidence_kind": evidence_kind,
        "safe": safe,
    }


def _seal(body: Mapping[str, object]) -> str:
    """Return the canonical-JSON SHA-256 digest of a certificate body."""
    serialised = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _barrier_margins(
    barrier: NeuralBarrier | None,
    replay_states: Sequence[FloatArray] | FloatArray | None,
) -> tuple[float | None, int]:
    """Worst barrier margin and violation count over the replay states."""
    if barrier is None:
        return None, 0
    if replay_states is None:
        raise ValueError("a barrier requires replay_states to evaluate")
    states = np.atleast_2d(np.asarray(replay_states, dtype=float))
    if states.shape[0] == 0:
        raise ValueError("replay_states must contain at least one state")
    margins = [float(barrier.value(np.asarray(row, dtype=float))) for row in states]
    worst = min(margins)
    violations = sum(1 for margin in margins if margin < 0.0)
    return worst, violations


def _bounded_constraint(
    threshold: float | None,
    *,
    required: bool,
    values: Sequence[float | None],
    higher_is_safe: bool,
) -> tuple[float | None, bool]:
    """Worst-case margin and verdict for one bounded safety constraint."""
    if threshold is None:
        return None, True
    present = [value for value in values if value is not None]
    if required and len(present) < len(values):
        return None, False
    if not present:
        return None, not required
    margin = min(present) - threshold if higher_is_safe else threshold - max(present)
    return margin, margin >= 0.0


def certify_candidate_safety(
    candidate: KnobPolicyCandidate,
    observations: Sequence[RewardObservation],
    constraints: SafetyConstraintConfig,
    *,
    barrier: NeuralBarrier | None = None,
    replay_states: Sequence[FloatArray] | FloatArray | None = None,
    forward_invariance: BarrierCertificate | None = None,
) -> CandidateSafetyCertificate:
    """Certify a candidate's safety over a replay window.

    The candidate is certified against a control-barrier function over the states
    it visited, against the Lyapunov, STL, and safety-cost bounds over the replay
    observations, and against an optional forward-invariance certificate. The
    result is content-addressed and carries no control action.

    Parameters
    ----------
    candidate : KnobPolicyCandidate
        The candidate being certified.
    observations : Sequence[RewardObservation]
        The per-step replay observations carrying the Lyapunov exponent, STL
        robustness, and safety cost. Must be non-empty.
    constraints : SafetyConstraintConfig
        The Lyapunov/STL/safety-cost bounds and their require-evidence flags.
    barrier : NeuralBarrier | None
        A control-barrier function ``h(x)``. When supplied, ``replay_states`` is
        required and the worst margin over those states is certified.
    replay_states : Sequence[FloatArray] | FloatArray | None
        The states the candidate visited, as a 2-D array or a sequence of state
        vectors. Required when ``barrier`` is supplied.
    forward_invariance : BarrierCertificate | None
        An optional forward-invariance certificate for the barrier. When it is
        verified and the replay stayed within its certified shell, the evidence
        is reported as formally proven.

    Returns
    -------
    CandidateSafetyCertificate
        The sealed certificate.

    Raises
    ------
    ValueError
        If ``observations`` is empty, or if ``barrier`` is supplied without
        ``replay_states`` (or with empty states).
    """
    if len(observations) == 0:
        raise ValueError("observations must be non-empty")

    barrier_worst_margin, barrier_violations = _barrier_margins(barrier, replay_states)

    lyapunov_margin, lyapunov_ok = _bounded_constraint(
        constraints.max_lyapunov_exponent,
        required=constraints.require_lyapunov,
        values=[obs.lyapunov_exponent for obs in observations],
        higher_is_safe=False,
    )
    stl_margin, stl_ok = _bounded_constraint(
        constraints.min_stl_robustness,
        required=constraints.require_stl,
        values=[obs.stl_robustness for obs in observations],
        higher_is_safe=True,
    )
    safety_cost_margin, safety_cost_ok = _bounded_constraint(
        constraints.max_safety_cost,
        required=constraints.require_safety_cost,
        values=[obs.safety_cost for obs in observations],
        higher_is_safe=False,
    )
    constraint_verdicts = {
        "lyapunov": lyapunov_ok,
        "stl": stl_ok,
        "safety_cost": safety_cost_ok,
    }

    forward_invariance_verified = (
        None if forward_invariance is None else forward_invariance.verified
    )
    within_shell = (
        forward_invariance is not None
        and forward_invariance.verified
        and barrier_worst_margin is not None
        and barrier_worst_margin >= forward_invariance.boundary_shell
    )
    evidence_kind = _FORMALLY_PROVEN if within_shell else _MEASURED

    safe = barrier_violations == 0 and all(constraint_verdicts.values())

    body = _certificate_body(
        barrier_worst_margin=barrier_worst_margin,
        barrier_violations=barrier_violations,
        lyapunov_margin=lyapunov_margin,
        stl_margin=stl_margin,
        safety_cost_margin=safety_cost_margin,
        constraint_verdicts=constraint_verdicts,
        forward_invariance_verified=forward_invariance_verified,
        evidence_kind=evidence_kind,
        safe=safe,
    )
    return CandidateSafetyCertificate(
        candidate=candidate,
        barrier_worst_margin=barrier_worst_margin,
        barrier_violations=barrier_violations,
        lyapunov_margin=lyapunov_margin,
        stl_margin=stl_margin,
        safety_cost_margin=safety_cost_margin,
        constraint_verdicts=constraint_verdicts,
        forward_invariance_verified=forward_invariance_verified,
        evidence_kind=evidence_kind,
        safe=safe,
        digest=_seal(body),
    )
