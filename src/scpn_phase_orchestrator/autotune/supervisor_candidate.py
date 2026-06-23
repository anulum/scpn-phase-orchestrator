# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — auditable supervisor-candidate bundle

"""Assemble the auditable supervisor-candidate evidence bundle.

A reviewer asked to trust a supervisor policy candidate has three questions, and
the autotune track answers each separately: is it *better* (the reward report and
a comparison against the incumbent), *why* these knobs (the per-knob attribution),
and is it *safe* (the candidate safety certificate). This module glues those
answers into one sealed, review-only artefact so the candidate carries its whole
case in a single record.

:func:`build_supervisor_candidate_bundle` scores the candidate, attributes it
against a baseline, certifies its safety, compares it against the incumbent it
would replace, and stamps the result with its numeric provenance (which compute
backend produced the numbers, and the parity tolerance they were checked to) and
its safety tier. The bundle is content-addressed by a canonical-JSON SHA-256 seal
and carries the safety certificate's evidence modality. It proposes nothing and
actuates nothing — it is the evidence a separate, human review consumes before a
candidate is promoted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.control_barrier import (
    BarrierCertificate,
    NeuralBarrier,
)
from scpn_phase_orchestrator.autotune.candidate_safety_certificate import (
    CandidateSafetyCertificate,
    certify_candidate_safety,
)
from scpn_phase_orchestrator.autotune.knob_attribution import (
    KnobAttributionConfig,
    KnobAttributionReport,
    attribute_knob_policy,
)
from scpn_phase_orchestrator.autotune.reward import (
    AutotuneRewardReport,
    KnobPolicyCandidate,
    RewardObservation,
    SafetyConstraintConfig,
)

__all__ = [
    "NumericProvenance",
    "SupervisorCandidateBundle",
    "SupervisorCandidateComparison",
    "build_supervisor_candidate_bundle",
]

CandidateEvaluator = Callable[[KnobPolicyCandidate], AutotuneRewardReport]
FloatArray = NDArray[np.float64]

_SCHEMA = "studio.supervisor_candidate.v1"


@dataclass(frozen=True)
class NumericProvenance:
    """Which compute backend produced a bundle's numbers, and to what tolerance.

    Parameters
    ----------
    active_backend : str
        The name of the compute backend that produced the numbers, e.g.
        ``"python"`` or ``"rust"``.
    parity_tolerance : float
        The non-negative absolute tolerance the backend was parity-checked to
        against the Python reference.
    """

    active_backend: str
    parity_tolerance: float

    def __post_init__(self) -> None:
        """Validate the provenance fields.

        Raises
        ------
        ValueError
            If ``active_backend`` is empty or ``parity_tolerance`` is negative.
        """
        if not self.active_backend:
            raise ValueError("active_backend must be a non-empty name")
        if self.parity_tolerance < 0.0:
            raise ValueError("parity_tolerance must be non-negative")

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-ready record of the provenance.

        Returns
        -------
        dict[str, object]
            A mapping with the active backend and the parity tolerance.
        """
        return {
            "active_backend": self.active_backend,
            "parity_tolerance": self.parity_tolerance,
        }


@dataclass(frozen=True)
class SupervisorCandidateComparison:
    """How a candidate's reward compares against the incumbent it would replace.

    Parameters
    ----------
    incumbent_reward : float
        The incumbent policy's total reward under the same evaluator.
    candidate_reward : float
        The candidate policy's total reward.
    reward_delta : float
        ``candidate_reward`` minus ``incumbent_reward``.
    component_deltas : Mapping[str, float]
        Per-component reward differences (candidate minus incumbent).
    improved : bool
        ``True`` when ``reward_delta`` is strictly positive.
    """

    incumbent_reward: float
    candidate_reward: float
    reward_delta: float
    component_deltas: Mapping[str, float]
    improved: bool

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-ready, deterministic record of the comparison.

        Returns
        -------
        dict[str, object]
            A mapping with the two rewards, the total and per-component deltas
            (sorted by component name), and the improvement flag.
        """
        return {
            "incumbent_reward": self.incumbent_reward,
            "candidate_reward": self.candidate_reward,
            "reward_delta": self.reward_delta,
            "component_deltas": {
                name: self.component_deltas[name]
                for name in sorted(self.component_deltas)
            },
            "improved": self.improved,
        }


@dataclass(frozen=True)
class SupervisorCandidateBundle:
    """The complete, sealed evidence bundle for one supervisor candidate.

    Parameters
    ----------
    candidate : KnobPolicyCandidate
        The candidate the bundle describes.
    reward : AutotuneRewardReport
        The candidate's reward report.
    attribution : KnobAttributionReport
        The per-knob attribution against the baseline.
    safety : CandidateSafetyCertificate
        The candidate's safety certificate.
    comparison : SupervisorCandidateComparison
        The comparison against the incumbent.
    safety_tier : str
        The declared safety tier of the deployment the candidate targets.
    numeric_provenance : NumericProvenance
        The compute provenance of the numbers in the bundle.
    evidence_kind : str
        The safety certificate's evidence modality, carried to the bundle level.
    digest : str
        Canonical-JSON SHA-256 content address of the bundle body.
    """

    candidate: KnobPolicyCandidate
    reward: AutotuneRewardReport
    attribution: KnobAttributionReport
    safety: CandidateSafetyCertificate
    comparison: SupervisorCandidateComparison
    safety_tier: str
    numeric_provenance: NumericProvenance
    evidence_kind: str
    digest: str

    @property
    def safe_and_improved(self) -> bool:
        """Return whether the candidate is both safe and an improvement.

        Returns
        -------
        bool
            ``True`` when the safety certificate is safe and the comparison
            shows an improvement over the incumbent.
        """
        return self.safety.safe and self.comparison.improved

    def to_audit_record(self) -> dict[str, object]:
        """Return the JSON-ready ``studio.supervisor_candidate.v1`` record.

        Returns
        -------
        dict[str, object]
            The schema-tagged bundle with the candidate, reward, attribution,
            safety, and comparison sub-records, the safety tier, the numeric
            provenance, the evidence kind, and the content-address digest.
        """
        body = _bundle_body(
            candidate=self.candidate,
            reward=self.reward,
            attribution=self.attribution,
            safety=self.safety,
            comparison=self.comparison,
            safety_tier=self.safety_tier,
            numeric_provenance=self.numeric_provenance,
            evidence_kind=self.evidence_kind,
        )
        return {**body, "digest": self.digest}


def _candidate_record(candidate: KnobPolicyCandidate) -> dict[str, object]:
    """Serialise a candidate's knobs to a JSON-ready record."""
    return {
        "K": np.asarray(candidate.K, dtype=float).ravel().tolist(),
        "alpha": np.asarray(candidate.alpha, dtype=float).ravel().tolist(),
        "zeta": np.asarray(candidate.zeta, dtype=float).ravel().tolist(),
        "Psi": np.asarray(candidate.Psi, dtype=float).ravel().tolist(),
        "channel_weights": [float(value) for value in candidate.channel_weights],
        "cross_channel_gains": [
            float(value) for value in candidate.cross_channel_gains
        ],
    }


def _bundle_body(
    *,
    candidate: KnobPolicyCandidate,
    reward: AutotuneRewardReport,
    attribution: KnobAttributionReport,
    safety: CandidateSafetyCertificate,
    comparison: SupervisorCandidateComparison,
    safety_tier: str,
    numeric_provenance: NumericProvenance,
    evidence_kind: str,
) -> dict[str, object]:
    """Build the deterministic, digest-free bundle body."""
    return {
        "schema": _SCHEMA,
        "candidate": _candidate_record(candidate),
        "reward": reward.to_audit_record(),
        "attribution": attribution.to_audit_record(),
        "safety": safety.to_audit_record(),
        "comparison": comparison.to_audit_record(),
        "safety_tier": safety_tier,
        "numeric_provenance": numeric_provenance.to_audit_record(),
        "evidence_kind": evidence_kind,
    }


def _seal(body: Mapping[str, object]) -> str:
    """Return the canonical-JSON SHA-256 digest of a bundle body."""
    serialised = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _compare(
    incumbent: AutotuneRewardReport,
    candidate: AutotuneRewardReport,
) -> SupervisorCandidateComparison:
    """Compare a candidate's reward report against the incumbent's."""
    names = set(candidate.components) | set(incumbent.components)
    component_deltas = {
        name: candidate.components.get(name, 0.0) - incumbent.components.get(name, 0.0)
        for name in names
    }
    reward_delta = candidate.reward - incumbent.reward
    return SupervisorCandidateComparison(
        incumbent_reward=incumbent.reward,
        candidate_reward=candidate.reward,
        reward_delta=reward_delta,
        component_deltas=component_deltas,
        improved=reward_delta > 0.0,
    )


def build_supervisor_candidate_bundle(
    candidate: KnobPolicyCandidate,
    baseline: KnobPolicyCandidate,
    incumbent: KnobPolicyCandidate,
    evaluate: CandidateEvaluator,
    *,
    observations: Sequence[RewardObservation],
    constraints: SafetyConstraintConfig,
    safety_tier: str,
    numeric_provenance: NumericProvenance,
    barrier: NeuralBarrier | None = None,
    replay_states: Sequence[FloatArray] | FloatArray | None = None,
    forward_invariance: BarrierCertificate | None = None,
    attribution_config: KnobAttributionConfig | None = None,
) -> SupervisorCandidateBundle:
    """Assemble the sealed, review-only supervisor-candidate evidence bundle.

    The candidate is scored, attributed against ``baseline``, certified for
    safety, and compared against ``incumbent`` under the same evaluator. The
    pieces are stamped with the numeric provenance and safety tier and sealed.
    The function proposes and actuates nothing.

    Parameters
    ----------
    candidate : KnobPolicyCandidate
        The candidate being bundled.
    baseline : KnobPolicyCandidate
        The reference the attribution credits knobs against.
    incumbent : KnobPolicyCandidate
        The policy the candidate would replace, scored for the comparison.
    evaluate : CandidateEvaluator
        A side-effect-free scorer returning a reward report for any candidate.
    observations : Sequence[RewardObservation]
        The replay observations driving the safety certificate.
    constraints : SafetyConstraintConfig
        The Lyapunov/STL/safety-cost bounds for the safety certificate.
    safety_tier : str
        The declared safety tier of the target deployment.
    numeric_provenance : NumericProvenance
        The compute provenance of the bundle's numbers.
    barrier : NeuralBarrier | None
        An optional control-barrier function for the safety certificate.
    replay_states : object
        The states the candidate visited, required when ``barrier`` is supplied.
    forward_invariance : BarrierCertificate | None
        An optional forward-invariance certificate for the barrier.
    attribution_config : KnobAttributionConfig | None
        Optional exact-versus-sampled settings for the attribution.

    Returns
    -------
    SupervisorCandidateBundle
        The sealed evidence bundle.
    """
    reward = evaluate(candidate)
    attribution = attribute_knob_policy(
        candidate, baseline, evaluate, config=attribution_config
    )
    safety = certify_candidate_safety(
        candidate,
        observations,
        constraints,
        barrier=barrier,
        replay_states=replay_states,
        forward_invariance=forward_invariance,
    )
    comparison = _compare(evaluate(incumbent), reward)
    body = _bundle_body(
        candidate=candidate,
        reward=reward,
        attribution=attribution,
        safety=safety,
        comparison=comparison,
        safety_tier=safety_tier,
        numeric_provenance=numeric_provenance,
        evidence_kind=safety.evidence_kind,
    )
    return SupervisorCandidateBundle(
        candidate=candidate,
        reward=reward,
        attribution=attribution,
        safety=safety,
        comparison=comparison,
        safety_tier=safety_tier,
        numeric_provenance=numeric_provenance,
        evidence_kind=safety.evidence_kind,
        digest=_seal(body),
    )
