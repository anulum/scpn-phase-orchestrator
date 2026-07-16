# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-SINDy discovery confidence

"""Honest confidence classification for phase-SINDy discovery.

A phase-SINDy fit recovers a Kuramoto-style coupling structure from the
operator's *own* time series. Fitting a model to the same data it was learnt
from is self-consistency, not independent validation, so this classifier is
built so that it *cannot* award the ``externally_validated`` tier — that tier
is reserved for clearing an independent-reference test on real data, which a
self-fit can never do. The ceiling here is ``partial``; the honest default is
``scaffold``.

The classifier is pure: it reads the numeric summary of a fit (its status, R²,
sample and node counts, and term counts) and returns a tier plus a discovery
*posture* with human-readable reasons. It performs no I/O, no fitting, and no
mutation, so it is trivially testable and deterministic.

Postures
--------
``discovered``
    A fit was performed, explains the data well (R² at or above the policy
    threshold), is well determined (enough derivative samples per parameter),
    and selected at least one active term. Tier ``partial``.
``insufficient_evidence``
    A fit was performed but the evidence is too weak to stand behind the
    recovered structure (poor R², under-determined, or no active terms). Tier
    ``scaffold``.
``refused``
    No fit was performed at all (the discovery step skipped this library).
    Tier ``scaffold``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from scpn_phase_orchestrator.binding.types import (
    VALIDATION_TIER_PARTIAL,
    VALIDATION_TIER_SCAFFOLD,
)

POSTURE_DISCOVERED = "discovered"
POSTURE_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
POSTURE_REFUSED = "refused"

FITTED_STATUS = "fitted"


@dataclass(frozen=True)
class SindyConfidencePolicy:
    """Thresholds that separate a credible discovery from weak evidence.

    Parameters
    ----------
    min_r_squared : float
        Smallest coefficient of determination a fit must reach before its
        recovered structure may be called ``discovered``. The default of
        ``0.9`` demands the model explain the large majority of the derivative
        variance.
    min_samples_per_parameter : float
        Smallest ratio of regressed derivative samples to per-node parameters a
        fit must reach before it is considered well determined. The default of
        ``5.0`` keeps the per-node regression comfortably over-determined.
    """

    min_r_squared: float = 0.9
    min_samples_per_parameter: float = 5.0


DEFAULT_SINDY_CONFIDENCE_POLICY = SindyConfidencePolicy()


@dataclass(frozen=True)
class SindyConfidence:
    """The honest confidence verdict for a single phase-SINDy fit.

    Parameters
    ----------
    tier : str
        Validation tier, drawn from the canonical vocabulary. Never
        ``externally_validated`` — a self-fit cannot earn it.
    posture : str
        Discovery posture: ``discovered``, ``insufficient_evidence`` or
        ``refused``.
    r_squared : float or None
        The scale-free fit quality the verdict was based on, or ``None`` when
        no fit was performed.
    samples_per_parameter : float or None
        Regressed derivative samples per per-node parameter, or ``None`` when
        no fit was performed or the parameter count was unknown.
    reasons : tuple of str
        Human-readable justifications for the verdict, in evaluation order.
    """

    tier: str
    posture: str
    r_squared: float | None
    samples_per_parameter: float | None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_audit_record(self) -> dict[str, Any]:
        """Return the JSON-safe confidence record.

        Returns
        -------
        dict
            A JSON-serialisable mapping of the verdict fields.
        """
        return {
            "tier": self.tier,
            "posture": self.posture,
            "r_squared": self.r_squared,
            "samples_per_parameter": self.samples_per_parameter,
            "reasons": list(self.reasons),
        }


def classify_phase_sindy_confidence(
    *,
    status: str,
    r_squared: float | None,
    sample_count: int,
    node_count: int,
    active_terms: int,
    total_terms: int,
    sparsity: float,
    policy: SindyConfidencePolicy = DEFAULT_SINDY_CONFIDENCE_POLICY,
) -> SindyConfidence:
    """Classify a phase-SINDy fit into an honest tier and discovery posture.

    Parameters
    ----------
    status : str
        The fit status from the discovery block. Any value other than
        ``"fitted"`` is a skip and yields the ``refused`` posture.
    r_squared : float or None
        The scale-free coefficient of determination of the fit, or ``None``
        when no fit was performed.
    sample_count : int
        Number of derivative samples actually regressed.
    node_count : int
        Number of oscillator nodes; equal to the per-node parameter count of
        the Kuramoto sine-difference library.
    active_terms : int
        Number of coefficients selected above the sparsity threshold.
    total_terms : int
        Total number of coefficients in the library.
    sparsity : float
        Support-sparsity fraction of the fit; carried through for the record
        but not itself a gate.
    policy : SindyConfidencePolicy, optional
        Thresholds separating a credible discovery from weak evidence.

    Returns
    -------
    SindyConfidence
        The tier, posture, the quantities the verdict rested on, and the
        ordered reasons.
    """
    if status != FITTED_STATUS:
        return SindyConfidence(
            tier=VALIDATION_TIER_SCAFFOLD,
            posture=POSTURE_REFUSED,
            r_squared=None,
            samples_per_parameter=None,
            reasons=(f"no phase-SINDy fit was performed (status={status!r})",),
        )

    samples_per_parameter = (
        None if node_count <= 0 else float(sample_count) / float(node_count)
    )

    reasons: list[str] = []
    if active_terms <= 0:
        reasons.append("the fit selected no active terms above the sparsity threshold")
    if r_squared is None:
        reasons.append("the fit reported no R² to judge explanatory power")
    elif r_squared < policy.min_r_squared:
        reasons.append(
            f"R² {r_squared:.4f} is below the discovery threshold "
            f"{policy.min_r_squared:.4f}"
        )
    if samples_per_parameter is None:
        reasons.append("the per-node parameter count is unknown")
    elif samples_per_parameter < policy.min_samples_per_parameter:
        reasons.append(
            f"under-determined: {samples_per_parameter:.2f} derivative samples "
            f"per parameter is below the required "
            f"{policy.min_samples_per_parameter:.2f}"
        )

    if reasons:
        return SindyConfidence(
            tier=VALIDATION_TIER_SCAFFOLD,
            posture=POSTURE_INSUFFICIENT_EVIDENCE,
            r_squared=r_squared,
            samples_per_parameter=samples_per_parameter,
            reasons=tuple(reasons),
        )

    # An empty reasons list means every gate above passed, which in turn means
    # r_squared and samples_per_parameter are both non-None floats; the checks
    # accumulate reasons rather than narrow, so cast to record that here.
    resolved_r_squared = cast(float, r_squared)
    resolved_samples_per_parameter = cast(float, samples_per_parameter)
    return SindyConfidence(
        tier=VALIDATION_TIER_PARTIAL,
        posture=POSTURE_DISCOVERED,
        r_squared=resolved_r_squared,
        samples_per_parameter=resolved_samples_per_parameter,
        reasons=(
            "self-consistent recovery: the fit explains the derivative variance "
            f"(R² {resolved_r_squared:.4f}) and is over-determined "
            f"({resolved_samples_per_parameter:.2f} samples per parameter); tier "
            "is capped at 'partial' because a self-fit is not external validation",
        ),
    )


def classify_phase_sindy_block(
    block: Mapping[str, Any],
    *,
    policy: SindyConfidencePolicy = DEFAULT_SINDY_CONFIDENCE_POLICY,
) -> SindyConfidence:
    """Classify a phase-SINDy evidence block mapping.

    A thin, pure adapter over :func:`classify_phase_sindy_confidence` that
    reads the fields emitted by the discovery report's ``phase_sindy`` block.

    Parameters
    ----------
    block : Mapping
        A ``phase_sindy`` evidence block carrying at least ``status``; fitted
        blocks additionally carry ``r_squared``, ``sample_count``,
        ``node_count``, ``active_terms``, ``total_terms`` and ``sparsity``.
    policy : SindyConfidencePolicy, optional
        Thresholds separating a credible discovery from weak evidence.

    Returns
    -------
    SindyConfidence
        The honest confidence verdict for the block.
    """
    status = str(block.get("status", ""))
    raw_r_squared = block.get("r_squared")
    r_squared = None if raw_r_squared is None else float(raw_r_squared)
    return classify_phase_sindy_confidence(
        status=status,
        r_squared=r_squared,
        sample_count=int(block.get("sample_count", 0)),
        node_count=int(block.get("node_count", 0)),
        active_terms=int(block.get("active_terms", 0)),
        total_terms=int(block.get("total_terms", 0)),
        sparsity=float(block.get("sparsity", 1.0)),
        policy=policy,
    )
