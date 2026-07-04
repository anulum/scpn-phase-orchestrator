# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — NERC PRC oscillation-monitoring compliance evidence

"""NERC PRC oscillation-monitoring compliance evidence from detected modes.

This is the audit-package end of the dVOC grid pack. The matrix-pencil estimator
(:mod:`~scpn_phase_orchestrator.monitor.oscillation_modes`) detects the
electromechanical modes of a ringdown and their damping ratios;
:func:`screen_oscillation_modes` screens those damping ratios for NERC
PRC-028-1 disturbance-data analysis and PRC-030-1 unexpected IBR event mitigation
workflows: a mode whose damping ratio sits below a few percent is poorly damped,
and a mode with non-positive damping is undamped (growing). The screening also
preserves an engineering mode-family label for each finding, so inter-area and
sub-synchronous signals stay visible in the hash-sealed, review-only
:class:`PRCOscillationEvidence` record.

The record is content-addressed with the same canonical-JSON SHA-256 hashing the
assurance-case bundle uses, so it can be referenced by a stable digest and any
later mutation is detectable. The capture timestamp is supplied by the caller (it
is the measurement time of the PMU/ringdown event, not a wall-clock reading taken
here) so the record is deterministic and reproducible.

This is a technical evidence-mapping aid, not a legal conformity assessment: it
links measured modal damping to the disturbance-monitoring and unexpected-event
mitigation workflows those standards support. The exact identifiers, thresholds,
and reporting obligations must be confirmed against the issued standards — see
:data:`PRC_OSCILLATION_DISCLAIMER`. The screening is review-only: it reads
detected modes and reports findings; it never changes bindings, layers, or
coupling.

References
----------
* NERC PRC-028-1 (disturbance monitoring and reporting for inverter-based
  resources) and PRC-030-1 (unexpected inverter-based resource event mitigation),
  developed under FERC Order 901.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.monitor.oscillation_modes import (
    DEFAULT_DAMPING_THRESHOLD,
    OscillationMode,
    classify_oscillation_band,
)

__all__ = [
    "ACCEPTABLE",
    "FLAGGED_FOR_REVIEW",
    "NO_EXCEEDANCE",
    "POORLY_DAMPED",
    "PRC_OSCILLATION_DISCLAIMER",
    "PRC_OSCILLATION_STANDARD",
    "UNDAMPED",
    "PRCModeFinding",
    "PRCOscillationEvidence",
    "screen_oscillation_modes",
]

#: Mode classification: damping at or below the undamped threshold (growing).
UNDAMPED = "undamped"
#: Mode classification: positive damping below the poorly-damped threshold.
POORLY_DAMPED = "poorly_damped"
#: Mode classification: damping at or above the poorly-damped threshold.
ACCEPTABLE = "acceptable"

#: Verdict when one or more modes are flagged (undamped or poorly damped).
FLAGGED_FOR_REVIEW = "flagged_for_review"
#: Verdict when no mode breaches the screening thresholds.
NO_EXCEEDANCE = "no_exceedance"

#: Standard family the screening evidence is mapped to.
PRC_OSCILLATION_STANDARD = (
    "NERC PRC-028-1 / PRC-030-1 "
    "(IBR disturbance monitoring and unexpected event mitigation)"
)

#: Review-only disclaimer carried by every evidence record.
PRC_OSCILLATION_DISCLAIMER = (
    "This oscillation-monitoring evidence record is a technical evidence-mapping "
    "artifact. It screens measured modal damping for NERC PRC-028-1 disturbance "
    "monitoring and PRC-030-1 unexpected IBR event mitigation workflows; it does "
    "not constitute legal advice, a conformity assessment, or a certification of "
    "compliance. The exact clause identifiers, damping thresholds, and reporting "
    "obligations must be confirmed against the issued standards, and conformance "
    "determined by a qualified assessor."
)


@dataclass(frozen=True, slots=True)
class PRCModeFinding:
    """The screening outcome for a single detected mode.

    Attributes
    ----------
    mode_index : int
        Position of the mode in the screened sequence.
    frequency_hz : float
        Modal oscillation frequency in hertz.
    damping_ratio : float
        Dimensionless damping ratio of the mode.
    amplitude : float
        Modal amplitude in the units of the source signal.
    mode_family : str
        Engineering oscillation family such as ``inter_area`` or
        ``sub_synchronous``.
    classification : str
        One of :data:`UNDAMPED`, :data:`POORLY_DAMPED`, or :data:`ACCEPTABLE`.
    flagged : bool
        Whether the mode breaches a screening threshold (undamped or poorly
        damped).
    """

    mode_index: int
    frequency_hz: float
    damping_ratio: float
    amplitude: float
    mode_family: str
    classification: str
    flagged: bool

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the finding.

        Returns
        -------
        dict[str, object]
            The mode index, frequency, damping ratio, amplitude, classification,
            and flagged status.
        """
        return {
            "mode_index": self.mode_index,
            "frequency_hz": self.frequency_hz,
            "damping_ratio": self.damping_ratio,
            "amplitude": self.amplitude,
            "mode_family": self.mode_family,
            "classification": self.classification,
            "flagged": self.flagged,
        }


@dataclass(frozen=True, slots=True)
class PRCOscillationEvidence:
    """A hash-sealed oscillation-monitoring compliance-evidence record.

    Attributes
    ----------
    event_id : str
        Caller-assigned identifier for the oscillation event.
    captured_at : str
        Measurement timestamp of the event, supplied by the caller.
    signal_source : str
        Identifier of the screened signal (a bus, tie-line, or order parameter).
    sampling_rate_hz : float
        Sampling rate of the ringdown the modes were estimated from.
    poorly_damped_threshold : float
        Damping ratio below which a positively-damped mode is flagged.
    undamped_threshold : float
        Damping ratio at or below which a mode is flagged undamped (growing).
    findings : tuple[PRCModeFinding, ...]
        Per-mode screening outcomes, in the order screened.
    mode_family_counts : Mapping[str, int]
        Read-only number of screened modes by engineering family.
    flagged_count : int
        Number of findings flagged for review.
    worst_damping_ratio : float | None
        Lowest damping ratio across the findings, or ``None`` when no modes were
        detected.
    verdict : str
        :data:`FLAGGED_FOR_REVIEW` if any mode is flagged, else
        :data:`NO_EXCEEDANCE`.
    standard : str
        The standard family the record is mapped to.
    disclaimer : str
        The review-only regulatory disclaimer.
    content_hash : str
        SHA-256 of the canonical record (excluding this field); computed on
        construction.
    """

    event_id: str
    captured_at: str
    signal_source: str
    sampling_rate_hz: float
    poorly_damped_threshold: float
    undamped_threshold: float
    findings: tuple[PRCModeFinding, ...]
    mode_family_counts: Mapping[str, int]
    flagged_count: int
    worst_damping_ratio: float | None
    verdict: str
    standard: str
    disclaimer: str
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Compute the content hash from the canonical evidence payload."""
        object.__setattr__(
            self, "mode_family_counts", MappingProxyType(dict(self.mode_family_counts))
        )
        object.__setattr__(
            self, "content_hash", canonical_record_hash(self._canonical_payload())
        )

    def _canonical_payload(self) -> dict[str, object]:
        """Return the canonical payload for the assurance record."""
        return {
            "event_id": self.event_id,
            "captured_at": self.captured_at,
            "signal_source": self.signal_source,
            "sampling_rate_hz": self.sampling_rate_hz,
            "poorly_damped_threshold": self.poorly_damped_threshold,
            "undamped_threshold": self.undamped_threshold,
            "findings": [finding.to_audit_record() for finding in self.findings],
            "mode_family_counts": dict(self.mode_family_counts),
            "flagged_count": self.flagged_count,
            "worst_damping_ratio": self.worst_damping_ratio,
            "verdict": self.verdict,
            "standard": self.standard,
            "disclaimer": self.disclaimer,
        }

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the whole evidence record.

        Returns
        -------
        dict[str, object]
            The canonical payload plus the computed ``content_hash``.
        """
        record = self._canonical_payload()
        record["content_hash"] = self.content_hash
        return record


def screen_oscillation_modes(
    modes: Sequence[OscillationMode],
    *,
    event_id: str,
    captured_at: str,
    signal_source: str,
    sampling_rate_hz: float,
    poorly_damped_threshold: float = DEFAULT_DAMPING_THRESHOLD,
    undamped_threshold: float = 0.0,
) -> PRCOscillationEvidence:
    """Screen detected oscillation modes into a PRC compliance-evidence record.

    Parameters
    ----------
    modes : Sequence[OscillationMode]
        Modes recovered from a ringdown, e.g. by
        :func:`~scpn_phase_orchestrator.monitor.oscillation_modes.estimate_oscillation_modes`.
    event_id : str
        Caller-assigned identifier for the oscillation event.
    captured_at : str
        Measurement timestamp of the event, supplied by the caller.
    signal_source : str
        Identifier of the screened signal.
    sampling_rate_hz : float
        Sampling rate of the ringdown, in hertz (``> 0``).
    poorly_damped_threshold : float
        Damping ratio below which a positively-damped mode is flagged poorly
        damped.
    undamped_threshold : float
        Damping ratio at or below which a mode is flagged undamped; must be below
        ``poorly_damped_threshold``.

    Returns
    -------
    PRCOscillationEvidence
        The hash-sealed, review-only screening record.

    Raises
    ------
    ValueError
        If an identifier is empty, the sampling rate is not positive, the
        thresholds are not ordered finite reals, or an element is not an
        :class:`~scpn_phase_orchestrator.monitor.oscillation_modes.OscillationMode`.
    """
    event = _non_empty_str(event_id, "event_id")
    captured = _non_empty_str(captured_at, "captured_at")
    source = _non_empty_str(signal_source, "signal_source")
    fs = _positive_real(sampling_rate_hz, "sampling_rate_hz")
    undamped = _real_scalar(undamped_threshold, "undamped_threshold")
    poorly = _real_scalar(poorly_damped_threshold, "poorly_damped_threshold")
    if not undamped < poorly:
        raise ValueError("undamped_threshold must be below poorly_damped_threshold")

    findings = tuple(
        _screen_mode(index, mode, undamped, poorly) for index, mode in enumerate(modes)
    )
    flagged_count = sum(1 for finding in findings if finding.flagged)
    worst = min((finding.damping_ratio for finding in findings), default=None)
    verdict = FLAGGED_FOR_REVIEW if flagged_count else NO_EXCEEDANCE
    return PRCOscillationEvidence(
        event_id=event,
        captured_at=captured,
        signal_source=source,
        sampling_rate_hz=fs,
        poorly_damped_threshold=poorly,
        undamped_threshold=undamped,
        findings=findings,
        mode_family_counts=_mode_family_counts(findings),
        flagged_count=flagged_count,
        worst_damping_ratio=worst,
        verdict=verdict,
        standard=PRC_OSCILLATION_STANDARD,
        disclaimer=PRC_OSCILLATION_DISCLAIMER,
    )


def _screen_mode(
    index: int, mode: object, undamped: float, poorly: float
) -> PRCModeFinding:
    """Return the screening mode for the assessment."""
    if not isinstance(mode, OscillationMode):
        raise ValueError(f"modes[{index}] must be an OscillationMode, got {mode!r}")
    classification, flagged = _classify(mode.damping_ratio, undamped, poorly)
    return PRCModeFinding(
        mode_index=index,
        frequency_hz=mode.frequency_hz,
        damping_ratio=mode.damping_ratio,
        amplitude=mode.amplitude,
        mode_family=classify_oscillation_band(mode.frequency_hz),
        classification=classification,
        flagged=flagged,
    )


def _mode_family_counts(findings: Sequence[PRCModeFinding]) -> dict[str, int]:
    """Return deterministic mode-family counts for the evidence payload."""
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.mode_family] = counts.get(finding.mode_family, 0) + 1
    return counts


def _classify(damping: float, undamped: float, poorly: float) -> tuple[str, bool]:
    """Return the classification for the oscillation assessment."""
    if damping <= undamped:
        return UNDAMPED, True
    if damping < poorly:
        return POORLY_DAMPED, True
    return ACCEPTABLE, False


def _non_empty_str(value: object, name: str) -> str:
    """Return ``value`` as a non-empty string, else raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_real(value: object, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    scalar = _real_scalar(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _real_scalar(value: object, name: str) -> float:
    """Return ``value`` as a finite real scalar, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar
