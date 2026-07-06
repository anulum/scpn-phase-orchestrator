# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — sealed, review-only grid early-warning advisory

"""A hash-sealed, review-only operator advisory for a live grid instability alarm.

This is the step from a live alarm to a *decision surface* — the pinnacle the streaming
monitor was built toward — done honestly. When the certified streaming monitor
(:class:`~scpn_phase_orchestrator.monitor.grid_modal_stream.GridModalStreamMonitor`)
raises a :class:`~scpn_phase_orchestrator.monitor.grid_modal_stream.StreamAlarm`, this
module turns it into a claim-bounded advisory record an operator can read: the growth
rate ``σ`` that crossed the certified threshold, the most-unstable bus, the alarm time,
the certified operating point, and — as a first-class sealed field — the detector's
**honest recall**, so the reader knows how much the detector misses.

The advisory is **passive and review-only**. It never actuates: every record carries
``non_actuating = True`` and ``actuating = False``, the same fail-closed stance the STL
runtime actuation gate takes. It exists to *inform* a human decision, not to make one.
The sealed recall is the point: at its certified streaming operating point the detector
leads only about a quarter of growing-instability episodes at a matched ten-percent
stream false alarm, so an advisory is a reason to look, never a guarantee, and —
critically — the *absence* of an advisory is not evidence of stability.

The record is content-addressed with the same canonical-JSON SHA-256 seal the
early-warning evidence, the assurance-case bundle, and the NERC PRC oscillation evidence
use (:func:`~scpn_phase_orchestrator.assurance._hashing.canonical_record_hash`). The
capture timestamp and any ground-truth onset are supplied by the caller (they are
properties of the measured event, not wall-clock readings taken here), so the record is
deterministic and reproducible; a reported lead is honest, including a non-positive lead
when the alarm was coincident with or later than the onset.

:func:`seal_grid_early_warning_advisory` is the neutral primitive;
:func:`advise_from_stream_alarm` is the thin adapter that reads the alarm claims and the
certified operating point straight off a live monitor.

References
----------
* Kundur 1994, *Power System Stability and Control* — small-signal (modal) stability:
  the growth rate ``σ`` the advisory surfaces is the dominant mode's eigenvalue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TYPE_CHECKING

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.monitor.grid_modal_stream import WHOLE_NETWORK_BUS

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from scpn_phase_orchestrator.monitor.grid_modal_stream import (
        GridModalStreamMonitor,
        StreamAlarm,
    )

__all__ = [
    "GRID_EARLY_WARNING_DISCLAIMER",
    "GRID_EARLY_WARNING_FRAMEWORK",
    "GRID_EARLY_WARNING_OBSERVABLE",
    "GridEarlyWarningAdvisory",
    "advise_from_stream_alarm",
    "seal_grid_early_warning_advisory",
]

#: Verdict on every sealed advisory (an advisory exists only for a raised alarm).
GRID_ADVISORY_RAISED = "grid_early_warning_advisory_raised"

#: The wide-area-monitoring instability quantity the advisory surfaces.
GRID_EARLY_WARNING_OBSERVABLE = (
    "exponential growth rate σ of the most unstable bus's cross-bus voltage deviation "
    "(the dominant electromechanical mode's eigenvalue real part)"
)

#: The stability framework the advisory maps to.
GRID_EARLY_WARNING_FRAMEWORK = (
    "Small-signal (modal) power-system stability (Kundur 1994): a growing mode is a "
    "positive eigenvalue real part, certified through a matched-false-alarm moat"
)

#: Review-only claim boundary carried by every sealed advisory.
GRID_EARLY_WARNING_DISCLAIMER = (
    "This grid early-warning advisory is a passive, review-only technical artefact. It "
    "surfaces a certified streaming detector's alarm — the growth rate σ, the most "
    "unstable bus, and the certified operating point — to inform a human operator's "
    "review; it never actuates and constitutes no operational, dispatch, or safety "
    "decision, nor any certification of compliance. The sealed certified recall is "
    "well below one, so an advisory is a reason to inspect and the absence of an "
    "advisory is not evidence of stability. Any reported lead is measured against a "
    "caller-supplied ground-truth onset and holds only for the recorded operating "
    "point; a non-positive lead means the alarm was at or after the onset."
)


@dataclass(frozen=True, slots=True)
class GridEarlyWarningAdvisory:
    """A hash-sealed, review-only grid early-warning operator advisory.

    Attributes
    ----------
    detector : str
        The detector family label, e.g. ``grid_modal_growth_stream``.
    observable : str
        The physical quantity the detector read.
    signal_source : str
        Provenance identifier of the live stream (feed, event, or scenario).
    captured_at : str
        Measurement timestamp of the alarm, supplied by the caller.
    sampling_rate_hz : float
        Stream sampling rate in hertz.
    window_seconds, step_seconds : float
        The certified streaming operating point: window length and re-scoring hop.
    persistence : int
        Consecutive above-threshold windows required before the alarm fired.
    aggregation : str
        The certified aggregation (``"focal"`` or ``"mean"``).
    recency_top : float
        The certified recency weighting the growth rate was fitted under.
    r2_gate : float
        The certified fit-quality gate; ``0.0`` when off.
    warning_sample : int
        Stream sample index the alarm fired at.
    warning_time_s : float
        Alarm time in seconds from the stream start.
    growth_rate : float
        The growth rate ``σ`` at the alarm window.
    growth_rate_threshold : float
        The certified matched-false-alarm threshold ``σ`` crossed.
    most_unstable_bus : int
        The most-unstable bus, or :data:`WHOLE_NETWORK_BUS` under the mean aggregation.
    transition_onset_sample : int | None
        Caller-supplied ground-truth onset sample, or ``None`` when unknown.
    lead_samples : int | None
        ``transition_onset_sample - warning_sample`` when both are known, else ``None``.
    lead_seconds : float | None
        ``lead_samples / sampling_rate_hz`` when defined, else ``None``.
    lead_is_early : bool
        ``True`` only when the lead is defined and strictly positive.
    certified_recall : float
        The honest fraction of growing-instability episodes the detector leads at the
        certified operating point — sealed so the operator sees the miss rate.
    certified_false_alarm : float
        The matched stream false-alarm rate the threshold was certified at.
    certified_operating_point : str
        Provenance of the certified operating point (the sealed artefact it came from).
    non_actuating : bool
        Always ``True``: the advisory never actuates.
    actuating : bool
        Always ``False``: no actuation path exists.
    verdict : str
        :data:`GRID_ADVISORY_RAISED`.
    framework : str
        The stability framework the advisory maps to.
    disclaimer : str
        The review-only claim boundary.
    content_hash : str
        SHA-256 of the canonical record (excluding this field); set on construction.
    """

    detector: str
    observable: str
    signal_source: str
    captured_at: str
    sampling_rate_hz: float
    window_seconds: float
    step_seconds: float
    persistence: int
    aggregation: str
    recency_top: float
    r2_gate: float
    warning_sample: int
    warning_time_s: float
    growth_rate: float
    growth_rate_threshold: float
    most_unstable_bus: int
    transition_onset_sample: int | None
    lead_samples: int | None
    lead_seconds: float | None
    lead_is_early: bool
    certified_recall: float
    certified_false_alarm: float
    certified_operating_point: str
    non_actuating: bool
    actuating: bool
    verdict: str
    framework: str
    disclaimer: str
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Compute the content hash from the canonical advisory payload."""
        object.__setattr__(
            self, "content_hash", canonical_record_hash(self._canonical_payload())
        )

    def _canonical_payload(self) -> dict[str, object]:
        """Return the canonical payload for the sealed advisory."""
        return {
            "detector": self.detector,
            "observable": self.observable,
            "signal_source": self.signal_source,
            "captured_at": self.captured_at,
            "sampling_rate_hz": self.sampling_rate_hz,
            "window_seconds": self.window_seconds,
            "step_seconds": self.step_seconds,
            "persistence": self.persistence,
            "aggregation": self.aggregation,
            "recency_top": self.recency_top,
            "r2_gate": self.r2_gate,
            "warning_sample": self.warning_sample,
            "warning_time_s": self.warning_time_s,
            "growth_rate": self.growth_rate,
            "growth_rate_threshold": self.growth_rate_threshold,
            "most_unstable_bus": self.most_unstable_bus,
            "transition_onset_sample": self.transition_onset_sample,
            "lead_samples": self.lead_samples,
            "lead_seconds": self.lead_seconds,
            "lead_is_early": self.lead_is_early,
            "certified_recall": self.certified_recall,
            "certified_false_alarm": self.certified_false_alarm,
            "certified_operating_point": self.certified_operating_point,
            "non_actuating": self.non_actuating,
            "actuating": self.actuating,
            "verdict": self.verdict,
            "framework": self.framework,
            "disclaimer": self.disclaimer,
        }

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the whole sealed advisory.

        Returns
        -------
        dict[str, object]
            The canonical payload plus the computed ``content_hash``.
        """
        record = self._canonical_payload()
        record["content_hash"] = self.content_hash
        return record


def seal_grid_early_warning_advisory(
    *,
    detector: str,
    observable: str,
    signal_source: str,
    captured_at: str,
    sampling_rate_hz: float,
    window_seconds: float,
    step_seconds: float,
    persistence: int,
    aggregation: str,
    recency_top: float,
    r2_gate: float,
    warning_sample: int,
    warning_time_s: float,
    growth_rate: float,
    growth_rate_threshold: float,
    most_unstable_bus: int,
    certified_recall: float,
    certified_false_alarm: float,
    certified_operating_point: str,
    transition_onset_sample: int | None = None,
) -> GridEarlyWarningAdvisory:
    """Seal a live grid instability alarm into a hash-addressed, review-only advisory.

    The neutral primitive: it depends only on the alarm claims and the certified
    operating point, so it seals any monitor configuration without importing its
    internals. It hard-wires the non-actuating stance and computes the honest lead.

    Parameters
    ----------
    detector, observable, signal_source, captured_at, certified_operating_point : str
        The detector label, the read quantity, the stream provenance, the
        caller-supplied timestamp, and the operating-point provenance; each non-empty.
    sampling_rate_hz, window_seconds, step_seconds : float
        The stream rate and the operating-point window and hop in seconds; each ``> 0``.
    persistence : int
        The certified persistence; a positive integer.
    aggregation : str
        ``"focal"`` or ``"mean"``.
    recency_top : float
        The certified recency weighting; a finite number ``>= 1``.
    r2_gate, certified_recall, certified_false_alarm : float
        The certified fit-quality gate and the honest recall and false-alarm rate; each
        a finite number in ``[0, 1]``.
    warning_sample : int
        The alarm's stream sample index; non-negative.
    warning_time_s, growth_rate, growth_rate_threshold : float
        The alarm time and the growth rate and threshold; each finite.
    most_unstable_bus : int
        The most-unstable bus, or :data:`WHOLE_NETWORK_BUS` under the mean aggregation.
    transition_onset_sample : int | None
        Caller-supplied ground-truth onset sample; enables the lead computation.

    Returns
    -------
    GridEarlyWarningAdvisory
        The hash-sealed, review-only advisory.

    Raises
    ------
    ValueError
        If an identifier is empty, a rate or window is not positive, ``persistence`` is
        not a positive integer, ``aggregation`` is unknown, a bounded rate leaves
        ``[0, 1]``, ``recency_top`` is below one, a sample index is negative, the bus is
        below the whole-network sentinel, or a reported real is not finite.
    """
    fs = _positive_real(sampling_rate_hz, "sampling_rate_hz")
    onset = _optional_non_negative_int(
        transition_onset_sample, "transition_onset_sample"
    )
    sample_idx = _non_negative_int(warning_sample, "warning_sample")
    if aggregation not in ("focal", "mean"):
        raise ValueError(f"aggregation must be 'mean' or 'focal', got {aggregation!r}")
    bus = _finite_int(most_unstable_bus, "most_unstable_bus")
    if bus < WHOLE_NETWORK_BUS:
        raise ValueError(f"most_unstable_bus must be >= {WHOLE_NETWORK_BUS}, got {bus}")
    lead_samples = None if onset is None else onset - sample_idx
    lead_seconds = None if lead_samples is None else lead_samples / fs
    lead_is_early = lead_samples is not None and lead_samples > 0

    return GridEarlyWarningAdvisory(
        detector=_non_empty_str(detector, "detector"),
        observable=_non_empty_str(observable, "observable"),
        signal_source=_non_empty_str(signal_source, "signal_source"),
        captured_at=_non_empty_str(captured_at, "captured_at"),
        sampling_rate_hz=fs,
        window_seconds=_positive_real(window_seconds, "window_seconds"),
        step_seconds=_positive_real(step_seconds, "step_seconds"),
        persistence=_positive_int(persistence, "persistence"),
        aggregation=aggregation,
        recency_top=_recency(recency_top),
        r2_gate=_unit_interval(r2_gate, "r2_gate"),
        warning_sample=sample_idx,
        warning_time_s=_finite_real(warning_time_s, "warning_time_s"),
        growth_rate=_finite_real(growth_rate, "growth_rate"),
        growth_rate_threshold=_finite_real(
            growth_rate_threshold, "growth_rate_threshold"
        ),
        most_unstable_bus=bus,
        transition_onset_sample=onset,
        lead_samples=lead_samples,
        lead_seconds=lead_seconds,
        lead_is_early=lead_is_early,
        certified_recall=_unit_interval(certified_recall, "certified_recall"),
        certified_false_alarm=_unit_interval(
            certified_false_alarm, "certified_false_alarm"
        ),
        certified_operating_point=_non_empty_str(
            certified_operating_point, "certified_operating_point"
        ),
        non_actuating=True,
        actuating=False,
        verdict=GRID_ADVISORY_RAISED,
        framework=GRID_EARLY_WARNING_FRAMEWORK,
        disclaimer=GRID_EARLY_WARNING_DISCLAIMER,
    )


def advise_from_stream_alarm(
    alarm: StreamAlarm,
    monitor: GridModalStreamMonitor,
    *,
    signal_source: str,
    captured_at: str,
    certified_recall: float,
    certified_false_alarm: float,
    certified_operating_point: str,
    observable: str = GRID_EARLY_WARNING_OBSERVABLE,
    detector: str = "grid_modal_growth_stream",
    transition_onset_sample: int | None = None,
) -> GridEarlyWarningAdvisory:
    """Seal an advisory straight from a live monitor's alarm and operating point.

    Reads the alarm claims (growth rate, threshold, most-unstable bus, sample and time)
    off the :class:`~scpn_phase_orchestrator.monitor.grid_modal_stream.StreamAlarm` and
    the certified operating point (rate, window, step, persistence, aggregation, recency
    weighting, gate) off the live monitor, so the advisory records exactly what fired
    with no hand-set constants beyond the caller-supplied provenance and honest rates.

    Parameters
    ----------
    alarm : StreamAlarm
        The lead event the monitor raised.
    monitor : GridModalStreamMonitor
        The monitor that raised it, read for its certified operating point.
    signal_source, captured_at, certified_operating_point : str
        The stream provenance, the caller-supplied capture timestamp, and the
        certified-operating-point provenance.
    certified_recall, certified_false_alarm : float
        The honest recall and matched false-alarm rate of the certified operating point.
    observable, detector : str
        The read-quantity and detector labels sealed into the record.
    transition_onset_sample : int | None
        Caller-supplied ground-truth onset sample; enables the lead computation.

    Returns
    -------
    GridEarlyWarningAdvisory
        The hash-sealed, review-only advisory for the alarm.
    """
    return seal_grid_early_warning_advisory(
        detector=detector,
        observable=observable,
        signal_source=signal_source,
        captured_at=captured_at,
        sampling_rate_hz=monitor.rate,
        window_seconds=monitor.window_seconds,
        step_seconds=monitor.step_seconds,
        persistence=monitor.persistence,
        aggregation=monitor.aggregation,
        recency_top=monitor.recency_top,
        r2_gate=monitor.r2_gate,
        warning_sample=alarm.sample_index,
        warning_time_s=alarm.time_s,
        growth_rate=alarm.score,
        growth_rate_threshold=alarm.threshold,
        most_unstable_bus=alarm.bus,
        certified_recall=certified_recall,
        certified_false_alarm=certified_false_alarm,
        certified_operating_point=certified_operating_point,
        transition_onset_sample=transition_onset_sample,
    )


def _non_empty_str(value: object, name: str) -> str:
    """Return ``value`` as a non-empty string, else raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_real(value: object, name: str) -> float:
    """Return ``value`` as a finite real scalar, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar


def _positive_real(value: object, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    scalar = _finite_real(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _unit_interval(value: object, name: str) -> float:
    """Return ``value`` as a finite real in ``[0, 1]``, else raise ``ValueError``."""
    scalar = _finite_real(value, name)
    if not 0.0 <= scalar <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {scalar}")
    return scalar


def _recency(value: object) -> float:
    """Return ``value`` as a finite recency weighting ``>= 1``, else raise."""
    scalar = _finite_real(value, "recency_top")
    if scalar < 1.0:
        raise ValueError(f"recency_top must be at least one, got {scalar}")
    return scalar


def _positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _finite_int(value: object, name: str) -> int:
    """Return ``value`` as an integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _non_negative_int(value: object, name: str) -> int:
    """Return ``value`` as a non-negative integer, else raise ``ValueError``."""
    result = _finite_int(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _optional_non_negative_int(value: object, name: str) -> int | None:
    """Return ``value`` as a non-negative int or ``None``, else raise."""
    if value is None:
        return None
    return _non_negative_int(value, name)
