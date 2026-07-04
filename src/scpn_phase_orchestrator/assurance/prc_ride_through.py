# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — NERC PRC-029 ride-through evidence screening

"""Review-only NERC PRC-029 ride-through evidence screening.

This module maps operator-provided high-side transformer voltage and frequency
time series into deterministic, hash-sealed evidence for NERC PRC-029-1
review. It implements the published Attachment 1 voltage ride-through tables for
AC-connected wind IBRs and all other IBRs, plus the Attachment 2 frequency
ride-through table, then records only technical screening findings. It does not
assert conformance, evaluate real/reactive-current performance, apply hardware
limitation exemptions, or replace qualified assessor review.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "AC_WIND_IBR",
    "ASSESSOR_REVIEW_REQUIRED",
    "CONTINUOUS_OPERATION_REGION",
    "DURATION_EXCEEDS_MINIMUM",
    "MANDATORY_OPERATION_REGION",
    "MAY_RIDE_THROUGH_ZONE",
    "MAY_TRIP_ZONE_OBSERVED",
    "OTHER_IBR",
    "PERMISSIVE_OPERATION_REGION",
    "PRC_RIDE_THROUGH_DISCLAIMER",
    "PRC_RIDE_THROUGH_STANDARD",
    "WITHIN_MINIMUM_DURATION",
    "WITHIN_REVIEW_ENVELOPE",
    "PRCRideThroughEvidence",
    "PRCRideThroughFinding",
    "screen_ride_through_samples",
]

AC_WIND_IBR = "ac_wind"
OTHER_IBR = "other_ibr"
_SUPPORTED_IBR_CATEGORIES = frozenset({AC_WIND_IBR, OTHER_IBR})

CONTINUOUS_OPERATION_REGION = "continuous_operation_region"
MANDATORY_OPERATION_REGION = "mandatory_operation_region"
PERMISSIVE_OPERATION_REGION = "permissive_operation_region"
MAY_RIDE_THROUGH_ZONE = "may_ride_through_zone"

WITHIN_MINIMUM_DURATION = "within_minimum_duration"
DURATION_EXCEEDS_MINIMUM = "duration_exceeds_minimum"
MAY_TRIP_ZONE_OBSERVED = "may_trip_zone_observed"

WITHIN_REVIEW_ENVELOPE = "within_review_envelope"
ASSESSOR_REVIEW_REQUIRED = "assessor_review_required"

PRC_RIDE_THROUGH_STANDARD = (
    "NERC PRC-029-1 "
    "(Frequency and Voltage Ride-through Requirements for Inverter-based Resources)"
)

PRC_RIDE_THROUGH_DISCLAIMER = (
    "This PRC-029-1 ride-through evidence record is a technical screening "
    "artifact for qualified review. It does not constitute legal advice, a "
    "conformity assessment, certification of compliance, or an evaluation of "
    "all PRC-029-1 obligations. Current/reactive-power performance, phase-jump "
    "conditions, hardware-limit exemptions, reporting duties, and final "
    "conformance must be reviewed against the issued standard by a qualified "
    "assessor."
)

_VOLTAGE_WINDOW_S = 10.0
_FREQUENCY_WINDOW_S = 600.0


@dataclass(frozen=True, slots=True)
class _RideThroughBand:
    """Internal threshold-band metadata used to aggregate observations."""

    name: str
    operation_region: str
    minimum_ride_through_s: float | None
    window_s: float | None


@dataclass(frozen=True, slots=True)
class _Segment:
    """Internal interval carrying one held measurement value."""

    start_s: float
    end_s: float
    value: float
    band: _RideThroughBand

    @property
    def duration_s(self) -> float:
        """Return the segment duration in seconds."""
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class PRCRideThroughFinding:
    """One aggregated PRC-029 voltage or frequency ride-through observation.

    Attributes
    ----------
    channel : str
        ``"voltage"`` or ``"frequency"``.
    band : str
        Deterministic threshold-band identifier.
    operation_region : str
        Operation-region class from the PRC-029 ride-through tables.
    start_s, end_s : float
        First and last time covered by the aggregate observation.
    duration_s : float
        Total observed duration in the band across the trace.
    window_duration_s : float
        Maximum cumulative duration inside the relevant PRC-029 assessment
        window.
    observed_min, observed_max : float
        Minimum and maximum observed measurement values inside the band.
    minimum_ride_through_s : float | None
        Published minimum ride-through duration for the band, or ``None`` for
        may-trip zones.
    window_s : float | None
        Assessment window used for cumulative-duration screening.
    classification : str
        Screening classification. This is review language, not a legal verdict.
    flagged : bool
        Whether the observation needs qualified assessor review.
    """

    channel: str
    band: str
    operation_region: str
    start_s: float
    end_s: float
    duration_s: float
    window_duration_s: float
    observed_min: float
    observed_max: float
    minimum_ride_through_s: float | None
    window_s: float | None
    classification: str
    flagged: bool

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the finding.

        Returns
        -------
        dict[str, object]
            Stable audit fields for one ride-through observation.
        """
        return {
            "channel": self.channel,
            "band": self.band,
            "operation_region": self.operation_region,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "window_duration_s": self.window_duration_s,
            "observed_min": self.observed_min,
            "observed_max": self.observed_max,
            "minimum_ride_through_s": self.minimum_ride_through_s,
            "window_s": self.window_s,
            "classification": self.classification,
            "flagged": self.flagged,
        }


@dataclass(frozen=True, slots=True)
class PRCRideThroughEvidence:
    """Hash-sealed PRC-029 ride-through screening evidence.

    Attributes
    ----------
    event_id : str
        Caller-assigned event identifier.
    captured_at : str
        Measurement timestamp supplied by the caller.
    signal_source : str
        Operator-facing source label.
    ibr_category : str
        PRC-029 voltage-table category: :data:`AC_WIND_IBR` or
        :data:`OTHER_IBR`.
    sample_count : int
        Number of time-series samples consumed.
    duration_s : float
        Elapsed time from first to last sample.
    findings : tuple[PRCRideThroughFinding, ...]
        Aggregated voltage and frequency screening observations.
    channel_counts : Mapping[str, int]
        Read-only number of observations by channel.
    flagged_count : int
        Number of observations that require qualified review.
    verdict : str
        Review-only verdict string.
    standard : str
        Standard family the record is mapped to.
    disclaimer : str
        Review-only disclaimer.
    content_hash : str
        SHA-256 of the canonical record excluding this field.
    """

    event_id: str
    captured_at: str
    signal_source: str
    ibr_category: str
    sample_count: int
    duration_s: float
    findings: tuple[PRCRideThroughFinding, ...]
    channel_counts: Mapping[str, int]
    flagged_count: int
    verdict: str
    standard: str
    disclaimer: str
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Freeze channel counts and compute the canonical content hash."""
        object.__setattr__(
            self, "channel_counts", MappingProxyType(dict(self.channel_counts))
        )
        object.__setattr__(
            self, "content_hash", canonical_record_hash(self._canonical_payload())
        )

    def _canonical_payload(self) -> dict[str, object]:
        """Return the canonical payload for hashing and JSON export."""
        return {
            "event_id": self.event_id,
            "captured_at": self.captured_at,
            "signal_source": self.signal_source,
            "ibr_category": self.ibr_category,
            "sample_count": self.sample_count,
            "duration_s": self.duration_s,
            "findings": [finding.to_audit_record() for finding in self.findings],
            "channel_counts": dict(self.channel_counts),
            "flagged_count": self.flagged_count,
            "verdict": self.verdict,
            "standard": self.standard,
            "disclaimer": self.disclaimer,
        }

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the evidence record.

        Returns
        -------
        dict[str, object]
            The canonical payload plus ``content_hash``.
        """
        record = self._canonical_payload()
        record["content_hash"] = self.content_hash
        return record


def screen_ride_through_samples(
    time_s: Sequence[object],
    voltage_pu: Sequence[object],
    frequency_hz: Sequence[object],
    *,
    event_id: str,
    captured_at: str,
    signal_source: str,
    ibr_category: str = OTHER_IBR,
) -> PRCRideThroughEvidence:
    """Screen voltage and frequency samples into PRC-029 review evidence.

    Parameters
    ----------
    time_s : Sequence[object]
        Monotonic sample times in seconds.
    voltage_pu : Sequence[object]
        Voltage measurements in per unit at the applicable PRC-029 measurement
        point.
    frequency_hz : Sequence[object]
        Frequency measurements in hertz at the applicable PRC-029 measurement
        point.
    event_id : str
        Caller-assigned event identifier.
    captured_at : str
        Measurement timestamp stamped into the evidence.
    signal_source : str
        Operator-facing source label.
    ibr_category : str
        Voltage ride-through table selector, either :data:`AC_WIND_IBR` or
        :data:`OTHER_IBR`.

    Returns
    -------
    PRCRideThroughEvidence
        Deterministic, review-only PRC-029 screening evidence.

    Raises
    ------
    ValueError
        If identifiers, category, samples, or time ordering are invalid.
    """
    event = _non_empty_str(event_id, "event_id")
    captured = _non_empty_str(captured_at, "captured_at")
    source = _non_empty_str(signal_source, "signal_source")
    category = _ibr_category(ibr_category)
    times = _as_real_array(time_s, "time_s")
    voltage = _as_real_array(voltage_pu, "voltage_pu")
    frequency = _as_real_array(frequency_hz, "frequency_hz")
    _validate_shapes(times, voltage, frequency)

    voltage_findings = _collect_channel_findings(
        "voltage",
        _segments(times, voltage, lambda value: _voltage_band(value, category)),
    )
    frequency_findings = _collect_channel_findings(
        "frequency",
        _segments(times, frequency, _frequency_band),
    )
    findings = (*voltage_findings, *frequency_findings)
    flagged_count = sum(1 for finding in findings if finding.flagged)
    verdict = ASSESSOR_REVIEW_REQUIRED if flagged_count else WITHIN_REVIEW_ENVELOPE
    return PRCRideThroughEvidence(
        event_id=event,
        captured_at=captured,
        signal_source=source,
        ibr_category=category,
        sample_count=int(times.shape[0]),
        duration_s=float(times[-1] - times[0]),
        findings=findings,
        channel_counts=_channel_counts(findings),
        flagged_count=flagged_count,
        verdict=verdict,
        standard=PRC_RIDE_THROUGH_STANDARD,
        disclaimer=PRC_RIDE_THROUGH_DISCLAIMER,
    )


def _segments(
    times: FloatArray,
    values: FloatArray,
    classifier: Callable[[float], _RideThroughBand],
) -> tuple[_Segment, ...]:
    """Return held-value sample intervals with ride-through band metadata."""
    return tuple(
        _Segment(
            start_s=float(times[index]),
            end_s=float(times[index + 1]),
            value=float(values[index]),
            band=classifier(float(values[index])),
        )
        for index in range(times.shape[0] - 1)
    )


def _collect_channel_findings(
    channel: str, segments: Sequence[_Segment]
) -> tuple[PRCRideThroughFinding, ...]:
    """Aggregate segments into deterministic per-band findings."""
    grouped: dict[str, list[_Segment]] = {}
    ordered_bands: dict[str, _RideThroughBand] = {}
    for segment in segments:
        band = segment.band
        if (
            band.operation_region == CONTINUOUS_OPERATION_REGION
            and band.minimum_ride_through_s is None
        ):
            continue
        grouped.setdefault(band.name, []).append(segment)
        ordered_bands.setdefault(band.name, band)

    findings: list[PRCRideThroughFinding] = []
    for band_name, band_segments in grouped.items():
        band = ordered_bands[band_name]
        values = [segment.value for segment in band_segments]
        duration = sum((segment.duration_s for segment in band_segments), start=0.0)
        window_duration = _max_window_duration(band_segments, band.window_s)
        classification, flagged = _classification(band, window_duration)
        findings.append(
            PRCRideThroughFinding(
                channel=channel,
                band=band.name,
                operation_region=band.operation_region,
                start_s=band_segments[0].start_s,
                end_s=band_segments[-1].end_s,
                duration_s=duration,
                window_duration_s=window_duration,
                observed_min=min(values),
                observed_max=max(values),
                minimum_ride_through_s=band.minimum_ride_through_s,
                window_s=band.window_s,
                classification=classification,
                flagged=flagged,
            )
        )
    return tuple(findings)


def _classification(
    band: _RideThroughBand, window_duration_s: float
) -> tuple[str, bool]:
    """Return review classification and flagged status for an aggregate band."""
    if band.operation_region == MAY_RIDE_THROUGH_ZONE:
        return MAY_TRIP_ZONE_OBSERVED, True
    minimum = band.minimum_ride_through_s
    if minimum is not None and window_duration_s > minimum:
        return DURATION_EXCEEDS_MINIMUM, True
    return WITHIN_MINIMUM_DURATION, False


def _max_window_duration(segments: Sequence[_Segment], window_s: float | None) -> float:
    """Return maximum cumulative segment duration inside a fixed window."""
    if window_s is None:
        return sum((segment.duration_s for segment in segments), start=0.0)
    candidates: set[float] = set()
    for segment in segments:
        candidates.add(segment.start_s)
        candidates.add(segment.end_s - window_s)
    maximum = 0.0
    for start in candidates:
        end = start + window_s
        total = 0.0
        for segment in segments:
            overlap = min(segment.end_s, end) - max(segment.start_s, start)
            if overlap > 0.0:
                total += overlap
        maximum = max(maximum, total)
    return maximum


def _channel_counts(findings: Sequence[PRCRideThroughFinding]) -> dict[str, int]:
    """Return deterministic counts by measurement channel."""
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.channel] = counts.get(finding.channel, 0) + 1
    return counts


def _voltage_band(voltage_pu: float, ibr_category: str) -> _RideThroughBand:
    """Return PRC-029 Attachment 1 voltage band metadata."""
    if ibr_category == AC_WIND_IBR:
        return _ac_wind_voltage_band(voltage_pu)
    return _other_ibr_voltage_band(voltage_pu)


def _ac_wind_voltage_band(voltage_pu: float) -> _RideThroughBand:
    """Return Attachment 1 Table 1 band metadata for AC-connected wind IBRs."""
    if voltage_pu > 1.20:
        return _band("ac_wind_voltage_gt_1_20", MAY_RIDE_THROUGH_ZONE, None, None)
    if voltage_pu >= 1.10:
        return _band(
            "ac_wind_voltage_ge_1_10_le_1_20",
            MANDATORY_OPERATION_REGION,
            1.0,
            _VOLTAGE_WINDOW_S,
        )
    if voltage_pu > 1.05:
        return _band(
            "ac_wind_voltage_gt_1_05_lt_1_10",
            CONTINUOUS_OPERATION_REGION,
            1800.0,
            _VOLTAGE_WINDOW_S,
        )
    if voltage_pu >= 0.90:
        return _band(
            "ac_wind_voltage_ge_0_90_le_1_05",
            CONTINUOUS_OPERATION_REGION,
            None,
            None,
        )
    if voltage_pu < 0.10:
        return _band(
            "ac_wind_voltage_lt_0_10",
            PERMISSIVE_OPERATION_REGION,
            0.16,
            _VOLTAGE_WINDOW_S,
        )
    if voltage_pu < 0.25:
        return _band(
            "ac_wind_voltage_lt_0_25_ge_0_10",
            MANDATORY_OPERATION_REGION,
            0.16,
            _VOLTAGE_WINDOW_S,
        )
    if voltage_pu < 0.50:
        return _band(
            "ac_wind_voltage_lt_0_50_ge_0_25",
            MANDATORY_OPERATION_REGION,
            1.20,
            _VOLTAGE_WINDOW_S,
        )
    if voltage_pu < 0.70:
        return _band(
            "ac_wind_voltage_lt_0_70_ge_0_50",
            MANDATORY_OPERATION_REGION,
            2.50,
            _VOLTAGE_WINDOW_S,
        )
    return _band(
        "ac_wind_voltage_lt_0_90_ge_0_70",
        MANDATORY_OPERATION_REGION,
        3.00,
        _VOLTAGE_WINDOW_S,
    )


def _other_ibr_voltage_band(voltage_pu: float) -> _RideThroughBand:
    """Return Attachment 1 Table 2 band metadata for other IBRs."""
    if voltage_pu > 1.20:
        return _band("other_ibr_voltage_gt_1_20", MAY_RIDE_THROUGH_ZONE, None, None)
    if voltage_pu > 1.10:
        return _band(
            "other_ibr_voltage_gt_1_10_le_1_20",
            MANDATORY_OPERATION_REGION,
            1.0,
            _VOLTAGE_WINDOW_S,
        )
    if voltage_pu > 1.05:
        return _band(
            "other_ibr_voltage_gt_1_05_le_1_10",
            CONTINUOUS_OPERATION_REGION,
            1800.0,
            _VOLTAGE_WINDOW_S,
        )
    if voltage_pu >= 0.90:
        return _band(
            "other_ibr_voltage_ge_0_90_le_1_05",
            CONTINUOUS_OPERATION_REGION,
            None,
            None,
        )
    if voltage_pu < 0.10:
        return _band(
            "other_ibr_voltage_lt_0_10",
            PERMISSIVE_OPERATION_REGION,
            0.32,
            _VOLTAGE_WINDOW_S,
        )
    if voltage_pu < 0.25:
        return _band(
            "other_ibr_voltage_lt_0_25_ge_0_10",
            MANDATORY_OPERATION_REGION,
            0.32,
            _VOLTAGE_WINDOW_S,
        )
    if voltage_pu < 0.50:
        return _band(
            "other_ibr_voltage_lt_0_50_ge_0_25",
            MANDATORY_OPERATION_REGION,
            1.20,
            _VOLTAGE_WINDOW_S,
        )
    if voltage_pu < 0.70:
        return _band(
            "other_ibr_voltage_lt_0_70_ge_0_50",
            MANDATORY_OPERATION_REGION,
            3.00,
            _VOLTAGE_WINDOW_S,
        )
    return _band(
        "other_ibr_voltage_lt_0_90_ge_0_70",
        MANDATORY_OPERATION_REGION,
        6.00,
        _VOLTAGE_WINDOW_S,
    )


def _frequency_band(frequency_hz: float) -> _RideThroughBand:
    """Return PRC-029 Attachment 2 frequency band metadata."""
    if frequency_hz > 61.8:
        return _band("frequency_gt_61_8", MAY_RIDE_THROUGH_ZONE, None, None)
    if frequency_hz > 61.2:
        return _band(
            "frequency_gt_61_2_le_61_8",
            MANDATORY_OPERATION_REGION,
            299.0,
            _FREQUENCY_WINDOW_S,
        )
    if frequency_hz >= 58.8:
        return _band(
            "frequency_ge_58_8_le_61_2",
            CONTINUOUS_OPERATION_REGION,
            None,
            None,
        )
    if frequency_hz < 57.0:
        return _band("frequency_lt_57_0", MAY_RIDE_THROUGH_ZONE, None, None)
    return _band(
        "frequency_lt_58_8_ge_57_0",
        MANDATORY_OPERATION_REGION,
        299.0,
        _FREQUENCY_WINDOW_S,
    )


def _band(
    name: str,
    operation_region: str,
    minimum_ride_through_s: float | None,
    window_s: float | None,
) -> _RideThroughBand:
    """Return immutable ride-through threshold metadata."""
    return _RideThroughBand(
        name=name,
        operation_region=operation_region,
        minimum_ride_through_s=minimum_ride_through_s,
        window_s=window_s,
    )


def _validate_shapes(
    times: FloatArray, voltage: FloatArray, frequency: FloatArray
) -> None:
    """Validate shared sample shape and monotonic time."""
    if times.shape[0] < 2:
        raise ValueError("ride-through screening requires at least two samples")
    if not (times.shape == voltage.shape == frequency.shape):
        raise ValueError(
            "time_s, voltage_pu, and frequency_hz must have the same length"
        )
    if bool(np.any(np.diff(times) <= 0.0)):
        raise ValueError("time_s must be strictly increasing")


def _as_real_array(values: Sequence[object], name: str) -> FloatArray:
    """Return a finite float array, rejecting bool and string aliases."""
    checked: list[float] = []
    for index, value in enumerate(values):
        checked.append(_real_scalar(value, f"{name}[{index}]"))
    return np.ascontiguousarray(checked, dtype=np.float64)


def _ibr_category(value: object) -> str:
    """Return a supported IBR category, else raise ``ValueError``."""
    if not isinstance(value, str) or value not in _SUPPORTED_IBR_CATEGORIES:
        supported = ", ".join(sorted(_SUPPORTED_IBR_CATEGORIES))
        raise ValueError(f"ibr_category must be one of {supported}, got {value!r}")
    return value


def _non_empty_str(value: object, name: str) -> str:
    """Return ``value`` as a non-empty string, else raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _real_scalar(value: object, name: str) -> float:
    """Return ``value`` as a finite real scalar, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar
