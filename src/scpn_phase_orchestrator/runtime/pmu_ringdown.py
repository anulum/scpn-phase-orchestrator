# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PMU ringdown PRC evidence ingestion

"""Review-only PRC oscillation screening for operator-provided PMU ringdowns.

The dVOC audit pack needs a real-data ingress surface before it can be validated
against reviewed operator captures. This module provides that boundary without
claiming live control: it reads a local CSV exported from a PMU or historian,
validates finite uniformly sampled frequency measurements, converts them to a
nominal-frequency deviation signal, runs the matrix-pencil oscillation estimator,
and seals the resulting PRC evidence with a source-file digest.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.assurance.prc_oscillation import (
    PRCOscillationEvidence,
    screen_oscillation_modes,
)
from scpn_phase_orchestrator.monitor.oscillation_modes import estimate_oscillation_modes

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "PMU_RINGDOWN_AUDIT_SCHEMA",
    "PMU_RINGDOWN_CLAIM_BOUNDARY",
    "PMURingdownEvidence",
    "screen_pmu_ringdown_csv",
]

PMU_RINGDOWN_AUDIT_SCHEMA = "scpn_pmu_ringdown_prc_audit_v1"
PMU_RINGDOWN_CLAIM_BOUNDARY = "review_only_offline_no_live_actuation"


@dataclass(frozen=True, slots=True)
class PMURingdownEvidence:
    """Hash-sealed PRC screening evidence for one PMU ringdown CSV.

    Attributes
    ----------
    schema : str
        Audit schema identifier.
    event_id : str
        Caller-assigned event identifier.
    captured_at : str
        Capture timestamp supplied by the caller.
    signal_source : str
        Operator-facing source label for the PMU or historian signal.
    source_name : str
        Basename of the screened CSV path.
    source_sha256 : str
        SHA-256 digest of the exact source CSV bytes.
    time_column, frequency_column : str
        CSV columns consumed by the parser.
    nominal_frequency_hz : float
        Frequency subtracted from the measured PMU frequency before estimation.
    sample_count : int
        Number of accepted samples.
    sampling_rate_hz : float
        Uniform sampling rate inferred from the timestamp column.
    duration_s : float
        Capture duration from first to last sample.
    prc_evidence : PRCOscillationEvidence
        Hash-sealed PRC screening evidence for the frequency-deviation signal.
    claim_boundary : str
        Review-only claim boundary.
    review_only : bool
        Always ``True`` for this ingestion surface.
    content_hash : str
        SHA-256 of the canonical record excluding this field.
    """

    schema: str
    event_id: str
    captured_at: str
    signal_source: str
    source_name: str
    source_sha256: str
    time_column: str
    frequency_column: str
    nominal_frequency_hz: float
    sample_count: int
    sampling_rate_hz: float
    duration_s: float
    prc_evidence: PRCOscillationEvidence
    claim_boundary: str = PMU_RINGDOWN_CLAIM_BOUNDARY
    review_only: bool = True
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Compute the content hash from the canonical evidence payload."""
        object.__setattr__(
            self, "content_hash", canonical_record_hash(self._canonical_payload())
        )

    def _canonical_payload(self) -> dict[str, object]:
        """Return the canonical payload for hashing and JSON export."""
        return {
            "schema": self.schema,
            "event_id": self.event_id,
            "captured_at": self.captured_at,
            "signal_source": self.signal_source,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "time_column": self.time_column,
            "frequency_column": self.frequency_column,
            "nominal_frequency_hz": self.nominal_frequency_hz,
            "sample_count": self.sample_count,
            "sampling_rate_hz": self.sampling_rate_hz,
            "duration_s": self.duration_s,
            "prc_evidence_hash": self.prc_evidence.content_hash,
            "prc_evidence": self.prc_evidence.to_audit_record(),
            "claim_boundary": self.claim_boundary,
            "review_only": self.review_only,
        }

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the PMU ringdown evidence.

        Returns
        -------
        dict[str, object]
            The canonical payload plus the computed ``content_hash``.
        """
        record = self._canonical_payload()
        record["content_hash"] = self.content_hash
        return record


def screen_pmu_ringdown_csv(
    path: str | Path,
    *,
    event_id: str,
    captured_at: str,
    signal_source: str,
    time_column: str = "time_s",
    frequency_column: str = "frequency_hz",
    nominal_frequency_hz: float = 60.0,
    min_samples: int = 8,
    sampling_jitter_tolerance: float = 1.0e-6,
    model_order: int | None = None,
) -> PMURingdownEvidence:
    """Screen a PMU frequency ringdown CSV into hash-sealed PRC evidence.

    Parameters
    ----------
    path : str | pathlib.Path
        CSV path with a timestamp column and a measured frequency column.
    event_id : str
        Caller-assigned event identifier for the PMU capture.
    captured_at : str
        Capture timestamp stamped into the PRC evidence.
    signal_source : str
        Operator-facing PMU or historian signal label.
    time_column : str
        Name of the timestamp column in seconds.
    frequency_column : str
        Name of the measured frequency column in hertz.
    nominal_frequency_hz : float
        Nominal grid frequency subtracted before mode estimation.
    min_samples : int
        Minimum accepted sample count. Must be at least four.
    sampling_jitter_tolerance : float
        Relative tolerance for timestamp uniformity.
    model_order : int | None
        Optional matrix-pencil model order forwarded to the estimator.

    Returns
    -------
    PMURingdownEvidence
        Deterministic, review-only PMU screening evidence.

    Raises
    ------
    ValueError
        If the CSV, timestamps, frequency samples, or scalar controls are invalid.
    """
    csv_path = Path(path)
    event = _non_empty_str(event_id, "event_id")
    captured = _non_empty_str(captured_at, "captured_at")
    source = _non_empty_str(signal_source, "signal_source")
    time_field = _non_empty_str(time_column, "time_column")
    frequency_field = _non_empty_str(frequency_column, "frequency_column")
    nominal = _positive_real(nominal_frequency_hz, "nominal_frequency_hz")
    sample_floor = _min_samples(min_samples)
    jitter = _non_negative_real(sampling_jitter_tolerance, "sampling_jitter_tolerance")

    source_bytes = csv_path.read_bytes()
    times, frequencies = _read_pmu_csv(csv_path, time_field, frequency_field)
    if times.shape[0] < sample_floor:
        raise ValueError(
            f"PMU ringdown CSV must contain at least {sample_floor} samples"
        )
    sample_interval = _uniform_sample_interval(times, time_field, jitter)
    sampling_rate = 1.0 / sample_interval
    deviation = np.ascontiguousarray(frequencies - nominal, dtype=np.float64)
    modes = estimate_oscillation_modes(
        deviation, sampling_rate, model_order=model_order
    )
    prc_evidence = screen_oscillation_modes(
        modes,
        event_id=event,
        captured_at=captured,
        signal_source=f"{source}/deviation",
        sampling_rate_hz=sampling_rate,
    )
    return PMURingdownEvidence(
        schema=PMU_RINGDOWN_AUDIT_SCHEMA,
        event_id=event,
        captured_at=captured,
        signal_source=source,
        source_name=csv_path.name,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        time_column=time_field,
        frequency_column=frequency_field,
        nominal_frequency_hz=nominal,
        sample_count=int(times.shape[0]),
        sampling_rate_hz=sampling_rate,
        duration_s=float(times[-1] - times[0]),
        prc_evidence=prc_evidence,
    )


def _read_pmu_csv(
    path: Path, time_column: str, frequency_column: str
) -> tuple[FloatArray, FloatArray]:
    """Return finite timestamp and frequency arrays from a PMU CSV."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        for column in (time_column, frequency_column):
            if column not in fieldnames:
                raise ValueError(
                    f"PMU ringdown CSV is missing required column {column}"
                )
        times: list[float] = []
        frequencies: list[float] = []
        for row_index, row in enumerate(reader, start=1):
            times.append(
                _finite_float(row[time_column], f"{time_column} row {row_index}")
            )
            frequencies.append(
                _finite_float(
                    row[frequency_column], f"{frequency_column} row {row_index}"
                )
            )
    return (
        np.ascontiguousarray(times, dtype=np.float64),
        np.ascontiguousarray(frequencies, dtype=np.float64),
    )


def _uniform_sample_interval(
    times: FloatArray, time_column: str, jitter_tolerance: float
) -> float:
    """Return the uniform sample interval, else raise ``ValueError``."""
    intervals = np.diff(times)
    if np.any(intervals <= 0.0):
        raise ValueError(f"{time_column} must be strictly increasing")
    sample_interval = float(np.median(intervals))
    tolerance = max(1.0e-12, jitter_tolerance * sample_interval)
    if float(np.max(np.abs(intervals - sample_interval))) > tolerance:
        raise ValueError(f"{time_column} must be uniformly sampled")
    return sample_interval


def _min_samples(value: object) -> int:
    """Return a valid minimum sample count, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"min_samples must be an integer, got {value!r}")
    if value < 4:
        raise ValueError("min_samples must be at least 4")
    return value


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


def _non_negative_real(value: object, name: str) -> float:
    """Return ``value`` as a non-negative finite real, else raise."""
    scalar = _real_scalar(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _finite_float(value: str | None, name: str) -> float:
    """Return a CSV field as a finite float, else raise ``ValueError``."""
    if value is None:
        raise ValueError(f"{name} must be a finite real, got None")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real, got {value!r}") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar


def _real_scalar(value: object, name: str) -> float:
    """Return ``value`` as a finite real scalar, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar
