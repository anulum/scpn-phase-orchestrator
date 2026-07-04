# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — IBR ride-through CSV evidence ingestion

"""CSV ingress for review-only PRC-029 ride-through screening.

The C2 power-grid audit pack consumes local operator exports only. This module
reads a timestamped voltage/frequency CSV, rejects malformed measurements before
publication, invokes :mod:`scpn_phase_orchestrator.assurance.prc_ride_through`,
and wraps the result with a source-file digest so the screened evidence can be
reproduced byte-for-byte.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.assurance.prc_ride_through import (
    OTHER_IBR,
    PRCRideThroughEvidence,
    screen_ride_through_samples,
)

__all__ = [
    "IBR_RIDE_THROUGH_AUDIT_SCHEMA",
    "IBR_RIDE_THROUGH_CLAIM_BOUNDARY",
    "IBRRideThroughCsvEvidence",
    "screen_ibr_ride_through_csv",
]

IBR_RIDE_THROUGH_AUDIT_SCHEMA = "scpn_ibr_ride_through_prc029_audit_v1"
IBR_RIDE_THROUGH_CLAIM_BOUNDARY = "review_only_offline_no_live_actuation"


@dataclass(frozen=True, slots=True)
class IBRRideThroughCsvEvidence:
    """Hash-sealed PRC-029 screening evidence for one operator CSV.

    Attributes
    ----------
    schema : str
        Audit schema identifier.
    event_id : str
        Caller-assigned event identifier.
    captured_at : str
        Measurement timestamp supplied by the caller.
    signal_source : str
        Operator-facing source label.
    source_name : str
        Basename of the screened CSV.
    source_sha256 : str
        SHA-256 digest of the exact source CSV bytes.
    time_column, voltage_column, frequency_column : str
        CSV columns consumed by the parser.
    ibr_category : str
        PRC-029 voltage-table category forwarded to the screener.
    sample_count : int
        Number of accepted samples.
    duration_s : float
        Elapsed time from first to last accepted sample.
    prc029_evidence : PRCRideThroughEvidence
        Hash-sealed PRC-029 screening evidence.
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
    voltage_column: str
    frequency_column: str
    ibr_category: str
    sample_count: int
    duration_s: float
    prc029_evidence: PRCRideThroughEvidence
    claim_boundary: str = IBR_RIDE_THROUGH_CLAIM_BOUNDARY
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
            "voltage_column": self.voltage_column,
            "frequency_column": self.frequency_column,
            "ibr_category": self.ibr_category,
            "sample_count": self.sample_count,
            "duration_s": self.duration_s,
            "prc029_evidence_hash": self.prc029_evidence.content_hash,
            "prc029_evidence": self.prc029_evidence.to_audit_record(),
            "claim_boundary": self.claim_boundary,
            "review_only": self.review_only,
        }

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the CSV evidence record.

        Returns
        -------
        dict[str, object]
            The canonical payload plus ``content_hash``.
        """
        record = self._canonical_payload()
        record["content_hash"] = self.content_hash
        return record


def screen_ibr_ride_through_csv(
    path: str | Path,
    *,
    event_id: str,
    captured_at: str,
    signal_source: str,
    ibr_category: str = OTHER_IBR,
    time_column: str = "time_s",
    voltage_column: str = "voltage_pu",
    frequency_column: str = "frequency_hz",
) -> IBRRideThroughCsvEvidence:
    """Screen an operator voltage/frequency CSV into PRC-029 evidence.

    Parameters
    ----------
    path : str | pathlib.Path
        CSV path containing timestamp, voltage, and frequency columns.
    event_id : str
        Caller-assigned event identifier.
    captured_at : str
        Measurement timestamp stamped into the evidence.
    signal_source : str
        Operator-facing source label.
    ibr_category : str
        PRC-029 voltage-table selector.
    time_column : str
        Timestamp column in seconds.
    voltage_column : str
        Voltage column in per unit.
    frequency_column : str
        Frequency column in hertz.

    Returns
    -------
    IBRRideThroughCsvEvidence
        Deterministic, review-only CSV evidence package.

    Raises
    ------
    ValueError
        If the CSV, identifiers, category, or column values are invalid.
    """
    csv_path = Path(path)
    event = _non_empty_str(event_id, "event_id")
    captured = _non_empty_str(captured_at, "captured_at")
    source = _non_empty_str(signal_source, "signal_source")
    time_field = _non_empty_str(time_column, "time_column")
    voltage_field = _non_empty_str(voltage_column, "voltage_column")
    frequency_field = _non_empty_str(frequency_column, "frequency_column")

    source_bytes = csv_path.read_bytes()
    times, voltage, frequency = _read_ibr_csv(
        csv_path, time_field, voltage_field, frequency_field
    )
    prc029_evidence = screen_ride_through_samples(
        times,
        voltage,
        frequency,
        event_id=event,
        captured_at=captured,
        signal_source=source,
        ibr_category=ibr_category,
    )
    return IBRRideThroughCsvEvidence(
        schema=IBR_RIDE_THROUGH_AUDIT_SCHEMA,
        event_id=event,
        captured_at=captured,
        signal_source=source,
        source_name=csv_path.name,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        time_column=time_field,
        voltage_column=voltage_field,
        frequency_column=frequency_field,
        ibr_category=prc029_evidence.ibr_category,
        sample_count=prc029_evidence.sample_count,
        duration_s=prc029_evidence.duration_s,
        prc029_evidence=prc029_evidence,
    )


def _read_ibr_csv(
    path: Path,
    time_column: str,
    voltage_column: str,
    frequency_column: str,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return finite timestamp, voltage, and frequency tuples from a CSV."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        for column in (time_column, voltage_column, frequency_column):
            if column not in fieldnames:
                raise ValueError(
                    f"IBR ride-through CSV is missing required column {column}"
                )
        times: list[float] = []
        voltage: list[float] = []
        frequency: list[float] = []
        for row_index, row in enumerate(reader, start=1):
            times.append(
                _finite_float(row[time_column], f"{time_column} row {row_index}")
            )
            voltage.append(
                _finite_float(row[voltage_column], f"{voltage_column} row {row_index}")
            )
            frequency.append(
                _finite_float(
                    row[frequency_column], f"{frequency_column} row {row_index}"
                )
            )
    return (tuple(times), tuple(voltage), tuple(frequency))


def _non_empty_str(value: object, name: str) -> str:
    """Return ``value`` as a non-empty string, else raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


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
