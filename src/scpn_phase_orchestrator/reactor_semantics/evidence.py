# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor semantic evidence primitives

"""Clock, calibration, uncertainty, quality, validity, and provenance records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from .vocabulary import (
    ClockKind,
    QualityState,
    ValidityState,
    finite_real,
    non_negative_integer,
    non_negative_real,
    probability,
    require_enum,
    require_exact_keys,
    require_identifier,
    require_sha256,
    require_text,
)


@dataclass(frozen=True, slots=True)
class ClockReference:
    """Timestamp and sampling identity for one observation."""

    domain: str
    kind: ClockKind
    epoch: str
    timestamp_ns: int
    sample_rate_hz: float
    latency_s: float
    picosecond_offset: int = 0
    synchronized_to: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            require_enum(self.kind, ClockKind, field="clock kind"),
        )
        object.__setattr__(
            self, "domain", require_identifier(self.domain, field="clock domain")
        )
        object.__setattr__(self, "epoch", require_identifier(self.epoch, field="epoch"))
        object.__setattr__(
            self,
            "timestamp_ns",
            non_negative_integer(self.timestamp_ns, field="timestamp_ns"),
        )
        sample_rate = non_negative_real(self.sample_rate_hz, field="sample_rate_hz")
        if sample_rate == 0.0:
            raise ValueError("sample_rate_hz must be positive")
        object.__setattr__(self, "sample_rate_hz", sample_rate)
        object.__setattr__(
            self,
            "latency_s",
            non_negative_real(self.latency_s, field="latency_s"),
        )
        picosecond_offset = non_negative_integer(
            self.picosecond_offset,
            field="picosecond_offset",
        )
        if picosecond_offset > 999:
            raise ValueError("picosecond_offset must be in [0, 999]")
        object.__setattr__(self, "picosecond_offset", picosecond_offset)
        if self.synchronized_to is not None:
            object.__setattr__(
                self,
                "synchronized_to",
                require_identifier(self.synchronized_to, field="synchronized_to"),
            )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible clock record."""
        return {
            "domain": self.domain,
            "epoch": self.epoch,
            "kind": self.kind.value,
            "latency_s": self.latency_s,
            "picosecond_offset": self.picosecond_offset,
            "sample_rate_hz": self.sample_rate_hz,
            "synchronized_to": self.synchronized_to,
            "timestamp_ns": self.timestamp_ns,
        }

    @classmethod
    def from_record(cls, payload: object) -> ClockReference:
        """Construct a clock record from strict serialized input."""
        record = require_exact_keys(
            payload,
            required=frozenset(
                {
                    "domain",
                    "epoch",
                    "kind",
                    "latency_s",
                    "picosecond_offset",
                    "sample_rate_hz",
                    "synchronized_to",
                    "timestamp_ns",
                }
            ),
            field="clock",
        )
        synchronized_to = record["synchronized_to"]
        if synchronized_to is not None and not isinstance(synchronized_to, str):
            raise ValueError("clock.synchronized_to must be a string or null")
        return cls(
            domain=cast(str, record["domain"]),
            kind=ClockKind(cast(str, record["kind"])),
            epoch=cast(str, record["epoch"]),
            timestamp_ns=cast(int, record["timestamp_ns"]),
            sample_rate_hz=cast(float, record["sample_rate_hz"]),
            latency_s=cast(float, record["latency_s"]),
            picosecond_offset=cast(int, record["picosecond_offset"]),
            synchronized_to=synchronized_to,
        )


@dataclass(frozen=True, slots=True)
class CalibrationReference:
    """Calibration and transfer-function provenance."""

    calibration_id: str
    transfer_function_id: str
    calibrated_at_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "calibration_id",
            require_identifier(self.calibration_id, field="calibration_id"),
        )
        object.__setattr__(
            self,
            "transfer_function_id",
            require_identifier(self.transfer_function_id, field="transfer_function_id"),
        )
        object.__setattr__(
            self,
            "calibrated_at_ns",
            non_negative_integer(self.calibrated_at_ns, field="calibrated_at_ns"),
        )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible calibration record."""
        return {
            "calibrated_at_ns": self.calibrated_at_ns,
            "calibration_id": self.calibration_id,
            "transfer_function_id": self.transfer_function_id,
        }

    @classmethod
    def from_record(cls, payload: object) -> CalibrationReference:
        """Construct a calibration reference from strict serialized input."""
        record = require_exact_keys(
            payload,
            required=frozenset(
                {"calibrated_at_ns", "calibration_id", "transfer_function_id"}
            ),
            field="calibration",
        )
        return cls(
            calibration_id=cast(str, record["calibration_id"]),
            transfer_function_id=cast(str, record["transfer_function_id"]),
            calibrated_at_ns=cast(int, record["calibrated_at_ns"]),
        )


@dataclass(frozen=True, slots=True)
class Uncertainty:
    """Uncertainty bounds shared by scalar and circular quantities."""

    standard_deviation: float
    confidence_level: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    circular_std_rad: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "standard_deviation",
            non_negative_real(self.standard_deviation, field="standard_deviation"),
        )
        object.__setattr__(
            self,
            "confidence_level",
            probability(self.confidence_level, field="confidence_level"),
        )
        if self.lower_bound is not None:
            object.__setattr__(
                self,
                "lower_bound",
                finite_real(self.lower_bound, field="lower_bound"),
            )
        if self.upper_bound is not None:
            object.__setattr__(
                self,
                "upper_bound",
                finite_real(self.upper_bound, field="upper_bound"),
            )
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("uncertainty lower_bound must be <= upper_bound")
        if self.circular_std_rad is not None:
            object.__setattr__(
                self,
                "circular_std_rad",
                non_negative_real(self.circular_std_rad, field="circular_std_rad"),
            )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible uncertainty record."""
        return {
            "circular_std_rad": self.circular_std_rad,
            "confidence_level": self.confidence_level,
            "lower_bound": self.lower_bound,
            "standard_deviation": self.standard_deviation,
            "upper_bound": self.upper_bound,
        }

    @classmethod
    def from_record(cls, payload: object) -> Uncertainty:
        """Construct uncertainty from strict serialized input."""
        record = require_exact_keys(
            payload,
            required=frozenset(
                {
                    "circular_std_rad",
                    "confidence_level",
                    "lower_bound",
                    "standard_deviation",
                    "upper_bound",
                }
            ),
            field="uncertainty",
        )
        return cls(
            standard_deviation=cast(float, record["standard_deviation"]),
            confidence_level=cast(float, record["confidence_level"]),
            lower_bound=_optional_float(record["lower_bound"], field="lower_bound"),
            upper_bound=_optional_float(record["upper_bound"], field="upper_bound"),
            circular_std_rad=_optional_float(
                record["circular_std_rad"],
                field="circular_std_rad",
            ),
        )


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """Signal or estimator quality with explicit flags."""

    state: QualityState
    flags: tuple[str, ...] = ()
    signal_to_noise: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state",
            require_enum(self.state, QualityState, field="quality state"),
        )
        object.__setattr__(
            self,
            "flags",
            tuple(
                require_identifier(flag, field="quality flag") for flag in self.flags
            ),
        )
        if len(set(self.flags)) != len(self.flags):
            raise ValueError("quality flags must be unique")
        if self.signal_to_noise is not None:
            object.__setattr__(
                self,
                "signal_to_noise",
                non_negative_real(self.signal_to_noise, field="signal_to_noise"),
            )
        if self.state is QualityState.VALID and self.flags:
            raise ValueError("valid quality cannot carry fault flags")

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible quality record."""
        return {
            "flags": list(self.flags),
            "signal_to_noise": self.signal_to_noise,
            "state": self.state.value,
        }

    @classmethod
    def from_record(cls, payload: object) -> QualityAssessment:
        """Construct a quality assessment from strict serialized input."""
        record = require_exact_keys(
            payload,
            required=frozenset({"flags", "signal_to_noise", "state"}),
            field="quality",
        )
        flags = record["flags"]
        if not isinstance(flags, list) or any(
            not isinstance(flag, str) for flag in flags
        ):
            raise ValueError("quality.flags must be a list of strings")
        return cls(
            state=QualityState(cast(str, record["state"])),
            flags=tuple(flags),
            signal_to_noise=_optional_float(
                record["signal_to_noise"],
                field="signal_to_noise",
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidityWindow:
    """Validity interval and fail-closed epistemic state."""

    state: ValidityState
    valid_from_ns: int
    valid_until_ns: int
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state",
            require_enum(self.state, ValidityState, field="validity state"),
        )
        object.__setattr__(
            self,
            "valid_from_ns",
            non_negative_integer(self.valid_from_ns, field="valid_from_ns"),
        )
        object.__setattr__(
            self,
            "valid_until_ns",
            non_negative_integer(self.valid_until_ns, field="valid_until_ns"),
        )
        if self.valid_until_ns < self.valid_from_ns:
            raise ValueError("valid_until_ns must be >= valid_from_ns")
        object.__setattr__(
            self,
            "reasons",
            tuple(
                require_text(reason, field="validity reason") for reason in self.reasons
            ),
        )
        if self.state is ValidityState.VALID and self.reasons:
            raise ValueError("valid state cannot carry invalidity reasons")
        if self.state is not ValidityState.VALID and not self.reasons:
            raise ValueError("non-valid state requires at least one reason")

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible validity record."""
        return {
            "reasons": list(self.reasons),
            "state": self.state.value,
            "valid_from_ns": self.valid_from_ns,
            "valid_until_ns": self.valid_until_ns,
        }

    @classmethod
    def from_record(cls, payload: object) -> ValidityWindow:
        """Construct validity from strict serialized input."""
        record = require_exact_keys(
            payload,
            required=frozenset({"reasons", "state", "valid_from_ns", "valid_until_ns"}),
            field="validity",
        )
        reasons = record["reasons"]
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) for reason in reasons
        ):
            raise ValueError("validity.reasons must be a list of strings")
        return cls(
            state=ValidityState(cast(str, record["state"])),
            valid_from_ns=cast(int, record["valid_from_ns"]),
            valid_until_ns=cast(int, record["valid_until_ns"]),
            reasons=tuple(reasons),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Immutable source identity for a reactor semantic object."""

    source_project: str
    component: str
    symbol: str
    artifact_uri: str
    sha256: str
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_project",
            require_identifier(self.source_project, field="source_project"),
        )
        object.__setattr__(
            self,
            "component",
            require_identifier(self.component, field="component"),
        )
        object.__setattr__(
            self, "symbol", require_identifier(self.symbol, field="symbol")
        )
        object.__setattr__(
            self,
            "artifact_uri",
            require_text(self.artifact_uri, field="artifact_uri"),
        )
        object.__setattr__(
            self,
            "sha256",
            require_sha256(self.sha256, field="sha256"),
        )
        attributes = tuple(
            sorted(
                (
                    require_identifier(key, field="provenance attribute key"),
                    require_text(value, field="provenance attribute value"),
                )
                for key, value in self.attributes
            )
        )
        if len({key for key, _ in attributes}) != len(attributes):
            raise ValueError("provenance attribute keys must be unique")
        object.__setattr__(self, "attributes", attributes)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible provenance record."""
        return {
            "artifact_uri": self.artifact_uri,
            "attributes": dict(self.attributes),
            "component": self.component,
            "sha256": self.sha256,
            "source_project": self.source_project,
            "symbol": self.symbol,
        }

    @classmethod
    def from_record(cls, payload: object) -> ProvenanceRecord:
        """Construct provenance from strict serialized input."""
        record = require_exact_keys(
            payload,
            required=frozenset(
                {
                    "artifact_uri",
                    "attributes",
                    "component",
                    "sha256",
                    "source_project",
                    "symbol",
                }
            ),
            field="provenance",
        )
        attributes = record["attributes"]
        if not isinstance(attributes, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in attributes.items()
        ):
            raise ValueError("provenance.attributes must be a string mapping")
        return cls(
            source_project=cast(str, record["source_project"]),
            component=cast(str, record["component"]),
            symbol=cast(str, record["symbol"]),
            artifact_uri=cast(str, record["artifact_uri"]),
            sha256=cast(str, record["sha256"]),
            attributes=tuple(attributes.items()),
        )


def _optional_float(value: object, *, field: str) -> float | None:
    """Return a finite optional float."""
    if value is None:
        return None
    return finite_real(value, field=field)


__all__ = [
    "CalibrationReference",
    "ClockReference",
    "ProvenanceRecord",
    "QualityAssessment",
    "Uncertainty",
    "ValidityWindow",
]
