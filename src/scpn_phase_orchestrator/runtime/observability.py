# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Runtime observability defaults

"""Default runtime observability for SPO serving surfaces.

The runtime layer always exposes deterministic Prometheus text metrics and a
validated OpenTelemetry-compatible export surface. OpenTelemetry remains an
optional backend dependency; when it is absent, traces and metrics become
validated no-ops rather than disabling runtime observability.
"""

from __future__ import annotations

import re
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

from scpn_phase_orchestrator.upde.metrics import UPDEState

if TYPE_CHECKING:
    from scpn_phase_orchestrator.monitor.twin_confidence import TwinConfidenceSummary

__all__ = [
    "MetricsExporter",
    "OTelExporter",
    "PrometheusEvidenceSource",
    "RuntimeMetricSnapshot",
    "RuntimeObservability",
]

_PROMETHEUS_PREFIX_RE = re.compile(r"^[A-Za-z_:][A-Za-z0-9_:]*$")
_SERVICE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_SPAN_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_CONTRACT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_OPERATOR_STATUS_VALUES = ("healthy", "warning", "degraded", "critical")

try:
    import opentelemetry.metrics as otel_metrics  # pragma: no cover
    import opentelemetry.trace as otel_trace  # pragma: no cover

    _HAS_OTEL = True  # pragma: no cover
except ModuleNotFoundError:  # pragma: no cover
    _HAS_OTEL = False


@dataclass(frozen=True)
class RuntimeMetricSnapshot:
    """Validated runtime metrics emitted by HTTP, gRPC, and CLI surfaces."""

    upde_state: UPDEState
    regime: str
    latency_ms: float
    step_idx: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.upde_state, UPDEState):
            raise ValueError("upde_state must be a UPDEState instance")
        _validated_regime(self.regime)
        _validated_latency_ms(self.latency_ms)
        if self.step_idx is not None:
            _validated_step_idx(self.step_idx)


class PrometheusEvidenceSource:
    """Protocol-like base for audit-record evidence exported as metrics."""

    def to_audit_record(self) -> Mapping[str, object]:
        """Return JSON-safe evidence fields for Prometheus exposition.

        Returns
        -------
        Mapping[str, object]
            Return JSON-safe evidence fields for Prometheus exposition.

        Raises
        ------
        NotImplementedError
            If the operation is not implemented.
        """
        raise NotImplementedError


class _NoOpSpan:
    """Minimal stand-in when opentelemetry-api is absent."""

    def set_attribute(self, key: str, value: object) -> None:
        """No-op: discard span attribute."""

    def set_status(self, status: object) -> None:
        """No-op: discard span status."""

    def end(self) -> None:
        """No-op: end span."""

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *_: object) -> None:
        pass


def _escape_label_value(value: str) -> str:
    """Return a Prometheus-escaped label value."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _validated_latency_ms(latency_ms: object) -> float:
    """Return the validated latency in milliseconds, else raise."""
    if (
        not isinstance(latency_ms, Real)
        or isinstance(latency_ms, bool)
        or not isfinite(float(latency_ms))
        or latency_ms < 0.0
    ):
        raise ValueError("latency_ms must be a finite non-negative real value")
    return float(latency_ms)


def _validated_regime(regime: object, *, allow_control: bool = False) -> str:
    """Return the validated regime label, else raise."""
    if not isinstance(regime, str) or not regime:
        raise ValueError("regime must be a non-empty string")
    allowed_controls = {"\n", "\r", "\t"} if allow_control else set()
    if any(
        (ord(char) < 32 or ord(char) == 127) and char not in allowed_controls
        for char in regime
    ):
        raise ValueError("regime must not contain control characters")
    return regime


def _validated_finite_metric(value: object, *, field: str) -> float:
    """Return ``value`` as a validated finite metric, else raise."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validated_step_idx(step_idx: object) -> int:
    """Return the validated step index, else raise."""
    if isinstance(step_idx, bool) or not isinstance(step_idx, Integral):
        raise ValueError("step_idx must be a non-negative integer")
    result = int(step_idx)
    if result < 0:
        raise ValueError("step_idx must be a non-negative integer")
    return result


def _validated_non_negative_count(value: object, *, field: str) -> int:
    """Return ``value`` as a validated non-negative count, else raise."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def _validated_optional_sequence(value: object, *, field: str) -> int | None:
    """Return the validated optional sequence number, or ``None``."""
    if value is None:
        return None
    return _validated_non_negative_count(value, field=field)


def _validated_contract_hash(value: object) -> str:
    """Return the validated contract hash, else raise."""
    if not isinstance(value, str) or not _CONTRACT_HASH_RE.fullmatch(value):
        raise ValueError("contract_hash must be a 64-character lowercase SHA-256 hex")
    return value


def _validated_operator_status(value: object) -> str:
    """Return the validated operator status, else raise."""
    if not isinstance(value, str) or value not in _OPERATOR_STATUS_VALUES:
        raise ValueError(
            "status must be one of healthy, warning, degraded, or critical"
        )
    return value


def _validated_label_mapping(value: object, *, field: str) -> dict[str, int]:
    """Return the validated metric label mapping, else raise."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    validated: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(f"{field} keys must be non-empty strings")
        if any(ord(ch) < 32 for ch in raw_key):
            raise ValueError(f"{field} keys must not contain control characters")
        validated[raw_key] = _validated_non_negative_count(
            raw_count,
            field=f"{field}.{raw_key}",
        )
    return validated


def _validated_reason_counts(value: object, *, field: str) -> dict[str, int]:
    """Return the validated per-reason counts, else raise."""
    if not isinstance(value, Iterable) or isinstance(value, str | bytes | Mapping):
        raise ValueError(f"{field} must be a sequence of non-empty strings")
    counts: dict[str, int] = {}
    for reason in value:
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{field} entries must be non-empty strings")
        if any(ord(ch) < 32 for ch in reason):
            raise ValueError(f"{field} entries must not contain control characters")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _validated_optional_residual(value: object, *, field: str) -> float | None:
    """Return the validated optional residual, or ``None``."""
    if value is None:
        return None
    result = _validated_finite_metric(value, field=field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _evidence_record(
    evidence: Mapping[str, object] | PrometheusEvidenceSource,
) -> Mapping[str, object]:
    """Build the observability evidence record."""
    if isinstance(evidence, Mapping):
        return evidence
    if not hasattr(evidence, "to_audit_record"):
        raise ValueError("evidence must be a mapping or expose to_audit_record()")
    record = evidence.to_audit_record()
    if not isinstance(record, Mapping):
        raise ValueError("evidence.to_audit_record() must return a mapping")
    return record


def _validated_otel_name(value: object, *, field: str) -> str:
    """Return the validated OpenTelemetry metric name, else raise."""
    if not isinstance(value, str) or not _SPAN_NAME_RE.fullmatch(value):
        raise ValueError(f"{field} must be a valid OpenTelemetry name")
    return value


def _validated_attributes(
    attributes: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Return the validated OpenTelemetry attributes, else raise."""
    if attributes is None:
        return None
    if not isinstance(attributes, Mapping):
        raise ValueError("attributes must be a mapping with non-empty string keys")
    validated: dict[str, object] = {}
    for key, value in attributes.items():
        if not isinstance(key, str) or not key or any(ord(ch) < 32 for ch in key):
            raise ValueError("attributes must be a mapping with non-empty string keys")
        if not isinstance(value, str | bool | int | float):
            raise ValueError("attributes values must be primitive telemetry values")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("attributes values must be finite")
        validated[key] = value
    return validated


class MetricsExporter:
    """Format UPDE state, regime, and latency as Prometheus text exposition."""

    def __init__(self, prefix: str = "spo") -> None:
        if not isinstance(prefix, str) or not _PROMETHEUS_PREFIX_RE.fullmatch(prefix):
            raise ValueError(
                "prefix must be a valid Prometheus metric prefix "
                "([A-Za-z_:][A-Za-z0-9_:]*)"
            )
        self._prefix = prefix

    def exposition_lines(
        self,
        upde_state: UPDEState,
        regime: str,
        latency_ms: float,
        *,
        step_idx: int | None = None,
    ) -> list[str]:
        """Build individual metric lines in Prometheus text format.

        Parameters
        ----------
        upde_state : UPDEState
            The UPDE state to record or export.
        regime : str
            The current control regime label.
        latency_ms : float
            Step latency in milliseconds.
        step_idx : int | None
            Zero-based simulation step index, or ``None``.

        Returns
        -------
        list[str]
            The individual Prometheus metric lines.
        """
        p = self._prefix
        regime_label = _escape_label_value(
            _validated_regime(regime, allow_control=True)
        )
        latency_ms = _validated_latency_ms(latency_ms)
        lines: list[str] = []

        r_values = [
            _validated_finite_metric(layer.R, field=f"layer {idx} R")
            for idx, layer in enumerate(upde_state.layers)
        ]
        r_global = sum(r_values) / len(r_values) if r_values else 0.0
        stability_proxy = _validated_finite_metric(
            upde_state.stability_proxy,
            field="stability_proxy",
        )
        pac_max = _validated_finite_metric(upde_state.pac_max, field="pac_max")

        lines.append(f"# HELP {p}_r_global Global Kuramoto order parameter R")
        lines.append(f"# TYPE {p}_r_global gauge")
        lines.append(f'{p}_r_global{{regime="{regime_label}"}} {r_global:.6f}')

        lines.append(f"# HELP {p}_stability_proxy Mean R across layers")
        lines.append(f"# TYPE {p}_stability_proxy gauge")
        lines.append(
            f'{p}_stability_proxy{{regime="{regime_label}"}} {stability_proxy:.6f}'
        )

        lines.append(f"# HELP {p}_pac_max Maximum phase-amplitude coupling")
        lines.append(f"# TYPE {p}_pac_max gauge")
        lines.append(f'{p}_pac_max{{regime="{regime_label}"}} {pac_max:.6f}')

        lines.append(f"# HELP {p}_latency_ms UPDE step latency in milliseconds")
        lines.append(f"# TYPE {p}_latency_ms gauge")
        lines.append(f'{p}_latency_ms{{regime="{regime_label}"}} {latency_ms:.3f}')

        lines.append(f"# HELP {p}_layer_count Number of active UPDE layers")
        lines.append(f"# TYPE {p}_layer_count gauge")
        lines.append(f"{p}_layer_count {len(upde_state.layers)}")

        if step_idx is not None:
            step_idx = _validated_step_idx(step_idx)
            lines.append(f"# HELP {p}_step Current runtime step index")
            lines.append(f"# TYPE {p}_step gauge")
            lines.append(f"{p}_step {step_idx}")

        for i, r_value in enumerate(r_values):
            lines.append(
                f'{p}_layer_r{{layer="{i}",regime="{regime_label}"}} {r_value:.6f}'
            )

        return lines

    def export(
        self,
        upde_state: UPDEState,
        regime: str,
        latency_ms: float,
        *,
        step_idx: int | None = None,
    ) -> str:
        """Return full Prometheus text exposition as a single string.

        Parameters
        ----------
        upde_state : UPDEState
            The UPDE state to record or export.
        regime : str
            The current control regime label.
        latency_ms : float
            Step latency in milliseconds.
        step_idx : int | None
            Zero-based simulation step index, or ``None``.

        Returns
        -------
        str
            The full Prometheus text exposition.
        """
        return (
            "\n".join(
                self.exposition_lines(
                    upde_state,
                    regime,
                    latency_ms,
                    step_idx=step_idx,
                )
            )
            + "\n"
        )

    def digital_twin_operator_evidence_lines(
        self,
        evidence: Mapping[str, object] | PrometheusEvidenceSource,
    ) -> list[str]:
        """Build Prometheus lines for live or replayed digital-twin evidence.

        Parameters
        ----------
        evidence : Mapping[str, object] | PrometheusEvidenceSource
            Live or replayed digital-twin operator evidence.

        Returns
        -------
        list[str]
            The Prometheus lines for the digital-twin evidence.
        """
        record = _evidence_record(evidence)
        p = self._prefix
        contract_hash = _escape_label_value(
            _validated_contract_hash(record.get("contract_hash"))
        )
        accepted_count = _validated_non_negative_count(
            record.get("accepted_count"),
            field="accepted_count",
        )
        rejected_count = _validated_non_negative_count(
            record.get("rejected_count"),
            field="rejected_count",
        )
        adapter_count = _validated_non_negative_count(
            record.get("adapter_count"),
            field="adapter_count",
        )
        unhealthy_adapter_count = _validated_non_negative_count(
            record.get("unhealthy_adapter_count"),
            field="unhealthy_adapter_count",
        )
        latest_sequence = _validated_optional_sequence(
            record.get("latest_sequence"),
            field="latest_sequence",
        )
        max_abs_twin_residual = _validated_optional_residual(
            record.get("max_abs_twin_residual"),
            field="max_abs_twin_residual",
        )
        status = _validated_operator_status(record.get("status"))
        capability_counts = _validated_label_mapping(
            record.get("capability_counts"),
            field="capability_counts",
        )
        direction_counts = _validated_label_mapping(
            record.get("direction_counts"),
            field="direction_counts",
        )
        mismatch_reason_counts = _validated_reason_counts(
            record.get("mismatch_reasons"),
            field="mismatch_reasons",
        )

        contract_label = f'contract_hash="{contract_hash}"'
        lines = [
            (
                f"# HELP {p}_digital_twin_sync_accepted_total "
                "Accepted digital-twin sync validations"
            ),
            f"# TYPE {p}_digital_twin_sync_accepted_total counter",
            (
                f"{p}_digital_twin_sync_accepted_total{{{contract_label}}} "
                f"{accepted_count}"
            ),
            (
                f"# HELP {p}_digital_twin_sync_rejected_total "
                "Rejected digital-twin sync validations"
            ),
            f"# TYPE {p}_digital_twin_sync_rejected_total counter",
            (
                f"{p}_digital_twin_sync_rejected_total{{{contract_label}}} "
                f"{rejected_count}"
            ),
            f"# HELP {p}_digital_twin_adapter_count Digital-twin adapter count",
            f"# TYPE {p}_digital_twin_adapter_count gauge",
            f"{p}_digital_twin_adapter_count{{{contract_label}}} {adapter_count}",
            (
                f"# HELP {p}_digital_twin_unhealthy_adapter_count "
                "Digital-twin adapters failing compatibility or health review"
            ),
            f"# TYPE {p}_digital_twin_unhealthy_adapter_count gauge",
            (
                f"{p}_digital_twin_unhealthy_adapter_count{{{contract_label}}} "
                f"{unhealthy_adapter_count}"
            ),
        ]
        if latest_sequence is not None:
            lines.extend(
                [
                    (
                        f"# HELP {p}_digital_twin_latest_sequence "
                        "Latest accepted digital-twin sequence"
                    ),
                    f"# TYPE {p}_digital_twin_latest_sequence gauge",
                    (
                        f"{p}_digital_twin_latest_sequence{{{contract_label}}} "
                        f"{latest_sequence}"
                    ),
                ]
            )
        if max_abs_twin_residual is not None:
            lines.extend(
                [
                    (
                        f"# HELP {p}_digital_twin_max_abs_residual "
                        "Maximum absolute digital-twin residual"
                    ),
                    f"# TYPE {p}_digital_twin_max_abs_residual gauge",
                    (
                        f"{p}_digital_twin_max_abs_residual{{{contract_label}}} "
                        f"{max_abs_twin_residual:.6f}"
                    ),
                ]
            )
        lines.extend(
            [
                (
                    f"# HELP {p}_digital_twin_status "
                    "One-hot digital-twin operator status"
                ),
                f"# TYPE {p}_digital_twin_status gauge",
            ]
        )
        for status_value in _OPERATOR_STATUS_VALUES:
            status_label = _escape_label_value(status_value)
            lines.append(
                f"{p}_digital_twin_status{{{contract_label},"
                f'status="{status_label}"}} {int(status == status_value)}'
            )
        for capability, count in sorted(capability_counts.items()):
            capability_label = _escape_label_value(capability)
            lines.append(
                f"{p}_digital_twin_capability_count{{{contract_label},"
                f'capability="{capability_label}"}} {count}'
            )
        for direction, count in sorted(direction_counts.items()):
            direction_label = _escape_label_value(direction)
            lines.append(
                f"{p}_digital_twin_direction_count{{{contract_label},"
                f'direction="{direction_label}"}} {count}'
            )
        for reason, count in sorted(mismatch_reason_counts.items()):
            reason_label = _escape_label_value(reason)
            lines.append(
                f"{p}_digital_twin_mismatch_reason_count{{{contract_label},"
                f'reason="{reason_label}"}} {count}'
            )
        return lines

    def export_digital_twin_operator_evidence(
        self,
        evidence: Mapping[str, object] | PrometheusEvidenceSource,
    ) -> str:
        """Return Prometheus text for digital-twin operator evidence.

        Parameters
        ----------
        evidence : Mapping[str, object] | PrometheusEvidenceSource
            Live or replayed digital-twin operator evidence.

        Returns
        -------
        str
            The Prometheus text for the digital-twin operator evidence.
        """
        return "\n".join(self.digital_twin_operator_evidence_lines(evidence)) + "\n"

    def export_twin_confidence(self, summary: TwinConfidenceSummary) -> str:
        """Return Prometheus text for a twin-confidence operator summary.

        Parameters
        ----------
        summary : TwinConfidenceSummary
            The operator-facing aggregate over scored twin-confidence ticks.

        Returns
        -------
        str
            Prometheus exposition text using this exporter's metric prefix.
        """
        from scpn_phase_orchestrator.monitor.twin_confidence import (
            twin_confidence_prometheus_text,
        )

        return twin_confidence_prometheus_text(summary, prefix=self._prefix)


class OTelExporter:
    """Instrument UPDE steps with OpenTelemetry spans and metrics.

    Falls back to validated no-op behaviour when ``opentelemetry-api`` is not
    installed, so runtime observability remains enabled by default without
    making OpenTelemetry a base dependency.
    """

    def __init__(self, service_name: str = "spo") -> None:
        if not isinstance(service_name, str) or not _SERVICE_NAME_RE.fullmatch(
            service_name
        ):
            raise ValueError(
                "service_name must start with a letter and contain only "
                "letters, digits, underscore, dot, or hyphen"
            )
        self._service_name = service_name
        self._enabled = _HAS_OTEL
        self._tracer: Any = None
        self._r_global_gauge: Any = None
        self._stability_gauge: Any = None
        self._step_counter: Any = None
        if self._enabled:  # pragma: no cover
            self._tracer = otel_trace.get_tracer(service_name)
            meter = otel_metrics.get_meter(service_name)
            self._r_global_gauge = meter.create_gauge(
                "spo.r_global",
                description="Global Kuramoto order parameter R",
                unit="1",
            )
            self._stability_gauge = meter.create_gauge(
                "spo.stability_proxy",
                description="Mean R across layers",
                unit="1",
            )
            self._step_counter = meter.create_counter(
                "spo.steps_total",
                description="Total UPDE integration steps",
                unit="1",
            )

    @property
    def enabled(self) -> bool:
        """True when opentelemetry-api is installed and active.

        Returns
        -------
        bool
            True when opentelemetry-api is installed and active.
        """
        return self._enabled

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, object] | None = None,
    ) -> Generator[Any, None, None]:
        """Trace span context manager. No-op when OTel is absent.

        Parameters
        ----------
        name : str
            The span or resource name.
        attributes : Mapping[str, object] | None
            Optional span attributes, or ``None``.

        Returns
        -------
        Generator[Any, None, None]
            A trace-span context manager (no-op without OpenTelemetry).
        """
        name = _validated_otel_name(name, field="span name")
        attributes = _validated_attributes(attributes)
        if not self._enabled:
            yield _NoOpSpan()
            return
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    def record_step(self, upde_state: UPDEState, step_idx: int) -> None:
        """Record metrics from a completed UPDE step.

        Parameters
        ----------
        upde_state : UPDEState
            The UPDE state to record or export.
        step_idx : int
            Zero-based simulation step index, or ``None``.
        """
        _validated_step_idx(step_idx)
        stability_proxy = _validated_finite_metric(
            upde_state.stability_proxy,
            field="stability_proxy",
        )
        regime = _validated_regime(upde_state.regime_id)
        if not self._enabled:
            return
        attrs = {"spo.regime": regime}
        self._r_global_gauge.set(stability_proxy, attrs)
        self._stability_gauge.set(stability_proxy, attrs)
        self._step_counter.add(1, attrs)

    def record_regime_change(self, old: str, new: str) -> None:
        """Emit a span event for regime transitions.

        Parameters
        ----------
        old : str
            The previous regime label.
        new : str
            The new regime label.
        """
        old = _validated_regime(old)
        new = _validated_regime(new)
        if not self._enabled:
            return
        with self._tracer.start_as_current_span("spo.regime_change") as span:
            span.set_attribute("spo.regime.old", old)
            span.set_attribute("spo.regime.new", new)


class RuntimeObservability:
    """Default observability facade used by runtime serving surfaces."""

    def __init__(
        self,
        *,
        service_name: str = "spo",
        metric_prefix: str = "spo",
        otel_exporter: OTelExporter | None = None,
    ) -> None:
        self._metrics = MetricsExporter(metric_prefix)
        self._otel = (
            otel_exporter if otel_exporter is not None else OTelExporter(service_name)
        )

    @property
    def otel_enabled(self) -> bool:
        """True when OpenTelemetry is installed and active.

        Returns
        -------
        bool
            True when OpenTelemetry is installed and active.
        """
        return self._otel.enabled

    def prometheus_text(self, snapshot: RuntimeMetricSnapshot) -> str:
        """Return default Prometheus text for a runtime metric snapshot.

        Parameters
        ----------
        snapshot : RuntimeMetricSnapshot
            The runtime metric snapshot.

        Returns
        -------
        str
            The default Prometheus text for the snapshot.
        """
        return self._metrics.export(
            snapshot.upde_state,
            snapshot.regime,
            snapshot.latency_ms,
            step_idx=snapshot.step_idx,
        )

    def digital_twin_prometheus_text(
        self,
        evidence: Mapping[str, object] | PrometheusEvidenceSource,
    ) -> str:
        """Return Prometheus text for live or replayed digital-twin evidence.

        Parameters
        ----------
        evidence : Mapping[str, object] | PrometheusEvidenceSource
            Live or replayed digital-twin operator evidence.

        Returns
        -------
        str
            The Prometheus text for the digital-twin evidence.
        """
        return self._metrics.export_digital_twin_operator_evidence(evidence)

    def twin_confidence_prometheus_text(self, summary: TwinConfidenceSummary) -> str:
        """Return Prometheus text for a twin-confidence operator summary.

        Parameters
        ----------
        summary : TwinConfidenceSummary
            The operator-facing aggregate over scored twin-confidence ticks.

        Returns
        -------
        str
            Prometheus exposition text for the twin-confidence summary.
        """
        return self._metrics.export_twin_confidence(summary)

    def record_step(self, snapshot: RuntimeMetricSnapshot) -> None:
        """Record a runtime step through the optional OpenTelemetry backend.

        Parameters
        ----------
        snapshot : RuntimeMetricSnapshot
            The runtime metric snapshot.
        """
        if snapshot.step_idx is None:
            return
        self._otel.record_step(snapshot.upde_state, snapshot.step_idx)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, object] | None = None,
    ) -> Generator[Any, None, None]:
        """Return a validated runtime span, no-op without OpenTelemetry.

        Parameters
        ----------
        name : str
            The span or resource name.
        attributes : Mapping[str, object] | None
            Optional span attributes, or ``None``.

        Returns
        -------
        Generator[Any, None, None]
            A validated runtime span (no-op without OpenTelemetry).
        """
        with self._otel.span(name, attributes) as span:
            yield span
