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
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any

from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = [
    "MetricsExporter",
    "OTelExporter",
    "RuntimeMetricSnapshot",
    "RuntimeObservability",
]

_PROMETHEUS_PREFIX_RE = re.compile(r"^[A-Za-z_:][A-Za-z0-9_:]*$")
_SERVICE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_SPAN_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")

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
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _validated_latency_ms(latency_ms: object) -> float:
    if (
        not isinstance(latency_ms, Real)
        or isinstance(latency_ms, bool)
        or not isfinite(float(latency_ms))
        or latency_ms < 0.0
    ):
        raise ValueError("latency_ms must be a finite non-negative real value")
    return float(latency_ms)


def _validated_regime(regime: object, *, allow_control: bool = False) -> str:
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
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validated_step_idx(step_idx: object) -> int:
    if isinstance(step_idx, bool) or not isinstance(step_idx, Integral):
        raise ValueError("step_idx must be a non-negative integer")
    result = int(step_idx)
    if result < 0:
        raise ValueError("step_idx must be a non-negative integer")
    return result


def _validated_otel_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SPAN_NAME_RE.fullmatch(value):
        raise ValueError(f"{field} must be a valid OpenTelemetry name")
    return value


def _validated_attributes(
    attributes: Mapping[str, object] | None,
) -> dict[str, object] | None:
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
        """Build individual metric lines in Prometheus text format."""
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
        """Return full Prometheus text exposition as a single string."""
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
        """True when opentelemetry-api is installed and active."""
        return self._enabled

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, object] | None = None,
    ) -> Generator[Any, None, None]:
        """Trace span context manager. No-op when OTel is absent."""
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
        """Record metrics from a completed UPDE step."""
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
        """Emit a span event for regime transitions."""
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
        """True when OpenTelemetry is installed and active."""
        return self._otel.enabled

    def prometheus_text(self, snapshot: RuntimeMetricSnapshot) -> str:
        """Return default Prometheus text for a runtime metric snapshot."""
        return self._metrics.export(
            snapshot.upde_state,
            snapshot.regime,
            snapshot.latency_ms,
            step_idx=snapshot.step_idx,
        )

    def record_step(self, snapshot: RuntimeMetricSnapshot) -> None:
        """Record a runtime step through the optional OpenTelemetry backend."""
        if snapshot.step_idx is None:
            return
        self._otel.record_step(snapshot.upde_state, snapshot.step_idx)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, object] | None = None,
    ) -> Generator[Any, None, None]:
        """Return a validated runtime span, no-op without OpenTelemetry."""
        with self._otel.span(name, attributes) as span:
            yield span
