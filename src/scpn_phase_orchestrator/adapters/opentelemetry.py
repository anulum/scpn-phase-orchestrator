# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — OpenTelemetry adapter

"""OpenTelemetry trace/metric export for production observability.

Install the ``otel`` extra to use this module::

    pip install scpn-phase-orchestrator[otel]

Without the extra, :class:`OTelExporter` falls back to a no-op
implementation that silently discards spans and metrics.
"""

from __future__ import annotations

import re
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["OTelExporter"]

_SERVICE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")

try:
    import opentelemetry.metrics as otel_metrics  # pragma: no cover
    import opentelemetry.trace as otel_trace  # pragma: no cover

    _HAS_OTEL = True  # pragma: no cover
except ModuleNotFoundError:  # pragma: no cover
    _HAS_OTEL = False


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


class OTelExporter:
    """Instrument UPDE steps with OpenTelemetry spans and metrics.

    Falls back to no-op when ``opentelemetry-api`` is not installed.
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
        self, name: str, attributes: dict | None = None
    ) -> Generator[Any, None, None]:
        """Trace span context manager. No-op when OTel is absent."""
        if not self._enabled:
            yield _NoOpSpan()
            return
        with self._tracer.start_as_current_span(name) as s:
            if attributes:
                for k, v in attributes.items():
                    s.set_attribute(k, v)
            yield s

    def record_step(self, upde_state: UPDEState, step_idx: int) -> None:
        """Record metrics from a completed UPDE step."""
        if not self._enabled:
            return
        attrs = {"spo.regime": upde_state.regime_id}
        self._r_global_gauge.set(upde_state.stability_proxy, attrs)
        self._stability_gauge.set(upde_state.stability_proxy, attrs)
        self._step_counter.add(1, attrs)

    def record_regime_change(self, old: str, new: str) -> None:
        """Emit a span event for regime transitions."""
        if not self._enabled:
            return
        with self._tracer.start_as_current_span("spo.regime_change") as s:
            s.set_attribute("spo.regime.old", old)
            s.set_attribute("spo.regime.new", new)
