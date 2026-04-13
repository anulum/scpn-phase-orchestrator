# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — OpenTelemetry adapter tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.adapters.opentelemetry import OTelExporter, _NoOpSpan
from scpn_phase_orchestrator.upde.metrics import LayerState, LockSignature, UPDEState


def _make_state() -> UPDEState:
    return UPDEState(
        layers=[
            LayerState(
                R=0.9,
                psi=0.5,
                mean_amplitude=1.0,
                lock_signatures={
                    1: LockSignature(
                        source_layer=0, target_layer=1, plv=0.85, mean_lag=0.01
                    )
                },
            ),
        ],
        regime_id="coherent",
        stability_proxy=0.9,
        cross_layer_alignment=np.eye(1),
    )


class TestNoOpSpan:
    def test_set_attribute_noop(self) -> None:
        span = _NoOpSpan()
        span.set_attribute("key", "value")

    def test_set_status_noop(self) -> None:
        span = _NoOpSpan()
        span.set_status(None)

    def test_end_noop(self) -> None:
        span = _NoOpSpan()
        span.end()

    def test_context_manager(self) -> None:
        with _NoOpSpan() as s:
            assert isinstance(s, _NoOpSpan)
            s.set_attribute("x", 1)


class TestOTelExporter:
    """Test OTelExporter in both enabled and disabled modes."""

    def test_enabled_reflects_otel_availability(self) -> None:
        exp = OTelExporter(service_name="test")
        # enabled depends on whether opentelemetry is installed
        assert isinstance(exp.enabled, bool)

    def test_span_context_manager(self) -> None:
        exp = OTelExporter()
        with exp.span("test_span") as s:
            # If otel is installed, s is a real span; otherwise _NoOpSpan
            assert s is not None

    def test_span_with_attributes(self) -> None:
        exp = OTelExporter()
        with exp.span("test", attributes={"k": "v"}) as s:
            s.set_attribute("extra", 42)

    def test_record_step_no_error(self) -> None:
        exp = OTelExporter()
        exp.record_step(_make_state(), step_idx=0)

    def test_record_step_repeated(self) -> None:
        exp = OTelExporter()
        state = _make_state()
        for i in range(10):
            exp.record_step(state, step_idx=i)

    def test_record_regime_change_no_error(self) -> None:
        exp = OTelExporter()
        exp.record_regime_change("coherent", "incoherent")

    def test_custom_service_name(self) -> None:
        exp = OTelExporter(service_name="custom_spo")
        assert exp._service_name == "custom_spo"

    def test_default_service_name(self) -> None:
        exp = OTelExporter()
        assert exp._service_name == "spo"

    def test_multiple_spans_nested(self) -> None:
        exp = OTelExporter()
        with exp.span("outer"), exp.span("inner") as s2:
            s2.set_attribute("depth", 2)

    def test_regime_change_multiple(self) -> None:
        exp = OTelExporter()
        exp.record_regime_change("coherent", "incoherent")
        exp.record_regime_change("incoherent", "chimera")
        exp.record_regime_change("chimera", "coherent")
