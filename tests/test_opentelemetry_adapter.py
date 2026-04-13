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


class TestOTelExporterNoOp:
    """Test OTelExporter without opentelemetry installed (no-op mode)."""

    def test_not_enabled(self) -> None:
        exp = OTelExporter(service_name="test")
        assert exp.enabled is False

    def test_span_yields_noop(self) -> None:
        exp = OTelExporter()
        with exp.span("test_span") as s:
            assert isinstance(s, _NoOpSpan)

    def test_span_with_attributes(self) -> None:
        exp = OTelExporter()
        with exp.span("test", attributes={"k": "v"}) as s:
            s.set_attribute("extra", 42)

    def test_record_step_noop(self) -> None:
        exp = OTelExporter()
        exp.record_step(_make_state(), step_idx=0)

    def test_record_step_repeated(self) -> None:
        exp = OTelExporter()
        state = _make_state()
        for i in range(10):
            exp.record_step(state, step_idx=i)

    def test_record_regime_change_noop(self) -> None:
        exp = OTelExporter()
        exp.record_regime_change("coherent", "incoherent")

    def test_custom_service_name(self) -> None:
        exp = OTelExporter(service_name="custom_spo")
        assert exp._service_name == "custom_spo"

    def test_default_service_name(self) -> None:
        exp = OTelExporter()
        assert exp._service_name == "spo"
