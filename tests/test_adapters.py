# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Adapter tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.opentelemetry import OTelExporter
from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter
from scpn_phase_orchestrator.adapters.scpn_control_bridge import SCPNControlBridge
from scpn_phase_orchestrator.upde.metrics import LayerState, LockSignature, UPDEState

# ── OTel adapter ──────────────────────────────────────────────────────


def test_otel_exporter_noop_without_sdk():
    exp = OTelExporter("test-svc")
    # Should not raise regardless of whether opentelemetry is installed
    state = UPDEState(
        layers=[LayerState(R=0.8, psi=0.0)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=0.8,
        regime_id="nominal",
    )
    exp.record_step(state, step_idx=0)
    exp.record_regime_change("nominal", "degraded")


def test_otel_exporter_span_context_manager():
    exp = OTelExporter("test-svc")
    with exp.span("test.span", {"key": "val"}) as s:
        s.set_attribute("extra", 42)


def test_otel_exporter_enabled_property():
    exp = OTelExporter("test-svc")
    assert isinstance(exp.enabled, bool)


def test_noop_span_methods():
    from scpn_phase_orchestrator.adapters.opentelemetry import _NoOpSpan

    s = _NoOpSpan()
    s.set_attribute("k", "v")
    s.set_status(None)
    s.end()
    with s as entered:
        assert entered is s


def test_otel_record_step_noop():
    exp = OTelExporter("test-svc")
    exp._enabled = False
    state = UPDEState(
        layers=[LayerState(R=0.5, psi=0.0)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=0.5,
        regime_id="degraded",
    )
    exp.record_step(state, step_idx=5)


def test_otel_record_regime_change_noop():
    exp = OTelExporter("test-svc")
    exp._enabled = False
    exp.record_regime_change("nominal", "critical")


def test_otel_span_noop_path():
    exp = OTelExporter("test-svc")
    exp._enabled = False
    with exp.span("test.noop") as s:
        s.set_attribute("x", 1)


def test_otel_enabled_record_step_with_mock():
    from unittest.mock import MagicMock

    exp = OTelExporter("test-svc")
    exp._enabled = True
    exp._r_global_gauge = MagicMock()
    exp._stability_gauge = MagicMock()
    exp._step_counter = MagicMock()

    state = UPDEState(
        layers=[LayerState(R=0.8, psi=0.0)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=0.8,
        regime_id="nominal",
    )
    exp.record_step(state, step_idx=0)
    exp._r_global_gauge.set.assert_called_once()
    exp._step_counter.add.assert_called_once()


def test_otel_enabled_regime_change_with_mock():
    from unittest.mock import MagicMock

    exp = OTelExporter("test-svc")
    exp._enabled = True
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span
    exp._tracer = mock_tracer

    exp.record_regime_change("nominal", "degraded")
    mock_tracer.start_as_current_span.assert_called_once_with("spo.regime_change")


def test_otel_enabled_span_with_mock():
    from unittest.mock import MagicMock

    exp = OTelExporter("test-svc")
    exp._enabled = True
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span
    exp._tracer = mock_tracer

    with exp.span("test.span", {"key": "val"}) as s:
        s.set_attribute("extra", 42)
    mock_span.set_attribute.assert_any_call("key", "val")


def test_otel_enabled_span_no_attributes():
    from unittest.mock import MagicMock

    exp = OTelExporter("test-svc")
    exp._enabled = True
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span
    exp._tracer = mock_tracer

    with exp.span("test.bare"):
        pass


# ── Prometheus adapter ────────────────────────────────────────────────


def test_prometheus_connection_error():
    adapter = PrometheusAdapter("http://localhost:9090", timeout=1.0)
    with pytest.raises(ConnectionError):
        adapter.fetch_metric("up", 0.0, 1.0, 0.1)


# ── SCPN Control Bridge ──────────────────────────────────────────────


def test_bridge_import_knm():
    bridge = SCPNControlBridge({})
    knm = np.array([[0.0, 0.3], [0.3, 0.0]])
    state = bridge.import_knm(knm)
    assert state.knm.shape == (2, 2)
    assert state.active_template == "scpn_import"


def test_bridge_import_knm_non_square_raises():
    bridge = SCPNControlBridge({})
    with pytest.raises(ValueError, match="square"):
        bridge.import_knm(np.array([[1, 2, 3]]))


def test_bridge_import_omega():
    bridge = SCPNControlBridge({})
    omega = bridge.import_omega(np.array([1.0, 2.0]))
    assert omega.shape == (2,)


def test_bridge_import_omega_non_positive_raises():
    bridge = SCPNControlBridge({})
    with pytest.raises(ValueError, match="positive"):
        bridge.import_omega(np.array([1.0, -0.5]))


def test_bridge_import_omega_2d_raises():
    bridge = SCPNControlBridge({})
    with pytest.raises(ValueError, match="1-D"):
        bridge.import_omega(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_bridge_export_state():
    bridge = SCPNControlBridge({})
    sig = LockSignature(source_layer=0, target_layer=1, plv=0.95, mean_lag=0.01)
    layers = [
        LayerState(R=0.8, psi=1.0, lock_signatures={"0_1": sig}),
        LayerState(R=0.7, psi=2.0),
    ]
    state = UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(2),
        stability_proxy=0.85,
        regime_id="nominal",
    )
    out = bridge.export_state(state)
    assert out["regime"] == "nominal"
    assert len(out["layers"]) == 2
    assert out["layers"][0]["locks"]["0_1"]["plv"] == 0.95
