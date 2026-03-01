from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.opentelemetry import OTelAdapter
from scpn_phase_orchestrator.adapters.prometheus import PrometheusAdapter
from scpn_phase_orchestrator.adapters.scpn_control_bridge import SCPNControlBridge
from scpn_phase_orchestrator.upde.metrics import LayerState, LockSignature, UPDEState


def test_otel_adapter_init():
    adapter = OTelAdapter(service_name="test-svc")
    assert adapter._service_name == "test-svc"


def test_otel_adapter_not_implemented():
    adapter = OTelAdapter("svc")
    with pytest.raises(NotImplementedError):
        adapter.extract_events([])


def test_prometheus_not_implemented():
    adapter = PrometheusAdapter("http://localhost:9090")
    with pytest.raises(NotImplementedError):
        adapter.fetch_metric("up", 0.0, 1.0, 0.1)


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
