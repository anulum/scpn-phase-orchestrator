# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SNN bridge tests

from __future__ import annotations

import contextlib
from typing import get_type_hints

import numpy as np

from scpn_phase_orchestrator.adapters.snn_bridge import (
    TAU_RC,
    TAU_REF,
    SNNControllerBridge,
)
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _make_state(r_values):
    return UPDEState(
        layers=[LayerState(R=r, psi=0.0) for r in r_values],
        cross_layer_alignment=np.eye(len(r_values)),
        stability_proxy=float(np.mean(r_values)) if r_values else 0.0,
        regime_id="nominal",
    )


def test_default_lif_params():
    bridge = SNNControllerBridge()
    assert bridge.tau_rc == TAU_RC
    assert bridge.tau_ref == TAU_REF


def test_upde_to_current():
    bridge = SNNControllerBridge()
    state = _make_state([0.5, 0.8, 0.3])
    currents = bridge.upde_state_to_input_current(state, i_scale=2.0)
    np.testing.assert_allclose(currents, [1.0, 1.6, 0.6])


def test_upde_to_current_empty():
    bridge = SNNControllerBridge()
    state = _make_state([])
    currents = bridge.upde_state_to_input_current(state)
    assert len(currents) == 0


def test_public_array_contracts_are_parameterised():
    for hint in [
        get_type_hints(SNNControllerBridge.upde_state_to_input_current)["return"],
        get_type_hints(SNNControllerBridge.spike_rates_to_actions)["rates"],
        get_type_hints(SNNControllerBridge.lif_rate_estimate)["currents"],
        get_type_hints(SNNControllerBridge.lif_rate_estimate)["return"],
    ]:
        assert "numpy.ndarray" in str(hint)
        assert "float64" in str(hint)


def test_spike_rates_above_threshold():
    bridge = SNNControllerBridge()
    rates = np.array([60.0, 40.0, 80.0])
    actions = bridge.spike_rates_to_actions(rates, [0, 1, 2], threshold_hz=50.0)
    assert len(actions) == 2
    assert actions[0].scope == "layer_0"
    assert actions[1].scope == "layer_2"


def test_spike_rates_below_threshold():
    bridge = SNNControllerBridge()
    rates = np.array([10.0, 20.0])
    actions = bridge.spike_rates_to_actions(rates, [0, 1], threshold_hz=50.0)
    assert len(actions) == 0


def test_spike_rate_action_value():
    bridge = SNNControllerBridge()
    rates = np.array([100.0])
    actions = bridge.spike_rates_to_actions(rates, [0], threshold_hz=50.0)
    expected_value = 0.05 * (100.0 - 50.0) / 50.0
    assert abs(actions[0].value - expected_value) < 1e-10


def test_lif_rate_superthreshold():
    bridge = SNNControllerBridge()
    currents = np.array([1.5, 2.0, 3.0])
    rates = bridge.lif_rate_estimate(currents)
    assert all(r > 0 for r in rates)
    assert rates[0] < rates[1] < rates[2]


def test_lif_rate_subthreshold():
    bridge = SNNControllerBridge()
    currents = np.array([0.5, 0.9, 1.0])
    rates = bridge.lif_rate_estimate(currents)
    assert all(r == 0.0 for r in rates)


def test_lif_rate_edge():
    bridge = SNNControllerBridge()
    currents = np.array([1.01])
    rates = bridge.lif_rate_estimate(currents)
    assert rates[0] > 0


def test_custom_lif_params():
    bridge = SNNControllerBridge(tau_rc=0.05, tau_ref=0.005)
    currents = np.array([2.0])
    rates = bridge.lif_rate_estimate(currents)
    assert rates[0] > 0


def test_numpy_network_builds():
    bridge = SNNControllerBridge(n_neurons=50)
    model = bridge.build_numpy_network(n_layers=3, seed=42)
    assert model.input_node.shape == (3,)
    assert model.output_node.shape == (3,)
    assert model.ensemble.n_neurons == 50
    assert model.ensemble.encoders.shape == (50, 3)


def test_nengo_network_alias():
    bridge = SNNControllerBridge(n_neurons=50)
    model = bridge.build_nengo_network(n_layers=3, seed=42)
    assert model.ensemble.n_neurons == 50


def test_lava_import_error():
    bridge = SNNControllerBridge()
    with contextlib.suppress(ImportError):
        bridge.build_lava_process(n_layers=3)


def test_lava_process_with_mock():
    from unittest.mock import MagicMock, patch

    mock_lif_mod = MagicMock()
    mock_lif_cls = MagicMock()
    mock_lif_mod.LIF = mock_lif_cls
    modules = {
        "lava": MagicMock(),
        "lava.proc": MagicMock(),
        "lava.proc.lif": MagicMock(),
        "lava.proc.lif.process": mock_lif_mod,
    }
    bridge = SNNControllerBridge(n_neurons=50)
    with patch.dict("sys.modules", modules):
        bridge.build_lava_process(n_layers=3)
    mock_lif_cls.assert_called_once()


def test_neuromorphic_schedule_manifest_is_deterministic_and_safe():
    state = _make_state([0.25, 0.75])
    state.cross_layer_alignment[0, 1] = 0.4
    bridge = SNNControllerBridge(n_neurons=32)

    manifest = bridge.build_neuromorphic_schedule_manifest(
        state,
        i_scale=2.0,
        threshold_hz=20.0,
    )
    repeated = bridge.build_neuromorphic_schedule_manifest(
        state,
        i_scale=2.0,
        threshold_hz=20.0,
    )

    assert manifest == repeated
    assert manifest["manifest_kind"] == "neuromorphic_schedule_manifest"
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "simulator_parity_passed"
    assert manifest["target_backends"] == ["lava", "pynn"]
    assert manifest["actuation_permitted"] is False
    assert manifest["hardware_write_permitted"] is False
    assert len(manifest["schedule_sha256"]) == 64
    assert manifest["populations"][0]["lava_process"] == "LIF"
    assert manifest["populations"][0]["pynn_cell"] == "IF_curr_exp"
    assert manifest["populations"][1]["input_current"] == 1.5
    assert manifest["projections"] == [
        {
            "source": "layer_0",
            "target": "layer_1",
            "weight": 0.4,
            "delay_ms": 1.0,
            "receptor_type": "excitatory",
        }
    ]
    assert manifest["simulator_parity"]["max_abs_rate_error_hz"] == 0.0
    assert manifest["operator_commands"] == [
        "review neuromorphic_schedule_manifest.json",
        "run Lava or PyNN simulator parity before hardware handoff",
    ]


def test_neuromorphic_schedule_manifest_rejects_invalid_state():
    bridge = SNNControllerBridge()
    state = UPDEState(
        layers=[LayerState(R=0.4, psi=0.0), LayerState(R=0.6, psi=0.0)],
        cross_layer_alignment=np.ones((1, 2)),
        stability_proxy=0.5,
        regime_id="nominal",
    )

    try:
        bridge.build_neuromorphic_schedule_manifest(state)
    except ValueError as exc:
        assert "cross_layer_alignment shape" in str(exc)
    else:
        raise AssertionError("invalid cross-layer shape must be rejected")


class TestSNNPipelineWiring:
    """Pipeline: UPDEState → SNN currents → LIF rates → actions."""

    def test_upde_state_to_snn_roundtrip(self):
        """UPDEState → upde_state_to_input_current → lif_rate_estimate →
        spike_rates_to_actions. Full neuro-feedback loop."""
        state = _make_state([0.3, 0.5, 0.7, 0.9])
        bridge = SNNControllerBridge(n_neurons=100)

        currents = bridge.upde_state_to_input_current(state)
        assert currents.shape == (4,)
        assert np.all(np.isfinite(currents))

        rates = bridge.lif_rate_estimate(currents)
        assert np.all(rates >= 0.0)

        actions = bridge.spike_rates_to_actions(rates, layer_assignments=[0, 1, 2, 3])
        assert isinstance(actions, list)
