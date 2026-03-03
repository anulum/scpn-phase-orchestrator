# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import contextlib

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
    # Higher current → higher rate
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


def test_nengo_import_error():
    bridge = SNNControllerBridge()
    with contextlib.suppress(ImportError):
        bridge.build_nengo_network(n_layers=3)


def test_lava_import_error():
    bridge = SNNControllerBridge()
    with contextlib.suppress(ImportError):
        bridge.build_lava_process(n_layers=3)
