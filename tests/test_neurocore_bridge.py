# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — sc-neurocore bridge tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.neurocore_bridge import (
    HAS_NEUROCORE,
    NeurocoreBridge,
)
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

pytestmark = pytest.mark.skipif(
    not HAS_NEUROCORE,
    reason="sc-neurocore not installed",
)


def _make_state(r_values: list[float]) -> UPDEState:
    layers = [LayerState(R=r, psi=0.0) for r in r_values]
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(len(layers)),
        stability_proxy=float(np.mean(r_values)),
        regime_id="nominal",
    )


class TestNeurocoreBridge:
    def test_init(self):
        bridge = NeurocoreBridge(n_layers=4, neurons_per_layer=4)
        assert len(bridge._neurons) == 16

    def test_step_returns_rates(self):
        bridge = NeurocoreBridge(n_layers=3, neurons_per_layer=4)
        state = _make_state([0.9, 0.5, 0.1])
        rates = bridge.step(state, n_substeps=50)
        assert rates.shape == (3,)
        assert np.all(rates >= 0.0)

    def test_high_coherence_higher_rate(self):
        bridge = NeurocoreBridge(
            n_layers=2,
            neurons_per_layer=8,
            current_scale=3.0,
        )
        state = _make_state([0.95, 0.1])
        rates = bridge.step(state, n_substeps=100)
        assert rates[0] >= rates[1]

    def test_rates_to_actions(self):
        bridge = NeurocoreBridge(n_layers=2)
        actions = bridge.rates_to_actions(np.array([100.0, 10.0]))
        assert len(actions) == 1
        assert actions[0].scope == "layer_0"
        assert actions[0].knob == "K"

    def test_step_and_act(self):
        bridge = NeurocoreBridge(n_layers=3, neurons_per_layer=4)
        state = _make_state([0.9, 0.9, 0.9])
        actions = bridge.step_and_act(state, n_substeps=50)
        assert isinstance(actions, list)

    def test_get_neuron_states(self):
        bridge = NeurocoreBridge(n_layers=2, neurons_per_layer=2)
        states = bridge.get_neuron_states()
        assert len(states) == 4
        assert "v" in states[0]

    def test_reset(self):
        bridge = NeurocoreBridge(n_layers=2, neurons_per_layer=2)
        state = _make_state([0.9, 0.9])
        bridge.step(state, n_substeps=10)
        bridge.reset()
        assert bridge._step_count == 0
        assert np.all(bridge._spike_counts == 0)


def test_has_neurocore_flag():
    assert isinstance(HAS_NEUROCORE, bool)
