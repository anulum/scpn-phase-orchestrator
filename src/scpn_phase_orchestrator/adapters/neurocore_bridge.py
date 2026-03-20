# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — sc-neurocore bridge

"""Bridge between sc-neurocore stochastic neurons and phase-orchestrator.

sc-neurocore provides StochasticLIFNeuron, SCIzhikevichNeuron, and
HomeostaticLIFNeuron with get_state()/step(current)/reset_state() API.

This bridge:
1. Maps UPDE layer R values to neuron input currents
2. Runs an ensemble of sc-neurocore neurons as a spiking controller
3. Converts spike rates back to orchestrator ControlActions

Install sc-neurocore: pip install sc-neurocore
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["NeurocoreBridge", "HAS_NEUROCORE"]

try:
    from sc_neurocore import StochasticLIFNeuron  # pragma: no cover

    HAS_NEUROCORE = True  # pragma: no cover
except ImportError:  # pragma: no cover
    HAS_NEUROCORE = False  # pragma: no cover


class NeurocoreBridge:
    """Live integration with sc-neurocore StochasticLIFNeuron ensemble.

    Each layer in the UPDE state maps to a group of stochastic LIF
    neurons. Layer coherence R drives input current; spike rates above
    threshold generate coupling boost actions.
    """

    def __init__(
        self,
        n_layers: int,
        neurons_per_layer: int = 8,
        current_scale: float = 2.0,
        spike_threshold_hz: float = 40.0,
    ) -> None:
        if not HAS_NEUROCORE:  # pragma: no cover
            msg = "sc-neurocore not installed. pip install sc-neurocore"
            raise ImportError(msg)
        self._n_layers = n_layers
        self._n_per = neurons_per_layer
        self._scale = current_scale
        self._threshold_hz = spike_threshold_hz
        self._dt = 0.001  # 1ms step

        self._neurons: list = []
        for _ in range(n_layers * neurons_per_layer):
            self._neurons.append(StochasticLIFNeuron())

        self._spike_counts = np.zeros(
            n_layers * neurons_per_layer,
            dtype=np.int64,
        )
        self._step_count = 0

    def step(self, state: UPDEState, n_substeps: int = 10) -> NDArray:
        """Run neuron ensemble for n_substeps, return per-layer spike rates.

        Each layer's R value is converted to input current. Neurons
        spike when membrane voltage exceeds threshold.
        """
        r_values = np.array(
            [ls.R for ls in state.layers[: self._n_layers]],
            dtype=np.float64,
        )
        currents = r_values * self._scale

        for _ in range(n_substeps):
            for layer_idx in range(self._n_layers):
                current = currents[layer_idx]
                for j in range(self._n_per):
                    nidx = layer_idx * self._n_per + j
                    spiked = self._neurons[nidx].step(current)
                    if spiked:
                        self._spike_counts[nidx] += 1
            self._step_count += 1

        duration_s = self._step_count * self._dt
        if duration_s == 0:  # pragma: no cover
            return np.zeros(self._n_layers)

        rates = np.zeros(self._n_layers)
        for layer_idx in range(self._n_layers):
            start = layer_idx * self._n_per
            end = start + self._n_per
            total_spikes = self._spike_counts[start:end].sum()
            rates[layer_idx] = total_spikes / (self._n_per * duration_s)

        return rates

    def rates_to_actions(self, rates: NDArray) -> list[ControlAction]:
        """Convert per-layer spike rates to coupling boost actions."""
        actions: list[ControlAction] = []
        for layer_idx, rate in enumerate(rates):
            if rate > self._threshold_hz:
                excess = (rate - self._threshold_hz) / self._threshold_hz
                actions.append(
                    ControlAction(
                        knob="K",
                        scope=f"layer_{layer_idx}",
                        value=0.05 * min(excess, 2.0),
                        ttl_s=5.0,
                        justification=(f"neurocore layer {layer_idx}: {rate:.1f} Hz"),
                    )
                )
        return actions

    def step_and_act(
        self,
        state: UPDEState,
        n_substeps: int = 10,
    ) -> list[ControlAction]:
        """Step the ensemble and return control actions."""
        rates = self.step(state, n_substeps)
        return self.rates_to_actions(rates)

    def get_neuron_states(self) -> list[dict]:
        """Return voltage/refractory state for all neurons."""
        return [n.get_state() for n in self._neurons]

    def reset(self) -> None:
        """Reset all neurons and counters."""
        for n in self._neurons:
            n.reset_state()
        self._spike_counts[:] = 0
        self._step_count = 0
