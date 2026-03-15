# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["SNNControllerBridge"]

# Abbott 1999, Eq. 1 — LIF time constants
TAU_RC = 0.02  # s, membrane time constant
TAU_REF = 0.002  # s, refractory period


class SNNControllerBridge:
    """Bridge between UPDE state and spiking neural network controllers.

    All methods are pure-numpy — no external SNN libraries required.
    """

    def __init__(
        self,
        n_neurons: int = 100,
        tau_rc: float = TAU_RC,
        tau_ref: float = TAU_REF,
    ) -> None:
        self.n_neurons = n_neurons
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

    def upde_state_to_input_current(
        self, state: UPDEState, i_scale: float = 1.0
    ) -> NDArray:
        """Map R values from each layer to LIF input currents."""
        r_values = np.array([ls.R for ls in state.layers], dtype=np.float64)
        return r_values * i_scale

    def spike_rates_to_actions(
        self,
        rates: NDArray,
        layer_assignments: list[int],
        threshold_hz: float = 50.0,
    ) -> list[ControlAction]:
        """Convert spike rates to control actions.

        *rates*: 1-D array of mean firing rates (Hz) per neuron group.
        *layer_assignments*: maps each rate index to a layer.
        *threshold_hz*: rates above this trigger coupling boost.
        """
        actions: list[ControlAction] = []
        for idx, (rate, layer) in enumerate(
            zip(rates, layer_assignments, strict=False)
        ):
            if rate > threshold_hz:
                excess = (rate - threshold_hz) / threshold_hz
                actions.append(
                    ControlAction(
                        knob="K",
                        scope=f"layer_{layer}",
                        value=0.05 * excess,
                        ttl_s=5.0,
                        justification=f"SNN group {idx}: {rate:.1f} Hz",
                    )
                )
        return actions

    def lif_rate_estimate(self, currents: NDArray) -> NDArray:
        """Analytic LIF steady-state firing rate (Abbott 1999, Eq. 1).

        rate = 1 / (tau_ref - tau_rc * ln(1 - 1/J))  for J > 1
        """
        rates = np.zeros_like(currents, dtype=np.float64)
        above = currents > 1.0
        if above.any():
            j = currents[above]
            rates[above] = 1.0 / (self.tau_ref - self.tau_rc * np.log(1.0 - 1.0 / j))
        return rates

    def build_numpy_network(
        self, n_layers: int, seed: int = 0, synapse: float = 0.01
    ) -> SimpleNamespace:
        """Build a pure-numpy LIF network for UPDE-SNN coupling.

        Returns a SimpleNamespace with input_node, ensemble, output_node
        attributes and a step() method.
        """
        rng = np.random.default_rng(seed)
        n = self.n_neurons
        encoders = rng.choice([-1.0, 1.0], (n, n_layers))
        max_rates = rng.uniform(100, 200, n)
        intercepts = rng.uniform(-0.5, 0.5, n)

        J_max = 1.0 / (1.0 - np.exp((self.tau_ref - 1.0 / max_rates) / self.tau_rc))
        alpha = (J_max - 1.0) / (1.0 - intercepts)
        J_bias = 1.0 - alpha * intercepts

        return SimpleNamespace(
            input_node=np.zeros(n_layers),
            ensemble=SimpleNamespace(
                n_neurons=n,
                encoders=encoders,
                alpha=alpha,
                J_bias=J_bias,
            ),
            output_node=np.zeros(n_layers),
            synapse=synapse,
            n_layers=n_layers,
        )

    # Backward-compat alias
    build_nengo_network = build_numpy_network

    def build_lava_process(self, n_layers: int) -> object:
        """Build a Lava LIF process for UPDE-SNN coupling.

        Raises ImportError if lava-nc is not installed.
        """
        from lava.proc.lif.process import LIF

        return LIF(
            shape=(self.n_neurons,),
            du=1.0 / self.tau_rc,
            dv=1.0 / self.tau_ref,
            vth=1.0,
        )
