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
2. Runs a LIF ensemble matching sc-neurocore dynamics
3. Converts spike rates back to orchestrator ControlActions

Backend priority:
  1. Rust (spo_kernel.PyLIFEnsemble) — ~1000x faster than scalar Python
  2. NumPy vectorised — ~50-100x faster than scalar Python
  3. sc-neurocore scalar — per-neuron Python objects (validation only)

LIF parameters match sc-neurocore v3.13.3 defaults (Gerstner & Kistler
2002): v_rest=0, v_threshold=1, tau_mem=20ms, R=1, dt=1ms.

Install sc-neurocore: pip install sc-neurocore
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["NeurocoreBridge", "HAS_NEUROCORE"]

try:
    from sc_neurocore import StochasticLIFNeuron  # pragma: no cover

    HAS_NEUROCORE = True  # pragma: no cover
except ImportError:  # pragma: no cover
    HAS_NEUROCORE = False  # pragma: no cover

# LIF defaults matching sc-neurocore v3.13.3 (Gerstner & Kistler 2002)
_V_REST = 0.0
_V_RESET = 0.0
_V_THRESHOLD = 1.0
_TAU_MEM = 20.0
_LIF_DT = 1.0
_RESISTANCE = 1.0
_REFRACTORY_PERIOD = 0


class NeurocoreBridge:
    """Live integration with sc-neurocore StochasticLIFNeuron ensemble.

    Each layer in the UPDE state maps to a group of stochastic LIF
    neurons. Layer coherence R drives input current; spike rates above
    threshold generate coupling boost actions.

    Backend selection (automatic):
      - ``"rust"`` — Rust LIF via spo_kernel.PyLIFEnsemble (fastest)
      - ``"numpy"`` — vectorised numpy LIF integration
      - ``"scalar"`` — per-neuron sc-neurocore objects (requires sc-neurocore)

    Pass ``backend="numpy"`` or ``backend="scalar"`` to force a specific
    backend. Default: best available.
    """

    def __init__(
        self,
        n_layers: int,
        neurons_per_layer: int = 8,
        current_scale: float = 2.0,
        spike_threshold_hz: float = 40.0,
        noise_std: float = 0.0,
        backend: str = "auto",
    ) -> None:
        self._n_layers = n_layers
        self._n_per = neurons_per_layer
        self._n_total = n_layers * neurons_per_layer
        self._scale = current_scale
        self._threshold_hz = spike_threshold_hz
        self._dt = 0.001  # 1ms step (for rate Hz conversion)

        # Resolve backend
        if backend == "auto":
            backend = "rust" if _HAS_RUST else "numpy"
        self._backend = backend

        if backend == "rust":
            from spo_kernel import PyLIFEnsemble  # type: ignore[import-untyped]

            self._rust_ensemble = PyLIFEnsemble(n_layers, neurons_per_layer, noise_std)
        elif backend == "numpy":
            self._v = np.full(self._n_total, _V_REST)
            self._refractory = np.zeros(self._n_total, dtype=np.int32)
            self._noise_std = noise_std
            self._rng = np.random.default_rng()
        elif backend == "scalar":
            if not HAS_NEUROCORE:  # pragma: no cover
                msg = "sc-neurocore not installed. pip install sc-neurocore"
                raise ImportError(msg)
            self._neurons: list = []
            for _ in range(self._n_total):
                self._neurons.append(StochasticLIFNeuron())
        else:
            msg = (
                f"Unknown backend {backend!r}, "
                "expected 'auto', 'rust', 'numpy', or 'scalar'"
            )
            raise ValueError(msg)

        self._spike_counts = np.zeros(self._n_total, dtype=np.int64)
        self._step_count = 0

    def step(self, state: UPDEState, n_substeps: int = 10) -> NDArray:
        """Run neuron ensemble for n_substeps, return per-layer spike rates."""
        r_values = np.array(
            [ls.R for ls in state.layers[: self._n_layers]],
            dtype=np.float64,
        )
        layer_currents = r_values * self._scale

        if self._backend == "rust":
            return self._step_rust(layer_currents, n_substeps)
        if self._backend == "numpy":
            currents = np.repeat(layer_currents, self._n_per)
            self._step_numpy(currents, n_substeps)
        else:
            currents = np.repeat(layer_currents, self._n_per)
            self._step_scalar(currents, n_substeps)

        duration_s = self._step_count * self._dt
        if duration_s == 0:  # pragma: no cover
            return np.zeros(self._n_layers)

        spikes_2d = self._spike_counts.reshape(self._n_layers, self._n_per)
        layer_spikes = spikes_2d.sum(axis=1)
        return layer_spikes / (self._n_per * duration_s)

    def _step_rust(self, layer_currents: NDArray, n_substeps: int) -> NDArray:
        """Delegate to Rust PyLIFEnsemble — fastest path."""
        return np.asarray(self._rust_ensemble.step(layer_currents, n_substeps))

    def _step_numpy(self, currents: NDArray, n_substeps: int) -> None:
        """Vectorised LIF Euler-Maruyama integration over all neurons."""
        v = self._v
        refractory = self._refractory
        dt_over_tau = _LIF_DT / _TAU_MEM
        input_term = currents * (_RESISTANCE * _LIF_DT)

        for _ in range(n_substeps):
            ref_mask = refractory > 0
            v[ref_mask] = _V_REST
            refractory[ref_mask] -= 1

            active = ~ref_mask
            v[active] += -(v[active] - _V_REST) * dt_over_tau + input_term[active]

            if self._noise_std > 0:
                sqrt_dt = _LIF_DT**0.5
                v[active] += self._rng.normal(
                    0.0, self._noise_std * sqrt_dt, size=int(active.sum())
                )

            spiked = v >= _V_THRESHOLD
            self._spike_counts[spiked] += 1
            v[spiked] = _V_RESET
            refractory[spiked] = _REFRACTORY_PERIOD

            self._step_count += 1

    def _step_scalar(self, currents: NDArray, n_substeps: int) -> None:
        """Per-neuron stepping via sc-neurocore objects (slow, for validation)."""
        for _ in range(n_substeps):
            for nidx in range(self._n_total):
                spiked = self._neurons[nidx].step(float(currents[nidx]))
                if spiked:
                    self._spike_counts[nidx] += 1
            self._step_count += 1

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
        if self._backend == "rust":
            return self._rust_ensemble.get_neuron_states()
        if self._backend == "numpy":
            return [
                {"v": float(self._v[i]), "refractory": int(self._refractory[i])}
                for i in range(self._n_total)
            ]
        return [n.get_state() for n in self._neurons]

    def reset(self) -> None:
        """Reset all neurons and counters."""
        if self._backend == "rust":
            self._rust_ensemble.reset()
        elif self._backend == "numpy":
            self._v[:] = _V_REST
            self._refractory[:] = 0
        else:
            for n in self._neurons:
                n.reset_state()
        self._spike_counts[:] = 0
        self._step_count = 0

    @property
    def backend(self) -> str:
        """Active backend: 'rust', 'numpy', or 'scalar'."""
        return self._backend
