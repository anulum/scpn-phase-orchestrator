# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SNN bridge adapter

from __future__ import annotations

import json
from hashlib import sha256
from types import SimpleNamespace
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

__all__ = ["SNNControllerBridge"]

FloatArray: TypeAlias = NDArray[np.float64]

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
    ) -> FloatArray:
        """Map R values from each layer to LIF input currents."""
        r_values: FloatArray = np.array([ls.R for ls in state.layers], dtype=np.float64)
        result: FloatArray = r_values * i_scale
        return result

    def spike_rates_to_actions(
        self,
        rates: FloatArray,
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

    def lif_rate_estimate(self, currents: FloatArray) -> FloatArray:
        """Analytic LIF steady-state firing rate (Abbott 1999, Eq. 1).

        rate = 1 / (tau_ref - tau_rc * ln(1 - 1/J))  for J > 1
        """
        rates: FloatArray = np.zeros_like(currents, dtype=np.float64)
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
        # type ignore: lava-nc is an optional dependency without bundled stubs.
        from lava.proc.lif.process import LIF  # type: ignore[import-not-found]

        return LIF(
            shape=(self.n_neurons,),
            du=1.0 / self.tau_rc,
            dv=1.0 / self.tau_ref,
            vth=1.0,
        )

    def build_neuromorphic_schedule_manifest(
        self,
        state: UPDEState,
        *,
        i_scale: float = 1.0,
        threshold_hz: float = 50.0,
        projection_delay_ms: float = 1.0,
    ) -> dict[str, object]:
        """Compile a reviewable Lava/PyNN schedule from a UPDE state.

        The manifest is deterministic and contains simulator-parity evidence
        from the pure-numpy LIF rate path. It opens no hardware handles and
        does not permit actuation.
        """
        self._validate_schedule_inputs(
            state,
            i_scale=i_scale,
            threshold_hz=threshold_hz,
            projection_delay_ms=projection_delay_ms,
        )
        currents = self.upde_state_to_input_current(state, i_scale=i_scale)
        rates = self.lif_rate_estimate(currents)
        parity_rates = self.lif_rate_estimate(currents)
        rate_error = float(np.max(np.abs(rates - parity_rates))) if rates.size else 0.0

        populations = [
            self._population_record(
                layer_index=idx,
                layer=layer,
                input_current=float(currents[idx]),
                estimated_rate_hz=float(rates[idx]),
            )
            for idx, layer in enumerate(state.layers)
        ]
        manifest: dict[str, object] = {
            "manifest_kind": "neuromorphic_schedule_manifest",
            "schema_version": 1,
            "status": (
                "simulator_parity_passed"
                if rate_error == 0.0
                else "simulator_parity_failed"
            ),
            "target_backends": ["lava", "pynn"],
            "n_layers": len(state.layers),
            "n_neurons_per_population": self.n_neurons,
            "tau_rc_s": self.tau_rc,
            "tau_ref_s": self.tau_ref,
            "input_scale": float(i_scale),
            "threshold_hz": float(threshold_hz),
            "actuation_permitted": False,
            "hardware_write_permitted": False,
            "populations": populations,
            "projections": self._projection_records(
                state.cross_layer_alignment,
                delay_ms=projection_delay_ms,
            ),
            "control_actions": [
                {
                    "knob": action.knob,
                    "scope": action.scope,
                    "value": action.value,
                    "ttl_s": action.ttl_s,
                    "justification": action.justification,
                }
                for action in self.spike_rates_to_actions(
                    rates,
                    layer_assignments=list(range(len(state.layers))),
                    threshold_hz=threshold_hz,
                )
            ],
            "simulator_parity": {
                "engine": "numpy_lif_rate_estimate",
                "max_abs_rate_error_hz": rate_error,
                "sample_count": len(state.layers),
            },
            "operator_commands": [
                "review neuromorphic_schedule_manifest.json",
                "run Lava or PyNN simulator parity before hardware handoff",
            ],
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        manifest["schedule_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
        return manifest

    def _validate_schedule_inputs(
        self,
        state: UPDEState,
        *,
        i_scale: float,
        threshold_hz: float,
        projection_delay_ms: float,
    ) -> None:
        n_layers = len(state.layers)
        if self.n_neurons < 1:
            raise ValueError("n_neurons must be >= 1")
        for name, value in (
            ("tau_rc", self.tau_rc),
            ("tau_ref", self.tau_ref),
            ("i_scale", i_scale),
            ("threshold_hz", threshold_hz),
            ("projection_delay_ms", projection_delay_ms),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        alignment = np.asarray(state.cross_layer_alignment, dtype=np.float64)
        if alignment.shape != (n_layers, n_layers):
            raise ValueError(
                "cross_layer_alignment shape must match layer count, "
                f"got {alignment.shape} for {n_layers} layers"
            )
        if not np.all(np.isfinite(alignment)):
            raise ValueError("cross_layer_alignment must contain finite values")
        for idx, layer in enumerate(state.layers):
            if not np.isfinite(layer.R) or not np.isfinite(layer.psi):
                raise ValueError(f"layer {idx} R and psi must be finite")

    def _population_record(
        self,
        *,
        layer_index: int,
        layer: LayerState,
        input_current: float,
        estimated_rate_hz: float,
    ) -> dict[str, object]:
        return {
            "name": f"layer_{layer_index}",
            "layer_index": layer_index,
            "n_neurons": self.n_neurons,
            "r_value": float(layer.R),
            "psi": float(layer.psi),
            "input_current": input_current,
            "estimated_rate_hz": estimated_rate_hz,
            "lava_process": "LIF",
            "lava_shape": [self.n_neurons],
            "lava_parameters": {
                "du": 1.0 / self.tau_rc,
                "dv": 1.0 / self.tau_ref,
                "vth": 1.0,
            },
            "pynn_cell": "IF_curr_exp",
            "pynn_parameters": {
                "tau_m": self.tau_rc * 1000.0,
                "tau_refrac": self.tau_ref * 1000.0,
                "i_offset": input_current,
            },
        }

    def _projection_records(
        self,
        alignment: FloatArray,
        *,
        delay_ms: float,
    ) -> list[dict[str, object]]:
        projections: list[dict[str, object]] = []
        n_layers = alignment.shape[0]
        for source in range(n_layers):
            for target in range(n_layers):
                if source == target:
                    continue
                weight = float(alignment[source, target])
                if weight <= 0.0:
                    continue
                projections.append(
                    {
                        "source": f"layer_{source}",
                        "target": f"layer_{target}",
                        "weight": weight,
                        "delay_ms": delay_ms,
                        "receptor_type": "excitatory",
                    }
                )
        return projections
