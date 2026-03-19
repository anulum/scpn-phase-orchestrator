# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Quantum-control bridge adapter

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

__all__ = ["QuantumControlBridge"]

TWO_PI = 2.0 * np.pi


class QuantumControlBridge:
    """Adapter between scpn-quantum-control artifacts and phase-orchestrator types.

    Pure dict/array methods (import_artifact, export_artifact) work without
    scpn-quantum-control installed.  Circuit methods require the package.
    """

    def __init__(self, n_oscillators: int, trotter_order: int = 1):
        if n_oscillators < 1:
            raise ValueError(f"n_oscillators must be >= 1, got {n_oscillators}")
        self._n = n_oscillators
        self._trotter_order = trotter_order

    def import_artifact(self, artifact_dict: dict) -> UPDEState:
        """Convert a scpn-quantum-control result dict into UPDEState.

        Expected keys: 'phases' (1-D array), 'fidelity' (float),
        optional 'regime' (str), 'layer_assignments' (list of lists of int).
        """
        phases = np.asarray(artifact_dict["phases"], dtype=np.float64) % TWO_PI
        fidelity = float(artifact_dict.get("fidelity", 0.0))

        layer_assignments = artifact_dict.get("layer_assignments")
        if layer_assignments is None:
            mid = len(phases) // 2
            layer_assignments = [list(range(mid)), list(range(mid, len(phases)))]

        layers: list[LayerState] = []
        for group in layer_assignments:
            if len(group) == 0:
                layers.append(LayerState(R=0.0, psi=0.0))
                continue
            z = np.exp(1j * phases[group])
            order = z.mean()
            r_val = float(np.abs(order))
            psi_val = float(np.angle(order) % TWO_PI)
            layers.append(LayerState(R=r_val, psi=psi_val))

        n_layers = len(layers)
        cross = np.eye(n_layers, dtype=np.float64)
        regime = str(artifact_dict.get("regime", "NOMINAL"))

        return UPDEState(
            layers=layers,
            cross_layer_alignment=cross,
            stability_proxy=fidelity,
            regime_id=regime,
        )

    def export_artifact(self, state: UPDEState) -> dict:
        """Convert UPDEState back to a dict compatible with scpn-quantum-control."""
        return {
            "regime": state.regime_id,
            "fidelity": state.stability_proxy,
            "layers": [{"R": ls.R, "psi": ls.psi} for ls in state.layers],
            "cross_alignment": state.cross_layer_alignment.tolist(),
        }

    def import_knm(self, knm_array: NDArray) -> CouplingState:
        """Wrap a coupling matrix from quantum calibration into CouplingState."""
        knm = np.asarray(knm_array, dtype=np.float64)
        if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
            raise ValueError(f"Knm must be square, got shape {knm.shape}")
        n = knm.shape[0]
        return CouplingState(
            knm=knm,
            alpha=np.zeros((n, n), dtype=np.float64),
            active_template="quantum_import",
        )

    def build_hamiltonian(self, knm: NDArray, omegas: NDArray) -> object:
        """Build Kuramoto XY Hamiltonian as SparsePauliOp.

        Requires scpn-quantum-control.
        """
        from scpn_quantum_control import knm_to_hamiltonian  # pragma: no cover

        return knm_to_hamiltonian(knm, omegas)  # pragma: no cover

    def orchestrator_to_quantum(
        self,
        state: UPDEState,
    ) -> NDArray:
        """Convert orchestrator UPDEState to quantum phase array.

        Requires scpn-quantum-control.
        """
        from scpn_quantum_control import (  # pragma: no cover
            orchestrator_to_quantum_phases,
        )

        payload = self.export_artifact(state)  # pragma: no cover
        layer_phases = {  # pragma: no cover
            f"layer_{i}": ls["psi"] for i, ls in enumerate(payload["layers"])
        }
        return orchestrator_to_quantum_phases(layer_phases)  # pragma: no cover

    def quantum_to_orchestrator(
        self,
        quantum_theta: NDArray,
    ) -> dict:
        """Convert quantum phase array back to orchestrator-compatible dict.

        Requires scpn-quantum-control.
        """
        from scpn_quantum_control import (  # pragma: no cover
            quantum_to_orchestrator_phases,
        )

        return quantum_to_orchestrator_phases(quantum_theta)  # pragma: no cover
