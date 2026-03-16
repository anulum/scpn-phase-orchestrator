# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

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

    def build_quantum_circuit(
        self, knm: NDArray, omegas: NDArray, time: float, trotter_steps: int = 10
    ) -> object:
        """Build a Trotterised XY-Hamiltonian circuit.

        Requires scpn-quantum-control.  Returns a Qiskit QuantumCircuit.
        """
        try:
            from scpn_quantum_control.circuits import xy_trotter_circuit
        except ImportError as exc:
            raise ImportError(
                "scpn-quantum-control is required for circuit construction. "
                "Install with: pip install scpn-quantum-control"
            ) from exc
        return xy_trotter_circuit(  # pragma: no cover
            knm=knm,
            omegas=omegas,
            time=time,
            trotter_steps=trotter_steps,
            trotter_order=self._trotter_order,
        )

    def extract_phases_from_statevector(self, statevector: object) -> NDArray:
        """Extract per-qubit phases from a Qiskit Statevector.

        Requires scpn-quantum-control.
        """
        try:
            from scpn_quantum_control.analysis import statevector_to_phases
        except ImportError as exc:
            raise ImportError(
                "scpn-quantum-control is required for statevector analysis. "
                "Install with: pip install scpn-quantum-control"
            ) from exc
        phases = statevector_to_phases(statevector)  # pragma: no cover
        return np.asarray(phases, dtype=np.float64) % TWO_PI  # pragma: no cover
