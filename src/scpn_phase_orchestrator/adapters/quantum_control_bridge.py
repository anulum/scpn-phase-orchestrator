# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Quantum-control bridge adapter

"""Quantum-control bridge for reviewable Hamiltonian and phase handoffs.

The bridge imports quantum phase artifacts into UPDE diagnostics, exports UPDE
state summaries, validates coupling/frequency arrays, and can build deterministic
OpenQASM manifest handoffs with parity hashes and actuation disabled. Live
Hamiltonian or Q-UPDE execution is delegated only when external quantum-control
packages are explicitly imported by the called method.
"""

from __future__ import annotations

import json
from hashlib import sha256
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

__all__ = ["QuantumControlBridge"]

TWO_PI = 2.0 * np.pi
FloatArray: TypeAlias = NDArray[np.float64]


def _require_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= 1")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return parsed


def _require_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast("dict[str, object]", value)


def _require_positive_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _require_fidelity(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and in the range [0.0, 1.0]")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be finite and in the range [0.0, 1.0]")
    return parsed


def _require_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _finite_array(value: object, *, name: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _validate_layer_assignments(
    layer_assignments: object,
    *,
    n_phases: int,
) -> list[list[int]]:
    if not isinstance(layer_assignments, list):
        raise ValueError("layer_assignments must be a list of index groups")

    seen: set[int] = set()
    validated: list[list[int]] = []
    for group in layer_assignments:
        if not isinstance(group, list):
            raise ValueError("layer_assignments must contain list groups")
        validated_group: list[int] = []
        for index in group:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise ValueError("layer_assignments must contain integer indexes")
            parsed = int(index)
            if parsed < 0 or parsed >= n_phases:
                raise ValueError("layer_assignments index out of phase range")
            if parsed in seen:
                raise ValueError("layer_assignments must not repeat phase indexes")
            seen.add(parsed)
            validated_group.append(parsed)
        validated.append(validated_group)

    if len(seen) != n_phases:
        raise ValueError("layer_assignments must cover every phase index exactly once")

    return validated


def _validate_upde_state(state: object) -> tuple[list[LayerState], FloatArray]:
    if not isinstance(state, UPDEState):
        raise ValueError("state must be a UPDEState")

    if not isinstance(state.layers, list):
        raise ValueError("UPDEState.layers must be a list")

    for index, layer in enumerate(state.layers):
        if not isinstance(layer, LayerState):
            raise ValueError(f"UPDEState.layers[{index}] must be a LayerState")
        _require_finite_real(layer.R, name=f"UPDEState.layers[{index}].R")
        _require_finite_real(layer.psi, name=f"UPDEState.layers[{index}].psi")

    cross_layer_alignment = _finite_array(
        state.cross_layer_alignment,
        name="cross_layer_alignment",
    )
    if cross_layer_alignment.ndim != 2:
        raise ValueError("cross_layer_alignment must be a square matrix")
    if cross_layer_alignment.shape[0] != cross_layer_alignment.shape[1]:
        raise ValueError("cross_layer_alignment must be a square matrix")

    n_layers = len(state.layers)
    if cross_layer_alignment.shape != (n_layers, n_layers):
        raise ValueError("cross_layer_alignment shape must match number of layers")

    return state.layers, cross_layer_alignment


class QuantumControlBridge:
    """Adapter between scpn-quantum-control artifacts and phase-orchestrator types.

    The QuantumControlBridge enables the mapping of classical Kuramoto
    phase dynamics onto Quantum Hardware (isomorphic XY spin Hamiltonian).
    It supports Hamiltonian construction, Trotterized time evolution (Q-UPDE),
    and variational synchronization minimization.
    """

    def __init__(self, n_oscillators: int, trotter_order: int = 1):
        n_oscillators = _require_positive_integer(n_oscillators, name="n_oscillators")
        if isinstance(trotter_order, bool) or not isinstance(trotter_order, Integral):
            raise ValueError("trotter_order must be an integer >= 1")
        if trotter_order < 1:
            raise ValueError("trotter_order must be an integer >= 1")
        self._n: int = int(n_oscillators)
        self._trotter_order: int = int(trotter_order)

    def import_artifact(self, artifact_dict: dict) -> UPDEState:
        """Convert a scpn-quantum-control result dict into UPDEState."""
        artifact = _require_mapping(artifact_dict, name="artifact_dict")
        if "phases" not in artifact:
            raise ValueError("artifact_dict must include 'phases'")
        phases = _finite_array(artifact["phases"], name="phases")
        if phases.shape != (self._n,):
            raise ValueError(
                f"phases shape {phases.shape} does not match n_oscillators={self._n}"
            )
        phases = phases % TWO_PI
        fidelity = _require_fidelity(artifact.get("fidelity", 0.0), name="fidelity")

        layer_assignments = artifact.get("layer_assignments")
        if layer_assignments is None:
            mid = len(phases) // 2
            layer_assignments = [list(range(mid)), list(range(mid, len(phases)))]
        layer_assignments = _validate_layer_assignments(
            layer_assignments,
            n_phases=len(phases),
        )

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
        regime = str(artifact.get("regime", "NOMINAL"))

        return UPDEState(
            layers=layers,
            cross_layer_alignment=cross,
            stability_proxy=fidelity,
            regime_id=regime,
        )

    def export_artifact(self, state: UPDEState) -> dict:
        """Convert UPDEState back to a dict compatible with scpn-quantum-control."""
        layers, cross_layer_alignment = _validate_upde_state(state)
        fidelity = _require_finite_real(
            state.stability_proxy,
            name="state.stability_proxy",
        )
        return {
            "regime": state.regime_id,
            "fidelity": fidelity,
            "layers": [{"R": ls.R, "psi": ls.psi} for ls in layers],
            "cross_alignment": cross_layer_alignment.tolist(),
        }

    def import_knm(self, knm_array: FloatArray) -> CouplingState:
        """Wrap a coupling matrix from quantum calibration into CouplingState."""
        knm = _finite_array(knm_array, name="Knm")
        if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
            raise ValueError(f"Knm must be square, got shape {knm.shape}")
        if knm.shape != (self._n, self._n):
            raise ValueError(
                f"Knm shape {knm.shape} does not match n_oscillators={self._n}"
            )
        n = knm.shape[0]
        return CouplingState(
            knm=knm.copy(),
            alpha=np.zeros((n, n), dtype=np.float64),
            active_template="quantum_import",
        )

    def build_quantum_compiler_manifest(
        self,
        knm: FloatArray,
        omegas: FloatArray,
        *,
        dt: float,
    ) -> dict[str, object]:
        """Return a deterministic OpenQASM handoff with parity evidence.

        The manifest is dependency-free review output for Qiskit/PennyLane
        simulator handoff. It does not execute on a QPU and does not permit
        live actuation.
        """
        knm_array, omega_array = self._validate_compiler_inputs(knm, omegas, dt=dt)
        frequency_terms: list[dict[str, object]] = [
            {
                "qubit": idx,
                "omega": float(omega_array[idx]),
                "angle": float(omega_array[idx] * dt),
            }
            for idx in range(self._n)
        ]
        coupling_terms = self._quantum_coupling_terms(knm_array, dt=dt)
        openqasm = self._render_openqasm(frequency_terms, coupling_terms)
        qasm_hash = sha256(openqasm.encode("utf-8")).hexdigest()
        parity = self._quantum_compiler_parity(
            omega_array,
            frequency_terms,
            knm_array,
            coupling_terms,
        )
        manifest: dict[str, object] = {
            "manifest_kind": "quantum_compiler_manifest",
            "schema_version": 1,
            "status": (
                "co_simulation_parity_passed"
                if parity["max_abs_frequency_error"] == 0.0
                and parity["max_abs_coupling_error"] == 0.0
                else "co_simulation_parity_failed"
            ),
            "target_backends": ["qiskit_openqasm3", "pennylane_qasm"],
            "n_qubits": self._n,
            "trotter_order": self._trotter_order,
            "dt": float(dt),
            "qpu_execution_permitted": False,
            "actuation_permitted": False,
            "frequency_terms": frequency_terms,
            "coupling_terms": coupling_terms,
            "openqasm": openqasm,
            "qasm_sha256": qasm_hash,
            "co_simulation_parity": parity,
            "operator_commands": [
                "review quantum_compiler_manifest.json",
                "run Qiskit or PennyLane simulator parity before QPU handoff",
            ],
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        manifest["manifest_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
        return manifest

    def build_hamiltonian(self, knm: FloatArray, omegas: FloatArray) -> object:
        """Build Kuramoto XY Hamiltonian as SparsePauliOp.

        Requires scpn-quantum-control.
        """
        knm, omegas = self._validate_knm_omegas(knm, omegas)
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_hamiltonian

        return knm_to_hamiltonian(knm, omegas)

    def solve_q_upde(
        self,
        knm: FloatArray,
        omegas: FloatArray,
        t_max: float = 1.0,
        dt: float = 0.1,
        trotter_per_step: int = 5,
    ) -> dict:
        """Execute Trotterized quantum simulation of the phase network (Q-UPDE).

        This method maps the classical sin(delta theta) interaction to
        the XY spin exchange interaction (XX + YY) and natural frequencies
        to Z-axis magnetic fields.

        Requires scpn-quantum-control.
        """
        t_max = _require_positive_real(t_max, name="t_max")
        dt = _require_positive_real(dt, name="dt")
        trotter_per_step = _require_positive_integer(
            trotter_per_step,
            name="trotter_per_step",
        )
        knm, omegas = self._validate_knm_omegas(knm, omegas)
        from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

        solver = QuantumKuramotoSolver(
            n_oscillators=len(omegas),
            K_coupling=knm,
            omega_natural=omegas,
            trotter_order=self._trotter_order,
        )
        return cast(
            "dict[str, object]",
            solver.run(t_max=t_max, dt=dt, trotter_per_step=trotter_per_step),
        )

    def orchestrator_to_quantum(
        self,
        state: UPDEState,
    ) -> FloatArray:
        """Convert orchestrator UPDEState to quantum phase array."""
        from scpn_quantum_control import (  # noqa: PLC0415
            orchestrator_to_quantum_phases,
        )

        payload = self.export_artifact(state)
        layer_phases = {
            f"layer_{i}": ls["psi"] for i, ls in enumerate(payload["layers"])
        }
        return cast("FloatArray", orchestrator_to_quantum_phases(layer_phases))

    def quantum_to_orchestrator(
        self,
        quantum_theta: FloatArray,
    ) -> dict:
        """Convert quantum phase array back to orchestrator-compatible dict."""
        from scpn_quantum_control import (  # noqa: PLC0415
            quantum_to_orchestrator_phases,
        )

        return cast("dict", quantum_to_orchestrator_phases(quantum_theta))

    def _validate_compiler_inputs(
        self,
        knm: FloatArray,
        omegas: FloatArray,
        *,
        dt: float,
    ) -> tuple[FloatArray, FloatArray]:
        knm_array, omega_array = self._validate_knm_omegas(knm, omegas)
        _require_positive_real(dt, name="dt")
        return knm_array, omega_array

    def _validate_knm_omegas(
        self,
        knm: FloatArray,
        omegas: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        knm_array = _finite_array(knm, name="knm")
        omega_array = _finite_array(omegas, name="omegas")
        if knm_array.shape != (self._n, self._n):
            raise ValueError(
                f"knm shape {knm_array.shape} does not match n_oscillators={self._n}"
            )
        if omega_array.shape != (self._n,):
            raise ValueError(
                f"omegas shape {omega_array.shape} does not match "
                f"n_oscillators={self._n}"
            )
        return knm_array.copy(), omega_array.copy()

    def _quantum_coupling_terms(
        self,
        knm: FloatArray,
        *,
        dt: float,
    ) -> list[dict[str, object]]:
        terms: list[dict[str, object]] = []
        for source in range(self._n):
            for target in range(source + 1, self._n):
                forward = float(knm[source, target])
                reverse = float(knm[target, source])
                if forward == 0.0 and reverse == 0.0:
                    continue
                coupling = 0.5 * (forward + reverse)
                if coupling == 0.0:
                    continue
                angle = coupling * dt
                terms.append(
                    {
                        "source": source,
                        "target": target,
                        "forward_coupling": forward,
                        "reverse_coupling": reverse,
                        "symmetric_coupling": coupling,
                        "xx_angle": float(angle),
                        "yy_angle": float(angle),
                    }
                )
        return terms

    def _render_openqasm(
        self,
        frequency_terms: list[dict[str, object]],
        coupling_terms: list[dict[str, object]],
    ) -> str:
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{self._n}] q;",
        ]
        for term in frequency_terms:
            lines.append(f"rz({_qasm_float(term['angle'])}) q[{term['qubit']}];")
        for term in coupling_terms:
            source = term["source"]
            target = term["target"]
            lines.append(
                f"rxx({_qasm_float(term['xx_angle'])}) q[{source}], q[{target}];"
            )
            lines.append(
                f"ryy({_qasm_float(term['yy_angle'])}) q[{source}], q[{target}];"
            )
        return "\n".join(lines) + "\n"

    def _quantum_compiler_parity(
        self,
        omegas: FloatArray,
        frequency_terms: list[dict[str, object]],
        knm: FloatArray,
        coupling_terms: list[dict[str, object]],
    ) -> dict[str, object]:
        frequency_error = 0.0
        for term in frequency_terms:
            qubit = _int_field(term, "qubit")
            frequency_error = max(
                frequency_error,
                abs(_float_field(term, "omega") - float(omegas[qubit])),
            )
        coupling_error = 0.0
        for term in coupling_terms:
            source = _int_field(term, "source")
            target = _int_field(term, "target")
            expected = 0.5 * (float(knm[source, target]) + float(knm[target, source]))
            coupling_error = max(
                coupling_error,
                abs(_float_field(term, "symmetric_coupling") - expected),
            )
        return {
            "engine": "deterministic_xy_term_reconstruction",
            "max_abs_frequency_error": frequency_error,
            "max_abs_coupling_error": coupling_error,
            "term_count": len(frequency_terms) + len(coupling_terms),
        }


def _qasm_float(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("QASM angle must be numeric")
    return f"{float(value):.12f}"


def _int_field(mapping: dict[str, object], key: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _float_field(mapping: dict[str, object], key: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)
