# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling matrix builder

"""Coupling-matrix builders for generic and SCPN-layer topologies.

`CouplingBuilder` constructs deterministic `K_nm`, `alpha`, and optional
amplitude-coupling snapshots from validated scalar parameters. The Rust-backed
path and NumPy fallback share the same public contract: finite non-boolean
inputs, positive layer counts, zero diagonals, and explicit template labels for
runtime/audit reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST

FloatArray: TypeAlias = NDArray[np.float64]


__all__ = [
    "CouplingState",
    "CouplingBuilder",
    "SCPN_LAYER_TIMESCALES",
    "SCPN_LAYER_NAMES",
    "SCPN_CALIBRATION_ANCHORS",
]

# 16-layer SCPN hierarchy timescales (seconds).
# Source: HolonomicAtlas/src/knm_tools/knm_matrix_calculator.py,
# Paper 0 (UPDE framework), Paper 4 (synchronization).
SCPN_LAYER_TIMESCALES: dict[int, float] = {
    1: 0.1,  # Quantum Biological (~100ms)
    2: 0.004,  # Neurochemical (~25ms, 40Hz)
    3: 3600.0,  # Genomic (~1 hour)
    4: 2.0,  # Synchronization (~2s)
    5: 1.0,  # Psychoemotional (~1s)
    6: 86400.0,  # Planetary (~24h)
    7: 10.0,  # Geometrical-Symbolic (~10s)
    8: 31557600.0,  # Cosmic Phase Locking (~1 year)
    9: 3.154e9,  # Memory Imprint (~100 years)
    10: 1.0,  # Boundary Control (~1s)
    11: 86400.0,  # Noospheric (~24h)
    12: 31557600.0,  # Ecological Gaian (~1 year)
    13: 0.001,  # Source Field (~1ms)
    14: 1e-20,  # Transdimensional (~Planck)
    15: 1.0,  # Consilium Oversoul (~1s)
    16: 1.0,  # Meta Director (operational anchor; adjacency handled explicitly)
}

SCPN_LAYER_NAMES: dict[int, str] = {
    1: "Quantum",
    2: "Neural",
    3: "Genomic",
    4: "Tissue",
    5: "Psycho",
    6: "Planetary",
    7: "Symbolic",
    8: "Cosmic",
    9: "Memory",
    10: "Boundary",
    11: "Noospheric",
    12: "Gaian",
    13: "Source",
    14: "Transdim",
    15: "Consilium",
    16: "Meta",
}

# Calibration anchors from Paper 0 / HolonomicAtlas.
# These override the computed values for adjacent layers 1-5.
SCPN_CALIBRATION_ANCHORS: dict[tuple[int, int], float] = {
    (1, 2): 0.302,
    (2, 3): 0.201,
    (3, 4): 0.252,
    (4, 5): 0.154,
}


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be >= 1 as a non-boolean integer, got {value!r}")
    return int(value)


def _validate_finite_float(
    value: object,
    *,
    name: str,
    lower_bound: float | None = None,
    inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if lower_bound is not None:
        if inclusive and coerced < lower_bound:
            raise ValueError(f"{name} must be >= {lower_bound}, got {value!r}")
        if not inclusive and coerced <= lower_bound:
            raise ValueError(f"{name} must be > {lower_bound}, got {value!r}")
    return coerced


def _validate_layer_index(value: object, *, name: str, n_layers: int) -> int:
    index = _validate_positive_int(value, name=name)
    if index > n_layers:
        raise ValueError(f"{name} must be in [1, {n_layers}], got {value!r}")
    return index


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON object key is not allowed: {key!r}")
        record[key] = value
    return record


def _loads_knm_json(payload: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError:
        raise
    except ValueError as exc:
        raise ValueError("K_nm JSON must be canonical finite JSON") from exc


def _validate_coupling_output(
    knm: object,
    alpha: object,
    *,
    n_layers: int,
) -> tuple[FloatArray, FloatArray]:
    try:
        knm_array = np.asarray(knm, dtype=np.float64).reshape(n_layers, n_layers)
        alpha_array = np.asarray(alpha, dtype=np.float64).reshape(n_layers, n_layers)
    except (TypeError, ValueError) as exc:
        raise ValueError("coupling builder output must match requested shape") from exc
    if not np.all(np.isfinite(knm_array)):
        raise ValueError("coupling builder output K_nm must contain only finite values")
    if not np.all(np.isfinite(alpha_array)):
        raise ValueError(
            "coupling builder output alpha must contain only finite values"
        )
    if np.any(knm_array < 0.0):
        raise ValueError("coupling builder output K_nm must be non-negative")
    if not np.allclose(knm_array, knm_array.T, rtol=1e-12, atol=1e-12):
        raise ValueError("coupling builder output K_nm must be symmetric")
    if not np.allclose(np.diag(knm_array), 0.0, rtol=0.0, atol=1e-15):
        raise ValueError("coupling builder output K_nm diagonal must be zero")
    return (
        np.ascontiguousarray(knm_array, dtype=np.float64),
        np.ascontiguousarray(alpha_array, dtype=np.float64),
    )


@dataclass(frozen=True)
class CouplingState:
    """Immutable snapshot of phase/amplitude coupling matrices and template."""

    knm: FloatArray
    alpha: FloatArray
    active_template: str
    knm_r: FloatArray | None = None


class CouplingBuilder:
    """Builds Knm coupling matrices."""

    def build(
        self, n_layers: int, base_strength: float, decay_alpha: float
    ) -> CouplingState:
        """Build an exponentially decayed phase-coupling matrix.

        Parameters
        ----------
        n_layers
            Number of hierarchy layers or oscillators represented in the
            square coupling matrix.
        base_strength
            Coupling strength before distance decay is applied.
        decay_alpha
            Non-negative exponential decay coefficient in
            ``exp(-decay_alpha * |i - j|)``.

        Returns
        -------
        CouplingState
            Coupling snapshot with ``knm`` and ``alpha`` matrices of shape
            ``(n_layers, n_layers)``. The diagonal of ``knm`` is zero and
            ``alpha`` is initialised to zeros.

        Notes
        -----
        When the Rust extension is available, construction dispatches to
        ``spo_kernel.PyCouplingBuilder`` and preserves the same output
        contract as the NumPy fallback.
        """
        n_layers = _validate_positive_int(n_layers, name="n_layers")
        base_strength = _validate_finite_float(
            base_strength,
            name="base_strength",
            lower_bound=0.0,
        )
        decay_alpha = _validate_finite_float(
            decay_alpha,
            name="decay_alpha",
            lower_bound=0.0,
        )
        if _HAS_RUST:  # pragma: no cover
            from spo_kernel import PyCouplingBuilder

            try:
                d = PyCouplingBuilder().build(n_layers, base_strength, decay_alpha)
                n = _validate_positive_int(d["n"], name="rust n_layers")
                if n != n_layers:
                    raise ValueError("Rust coupling output layer count mismatch")
                rust_knm, rust_alpha = _validate_coupling_output(
                    d["knm"], d["alpha"], n_layers=n_layers
                )
                return CouplingState(
                    knm=rust_knm, alpha=rust_alpha, active_template="default"
                )
            except Exception as exc:
                _fallback_reason = exc
        idx = np.arange(n_layers)
        dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
        knm = base_strength * np.exp(-decay_alpha * dist)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n_layers, n_layers), dtype=np.float64)
        return CouplingState(knm=knm, alpha=alpha, active_template="default")

    def build_scpn_physics(
        self,
        k_base: float = 0.45,
        alpha_decay: float = 0.3,
    ) -> CouplingState:
        """Build 16×16 K_nm using SCPN layer physics.

        Three coupling mechanisms (Paper 0, HolonomicAtlas v2.4.0):
        - Adjacent: timescale matching with calibration anchors
        - Near-neighbor (|n-m|=2): geometric mean of intermediate path
        - Distant (|n-m|>=3): exponential decay with cross-hierarchy boosts

        Returns CouplingState with 16×16 matrix.
        """
        k_base = _validate_finite_float(
            k_base,
            name="k_base",
            lower_bound=0.0,
            inclusive=False,
        )
        alpha_decay = _validate_finite_float(
            alpha_decay,
            name="alpha_decay",
            lower_bound=0.0,
        )
        K = np.zeros((16, 16))

        # Pass 1: Adjacent layers. Use anchors where available.
        for n in range(1, 16):
            m = n + 1
            if (n, m) in SCPN_CALIBRATION_ANCHORS:
                val = SCPN_CALIBRATION_ANCHORS[(n, m)]
            else:
                val = self._adjacent_coupling(n, m, k_base)
            K[n - 1, m - 1] = val
            K[m - 1, n - 1] = val

        # Pass 2: Near-neighbor (|n-m|=2), geometric mean of intermediate path
        for n in range(1, 15):
            m = n + 2
            mid = (n + m) // 2
            k1 = K[n - 1, mid - 1]
            k2 = K[mid - 1, m - 1]
            val = float(np.sqrt(k1 * k2))
            # Frequency penalty for large timescale mismatch
            tau_n = SCPN_LAYER_TIMESCALES.get(n, 1.0)
            tau_m = SCPN_LAYER_TIMESCALES.get(m, 1.0)
            if n != 16 and m != 16 and tau_n > 0 and tau_m > 0:
                omega_n = 2.0 * np.pi / tau_n
                omega_m = 2.0 * np.pi / tau_m
                omega_avg = (omega_n + omega_m) / 2.0
                if omega_avg > 0:
                    penalty = 1.0 + abs(omega_n - omega_m) / omega_avg * 0.1
                    val /= penalty
            val = float(np.clip(val, 0.01, 0.4))
            K[n - 1, m - 1] = val
            K[m - 1, n - 1] = val

        # Pass 3: Distant (|n-m|>=3), exponential decay
        for n in range(1, 17):
            for m in range(n + 3, 17):
                dist = abs(n - m)
                val = k_base * np.exp(-alpha_decay * dist)
                val = float(np.clip(val, 0.001, 0.2))
                K[n - 1, m - 1] = val
                K[m - 1, n - 1] = val

        # Cross-hierarchy boosts (Paper 0, Section 5)
        _set_symmetric(K, 1, 16, max(K[0, 15], 0.05))  # Quantum-Meta
        _set_symmetric(K, 5, 7, max(K[4, 6], 0.15))  # Psycho-Symbolic

        alpha = np.zeros((16, 16), dtype=np.float64)
        return CouplingState(knm=K, alpha=alpha, active_template="scpn_physics")

    def apply_handshakes(
        self, state: CouplingState, handshakes_path: str | Path
    ) -> CouplingState:
        """Overlay documented inter-layer couplings from JSON spec.

        Reads the KNM_MATRIX_COMPLETE_SPECIFICATION.json format:
        each entry has from_layer, to_layer, coupling_strength.
        Negative values are preserved (inhibitory coupling).
        """
        path = Path(handshakes_path)
        data = _loads_knm_json(path.read_text(encoding="utf-8"))
        matrix = data.get("matrix")
        if not isinstance(matrix, list):
            raise ValueError("handshake matrix must be a list")
        knm = state.knm.copy()
        n_layers = knm.shape[0]
        for idx, entry in enumerate(matrix):
            if not isinstance(entry, dict):
                raise ValueError(f"matrix[{idx}] must be a mapping")
            fr = _validate_layer_index(
                entry.get("from_layer"),
                name="from_layer",
                n_layers=n_layers,
            )
            to = _validate_layer_index(
                entry.get("to_layer"),
                name="to_layer",
                n_layers=n_layers,
            )
            if fr == to:
                raise ValueError("handshake self-coupling entries are not physical")
            strength = _validate_finite_float(
                entry.get("coupling_strength"),
                name="coupling_strength",
            )
            knm[fr - 1, to - 1] = strength
            # Symmetric unless negative (directional inhibition)
            if strength >= 0:
                knm[to - 1, fr - 1] = strength
        return CouplingState(
            knm=knm,
            alpha=state.alpha.copy(),
            active_template="scpn_handshakes",
            knm_r=state.knm_r,
        )

    def build_with_amplitude(
        self,
        n_layers: int,
        base_strength: float,
        decay_alpha: float,
        amp_strength: float,
        amp_decay: float,
    ) -> CouplingState:
        """Build phase + amplitude coupling matrices together."""
        amp_strength = _validate_finite_float(
            amp_strength,
            name="amp_strength",
            lower_bound=0.0,
        )
        amp_decay = _validate_finite_float(
            amp_decay,
            name="amp_decay",
            lower_bound=0.0,
        )
        phase = self.build(n_layers, base_strength, decay_alpha)
        idx = np.arange(n_layers)
        dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
        knm_r = amp_strength * np.exp(-amp_decay * dist)
        np.fill_diagonal(knm_r, 0.0)
        return CouplingState(
            knm=phase.knm,
            alpha=phase.alpha,
            active_template=phase.active_template,
            knm_r=knm_r,
        )

    def switch_template(
        self,
        state: CouplingState,
        template_name: str,
        templates: dict[str, FloatArray],
    ) -> CouplingState:
        """Replace the active K_nm with a named template matrix."""
        if template_name not in templates:
            raise KeyError(f"Template {template_name!r} not found")
        template = np.asarray(templates[template_name], dtype=np.float64)
        if template.shape != state.knm.shape:
            raise ValueError(
                f"template shape {template.shape}, expected {state.knm.shape}"
            )
        if not np.all(np.isfinite(template)):
            raise ValueError("template values must be finite")
        if not np.allclose(np.diag(template), 0.0, rtol=0.0, atol=1e-15):
            raise ValueError("template self-coupling diagonal must be zero")
        return CouplingState(
            knm=template.copy(),
            alpha=state.alpha.copy(),
            active_template=template_name,
            knm_r=state.knm_r,
        )

    @staticmethod
    def _adjacent_coupling(n: int, m: int, k_base: float) -> float:
        """Adjacent layer coupling via timescale matching."""
        if n == 16 or m == 16:
            return 0.2
        tau_n = SCPN_LAYER_TIMESCALES[n]
        tau_m = SCPN_LAYER_TIMESCALES[m]
        if not (
            np.isfinite(tau_n) and np.isfinite(tau_m) and tau_n > 0.0 and tau_m > 0.0
        ):
            raise ValueError("layer timescales must be finite and positive")
        mismatch = abs(np.log(tau_n / tau_m))
        # Adjusted for stiff biological hierarchies (beta=0.05)
        val = k_base / (1.0 + 0.05 * mismatch)
        return float(np.clip(val, 0.1, 0.5))


def _set_symmetric(K: FloatArray, n: int, m: int, val: float) -> None:
    """Set K[n-1, m-1] and K[m-1, n-1] to val (1-indexed layer numbers)."""
    K[n - 1, m - 1] = val
    K[m - 1, n - 1] = val
