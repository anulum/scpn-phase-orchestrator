# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plasma-control bridge adapter

"""Plasma-control bridge for telemetry import, coupling expansion, and review.

The bridge converts plasma-layer coupling matrices, phase snapshots, natural
frequencies, Lyapunov verdicts, and proposed control action dictionaries into
SPO-compatible data structures under strict finite-shape validation. It also
checks local plasma invariants for review. The adapter performs no live plasma
actuation and does not require ``scpn-control`` to be installed.
"""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

__all__ = ["PlasmaControlBridge"]

TWO_PI = 2.0 * np.pi

# Physics thresholds (fallback when scpn-control is unavailable)
Q_MIN_STABLE = 1.0  # Kruskal-Shafranov limit
BETA_N_LIMIT = 2.8  # Troyon no-wall limit
GREENWALD_LIMIT = 1.2  # Greenwald density fraction
_PLASMA_LAYER_COUNT = 8
FloatArray: TypeAlias = NDArray[np.float64]


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _finite_array(value: object, *, name: str) -> FloatArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if raw.dtype.kind == "b":
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _finite_real(value: object, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real value")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real value")
    return result


def _finite_positive_real(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_non_negative_real(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _unit_interval(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _label(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or any(ord(ch) < 32 for ch in value):
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_layer_sizes(value: object, *, n_phases: int, n_layers: int) -> list[int]:
    if not isinstance(value, list) or len(value) != n_layers:
        raise ValueError("layer_sizes must contain one size per configured layer")
    sizes = [_validate_positive_or_zero_int(size, name="layer_sizes") for size in value]
    if sum(sizes) != n_phases:
        raise ValueError("layer_sizes must sum to phase count")
    return sizes


def _validate_positive_or_zero_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must contain non-negative integers")
    return int(value)


class PlasmaControlBridge:
    """Adapter between scpn-control plasma telemetry and phase-orchestrator types.

    All methods work without scpn-control installed (pure numpy + dict).
    """

    _n_layers: int

    def __init__(self, n_layers: int = 8):
        n_layers = _validate_positive_int(n_layers, name="n_layers")
        if n_layers > _PLASMA_LAYER_COUNT:
            raise ValueError(
                f"n_layers must be <= {_PLASMA_LAYER_COUNT}, got {n_layers!r}"
            )
        self._n_layers = n_layers

    def import_knm_spec(self, knm_spec_or_dict: object) -> CouplingState:
        """Expand an (L, L) layer-coupling matrix to (N, N) via Kronecker replication.

        Accepts a dict with 'matrix' key (list of lists or ndarray) and optional
        'n_osc_per_layer' (int, default 2), or a raw ndarray of shape (L, L).
        """
        if isinstance(knm_spec_or_dict, dict):
            layer_knm = _finite_array(knm_spec_or_dict["matrix"], name="Layer Knm")
            n_per = _validate_positive_int(
                knm_spec_or_dict.get("n_osc_per_layer", 2),
                name="n_osc_per_layer",
            )
        else:
            layer_knm = _finite_array(knm_spec_or_dict, name="Layer Knm")
            n_per = 2

        if layer_knm.ndim != 2 or layer_knm.shape[0] != layer_knm.shape[1]:
            raise ValueError(f"Layer Knm must be square, got shape {layer_knm.shape}")
        if layer_knm.shape != (self._n_layers, self._n_layers):
            raise ValueError(
                f"Layer Knm shape {layer_knm.shape} must match "
                f"n_layers={self._n_layers}"
            )
        if not np.allclose(np.diag(layer_knm), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("Layer Knm self-coupling diagonal must be zero")

        # Kronecker expansion: each layer block shares the inter-layer coupling
        n_total = layer_knm.shape[0] * n_per
        knm = np.kron(layer_knm, np.ones((n_per, n_per), dtype=np.float64)).astype(
            np.float64
        )
        np.fill_diagonal(knm, 0.0)

        return CouplingState(
            knm=knm,
            alpha=np.zeros((n_total, n_total), dtype=np.float64),
            active_template="plasma_import",
        )

    def import_plasma_omega(self, n_osc_per_layer: int = 1) -> FloatArray:
        """Generate natural frequencies spanning plasma timescales.

        Returns frequencies ordered: micro_turbulence(fast) → plasma_wall(slow).
        """
        n_osc_per_layer = _validate_positive_int(
            n_osc_per_layer,
            name="n_osc_per_layer",
        )
        # 8 characteristic plasma timescales (normalised, rad/s)
        layer_omegas = np.array(
            [
                10.0,  # micro_turbulence
                8.0,  # zonal_flow
                3.0,  # mhd_tearing
                5.0,  # sawtooth_elm
                0.5,  # transport_barrier
                0.3,  # current_profile
                0.1,  # global_equilibrium
                1.0,  # plasma_wall
            ],
            dtype=np.float64,
        )[: self._n_layers]

        if n_osc_per_layer == 1:
            return layer_omegas
        return np.repeat(layer_omegas, n_osc_per_layer)

    def import_snapshot(self, tick_result: dict) -> UPDEState:
        """Convert an scpn-control tick result dict to UPDEState.

        Expected keys: 'phases' (1-D array), optional 'regime', 'layer_sizes'.
        """
        if not isinstance(tick_result, dict):
            raise ValueError("tick_result must be a dict")
        phases = _finite_array(tick_result["phases"], name="phases")
        if phases.ndim != 1:
            raise ValueError("phases must be a 1-D array")
        if phases.size == 0:
            raise ValueError("phases must not be empty")
        phases = phases % TWO_PI
        regime = _label(tick_result.get("regime", "NOMINAL"), name="regime")
        layer_sizes = tick_result.get("layer_sizes")

        if layer_sizes is None:
            n_per = max(1, len(phases) // self._n_layers)
            layer_sizes = [n_per] * self._n_layers
        else:
            layer_sizes = _validate_layer_sizes(
                layer_sizes,
                n_phases=len(phases),
                n_layers=self._n_layers,
            )

        layers: list[LayerState] = []
        offset = 0
        for size in layer_sizes:
            group = phases[offset : offset + size]
            offset += size
            if len(group) == 0:
                layers.append(LayerState(R=0.0, psi=0.0))
                continue
            z = np.exp(1j * group)
            order = z.mean()
            r_val = float(np.abs(order))
            psi_val = float(np.angle(order) % TWO_PI)
            layers.append(LayerState(R=r_val, psi=psi_val))

        n_l = len(layers)
        cross = np.eye(n_l, dtype=np.float64)
        stability = _unit_interval(tick_result.get("stability", 0.5), name="stability")

        return UPDEState(
            layers=layers,
            cross_layer_alignment=cross,
            stability_proxy=stability,
            regime_id=regime,
        )

    def import_lyapunov_verdict(self, verdict_or_dict: object) -> dict:
        """Map a Lyapunov verdict to a boundary-compatible signal dict.

        Accepts dict with 'score' (float in [0,1]).
        """
        if isinstance(verdict_or_dict, dict):
            score = _unit_interval(verdict_or_dict.get("score", 0.0), name="score")
        else:
            score = _unit_interval(getattr(verdict_or_dict, "score", 0.0), name="score")
        return {
            "lyapunov_score": score,
            "stable": score > 0.3,
        }

    def export_control_actions(self, actions: list) -> dict:
        """Package a list of control action dicts for scpn-control consumption."""
        if not isinstance(actions, list):
            raise ValueError("actions must be a list of dicts")
        exported: list[dict[str, object]] = []
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("actions must contain dict entries")
            knob = _label(action.get("knob", "K"), name="knob")
            scope = _label(action.get("scope", "global"), name="scope")
            value = _finite_real(action.get("value", 0.0), name="value")
            exported.append({"knob": knob, "scope": scope, "value": value})
        return {
            "actions": exported,
        }

    def check_physics_invariants(self, values: dict) -> list[dict]:
        """Check plasma physics invariants against local thresholds.

        Returns a list of violation dicts (empty if all invariants hold).
        """
        if not isinstance(values, dict):
            raise ValueError("physics invariant values must be a dict")
        violations: list[dict] = []
        q_min = values.get("q_min")
        if q_min is not None:
            q_min = _finite_positive_real(q_min, name="physics invariant q_min")
        if q_min is not None and q_min < Q_MIN_STABLE:
            violations.append(
                {
                    "variable": "q_min",
                    "value": q_min,
                    "threshold": Q_MIN_STABLE,
                    "severity": "hard",
                    "message": f"q_min={q_min:.3f} < {Q_MIN_STABLE}",
                }
            )
        beta_n = values.get("beta_n")
        if beta_n is not None:
            beta_n = _finite_non_negative_real(
                beta_n,
                name="physics invariant beta_n",
            )
        if beta_n is not None and beta_n > BETA_N_LIMIT:
            violations.append(
                {
                    "variable": "beta_n",
                    "value": beta_n,
                    "threshold": BETA_N_LIMIT,
                    "severity": "hard",
                    "message": f"beta_n={beta_n:.3f} > {BETA_N_LIMIT} (Troyon no-wall)",
                }
            )
        greenwald = values.get("greenwald")
        if greenwald is not None:
            greenwald = _finite_non_negative_real(
                greenwald,
                name="physics invariant greenwald",
            )
        if greenwald is not None and greenwald > GREENWALD_LIMIT:
            violations.append(
                {
                    "variable": "greenwald",
                    "value": greenwald,
                    "threshold": GREENWALD_LIMIT,
                    "severity": "hard",
                    "message": f"greenwald={greenwald:.3f} > {GREENWALD_LIMIT}",
                }
            )
        return violations
