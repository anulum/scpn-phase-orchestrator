# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plasma-control bridge adapter

from __future__ import annotations

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


class PlasmaControlBridge:
    """Adapter between scpn-control plasma telemetry and phase-orchestrator types.

    All methods work without scpn-control installed (pure numpy + dict).
    """

    def __init__(self, n_layers: int = 8):
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self._n_layers = n_layers

    def import_knm_spec(self, knm_spec_or_dict: object) -> CouplingState:
        """Expand an (L, L) layer-coupling matrix to (N, N) via Kronecker replication.

        Accepts a dict with 'matrix' key (list of lists or ndarray) and optional
        'n_osc_per_layer' (int, default 2), or a raw ndarray of shape (L, L).
        """
        if isinstance(knm_spec_or_dict, dict):
            layer_knm = np.asarray(knm_spec_or_dict["matrix"], dtype=np.float64)
            n_per = int(knm_spec_or_dict.get("n_osc_per_layer", 2))
        else:
            layer_knm = np.asarray(knm_spec_or_dict, dtype=np.float64)
            n_per = 2

        if layer_knm.ndim != 2 or layer_knm.shape[0] != layer_knm.shape[1]:
            raise ValueError(f"Layer Knm must be square, got shape {layer_knm.shape}")

        # Kronecker expansion: each layer block shares the inter-layer coupling
        n_total = layer_knm.shape[0] * n_per
        knm = np.kron(layer_knm, np.ones((n_per, n_per), dtype=np.float64))
        np.fill_diagonal(knm, 0.0)

        return CouplingState(
            knm=knm,
            alpha=np.zeros((n_total, n_total), dtype=np.float64),
            active_template="plasma_import",
        )

    def import_plasma_omega(self, n_osc_per_layer: int = 1) -> NDArray:
        """Generate natural frequencies spanning plasma timescales.

        Returns frequencies ordered: micro_turbulence(fast) → plasma_wall(slow).
        """
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
        phases = np.asarray(tick_result["phases"], dtype=np.float64) % TWO_PI
        regime = str(tick_result.get("regime", "NOMINAL"))
        layer_sizes = tick_result.get("layer_sizes")

        if layer_sizes is None:
            n_per = max(1, len(phases) // self._n_layers)
            layer_sizes = [n_per] * self._n_layers

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
        stability = float(tick_result.get("stability", 0.5))

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
            score = float(verdict_or_dict.get("score", 0.0))
        else:
            score = float(getattr(verdict_or_dict, "score", 0.0))
        return {
            "lyapunov_score": score,
            "stable": score > 0.3,
        }

    def export_control_actions(self, actions: list) -> dict:
        """Package a list of control action dicts for scpn-control consumption."""
        return {
            "actions": [
                {
                    "knob": a.get("knob", "K"),
                    "scope": a.get("scope", "global"),
                    "value": float(a.get("value", 0.0)),
                }
                for a in actions
            ],
        }

    def check_physics_invariants(self, values: dict) -> list[dict]:
        """Check plasma physics invariants against local thresholds.

        Returns a list of violation dicts (empty if all invariants hold).
        """
        violations: list[dict] = []
        q_min = values.get("q_min")
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
