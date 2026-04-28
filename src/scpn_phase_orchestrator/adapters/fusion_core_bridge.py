# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fusion-Core bridge adapter

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["FusionCoreBridge"]

TWO_PI = 2.0 * np.pi

BETA_N_LIMIT = 2.8  # Troyon no-wall limit
Q_MIN_STABLE = 1.0  # Kruskal-Shafranov limit
TAU_E_REF_S = 3.0  # ITER confinement target, seconds
MHD_THRESHOLD = 1.0  # normalised MHD amplitude threshold

_OBS_NAMES = [
    "q_profile",
    "beta_n",
    "tau_e",
    "sawtooth_count",
    "elm_count",
    "mhd_amplitude",
]


class FusionCoreBridge:
    """Adapter between scpn-fusion-core equilibrium data and phase-orchestrator.

    All methods work without scpn-fusion-core (pure numpy + dict).
    """

    def __init__(self, n_layers: int = 6):
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self._n_layers = n_layers

    def observables_to_phases(self, snapshot: dict) -> NDArray:
        """Map 6 fusion observables to [0, 2*pi) phases.

        Observable → Phase formula:
          q_profile       → 2*pi*(q - q_min)/(q_max - q_min)
          beta_n          → 2*pi*beta_n/beta_limit
          tau_e           → 2*pi*tau_e/tau_ref
          sawtooth_count  → count*pi mod 2*pi
          elm_count       → count*pi mod 2*pi
          mhd_amplitude   → 2*pi*amplitude/threshold
        """
        q = float(snapshot.get("q_profile", 1.5))
        q_min = float(snapshot.get("q_min", 1.0))
        q_max = float(snapshot.get("q_max", 5.0))
        beta_n = float(snapshot.get("beta_n", 1.0))
        tau_e = float(snapshot.get("tau_e", 1.0))
        saw_count = int(snapshot.get("sawtooth_count", 0))
        elm_count = int(snapshot.get("elm_count", 0))
        mhd_amp = float(snapshot.get("mhd_amplitude", 0.0))

        denom_q = q_max - q_min if q_max != q_min else 1.0
        phases = np.array(
            [
                TWO_PI * np.clip((q - q_min) / denom_q, 0.0, 1.0),
                TWO_PI * np.clip(beta_n / BETA_N_LIMIT, 0.0, 1.0),
                TWO_PI * np.clip(tau_e / TAU_E_REF_S, 0.0, 1.0),
                (saw_count * np.pi) % TWO_PI,
                (elm_count * np.pi) % TWO_PI,
                TWO_PI * np.clip(mhd_amp / MHD_THRESHOLD, 0.0, 1.0),
            ],
            dtype=np.float64,
        )

        return phases[: self._n_layers]

    def phases_to_feedback(self, phases: NDArray, omegas: NDArray) -> dict:
        """Convert phase state back to feedback signals for the equilibrium solver."""
        n = min(len(phases), self._n_layers)
        z = np.exp(1j * phases[:n])
        order = z.mean()
        r = float(np.abs(order))
        psi = float(np.angle(order) % TWO_PI)
        return {
            "R_global": r,
            "mean_phase": psi,
            "mean_omega": float(np.mean(omegas[:n])),
            "n_oscillators": n,
        }

    def import_q_profile(self, q_profile_or_dict: object) -> dict:
        """Parse a q-profile from dict or scpn-fusion-core object.

        Returns normalised dict with keys: q_min, q_max, q_axis, q_edge.
        """
        if isinstance(q_profile_or_dict, dict):
            q_min = float(q_profile_or_dict.get("q_min", 1.0))
            q_max = float(q_profile_or_dict.get("q_max", 5.0))
            q_axis = float(q_profile_or_dict.get("q_axis", q_min))
            q_edge = float(q_profile_or_dict.get("q_edge", q_max))
        else:
            q_min = float(getattr(q_profile_or_dict, "q_min", 1.0))
            q_max = float(getattr(q_profile_or_dict, "q_max", 5.0))
            q_axis = float(getattr(q_profile_or_dict, "q_axis", q_min))
            q_edge = float(getattr(q_profile_or_dict, "q_edge", q_max))
        return {"q_min": q_min, "q_max": q_max, "q_axis": q_axis, "q_edge": q_edge}

    def import_equilibrium(self, kernel_result: dict) -> dict:
        """Extract equilibrium observables from a fusion kernel result dict."""
        return {
            "q_profile": float(kernel_result.get("q_profile", 1.5)),
            "beta_n": float(kernel_result.get("beta_n", 1.0)),
            "tau_e": float(kernel_result.get("tau_e", 1.0)),
            "sawtooth_count": int(kernel_result.get("sawtooth_count", 0)),
            "elm_count": int(kernel_result.get("elm_count", 0)),
            "mhd_amplitude": float(kernel_result.get("mhd_amplitude", 0.0)),
        }

    def check_stability(self, observables: dict) -> list[dict]:
        """Check fusion stability invariants.

        Returns a list of violation dicts (empty if all invariants hold).
        """
        violations: list[dict] = []
        q_min = observables.get("q_min")
        if q_min is None:
            q_min = observables.get("q_profile")
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
        beta_n = observables.get("beta_n")
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
        tau_ratio = observables.get("tau_e_ratio")
        if tau_ratio is not None and tau_ratio < 0.5:
            violations.append(
                {
                    "variable": "tau_e_ratio",
                    "value": tau_ratio,
                    "threshold": 0.5,
                    "severity": "soft",
                    "message": f"tau_e_ratio={tau_ratio:.3f} < 0.5",
                }
            )
        return violations
