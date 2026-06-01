# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fusion-Core bridge adapter

"""Fusion-Core bridge for phase encoding and stability-review diagnostics.

The bridge maps fusion equilibrium observables into bounded phase vectors,
returns aggregate phase feedback summaries, normalises q-profile/equilibrium
payloads, and checks local fusion stability invariants. It is pure NumPy/dict
code and does not require or call a live fusion solver; outputs are review
signals and feedback dictionaries for explicit downstream handoff.
"""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["FusionCoreBridge"]

TWO_PI = 2.0 * np.pi
FloatArray: TypeAlias = NDArray[np.float64]

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


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_vector(value: object, *, name: str) -> FloatArray:
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
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _validate_q_bounds(q_min: float, q_max: float) -> None:
    if q_max <= q_min:
        raise ValueError("q_max must be greater than q_min")


class FusionCoreBridge:
    """Adapter between scpn-fusion-core equilibrium data and phase-orchestrator.

    All methods work without scpn-fusion-core (pure numpy + dict).
    """

    _n_layers: int

    def __init__(self, n_layers: int = 6):
        if (
            isinstance(n_layers, bool)
            or not isinstance(n_layers, Integral)
            or not 1 <= n_layers <= len(_OBS_NAMES)
        ):
            raise ValueError(
                f"n_layers must be an integer in [1, {len(_OBS_NAMES)}], "
                f"got {n_layers!r}"
            )
        self._n_layers = int(n_layers)

    def observables_to_phases(self, snapshot: dict) -> FloatArray:
        """Map 6 fusion observables to [0, 2*pi) phases.

        Observable → Phase formula:
          q_profile       → 2*pi*(q - q_min)/(q_max - q_min)
          beta_n          → 2*pi*beta_n/beta_limit
          tau_e           → 2*pi*tau_e/tau_ref
          sawtooth_count  → count*pi mod 2*pi
          elm_count       → count*pi mod 2*pi
          mhd_amplitude   → 2*pi*amplitude/threshold
        """
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot must be a dict")
        q = _finite_positive_real(snapshot.get("q_profile", 1.5), name="q_profile")
        q_min = _finite_positive_real(snapshot.get("q_min", 1.0), name="q_min")
        q_max = _finite_positive_real(snapshot.get("q_max", 5.0), name="q_max")
        _validate_q_bounds(q_min, q_max)
        beta_n = _finite_non_negative_real(snapshot.get("beta_n", 1.0), name="beta_n")
        tau_e = _finite_non_negative_real(snapshot.get("tau_e", 1.0), name="tau_e")
        saw_count = _non_negative_int(
            snapshot.get("sawtooth_count", 0),
            name="sawtooth_count",
        )
        elm_count = _non_negative_int(snapshot.get("elm_count", 0), name="elm_count")
        mhd_amp = _finite_non_negative_real(
            snapshot.get("mhd_amplitude", 0.0),
            name="mhd_amplitude",
        )

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

    def phases_to_feedback(
        self,
        phases: FloatArray,
        omegas: FloatArray,
    ) -> dict:
        """Convert phase state back to feedback signals for the equilibrium solver."""
        phases = _finite_vector(phases, name="phases")
        omegas = _finite_vector(omegas, name="omegas")
        if omegas.size < min(phases.size, self._n_layers):
            raise ValueError("omegas length must cover feedback oscillator count")
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
            q_min = _finite_positive_real(
                q_profile_or_dict.get("q_min", 1.0),
                name="q_min",
            )
            q_max = _finite_positive_real(
                q_profile_or_dict.get("q_max", 5.0),
                name="q_max",
            )
            q_axis = _finite_real(
                q_profile_or_dict.get("q_axis", q_min),
                name="q_axis",
            )
            q_edge = _finite_real(
                q_profile_or_dict.get("q_edge", q_max),
                name="q_edge",
            )
        else:
            q_min = _finite_positive_real(
                getattr(q_profile_or_dict, "q_min", 1.0),
                name="q_min",
            )
            q_max = _finite_positive_real(
                getattr(q_profile_or_dict, "q_max", 5.0),
                name="q_max",
            )
            q_axis = _finite_real(
                getattr(q_profile_or_dict, "q_axis", q_min),
                name="q_axis",
            )
            q_edge = _finite_real(
                getattr(q_profile_or_dict, "q_edge", q_max),
                name="q_edge",
            )
        _validate_q_bounds(q_min, q_max)
        if not q_min <= q_axis <= q_max:
            raise ValueError("q_axis must be within q_min and q_max")
        if not q_min <= q_edge <= q_max:
            raise ValueError("q_edge must be within q_min and q_max")
        return {"q_min": q_min, "q_max": q_max, "q_axis": q_axis, "q_edge": q_edge}

    def import_equilibrium(self, kernel_result: dict) -> dict:
        """Extract equilibrium observables from a fusion kernel result dict."""
        if not isinstance(kernel_result, dict):
            raise ValueError("kernel_result must be a dict")
        return {
            "q_profile": _finite_positive_real(
                kernel_result.get("q_profile", 1.5),
                name="q_profile",
            ),
            "beta_n": _finite_non_negative_real(
                kernel_result.get("beta_n", 1.0),
                name="beta_n",
            ),
            "tau_e": _finite_non_negative_real(
                kernel_result.get("tau_e", 1.0),
                name="tau_e",
            ),
            "sawtooth_count": _non_negative_int(
                kernel_result.get("sawtooth_count", 0),
                name="sawtooth_count",
            ),
            "elm_count": _non_negative_int(
                kernel_result.get("elm_count", 0),
                name="elm_count",
            ),
            "mhd_amplitude": _finite_non_negative_real(
                kernel_result.get("mhd_amplitude", 0.0),
                name="mhd_amplitude",
            ),
        }

    def check_stability(self, observables: dict) -> list[dict]:
        """Check fusion stability invariants.

        Returns a list of violation dicts (empty if all invariants hold).
        """
        if not isinstance(observables, dict):
            raise ValueError("stability observables must be a dict")
        violations: list[dict] = []
        q_min = observables.get("q_min")
        if q_min is None:
            q_min = observables.get("q_profile")
        if q_min is not None:
            q_min = _finite_real(q_min, name="stability q_min")
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
        if beta_n is not None:
            beta_n = _finite_non_negative_real(beta_n, name="stability beta_n")
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
        if tau_ratio is not None:
            tau_ratio = _finite_non_negative_real(
                tau_ratio,
                name="stability tau_e_ratio",
            )
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
