# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Auto-tune pipeline: raw data → BindingSpec

"""Offline auto-tune pipeline from raw channels to inferred coupling settings.

``identify_binding_spec`` extracts per-channel phases and dominant frequencies,
estimates a non-negative coupling matrix, initializes zero phase lags, and asks
the universal prior for a critical-coupling estimate. The result is an
``AutoTuneResult`` for review or downstream proposal generation; the function
does not write a binding file, change runtime configuration, or activate the
inferred parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.autotune.coupling_est import estimate_coupling
from scpn_phase_orchestrator.autotune.phase_extract import extract_phases
from scpn_phase_orchestrator.coupling.prior import UniversalPrior

__all__ = ["identify_binding_spec", "AutoTuneResult"]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class AutoTuneResult:
    """Output of the auto-tune pipeline: inferred frequencies and coupling."""

    omegas: list[float]
    knm: FloatArray
    alpha: FloatArray
    n_layers: int
    dominant_freqs: list[float]
    K_c_estimate: float


def identify_binding_spec(
    time_series: FloatArray,
    fs: float,
    n_layers: int | None = None,
) -> AutoTuneResult:
    """Full auto-tune pipeline: raw multichannel data → coupling parameters.

    1. Phase extraction (Hilbert) per channel → ω_i
    2. Coupling estimation (least squares) → K_ij
    3. K_c estimate from universal prior

    Parameters
    ----------
    time_series : FloatArray
        (n_channels, n_samples) raw data.
    fs : float
        sampling frequency in Hz.
    n_layers : int | None
        override number of layers (default: n_channels).

    Returns
    -------
    AutoTuneResult
        The result.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    TypeError
        If an argument has the wrong type.
    """
    sample_rate = _positive_real(fs, "fs")
    data = np.atleast_2d(_real_time_series(time_series))
    n_ch, n_t = data.shape
    if n_t < 4:
        raise ValueError(f"time_series needs >= 4 samples, got {n_t}")
    if not np.all(np.isfinite(data)):
        raise ValueError("time_series must contain only finite values")
    if not np.any(np.abs(data - data.mean(axis=1, keepdims=True)) > 0.0):
        raise ValueError("time_series must contain non-zero temporal dynamics")

    if n_layers is None:
        n_layers = n_ch
    elif isinstance(n_layers, bool) or not isinstance(n_layers, Integral):
        raise TypeError(f"n_layers must be an integer, got {n_layers!r}")
    elif n_layers <= 0:
        raise ValueError("n_layers must be a positive integer")
    else:
        n_layers = int(n_layers)

    # Step 1: extract phases and frequencies per channel
    channel_phases = []
    dominant_freqs = []
    for ch in range(n_ch):
        pr = extract_phases(data[ch], sample_rate)
        channel_phases.append(pr.phases)
        dominant_freqs.append(pr.dominant_freq)

    omegas = [2 * np.pi * f for f in dominant_freqs]

    # Step 2: estimate coupling from phase trajectories
    phase_matrix = np.array(channel_phases)
    dt = 1.0 / sample_rate
    knm = estimate_coupling(phase_matrix, np.array(omegas), dt)

    # Ensure non-negative coupling
    knm = np.maximum(knm, 0.0)
    np.fill_diagonal(knm, 0.0)

    alpha = np.zeros((n_ch, n_ch), dtype=np.float64)

    # Step 3: K_c from universal prior
    prior = UniversalPrior()
    kc_result = prior.estimate_Kc(np.array(omegas), n_ch)

    return AutoTuneResult(
        omegas=omegas,
        knm=knm,
        alpha=alpha,
        n_layers=n_layers,
        dominant_freqs=dominant_freqs,
        K_c_estimate=kc_result.K_c_estimate,
    )


def _real_time_series(time_series: object) -> FloatArray:
    raw = np.asarray(time_series)
    if raw.dtype == np.bool_ or _contains_alias(raw, (bool, np.bool_)):
        raise ValueError("time_series must not contain boolean values")
    if np.iscomplexobj(raw) or _contains_alias(raw, (complex, np.complexfloating)):
        raise ValueError("time_series must be real-valued")
    try:
        data: FloatArray = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_series must be real-valued") from exc
    return data


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real value")
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _contains_alias(raw: NDArray[np.generic], aliases: tuple[type, ...]) -> bool:
    if raw.dtype != object:
        return False
    return any(isinstance(item, aliases) for item in raw.ravel())
