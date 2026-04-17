# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Auto-tune pipeline: raw data → BindingSpec

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.autotune.coupling_est import estimate_coupling
from scpn_phase_orchestrator.autotune.phase_extract import extract_phases
from scpn_phase_orchestrator.coupling.prior import UniversalPrior

__all__ = ["identify_binding_spec", "AutoTuneResult"]


@dataclass
class AutoTuneResult:
    """Output of the auto-tune pipeline: inferred frequencies and coupling."""

    omegas: list[float]
    knm: NDArray
    alpha: NDArray
    n_layers: int
    dominant_freqs: list[float]
    K_c_estimate: float


def identify_binding_spec(
    time_series: NDArray,
    fs: float,
    n_layers: int | None = None,
) -> AutoTuneResult:
    """Full auto-tune pipeline: raw multichannel data → coupling parameters.

    1. Phase extraction (Hilbert) per channel → ω_i
    2. Coupling estimation (least squares) → K_ij
    3. K_c estimate from universal prior

    Args:
        time_series: (n_channels, n_samples) raw data.
        fs: sampling frequency in Hz.
        n_layers: override number of layers (default: n_channels).
    """
    data = np.atleast_2d(time_series)
    n_ch, n_t = data.shape

    if n_layers is None:
        n_layers = n_ch

    # Step 1: extract phases and frequencies per channel
    channel_phases = []
    dominant_freqs = []
    for ch in range(n_ch):
        pr = extract_phases(data[ch], fs)
        channel_phases.append(pr.phases)
        dominant_freqs.append(pr.dominant_freq)

    omegas = [2 * np.pi * f for f in dominant_freqs]

    # Step 2: estimate coupling from phase trajectories
    phase_matrix = np.array(channel_phases)
    dt = 1.0 / fs
    knm = estimate_coupling(phase_matrix, np.array(omegas), dt)

    # Ensure non-negative coupling
    knm = np.maximum(knm, 0.0)
    np.fill_diagonal(knm, 0.0)

    alpha = np.zeros((n_ch, n_ch), dtype=np.float64)

    # Step 3: K_c from universal prior
    prior = UniversalPrior()
    kc_result = prior.estimate_Kc(np.array(omegas), n_layers)

    return AutoTuneResult(
        omegas=omegas,
        knm=knm,
        alpha=alpha,
        n_layers=n_layers,
        dominant_freqs=dominant_freqs,
        K_c_estimate=kc_result.K_c_estimate,
    )
