# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Frequency identification via DMD

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["identify_frequencies", "FrequencyResult"]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class FrequencyResult:
    frequencies: FloatArray
    amplitudes: FloatArray
    layer_assignment: list[int]


def identify_frequencies(
    data: FloatArray,
    fs: float,
    n_modes: int | None = None,
    rank_threshold: float = 0.01,
) -> FrequencyResult:
    """Identify dominant frequencies from multichannel data via exact DMD.

    Uses SVD-based Dynamic Mode Decomposition: X' ≈ A X where
    A = X' V Σ⁻¹ Uᴴ, eigenvalues of A give frequencies.

    Args:
        data: (n_channels, n_samples) multichannel time series.
        fs: sampling frequency in Hz.
        n_modes: number of DMD modes to keep (default: auto from singular values).
        rank_threshold: singular value threshold for rank truncation.

    Returns:
        FrequencyResult with frequencies, amplitudes, and layer assignment
        (which channel maps to which frequency cluster).
    """
    data = np.atleast_2d(data)
    n_ch, n_t = data.shape
    if n_t < 3:
        raise ValueError(f"Need >= 3 time samples, got {n_t}")

    X = data[:, :-1]
    Xp = data[:, 1:]

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    if n_modes is None:
        n_modes = int(np.sum(rank_threshold * S[0] < S))
    n_modes = max(1, min(n_modes, len(S)))

    U_r = U[:, :n_modes]
    S_r = S[:n_modes]
    Vt_r = Vt[:n_modes, :]

    A_tilde = U_r.conj().T @ Xp @ Vt_r.conj().T @ np.diag(1.0 / S_r)

    evals, _ = np.linalg.eig(A_tilde)

    dt = 1.0 / fs
    freqs = np.abs(np.log(evals + 1e-30).imag / (2 * np.pi * dt))
    amps = np.abs(evals)

    sort_idx = np.argsort(-amps)
    freqs = freqs[sort_idx]
    amps = amps[sort_idx]

    # Assign channels to nearest DMD frequency
    from scpn_phase_orchestrator.autotune.phase_extract import extract_phases

    channel_freqs = []
    for ch in range(n_ch):
        pr = extract_phases(data[ch], fs)
        channel_freqs.append(pr.dominant_freq)

    layer_assignment = []
    for cf in channel_freqs:
        idx = int(np.argmin(np.abs(freqs - cf)))
        layer_assignment.append(idx)

    return FrequencyResult(
        frequencies=freqs,
        amplitudes=amps,
        layer_assignment=layer_assignment,
    )
