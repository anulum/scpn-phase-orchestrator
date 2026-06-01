# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Frequency identification via DMD

"""Dynamic Mode Decomposition frequency identification for multichannel data.

``identify_frequencies`` derives dominant modal frequencies and amplitudes from
SVD-truncated exact DMD, then assigns each channel to the nearest modal
frequency estimated from Hilbert phase extraction. The function requires enough
time samples for a forward-shift pair and returns a diagnostic result only; it
does not alter bindings, layers, or coupling matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["identify_frequencies", "FrequencyResult"]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class FrequencyResult:
    """Dominant DMD frequencies, modal amplitudes, and channel assignments."""

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
    sample_rate = _positive_real(fs, "fs")
    data = np.atleast_2d(_real_data(data))
    n_ch, n_t = data.shape
    if n_t < 3:
        raise ValueError(f"Need >= 3 time samples, got {n_t}")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values")
    if not np.any(np.abs(data - data.mean(axis=1, keepdims=True)) > 0.0):
        raise ValueError("data must contain non-zero temporal dynamics")
    mode_count = _validated_n_modes(n_modes)
    rank_cutoff = _non_negative_real(rank_threshold, "rank_threshold")

    X = data[:, :-1]
    Xp = data[:, 1:]

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if S.size == 0 or not np.isfinite(S[0]) or S[0] <= 0.0:
        raise ValueError("data must contain non-zero temporal dynamics")

    if mode_count is None:
        mode_count = int(np.sum(rank_cutoff * S[0] < S))
    mode_count = max(1, min(mode_count, len(S)))

    U_r = U[:, :mode_count]
    S_r = S[:mode_count]
    Vt_r = Vt[:mode_count, :]

    A_tilde = U_r.conj().T @ Xp @ Vt_r.conj().T @ np.diag(1.0 / S_r)

    evals, _ = np.linalg.eig(A_tilde)

    dt = 1.0 / sample_rate
    freqs = np.abs(np.log(evals + 1e-30).imag / (2 * np.pi * dt))
    amps = np.abs(evals)

    sort_idx = np.argsort(-amps)
    freqs = freqs[sort_idx]
    amps = amps[sort_idx]

    # Assign channels to nearest DMD frequency
    from scpn_phase_orchestrator.autotune.phase_extract import extract_phases

    channel_freqs = []
    for ch in range(n_ch):
        pr = extract_phases(data[ch], sample_rate)
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


def _real_data(data: object) -> FloatArray:
    raw = np.asarray(data)
    if raw.dtype == np.bool_ or _contains_alias(raw, (bool, np.bool_)):
        raise ValueError("data must not contain boolean values")
    if np.iscomplexobj(raw) or _contains_alias(raw, (complex, np.complexfloating)):
        raise ValueError("data must be real-valued")
    try:
        parsed: FloatArray = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("data must be real-valued") from exc
    return parsed


def _validated_n_modes(n_modes: int | None) -> int | None:
    if n_modes is None:
        return None
    if isinstance(n_modes, (bool, np.bool_)) or not isinstance(n_modes, Integral):
        raise ValueError("n_modes must be a positive integer or None")
    parsed = int(n_modes)
    if parsed <= 0:
        raise ValueError("n_modes must be a positive integer or None")
    return parsed


def _positive_real(value: object, name: str) -> float:
    parsed = _real_scalar(value, name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _non_negative_real(value: object, name: str) -> float:
    parsed = _real_scalar(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _real_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real value")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _contains_alias(raw: NDArray[np.generic], aliases: tuple[type, ...]) -> bool:
    if raw.dtype != object:
        return False
    return any(isinstance(item, aliases) for item in raw.ravel())
