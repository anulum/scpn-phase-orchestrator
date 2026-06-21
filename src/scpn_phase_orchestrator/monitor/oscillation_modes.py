# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Inter-area oscillation mode estimation (matrix pencil)

"""Inter-area oscillation mode estimation by the matrix-pencil method.

After a grid disturbance, the ringdown of a wide-area signal (a bus frequency, a
tie-line angle, the network order parameter) is a sum of damped sinusoids; each
sinusoid is an electromechanical *mode*. A mode's **damping ratio** is the
reliability quantity that matters — North American reliability standards
(NERC PRC-028 / the proposed PRC-030 oscillation monitoring rules) treat a mode
whose damping ratio falls below a few percent as a poorly-damped, flaggable
inter-area oscillation.

:func:`estimate_oscillation_modes` fits the ringdown with the matrix-pencil
method of Hua & Sarkar (1990): build a Hankel matrix from the samples, take its
singular value decomposition, keep the dominant subspace, and recover each mode's
discrete pole ``z = exp((−α + j·2πf)·Δt)`` as a generalised eigenvalue of the
pencil formed from the right singular vectors. The pole gives the modal frequency
``f = angle(z)·fs / 2π`` and damping ratio ``ζ = −ln|z| / hypot(ln|z|, angle(z))``;
a least-squares Vandermonde fit gives each mode's amplitude and phase. The
estimator is diagnostic only — it reads a signal and reports modes; it never
changes bindings, layers, or coupling.

Matrix pencil is preferred over Prony for ringdown analysis (it is a one-shot
SVD problem rather than a polynomial root-find, far less noise-sensitive) and is
distinct in purpose from ``autotune.freq_id`` (multichannel DMD that assigns
oscillator channels to modal frequencies for tuning): this estimator reports
single-signal modal *damping* for oscillation safety monitoring.

The whole estimate is one offline SVD + eigen-solve on a short ringdown window
(LAPACK via NumPy), not a per-step hot path, so it stays on the NumPy floor — the
same judgement as ``autotune.freq_id``.

References
----------
* Hua, Y. & Sarkar, T. K. 1990, *IEEE Trans. Acoust. Speech Signal Process.*
  38(5):814–824 — matrix pencil for estimating parameters of exponentially
  damped/undamped sinusoids in noise.
* Sarkar, T. K. & Pereira, O. 1995, *IEEE Antennas Propag. Mag.* 37(1):48–55 —
  using the matrix-pencil method to estimate the parameters of a sum of complex
  exponentials.
* NERC PRC-028 (oscillation monitoring) — damping-ratio screening of inter-area
  modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]

__all__ = [
    "DEFAULT_DAMPING_THRESHOLD",
    "OscillationMode",
    "estimate_oscillation_modes",
]

#: Damping ratio below which a mode is flagged poorly damped (PRC-028 screening).
DEFAULT_DAMPING_THRESHOLD = 0.03


@dataclass(frozen=True)
class OscillationMode:
    """A single damped-sinusoid mode recovered from a ringdown.

    Attributes
    ----------
    frequency_hz : float
        Modal oscillation frequency in hertz (``≥ 0``).
    damping_ratio : float
        Dimensionless damping ratio ``ζ``; ``> 0`` is stable, ``≤ 0`` is growing
        (unstable), small positive values are poorly damped.
    amplitude : float
        Modal amplitude in the units of the input signal (``≥ 0``).
    phase_rad : float
        Modal phase at the first sample, in radians on ``(−π, π]``.
    poorly_damped : bool
        Whether ``damping_ratio`` is below the screening threshold.
    """

    frequency_hz: float
    damping_ratio: float
    amplitude: float
    phase_rad: float
    poorly_damped: bool

    def to_dict(self) -> dict[str, bool | float]:
        """Return a JSON-serialisable mapping of the mode.

        Returns
        -------
        dict[str, bool | float]
            The frequency, damping ratio, amplitude, phase, and the
            poorly-damped flag.
        """
        return {
            "frequency_hz": self.frequency_hz,
            "damping_ratio": self.damping_ratio,
            "amplitude": self.amplitude,
            "phase_rad": self.phase_rad,
            "poorly_damped": self.poorly_damped,
        }


def estimate_oscillation_modes(
    signal: FloatArray,
    fs: float,
    *,
    model_order: int | None = None,
    pencil_factor: float = 0.4,
    damping_threshold: float = DEFAULT_DAMPING_THRESHOLD,
    energy_floor: float = 1.0e-3,
) -> tuple[OscillationMode, ...]:
    """Estimate damped oscillation modes of a ringdown by the matrix-pencil method.

    Parameters
    ----------
    signal : FloatArray
        Real ringdown samples, uniformly sampled, length ``≥ 4``.
    fs : float
        Sampling frequency in hertz (``> 0``).
    model_order : int | None
        Number of modes (real exponentials plus conjugate pairs) to recover. When
        ``None`` the order is chosen from the singular-value spectrum (values
        above ``1e-3`` of the largest). Capped to the pencil dimensions.
    pencil_factor : float
        Pencil parameter as a fraction of the sample count, ``L = round(factor·N)``;
        Hua & Sarkar recommend ``0.33``–``0.5`` for best noise rejection. Must lie
        in ``(0, 1)`` and yield ``1 ≤ L ≤ N − 1``.
    damping_threshold : float
        Damping ratio below which a mode is flagged ``poorly_damped``.
    energy_floor : float
        Modes whose amplitude is below ``energy_floor`` times the largest modal
        amplitude are discarded as numerical noise.

    Returns
    -------
    tuple[OscillationMode, ...]
        The recovered modes, ordered by descending amplitude. Conjugate pairs are
        merged into one positive-frequency mode; near-zero-frequency real modes
        (pure decay) are reported with ``frequency_hz = 0``.

    Raises
    ------
    ValueError
        If the signal, sampling rate, or parameters are invalid.
    """
    samples = _validate_signal(signal)
    sample_rate = _positive_real(fs, "fs")
    threshold = _real_scalar(damping_threshold, "damping_threshold")
    floor = _non_negative_real(energy_floor, "energy_floor")
    n = samples.shape[0]
    pencil = _validated_pencil(pencil_factor, n)

    poles = _matrix_pencil_poles(samples, pencil, _validated_order(model_order))
    if poles.size == 0:
        return ()
    amplitudes, phases = _modal_residues(samples, poles)
    modes = _assemble_modes(poles, amplitudes, phases, sample_rate, threshold)
    return _prune_and_sort(modes, floor)


def _matrix_pencil_poles(
    samples: FloatArray, pencil: int, order: int | None
) -> ComplexArray:
    """Recover discrete poles ``z`` via the Hua–Sarkar matrix-pencil eigenproblem."""
    n = samples.shape[0]
    rows = n - pencil
    # Hankel matrix Y, shape (n - L) x (L + 1): Y[i, j] = signal[i + j].
    hankel = np.empty((rows, pencil + 1), dtype=np.float64)
    for j in range(pencil + 1):
        hankel[:, j] = samples[j : j + rows]
    # SVD; the right singular vectors span the signal subspace.
    _u, singular, vh = np.linalg.svd(hankel, full_matrices=False)
    rank = _effective_order(singular, order, pencil)
    if rank == 0:
        return np.empty(0, dtype=np.complex128)
    v = vh[:rank, :].conj().T  # (L + 1) x rank
    v1 = v[:-1, :]  # drop last row
    v2 = v[1:, :]  # drop first row
    # Poles are the eigenvalues of the pencil V1^+ V2 (Sarkar & Pereira 1995).
    pencil_matrix = np.linalg.pinv(v1) @ v2
    return np.asarray(np.linalg.eigvals(pencil_matrix), dtype=np.complex128)


def _effective_order(singular: FloatArray, order: int | None, pencil: int) -> int:
    """Resolve the model order from the singular spectrum or an explicit request."""
    available = int(min(singular.shape[0], pencil))
    if order is not None:
        return int(min(order, available))
    if singular.shape[0] == 0 or singular[0] == 0.0:
        return 0
    significant = int(np.count_nonzero(singular > 1.0e-3 * singular[0]))
    return int(min(max(significant, 1), available))


def _modal_residues(
    samples: FloatArray, poles: ComplexArray
) -> tuple[FloatArray, FloatArray]:
    """Least-squares Vandermonde fit of complex residues, returned as |R|·2 and ∠R."""
    n = samples.shape[0]
    exponents = np.arange(n, dtype=np.float64)
    vander = poles[np.newaxis, :] ** exponents[:, np.newaxis]  # n x order
    residues, _residual, _rank, _sv = np.linalg.lstsq(
        vander, samples.astype(np.complex128), rcond=None
    )
    return np.abs(residues), np.angle(residues)


def _assemble_modes(
    poles: ComplexArray,
    amplitudes: FloatArray,
    phases: FloatArray,
    fs: float,
    threshold: float,
) -> list[OscillationMode]:
    """Convert each pole+residue into a physical mode, merging conjugate pairs."""
    modes: list[OscillationMode] = []
    seen_conjugate = np.zeros(poles.shape[0], dtype=bool)
    for i, pole in enumerate(poles):
        if seen_conjugate[i]:
            continue
        amplitude = float(amplitudes[i])
        # A complex pole has a conjugate partner carrying half the real signal's
        # energy each; merge them into one real mode of twice the amplitude.
        members = [i]
        partner = _conjugate_partner(poles, i, seen_conjugate)
        if partner is not None:
            seen_conjugate[partner] = True
            amplitude += float(amplitudes[partner])
            members.append(partner)
        magnitude = float(np.abs(pole))
        angle = abs(float(np.angle(pole)))
        frequency = angle * fs / (2.0 * np.pi)
        # ζ = −ln|z| / hypot(ln|z|, ∠z); clamp keeps a zero/unit pole finite.
        log_mag = float(np.log(max(magnitude, 1.0e-300)))
        denom = float(np.hypot(log_mag, angle)) or 1.0
        damping = -log_mag / denom
        # Report the phase of the positive-frequency member for determinism.
        positive = max(members, key=lambda k: float(np.angle(poles[k])))
        modes.append(
            OscillationMode(
                frequency_hz=frequency,
                damping_ratio=damping,
                amplitude=amplitude,
                phase_rad=float(phases[positive]),
                poorly_damped=damping < threshold,
            )
        )
    return modes


def _conjugate_partner(
    poles: ComplexArray, index: int, seen: NDArray[np.bool_]
) -> int | None:
    """Index of the unclaimed complex-conjugate partner of ``poles[index]``, if any."""
    target = np.conj(poles[index])
    if abs(poles[index].imag) < 1.0e-9:
        return None
    best: int | None = None
    best_dist = 1.0e-6
    for j in range(poles.shape[0]):
        if j == index or seen[j]:
            continue
        dist = float(np.abs(poles[j] - target))
        if dist < best_dist:
            best_dist = dist
            best = j
    return best


def _prune_and_sort(
    modes: list[OscillationMode], energy_floor: float
) -> tuple[OscillationMode, ...]:
    """Drop sub-floor amplitudes and order modes by descending amplitude.

    The caller only reaches here with at least one mode (an empty pole set
    returns early), so ``modes`` is non-empty.
    """
    peak = max(mode.amplitude for mode in modes)
    cutoff = energy_floor * peak
    kept = [mode for mode in modes if mode.amplitude >= cutoff]
    kept.sort(key=lambda mode: mode.amplitude, reverse=True)
    return tuple(kept)


def _validate_signal(signal: object) -> FloatArray:
    raw = np.asarray(signal)
    if raw.dtype == np.bool_:
        raise ValueError("signal must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError("signal must be real-valued")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("signal must be a real float array") from exc
    if array.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got shape {array.shape}")
    if array.shape[0] < 4:
        raise ValueError("signal must have at least 4 samples")
    if not np.all(np.isfinite(array)):
        raise ValueError("signal must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validated_pencil(pencil_factor: object, n: int) -> int:
    factor = _real_scalar(pencil_factor, "pencil_factor")
    if not 0.0 < factor < 1.0:
        raise ValueError("pencil_factor must lie in (0, 1)")
    return min(max(int(round(factor * n)), 1), n - 1)


def _validated_order(order: object) -> int | None:
    if order is None:
        return None
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral):
        raise ValueError("model_order must be a positive integer or None")
    parsed = int(order)
    if parsed < 1:
        raise ValueError("model_order must be a positive integer or None")
    return parsed


def _positive_real(value: object, name: str) -> float:
    scalar = _real_scalar(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _non_negative_real(value: object, name: str) -> float:
    scalar = _real_scalar(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _real_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar
