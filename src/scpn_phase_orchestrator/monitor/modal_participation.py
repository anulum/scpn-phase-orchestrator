# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Modal participation and damping controllability

"""Model-based modal participation and damping controllability of the phase network.

Where :mod:`~scpn_phase_orchestrator.monitor.oscillation_modes` recovers modal
*damping* from a measured ringdown (data-driven), this module answers the two
questions a damping-recommendation engine needs from the network *model*: for a
poorly-damped inter-area mode, **which oscillators swing in it** (mode shape and
participation factors → where the mode is observable) and **which actuators can
damp it** (modal controllability → where to act). A single ringdown signal cannot
give those — they come from the eigenstructure of the small-signal state matrix.

The small-signal state matrix of the Sakaguchi–Kuramoto network is the Jacobian
of the engine's coupling dynamics ``θ̇_i = ω_i + Σ_j K_ij sin(θ_j − θ_i − α_ij)
+ ζ sin(Ψ − θ_i)`` linearised about an operating point ``θ*``
(:func:`phase_network_jacobian`): ``J_ik = K_ik cos(θ_k* − θ_i* − α_ik)`` off the
diagonal and ``J_ii = −Σ_{k≠i} K_ik cos(θ_k* − θ_i* − α_ik) − ζ cos(Ψ − θ_i*)`` on
it. For a symmetric, lag-free network the Jacobian is symmetric
negative-semidefinite (overdamped, real eigenvalues); the oscillatory inter-area
modes that reliability standards screen for appear precisely when the phase lag
``α`` or a directed (asymmetric) coupling break that symmetry — the regime the
SCPN engine runs in.

:func:`analyse_network_modes` performs the small-signal modal analysis of any
continuous-time state matrix ``A`` (so a second-order swing/companion form can be
fed in directly too). It eigen-decomposes ``A`` (LAPACK ``geev`` via NumPy), reads
each eigenvalue ``λ = σ + jω`` as a mode of frequency ``f = |ω| / 2π`` and damping
ratio ``ζ = −σ / |λ|`` (Kundur 1994, §12), and from the right eigenvector ``φ_i``
and the matching left eigenvector ``ψ_i`` (the rows of ``A``'s eigenvector
inverse) forms the dimensionless **participation factor** ``p_ki = φ_ki · ψ_ik``
(Pérez-Arriaga, Verghese & Schweppe 1982) and the **modal controllability**
``|ψ_i · b_j|`` of mode ``i`` from input ``j``. Complex eigenvalues of a real
matrix occur in conjugate pairs; only the non-negative-frequency member of each
pair is reported. The analysis is diagnostic only — it reads a model and reports
modes; it never changes bindings, layers, or coupling.

The whole analysis is one offline eigen-decomposition of a modest state matrix
(LAPACK via NumPy), not a per-step hot path, so it stays on the NumPy floor — the
same judgement as ``oscillation_modes`` and ``autotune.freq_id``.

References
----------
* Kundur, P. 1994, *Power System Stability and Control* (McGraw-Hill), §12 —
  small-signal stability, eigenvalues, mode shapes, participation factors.
* Pérez-Arriaga, I. J., Verghese, G. C. & Schweppe, F. C. 1982,
  *IEEE Trans. Power App. Syst.* PAS-101(9):3117–3125 — selective modal analysis;
  participation factors.
* Dörfler, F. & Bullo, F. 2014, *Automatica* 50(6):1539–1564 — synchronisation in
  networks of phase oscillators; the Kuramoto stability Jacobian.
* NERC PRC-028 (oscillation monitoring) — damping-ratio screening of inter-area
  modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.oscillation_modes import (
    DEFAULT_DAMPING_THRESHOLD,
)

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]

__all__ = [
    "NetworkMode",
    "analyse_network_modes",
    "phase_network_jacobian",
]

#: Eigenvalues with modulus below this are treated as the marginal symmetry mode.
_ZERO_EIGENVALUE = 1.0e-12


@dataclass(frozen=True, eq=False)
class NetworkMode:
    """One small-signal mode of a phase-oscillator network.

    Equality is identity-based (``eq=False``): the array fields make value
    equality ambiguous, and modes are compared field by field, never as wholes.

    Attributes
    ----------
    eigenvalue : complex
        The continuous-time eigenvalue ``σ + jω`` (rad/s) of the state matrix,
        taken from the non-negative-frequency member of its conjugate pair.
    frequency_hz : float
        Modal oscillation frequency ``|ω| / 2π`` in hertz (``≥ 0``); ``0`` for a
        non-oscillatory (real-eigenvalue) mode.
    damping_ratio : float
        Dimensionless damping ratio ``ζ = −σ / |λ|``; ``> 0`` is stable, ``< 0``
        is growing (unstable), ``0`` is the marginal global-phase mode.
    mode_shape : ComplexArray
        Right eigenvector, unit Euclidean norm and phase-anchored so the
        largest-magnitude entry is real-positive: per-oscillator relative
        amplitude (``|·|``) and relative phase (``∠``) of the swing.
    participation : FloatArray
        Real participation factors over oscillators, ``≥ 0`` and summing to ``1``;
        ``participation[k]`` measures how much oscillator ``k`` shapes the mode.
    dominant_state : int
        Index of the oscillator with the largest participation factor.
    controllability : FloatArray | None
        Per-input modal controllability ``|ψ_i^{H} b_j|`` when an input matrix is
        supplied, otherwise ``None``; larger means input ``j`` damps the mode more.
    dominant_input : int | None
        Index of the most effective input, or ``None`` without an input matrix.
    poorly_damped : bool
        Whether ``damping_ratio`` is below the screening threshold.
    """

    eigenvalue: complex
    frequency_hz: float
    damping_ratio: float
    mode_shape: ComplexArray
    participation: FloatArray
    dominant_state: int
    controllability: FloatArray | None
    dominant_input: int | None
    poorly_damped: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping of the mode.

        Returns
        -------
        dict[str, object]
            The eigenvalue (as ``[real, imag]``), frequency, damping ratio, mode
            shape (as ``[real, imag]`` pairs), participation factors, dominant
            oscillator, per-input controllability (or ``None``), dominant input,
            and the poorly-damped flag.
        """
        controllability = (
            None
            if self.controllability is None
            else [float(value) for value in self.controllability]
        )
        return {
            "eigenvalue": [float(self.eigenvalue.real), float(self.eigenvalue.imag)],
            "frequency_hz": self.frequency_hz,
            "damping_ratio": self.damping_ratio,
            "mode_shape": [
                [float(value.real), float(value.imag)] for value in self.mode_shape
            ],
            "participation": [float(value) for value in self.participation],
            "dominant_state": self.dominant_state,
            "controllability": controllability,
            "dominant_input": self.dominant_input,
            "poorly_damped": self.poorly_damped,
        }


def analyse_network_modes(
    state_matrix: FloatArray,
    *,
    input_matrix: FloatArray | None = None,
    damping_threshold: float = DEFAULT_DAMPING_THRESHOLD,
) -> tuple[NetworkMode, ...]:
    """Decompose a continuous-time state matrix into damped modes with participation.

    Parameters
    ----------
    state_matrix : FloatArray
        Real square Jacobian ``A = ∂ẋ/∂x`` of the linearised dynamics, shape
        ``(N, N)``; build it from a phase network with
        :func:`phase_network_jacobian`.
    input_matrix : FloatArray | None
        Real input matrix ``B``, shape ``(N, M)``, mapping ``M`` actuator inputs
        into the state derivative; when given, each mode reports per-input modal
        controllability. ``None`` skips the controllability analysis.
    damping_threshold : float
        Damping ratio below which a mode is flagged ``poorly_damped``.

    Returns
    -------
    tuple[NetworkMode, ...]
        One mode per non-negative-frequency eigenvalue, ordered by ascending
        damping ratio (least-damped, most critical first). The marginal
        global-phase mode of a Kuramoto Jacobian appears with
        ``frequency_hz = 0`` and ``damping_ratio = 0``.

    Raises
    ------
    ValueError
        If the state matrix or input matrix is invalid, or the state matrix is
        defective (not diagonalisable, so left eigenvectors do not exist).
    """
    matrix = _validate_square_matrix(state_matrix, "state_matrix")
    threshold = _real_scalar(damping_threshold, "damping_threshold")
    inputs = _validate_input_matrix(input_matrix, matrix.shape[0])

    raw_eigenvalues, raw_right = np.linalg.eig(matrix)
    eigenvalues = raw_eigenvalues.astype(np.complex128)
    right = raw_right.astype(np.complex128)
    # A defective (non-diagonalisable) matrix has a rank-deficient eigenvector
    # matrix, so its left eigenvectors — and the participation factors built from
    # them — do not exist.
    if np.linalg.matrix_rank(right) < right.shape[0]:
        raise ValueError("state_matrix is defective (not diagonalisable)")
    left = np.linalg.inv(right).astype(np.complex128)

    modes = [
        _build_mode(index, eigenvalues, right, left, inputs, threshold)
        for index in range(eigenvalues.shape[0])
        if eigenvalues[index].imag >= 0.0
    ]
    modes.sort(key=lambda mode: mode.damping_ratio)
    return tuple(modes)


def phase_network_jacobian(
    coupling: FloatArray,
    phases: FloatArray,
    *,
    phase_lag: FloatArray | None = None,
    drive_strength: float = 0.0,
    drive_phase: float = 0.0,
) -> FloatArray:
    """Build the Sakaguchi–Kuramoto small-signal Jacobian at an operating point.

    The Jacobian is the state matrix of the engine's coupling dynamics linearised
    about ``phases``; feed it to :func:`analyse_network_modes`. It matches the
    integrator's derivative exactly: ``J_ik = K_ik cos(θ_k − θ_i − α_ik)`` off the
    diagonal and ``J_ii = −Σ_{k≠i} K_ik cos(θ_k − θ_i − α_ik) − ζ cos(Ψ − θ_i)``.

    Parameters
    ----------
    coupling : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``, with a zero diagonal (no
        self-coupling), as the integrator requires.
    phases : FloatArray
        Operating-point phases ``θ*`` in radians, shape ``(N,)`` — typically a
        synchronised fixed point reached by running the engine.
    phase_lag : FloatArray | None
        Sakaguchi phase-lag matrix ``α`` in radians, shape ``(N, N)``; ``None``
        means no lag (zeros).
    drive_strength : float
        External-drive strength ``ζ``; the default ``0`` gives the free-network
        Jacobian whose modes are the inter-area oscillations.
    drive_phase : float
        External-drive reference phase ``Ψ`` in radians (used only when
        ``drive_strength`` is non-zero).

    Returns
    -------
    FloatArray
        The ``(N, N)`` small-signal Jacobian.

    Raises
    ------
    ValueError
        If the coupling matrix, phases, or phase-lag matrix are invalid, or the
        coupling diagonal is non-zero.
    """
    matrix = _validate_square_matrix(coupling, "coupling")
    n = matrix.shape[0]
    angles = _validate_vector(phases, "phases", n)
    lag = _validate_phase_lag(phase_lag, n)
    strength = _real_scalar(drive_strength, "drive_strength")
    reference = _real_scalar(drive_phase, "drive_phase")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1.0e-15):
        raise ValueError("coupling self-coupling diagonal must be zero")

    differences = angles[np.newaxis, :] - angles[:, np.newaxis] - lag
    jacobian = matrix * np.cos(differences)
    row_sums = jacobian.sum(axis=1)
    drive = strength * np.cos(reference - angles)
    np.fill_diagonal(jacobian, -(row_sums + drive))
    return np.ascontiguousarray(jacobian, dtype=np.float64)


def _build_mode(
    index: int,
    eigenvalues: ComplexArray,
    right: ComplexArray,
    left: ComplexArray,
    inputs: FloatArray | None,
    threshold: float,
) -> NetworkMode:
    """Assemble one :class:`NetworkMode` from eigenvalue ``index`` of the spectrum."""
    eigenvalue = eigenvalues[index]
    magnitude = float(np.abs(eigenvalue))
    frequency = abs(float(eigenvalue.imag)) / (2.0 * np.pi)
    if magnitude > _ZERO_EIGENVALUE:
        damping = -float(eigenvalue.real) / magnitude
    else:
        damping = 0.0
    shape = _phase_anchor(right[:, index])
    weights = np.abs(right[:, index] * left[index, :])
    participation = weights / weights.sum()
    dominant_state = int(np.argmax(participation))
    if inputs is None:
        controllability: FloatArray | None = None
        dominant_input: int | None = None
    else:
        controllability = np.abs(left[index, :] @ inputs.astype(np.complex128))
        dominant_input = int(np.argmax(controllability))
    return NetworkMode(
        eigenvalue=complex(eigenvalue),
        frequency_hz=frequency,
        damping_ratio=damping,
        mode_shape=shape,
        participation=np.ascontiguousarray(participation, dtype=np.float64),
        dominant_state=dominant_state,
        controllability=controllability,
        dominant_input=dominant_input,
        poorly_damped=damping < threshold,
    )


def _phase_anchor(vector: ComplexArray) -> ComplexArray:
    """Return ``vector`` at unit norm with its dominant entry rotated real-positive."""
    unit = vector / np.linalg.norm(vector)
    anchor = unit[int(np.argmax(np.abs(unit)))]
    rotated = unit * (np.conj(anchor) / np.abs(anchor))
    return np.ascontiguousarray(rotated, dtype=np.complex128)


def _validate_square_matrix(value: object, name: str) -> FloatArray:
    array = _validate_real_array(value, name)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got shape {array.shape}")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must have at least one row")
    return array


def _validate_input_matrix(value: object | None, rows: int) -> FloatArray | None:
    if value is None:
        return None
    array = _validate_real_array(value, "input_matrix")
    if array.ndim != 2:
        raise ValueError(f"input_matrix must be 2-D, got shape {array.shape}")
    if array.shape[0] != rows:
        raise ValueError(
            f"input_matrix must have {rows} rows to match state_matrix, "
            f"got {array.shape[0]}"
        )
    if array.shape[1] < 1:
        raise ValueError("input_matrix must have at least one column")
    return array


def _validate_vector(value: object, name: str, length: int) -> FloatArray:
    array = _validate_real_array(value, name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {array.shape[0]}")
    return array


def _validate_phase_lag(value: object | None, n: int) -> FloatArray:
    if value is None:
        return np.zeros((n, n), dtype=np.float64)
    array = _validate_real_array(value, "phase_lag")
    if array.shape != (n, n):
        raise ValueError(f"phase_lag must have shape {(n, n)}, got {array.shape}")
    return array


def _validate_real_array(value: object, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real float array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _real_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar
