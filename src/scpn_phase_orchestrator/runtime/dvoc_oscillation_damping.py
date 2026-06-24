# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — closed-loop Koopman-MPC oscillation damping

"""Close the dVOC loop: detect a poorly-damped mode, damp it, prove it damped.

This integration wires the whole dVOC chain into one reviewable pipeline. An
underdamped oscillator rings down; the matrix-pencil estimator
(``monitor.oscillation_modes``) detects its electromechanical mode and the NERC
PRC screener (``assurance.prc_oscillation``) flags it as poorly damped. An
EDMD-with-control Koopman predictor (``monitor.koopman_edmd``) is then fitted from
input-excited snapshots and driven in closed loop by the condensed Koopman MPC
(``actuation.koopman_mpc``); the controlled ringdown is re-screened, and the
weakest mode is now better damped. The result carries both hash-sealed PRC
evidence records, so the damping improvement is auditable end to end.

The pipeline is review-only and offline: it operates on a caller-supplied
discrete-time plant ``x_{k+1} = A x_k + B u_k`` and emits evidence; it performs no
live actuation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from scpn_phase_orchestrator.actuation.koopman_mpc import (
    KoopmanMPCConfig,
    KoopmanMPCController,
)
from scpn_phase_orchestrator.assurance.prc_oscillation import (
    PRCOscillationEvidence,
    screen_oscillation_modes,
)
from scpn_phase_orchestrator.monitor.koopman_edmd import (
    KoopmanDictionary,
    KoopmanPredictor,
    fit_koopman_predictor,
)
from scpn_phase_orchestrator.monitor.oscillation_modes import estimate_oscillation_modes

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "OscillationDampingResult",
    "damp_oscillation",
    "underdamped_oscillator",
]


def underdamped_oscillator(
    *, frequency_hz: float, damping_ratio: float, dt: float
) -> tuple[FloatArray, FloatArray]:
    """Build a discrete-time underdamped second-order oscillator with control.

    The continuous plant is ``ẍ + 2ζω ẋ + ω² x = u`` with ``ω = 2π·f``, written in
    state form ``[x, ẋ]`` and discretised by exact zero-order hold.

    Parameters
    ----------
    frequency_hz : float
        Natural frequency ``f`` in hertz.
    damping_ratio : float
        Open-loop damping ratio ``ζ`` (a small value is poorly damped).
    dt : float
        Sampling interval.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The discrete state matrix ``A`` of shape ``(2, 2)`` and input matrix
        ``B`` of shape ``(2, 1)``.

    Raises
    ------
    ValueError
        If ``frequency_hz``, ``dt`` are not positive or ``damping_ratio`` is
        negative.
    """
    if frequency_hz <= 0.0 or dt <= 0.0:
        raise ValueError("frequency_hz and dt must be positive")
    if damping_ratio < 0.0:
        raise ValueError("damping_ratio must be non-negative")
    omega = 2.0 * np.pi * frequency_hz
    cont_state = np.array([[0.0, 1.0], [-(omega**2), -2.0 * damping_ratio * omega]])
    cont_input = np.array([[0.0], [1.0]])
    # Exact zero-order-hold discretisation via the augmented matrix exponential.
    augmented = np.zeros((3, 3))
    augmented[:2, :2] = cont_state
    augmented[:2, 2:] = cont_input
    discrete = expm(augmented * dt)
    state_matrix = np.ascontiguousarray(discrete[:2, :2], dtype=np.float64)
    input_matrix = np.ascontiguousarray(discrete[:2, 2:], dtype=np.float64)
    return state_matrix, input_matrix


@dataclass(frozen=True)
class OscillationDampingResult:
    """The before/after evidence of a closed-loop oscillation-damping run.

    Parameters
    ----------
    uncontrolled_signal, controlled_signal : numpy.ndarray
        The observed ringdown coordinate without and with Koopman MPC.
    uncontrolled_damping_ratio, controlled_damping_ratio : float
        The weakest detected modal damping ratio before and after control.
    before_evidence, after_evidence : PRCOscillationEvidence
        The hash-sealed PRC screening records of the two ringdowns.
    damping_improved : bool
        Whether the controlled ringdown is better damped than the open-loop one.
    fit_residual : float
        Root-mean-square one-step residual of the fitted Koopman predictor.
    """

    uncontrolled_signal: FloatArray
    controlled_signal: FloatArray
    uncontrolled_damping_ratio: float
    controlled_damping_ratio: float
    before_evidence: PRCOscillationEvidence
    after_evidence: PRCOscillationEvidence
    damping_improved: bool
    fit_residual: float


def _open_loop_ringdown(
    state_matrix: FloatArray, initial_state: FloatArray, horizon: int
) -> FloatArray:
    """Return the open-loop ringdown response of the plant."""
    state = initial_state
    signal = [float(state[0])]
    for _ in range(horizon):
        state = state_matrix @ state
        signal.append(float(state[0]))
    return np.asarray(signal, dtype=np.float64)


def _fit_plant_koopman(
    state_matrix: FloatArray,
    input_matrix: FloatArray,
    *,
    scale: float,
    samples: int,
    seed: int,
) -> KoopmanPredictor:
    """Fit the Koopman plant model from the ringdown data."""
    rng = np.random.default_rng(seed)
    state_dim = state_matrix.shape[0]
    input_dim = input_matrix.shape[1]
    states = rng.normal(0.0, scale, size=(samples, state_dim))
    inputs = rng.normal(0.0, scale, size=(samples, input_dim))
    next_states = states @ state_matrix.T + inputs @ input_matrix.T
    dictionary = KoopmanDictionary(kind="identity", state_dim=state_dim)
    return fit_koopman_predictor(states, next_states, inputs, dictionary=dictionary)


def _closed_loop_ringdown(
    state_matrix: FloatArray,
    input_matrix: FloatArray,
    controller: KoopmanMPCController,
    initial_state: FloatArray,
    horizon: int,
) -> FloatArray:
    """Return the closed-loop ringdown response under control."""
    state = initial_state
    input_dim = input_matrix.shape[1]
    previous_input = np.zeros(input_dim, dtype=np.float64)
    reference = np.zeros(state_matrix.shape[0], dtype=np.float64)
    signal = [float(state[0])]
    for _ in range(horizon):
        decision = controller.solve(
            state, reference=reference, previous_input=previous_input
        )
        control = decision.proposed_input
        state = state_matrix @ state + input_matrix @ control
        previous_input = control
        signal.append(float(state[0]))
    return np.asarray(signal, dtype=np.float64)


def _weakest_damping(signal: FloatArray, fs: float) -> float:
    """Return the weakest (least-damped) mode of the response."""
    modes = estimate_oscillation_modes(signal, fs)
    if not modes:
        # A signal with no detectable oscillatory mode is fully damped.
        return 1.0
    return min(mode.damping_ratio for mode in modes)


def damp_oscillation(
    state_matrix: FloatArray,
    input_matrix: FloatArray,
    *,
    initial_state: FloatArray,
    horizon: int,
    fs: float,
    captured_at: str,
    config: KoopmanMPCConfig | None = None,
    event_prefix: str = "dvoc-damping",
    training_scale: float = 1.0,
    training_samples: int = 400,
    seed: int = 0,
) -> OscillationDampingResult:
    """Damp a plant's oscillation with Koopman MPC and prove it with PRC evidence.

    Parameters
    ----------
    state_matrix, input_matrix : numpy.ndarray
        The discrete-time plant ``A`` ``(n, n)`` and ``B`` ``(n, m)``.
    initial_state : numpy.ndarray
        The perturbed initial state ``x_0`` of shape ``(n,)``.
    horizon : int
        Number of ringdown steps to simulate for each pass.
    fs : float
        The sampling rate in hertz used by the mode estimator and PRC screen.
    captured_at : str
        ISO-8601 capture timestamp stamped into the evidence records.
    config : KoopmanMPCConfig | None
        The MPC configuration; a damping-oriented default is used if omitted.
    event_prefix : str
        Prefix for the two PRC evidence event identifiers.
    training_scale : float
        Standard deviation of the random snapshots used to fit the predictor.
    training_samples : int
        Number of input-excited snapshots used to fit the predictor.
    seed : int
        Seed for the snapshot sampler.

    Returns
    -------
    OscillationDampingResult
        The before/after signals, weakest damping ratios, both PRC evidence
        records, the improvement flag, and the predictor fit residual.

    Raises
    ------
    ValueError
        If the plant, initial state, or horizon are inconsistent.
    """
    state = np.ascontiguousarray(np.asarray(initial_state, dtype=np.float64).ravel())
    if state.shape[0] != state_matrix.shape[0]:
        raise ValueError("initial_state length must match the state dimension")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    uncontrolled = _open_loop_ringdown(state_matrix, state, horizon)
    before_modes = estimate_oscillation_modes(uncontrolled, fs)
    before_evidence = screen_oscillation_modes(
        before_modes,
        event_id=f"{event_prefix}-open-loop",
        captured_at=captured_at,
        signal_source="koopman-mpc/open-loop-ringdown",
        sampling_rate_hz=fs,
    )

    predictor = _fit_plant_koopman(
        state_matrix,
        input_matrix,
        scale=training_scale,
        samples=training_samples,
        seed=seed,
    )
    mpc_config = config if config is not None else _default_damping_config(horizon)
    controller = KoopmanMPCController(predictor, mpc_config)
    controlled = _closed_loop_ringdown(
        state_matrix, input_matrix, controller, state, horizon
    )
    after_modes = estimate_oscillation_modes(controlled, fs)
    after_evidence = screen_oscillation_modes(
        after_modes,
        event_id=f"{event_prefix}-closed-loop",
        captured_at=captured_at,
        signal_source="koopman-mpc/closed-loop-ringdown",
        sampling_rate_hz=fs,
    )

    before_damping = _weakest_damping(uncontrolled, fs)
    after_damping = _weakest_damping(controlled, fs)
    return OscillationDampingResult(
        uncontrolled_signal=uncontrolled,
        controlled_signal=controlled,
        uncontrolled_damping_ratio=before_damping,
        controlled_damping_ratio=after_damping,
        before_evidence=before_evidence,
        after_evidence=after_evidence,
        damping_improved=after_damping > before_damping,
        fit_residual=predictor.fit_residual,
    )


def _default_damping_config(horizon: int) -> KoopmanMPCConfig:
    """Return the default oscillation-damping configuration."""
    return KoopmanMPCConfig(
        horizon=min(horizon, 20),
        output_weight=1.0,
        input_weight=1.0e-3,
        terminal_weight=10.0,
        input_lower=-50.0,
        input_upper=50.0,
    )
