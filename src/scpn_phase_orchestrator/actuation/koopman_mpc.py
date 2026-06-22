# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — review-only Koopman model predictive controller

"""A convex Koopman model-predictive controller for the dVOC oscillation pack.

A fitted Koopman predictor (``monitor.koopman_edmd``) supplies a *linear* model
``z_{k+1}=Az_k+Bu_k``, ``y=Cz`` of an otherwise nonlinear system. Linear model
predictive control over that lifted model is therefore a single **convex**
quadratic programme, which this controller builds in condensed form (Korda &
Mezić 2018, eq. 24): the lifted states are eliminated so the decision variable is
the input sequence ``U`` alone, and the online cost is independent of the lift
dimension ``N``.

Over a horizon ``H`` the predicted outputs stack as ``Y = Ψ ψ(x_k) + Θ U`` with

    Ψ_i = C Aⁱ,        Θ_{i,j} = C A^{i-1-j} B   (j < i),

and the controller minimises the tracking and effort cost

    Σ_{i=1}^{H} (y_i − r)ᵀ Q (y_i − r) + u_{i-1}ᵀ R u_{i-1} + (y_H − r)ᵀ Q_f (y_H − r)

subject to actuator bounds ``u_min ≤ u_i ≤ u_max`` and optional move limits
``|u_i − u_{i-1}| ≤ Δ``. The quadratic programme is solved by the deterministic
ADMM floor of the QP layer so the decision is reproducible and content-hashable.

This controller is **review-only**: it returns a proposed input sequence sealed
into a content-addressed :class:`KoopmanMPCDecision`; it never actuates. The
first proposed input is the action a downstream safety envelope
(``actuation.foundation_model_governor`` / ``actuation.control_barrier``) admits,
constrains, or rejects before any hardware sees it.

References
----------
* Korda & Mezić 2018, *Automatica* 93, 149-160 (arXiv:1611.03537) — linear
  predictors for nonlinear dynamical systems: Koopman operator meets MPC.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.koopman_edmd import KoopmanPredictor

from ._qp import solve_qp

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "KoopmanMPCConfig",
    "KoopmanMPCController",
    "KoopmanMPCDecision",
]

_HASH_DECIMALS = 9


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def _int_at_least(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _positive_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _non_negative_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _diagonal_weight(value: object, *, name: str, dim: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        weight = float(array)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return np.full(dim, weight, dtype=np.float64)
    flat = array.ravel()
    if flat.shape[0] != dim:
        raise ValueError(f"{name} must be a scalar or a length-{dim} vector")
    if not np.all(np.isfinite(flat)) or np.any(flat < 0.0):
        raise ValueError(f"{name} must contain finite non-negative weights")
    return np.ascontiguousarray(flat, dtype=np.float64)


def _bound_vector(value: object, *, name: str, dim: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        flat = np.full(dim, float(array), dtype=np.float64)
    else:
        flat = array.ravel()
    if flat.shape[0] != dim:
        raise ValueError(f"{name} must be a scalar or a length-{dim} vector")
    if np.any(np.isnan(flat)):
        raise ValueError(f"{name} must not contain NaN values")
    return np.ascontiguousarray(flat, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KoopmanMPCConfig:
    """Cost and constraint specification for the Koopman MPC.

    Parameters
    ----------
    horizon : int
        The prediction horizon ``H`` (number of steps).
    output_weight : float | numpy.ndarray
        The output tracking weight ``Q`` (scalar or per-output diagonal).
    input_weight : float | numpy.ndarray
        The input effort weight ``R`` (scalar or per-input diagonal).
    terminal_weight : float
        A non-negative multiplier on ``Q`` for the terminal stage.
    input_lower, input_upper : float | numpy.ndarray
        The actuator bounds (scalar or per-input).
    move_limit : float | None
        An optional symmetric per-step move limit ``|u_i − u_{i-1}| ≤ Δ``.
    """

    horizon: int
    output_weight: float | FloatArray = 1.0
    input_weight: float | FloatArray = 1.0e-2
    terminal_weight: float = 1.0
    input_lower: float | FloatArray = -np.inf
    input_upper: float | FloatArray = np.inf
    move_limit: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "horizon", _int_at_least(self.horizon, name="horizon", minimum=1)
        )
        object.__setattr__(
            self,
            "terminal_weight",
            _non_negative_real(self.terminal_weight, name="terminal_weight"),
        )
        if self.move_limit is not None:
            object.__setattr__(
                self, "move_limit", _positive_real(self.move_limit, name="move_limit")
            )


# --------------------------------------------------------------------------- #
# Decision                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KoopmanMPCDecision:
    """A review-only Koopman-MPC proposal, sealed by a content hash.

    Parameters
    ----------
    proposed_input : numpy.ndarray
        The first input ``u_0`` of the optimal sequence, shape ``(m,)`` — the
        action handed to the safety envelope.
    input_plan : numpy.ndarray
        The full optimal input sequence, shape ``(H, m)``.
    predicted_outputs : numpy.ndarray
        The predicted output trajectory ``y_1 … y_H``, shape ``(H, n)``.
    objective : float
        The optimal quadratic-programme objective value.
    status : str
        ``"OPTIMAL"`` if the solver converged, otherwise ``"MAX_ITER"``.
    active_bounds : bool
        Whether any element of ``proposed_input`` sits on an actuator bound.
    content_hash : str
        The SHA-256 of the canonical, rounded decision payload.
    """

    proposed_input: FloatArray
    input_plan: FloatArray
    predicted_outputs: FloatArray
    objective: float
    status: str
    active_bounds: bool
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "content_hash", _canonical_hash(self._canonical_payload())
        )

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "proposed_input": _rounded(self.proposed_input),
            "input_plan": _rounded(self.input_plan),
            "predicted_outputs": _rounded(self.predicted_outputs),
            "objective": round(float(self.objective), _HASH_DECIMALS),
            "status": self.status,
            "active_bounds": bool(self.active_bounds),
        }


# --------------------------------------------------------------------------- #
# Controller                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KoopmanMPCController:
    """A condensed convex Koopman model-predictive controller.

    Parameters
    ----------
    predictor : KoopmanPredictor
        The fitted Koopman linear predictor supplying ``(A, B, C)``.
    config : KoopmanMPCConfig
        The cost and constraint specification.
    """

    predictor: KoopmanPredictor
    config: KoopmanMPCConfig

    def solve(
        self,
        current_state: FloatArray,
        *,
        reference: FloatArray | None = None,
        previous_input: FloatArray | None = None,
    ) -> KoopmanMPCDecision:
        """Solve the MPC programme and return a review-only proposal.

        Parameters
        ----------
        current_state : numpy.ndarray
            The current physical state ``x_k`` of shape ``(n,)``.
        reference : numpy.ndarray | None
            The constant output set-point ``r`` of shape ``(n,)``; defaults to
            the origin (oscillation damping).
        previous_input : numpy.ndarray | None
            The previously applied input, required only when a move limit is
            configured.

        Returns
        -------
        KoopmanMPCDecision
            The sealed proposal.

        Raises
        ------
        ValueError
            If the shapes are inconsistent or a move limit is set without a
            previous input.
        """
        horizon = self.config.horizon
        n_state = self.predictor.state_dim
        n_input = self.predictor.input_dim

        z0 = self.predictor.lift(current_state)
        target = (
            np.zeros(n_state, dtype=np.float64)
            if reference is None
            else _bound_vector(reference, name="reference", dim=n_state)
        )
        output_q = _diagonal_weight(
            self.config.output_weight, name="output_weight", dim=n_state
        )
        input_r = _diagonal_weight(
            self.config.input_weight, name="input_weight", dim=n_input
        )
        lower = _bound_vector(self.config.input_lower, name="input_lower", dim=n_input)
        upper = _bound_vector(self.config.input_upper, name="input_upper", dim=n_input)
        if np.any(upper < lower):
            raise ValueError("input_upper must not be below input_lower")

        psi, theta = _condensed_prediction(self.predictor, horizon)
        free_output = psi @ z0  # (H * n,)
        target_stack = np.tile(target, horizon)

        stage_q = np.tile(output_q, horizon)
        stage_q[(horizon - 1) * n_state :] *= self.config.terminal_weight
        stage_r = np.tile(input_r, horizon)

        # ½ Uᵀ P U + qᵀ U with P = 2(Θᵀ diag(Q) Θ + diag(R)).
        weighted_theta = theta * stage_q[:, None]
        hessian = 2.0 * (theta.T @ weighted_theta + np.diag(stage_r))
        hessian = 0.5 * (hessian + hessian.T)
        linear = 2.0 * theta.T @ (stage_q * (free_output - target_stack))

        constraint, con_lower, con_upper = _build_constraints(
            horizon, n_input, lower, upper, self.config.move_limit, previous_input
        )
        solution = solve_qp(hessian, linear, constraint, con_lower, con_upper)

        plan = solution.x.reshape(horizon, n_input)
        outputs = (free_output + theta @ solution.x).reshape(horizon, n_state)
        proposed = np.ascontiguousarray(plan[0], dtype=np.float64)
        on_bound = bool(
            np.any(np.isclose(proposed, lower) & np.isfinite(lower))
            or np.any(np.isclose(proposed, upper) & np.isfinite(upper))
        )
        return KoopmanMPCDecision(
            proposed_input=proposed,
            input_plan=np.ascontiguousarray(plan, dtype=np.float64),
            predicted_outputs=np.ascontiguousarray(outputs, dtype=np.float64),
            objective=solution.objective,
            status="OPTIMAL" if solution.converged else "MAX_ITER",
            active_bounds=on_bound,
        )


def _condensed_prediction(
    predictor: KoopmanPredictor, horizon: int
) -> tuple[FloatArray, FloatArray]:
    """Build the condensed prediction matrices ``Ψ`` and ``Θ`` (Korda eq. 24)."""
    state_matrix = predictor.state_matrix
    input_matrix = predictor.input_matrix
    output_matrix = predictor.output_matrix
    lift_dim = predictor.lift_dim
    n_state = predictor.state_dim
    n_input = predictor.input_dim

    # Impulse-response blocks C A^p B for p = 0 … horizon-1.
    impulse: list[FloatArray] = []
    power = np.eye(lift_dim, dtype=np.float64)
    psi = np.zeros((horizon * n_state, lift_dim), dtype=np.float64)
    for i in range(horizon):
        power = power @ state_matrix if i else state_matrix
        psi[i * n_state : (i + 1) * n_state, :] = output_matrix @ power
        impulse.append(output_matrix @ (power @ input_matrix))

    theta = np.zeros((horizon * n_state, horizon * n_input), dtype=np.float64)
    for i in range(horizon):
        for j in range(i + 1):
            block = impulse[i - j]
            theta[i * n_state : (i + 1) * n_state, j * n_input : (j + 1) * n_input] = (
                block
            )
    return psi, theta


def _build_constraints(
    horizon: int,
    n_input: int,
    lower: FloatArray,
    upper: FloatArray,
    move_limit: float | None,
    previous_input: FloatArray | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Assemble the box (and optional move-limit) constraints on ``U``."""
    n_vars = horizon * n_input
    rows: list[FloatArray] = [np.eye(n_vars, dtype=np.float64)]
    con_lower = [np.tile(lower, horizon)]
    con_upper = [np.tile(upper, horizon)]
    if move_limit is not None:
        if previous_input is None:
            raise ValueError("a move limit requires previous_input")
        prev = _bound_vector(previous_input, name="previous_input", dim=n_input)
        diff = np.zeros((n_vars, n_vars), dtype=np.float64)
        offset = np.zeros(n_vars, dtype=np.float64)
        for i in range(horizon):
            block = slice(i * n_input, (i + 1) * n_input)
            diff[block, block] = np.eye(n_input)
            if i == 0:
                offset[block] = prev
            else:
                prev_block = slice((i - 1) * n_input, i * n_input)
                diff[block, prev_block] = -np.eye(n_input)
        rows.append(diff)
        con_lower.append(offset - move_limit)
        con_upper.append(offset + move_limit)
    return (
        np.ascontiguousarray(np.vstack(rows), dtype=np.float64),
        np.ascontiguousarray(np.concatenate(con_lower), dtype=np.float64),
        np.ascontiguousarray(np.concatenate(con_upper), dtype=np.float64),
    )


def _rounded(array: FloatArray) -> list[float]:
    return [round(float(value), _HASH_DECIMALS) for value in np.ravel(array)]


def _canonical_hash(record: dict[str, object]) -> str:
    payload = json.dumps(record, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
