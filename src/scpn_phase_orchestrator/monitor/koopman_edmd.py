# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Koopman EDMD-with-control linear predictor

"""Extended Dynamic Mode Decomposition with control — a data-driven linear predictor.

Koopman operator theory lifts a nonlinear controlled system ``x_{k+1} = f(x_k,
u_k)`` into a higher-dimensional space of observables ``ψ(x)`` where the
evolution is approximately *linear*. Extended Dynamic Mode Decomposition (EDMD)
fits that lifted linear model from data; with the control extension of
Korda & Mezić (2018) the fitted model is a linear controlled system

    z_{k+1} = A z_k + B u_k,        x̂_k = C z_k,        z_0 = ψ(x_0),

with ``A ∈ ℝ^{N×N}``, ``B ∈ ℝ^{N×m}``, ``C ∈ ℝ^{n×N}`` and ``N`` the dictionary
size. Given snapshot triples ``(x_i, u_i, y_i)`` with ``y_i = f(x_i, u_i)`` the
matrices are the (regularised) least-squares solutions

    [A, B] = argmin Σ_i ‖ψ(y_i) − A ψ(x_i) − B u_i‖²              (Korda eq. 17)
    C      = argmin Σ_i ‖x_i − C ψ(x_i)‖²                          (Korda eq. 20)

solved in closed form through the lifted data matrices (Korda eq. 22). When the
dictionary contains the state coordinates themselves, ``C`` reduces to the
selection ``[I, 0]`` and the recovery is exact on the training data; we still
fit ``C`` so the predictor degrades gracefully for partial dictionaries.

The lifted predictor is the model layer of the grid-forming (dVOC) oscillation
pack: it feeds the convex Koopman-MPC controller (``actuation.koopman_mpc``)
whose quadratic programme is built directly from ``(A, B, C)`` and whose online
cost is independent of the lift dimension ``N``.

The heavy step is the least-squares solve over the lifted matrices; it runs on
the standard five-backend chain (Rust → Mojo → Julia → Go → Python). The
dictionary lift and the predictor roll-out are control flow over that kernel and
stay Python-side.

References
----------
* Korda & Mezić 2018, *Automatica* 93, 149-160 (arXiv:1611.03537) — linear
  predictors for nonlinear dynamical systems: Koopman operator meets MPC.
* Williams, Kevrekidis & Rowley 2015, *J. Nonlinear Sci.* 25, 1307 — a
  data-driven approximation of the Koopman operator (EDMD).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "KoopmanDictionary",
    "KoopmanObservables",
    "KoopmanPredictor",
    "fit_koopman_predictor",
    "lift_states",
]


class KoopmanObservables(Protocol):
    """The observable-map interface consumed by the EDMD fit and predictor.

    A Koopman observable map declares its original state dimension ``n`` and
    lifts a batch of states ``(K, n)`` to observables ``(K, N)``. Both the
    analytic :class:`KoopmanDictionary` and the learned
    ``monitor.phase_koopman.LearnedKoopmanDictionary`` satisfy it, so the EDMD
    fit and the rolled-out predictor are agnostic to how the lift is produced.
    """

    @property
    def state_dim(self) -> int:
        """The original state dimension ``n``.

        Returns
        -------
        int
            The original state dimension ``n``.
        """
        ...

    def lift(self, states: FloatArray) -> FloatArray:
        """Lift a batch of states ``(K, n)`` to observables ``(K, N)``.

        Parameters
        ----------
        states : numpy.ndarray
            The state batch of shape ``(K, n)``.

        Returns
        -------
        numpy.ndarray
            The lifted batch of shape ``(K, N)``.
        """
        ...


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")

_DICTIONARY_KINDS = ("identity", "polynomial", "rbf", "phase")


# --------------------------------------------------------------------------- #
# Backend chain                                                               #
# --------------------------------------------------------------------------- #
def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import koopman_edmd_solve_rust

    def _solve(
        x_lift: FloatArray,
        inputs: FloatArray,
        y_lift: FloatArray,
        states: FloatArray,
        regularisation: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        samples, n_lift = x_lift.shape
        n_input = inputs.shape[1]
        n_state = states.shape[1]
        a, b, c = koopman_edmd_solve_rust(
            np.ascontiguousarray(x_lift.ravel(), dtype=np.float64),
            np.ascontiguousarray(inputs.ravel(), dtype=np.float64),
            np.ascontiguousarray(y_lift.ravel(), dtype=np.float64),
            np.ascontiguousarray(states.ravel(), dtype=np.float64),
            int(samples),
            int(n_lift),
            int(n_input),
            int(n_state),
            float(regularisation),
        )
        return (
            np.asarray(a, dtype=np.float64).reshape(n_lift, n_lift),
            np.asarray(b, dtype=np.float64).reshape(n_lift, n_input),
            np.asarray(c, dtype=np.float64).reshape(n_state, n_lift),
        )

    return {"edmd_solve": _solve}


def _load_mojo_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._koopman_edmd_mojo import (
        _ensure_exe,
        koopman_edmd_solve_mojo,
    )

    _ensure_exe()
    return {"edmd_solve": koopman_edmd_solve_mojo}


def _load_julia_fns() -> dict[str, object]:
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._koopman_edmd_julia import (
        koopman_edmd_solve_julia,
    )

    return {"edmd_solve": koopman_edmd_solve_julia}


def _load_go_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._koopman_edmd_go import (
        _load_lib,
        koopman_edmd_solve_go,
    )

    _load_lib()
    return {"edmd_solve": koopman_edmd_solve_go}


_LOADERS: dict[str, Callable[[], dict[str, object]]] = {
    "rust": _load_rust_fns,
    "mojo": _load_mojo_fns,
    "julia": _load_julia_fns,
    "go": _load_go_fns,
}
_BACKEND_CACHE: dict[str, dict[str, object]] = {}


def _load_backend(name: str) -> dict[str, object]:
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    _BACKEND_CACHE.clear()
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch(fn_name: str) -> object | None:
    """Return the fastest available backend function, or ``None`` for Python.

    The chain is walked active-first; a backend that lacks ``fn_name`` or fails
    to load is skipped so the dispatch never crashes or silently diverges.
    """
    ordered_backends = [ACTIVE_BACKEND, *AVAILABLE_BACKENDS]
    seen: set[str] = set()
    for name in ordered_backends:
        if name in seen:
            continue
        seen.add(name)
        if name == "python":
            return None
        try:
            fn = _load_backend(name)[fn_name]
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        if fn is not None:
            return fn
    return None


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def _contains_boolean_alias(value: object) -> bool:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _validate_matrix(value: object, *, name: str) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real-valued 2-D array") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_vector(value: object, *, name: str) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = raw.astype(np.float64, copy=True).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real-valued 1-D array") from exc
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_int_at_least(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _validate_non_negative_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


# --------------------------------------------------------------------------- #
# Observable dictionary                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KoopmanDictionary:
    """A dictionary of observables ``ψ : ℝ^n → ℝ^N`` for the lift.

    Parameters
    ----------
    kind : str
        One of ``"identity"``, ``"polynomial"``, ``"rbf"`` or ``"phase"``.
    state_dim : int
        The state dimension ``n``.
    degree : int
        Polynomial degree (``"polynomial"`` only); monomials up to and
        including this total degree are appended to the state.
    centres : numpy.ndarray | None
        RBF centres of shape ``(n_centres, n)`` (``"rbf"`` only).
    width : float
        Gaussian RBF width ``σ`` (``"rbf"`` only).
    include_constant : bool
        Whether to prepend the constant observable ``1``.

    Notes
    -----
    The ``"phase"`` dictionary is tuned for phase oscillators: it appends the
    first-harmonic Fourier features ``cos θ_i``, ``sin θ_i`` and the global
    Kuramoto order-parameter components ``R cos Ψ``, ``R sin Ψ``, which render
    the Sakaguchi–Kuramoto vector field close to linear in the lifted space.
    """

    kind: str
    state_dim: int
    degree: int = 2
    centres: FloatArray | None = None
    width: float = 1.0
    include_constant: bool = True
    output_dim: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.kind not in _DICTIONARY_KINDS:
            raise ValueError("kind must be one of: " + ", ".join(_DICTIONARY_KINDS))
        state_dim = _validate_int_at_least(self.state_dim, name="state_dim", minimum=1)
        object.__setattr__(self, "state_dim", state_dim)
        degree = _validate_int_at_least(self.degree, name="degree", minimum=1)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(
            self, "width", _validate_non_negative_real(self.width, name="width")
        )
        if self.kind == "rbf":
            if self.centres is None:
                raise ValueError("rbf dictionary requires centres")
            centres = _validate_matrix(self.centres, name="centres")
            if centres.shape[1] != state_dim:
                raise ValueError(
                    f"centres must have {state_dim} columns, got {centres.shape[1]}"
                )
            if self.width <= 0.0:
                raise ValueError("rbf dictionary requires a positive width")
            object.__setattr__(self, "centres", centres)
        elif self.centres is not None:
            object.__setattr__(
                self, "centres", _validate_matrix(self.centres, name="centres")
            )
        object.__setattr__(self, "output_dim", self._compute_output_dim())

    def _compute_output_dim(self) -> int:
        constant = 1 if self.include_constant else 0
        if self.kind == "identity":
            return constant + self.state_dim
        if self.kind == "polynomial":
            return constant + self._polynomial_feature_count()
        if self.kind == "rbf":
            centres = cast("FloatArray", self.centres)
            return constant + self.state_dim + int(centres.shape[0])
        # phase: state + (cos, sin) per oscillator + 2 order-parameter terms
        return constant + self.state_dim + 2 * self.state_dim + 2

    def _polynomial_feature_count(self) -> int:
        from math import comb

        # Number of monomials of total degree 1..degree in state_dim variables.
        return sum(comb(self.state_dim + d - 1, d) for d in range(1, self.degree + 1))

    def lift(self, states: FloatArray) -> FloatArray:
        """Lift a batch of states ``(K, n)`` to observables ``(K, N)``.

        Parameters
        ----------
        states : numpy.ndarray
            State batch of shape ``(K, n)``.

        Returns
        -------
        numpy.ndarray
            The lifted batch of shape ``(K, output_dim)``.

        Raises
        ------
        ValueError
            If ``states`` is not a finite ``(K, state_dim)`` array.
        """
        matrix = _validate_matrix(states, name="states")
        if matrix.shape[1] != self.state_dim:
            raise ValueError(
                f"states must have {self.state_dim} columns, got {matrix.shape[1]}"
            )
        if self.kind == "identity":
            features = matrix
        elif self.kind == "polynomial":
            features = self._lift_polynomial(matrix)
        elif self.kind == "rbf":
            features = self._lift_rbf(matrix)
        else:
            features = self._lift_phase(matrix)
        if self.include_constant:
            constant = np.ones((matrix.shape[0], 1), dtype=np.float64)
            features = np.hstack((constant, features))
        return np.ascontiguousarray(features, dtype=np.float64)

    def _lift_polynomial(self, states: FloatArray) -> FloatArray:
        from itertools import combinations_with_replacement

        columns: list[FloatArray] = []
        n = self.state_dim
        for total_degree in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(n), total_degree):
                term = np.ones(states.shape[0], dtype=np.float64)
                for index in combo:
                    term = term * states[:, index]
                columns.append(term)
        return np.column_stack(columns)

    def _lift_rbf(self, states: FloatArray) -> FloatArray:
        centres = cast("FloatArray", self.centres)
        # squared distances (K, n_centres) without forming the full 3-tensor
        sq = (
            np.sum(states**2, axis=1)[:, None]
            - 2.0 * states @ centres.T
            + np.sum(centres**2, axis=1)[None, :]
        )
        rbf = np.exp(-np.maximum(sq, 0.0) / (2.0 * self.width**2))
        return np.hstack((states, rbf))

    def _lift_phase(self, states: FloatArray) -> FloatArray:
        cos = np.cos(states)
        sin = np.sin(states)
        order = np.mean(np.exp(1j * states), axis=1)
        order_real = np.real(order)[:, None]
        order_imag = np.imag(order)[:, None]
        return np.hstack((states, cos, sin, order_real, order_imag))


def lift_states(dictionary: KoopmanDictionary, states: FloatArray) -> FloatArray:
    """Lift ``states`` through ``dictionary`` — a free-function alias of ``lift``.

    Parameters
    ----------
    dictionary : KoopmanDictionary
        The observable dictionary.
    states : numpy.ndarray
        State batch of shape ``(K, n)``.

    Returns
    -------
    numpy.ndarray
        The lifted batch of shape ``(K, output_dim)``.
    """
    return dictionary.lift(states)


# --------------------------------------------------------------------------- #
# EDMD solve (the dispatched kernel)                                          #
# --------------------------------------------------------------------------- #
def _edmd_solve_reference(
    x_lift: FloatArray,
    inputs: FloatArray,
    y_lift: FloatArray,
    states: FloatArray,
    regularisation: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Closed-form Tikhonov-regularised EDMD-with-control least squares.

    Solves ``[A, B] = Y_lift Φ^T (Φ Φ^T + ρ I)^{-1}`` with ``Φ = [X_lift; U]``
    (Korda eq. 22) and ``C = X X_lift^T (X_lift X_lift^T + ρ I)^{-1}``, using the
    row-snapshot convention (``X_lift`` is ``(K, N)``).
    """
    n_lift = x_lift.shape[1]
    phi = np.hstack((x_lift, inputs))  # (K, N + m)
    gram = phi.T @ phi
    gram[np.diag_indices_from(gram)] += regularisation
    cross = y_lift.T @ phi  # (N, N + m)
    ab = np.linalg.solve(gram, cross.T).T  # (N, N + m)
    a = np.ascontiguousarray(ab[:, :n_lift], dtype=np.float64)
    b = np.ascontiguousarray(ab[:, n_lift:], dtype=np.float64)

    lift_gram = x_lift.T @ x_lift
    lift_gram[np.diag_indices_from(lift_gram)] += regularisation
    c = np.linalg.solve(lift_gram, (states.T @ x_lift).T).T
    return a, b, np.ascontiguousarray(c, dtype=np.float64)


def _edmd_solve(
    x_lift: FloatArray,
    inputs: FloatArray,
    y_lift: FloatArray,
    states: FloatArray,
    regularisation: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    backend = _dispatch("edmd_solve")
    if backend is None:
        return _edmd_solve_reference(x_lift, inputs, y_lift, states, regularisation)
    solver = cast(
        "Callable[[FloatArray, FloatArray, FloatArray, FloatArray, float], "
        "tuple[FloatArray, FloatArray, FloatArray]]",
        backend,
    )
    return solver(x_lift, inputs, y_lift, states, regularisation)


# --------------------------------------------------------------------------- #
# Predictor                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KoopmanPredictor:
    """A fitted Koopman linear predictor ``z_{k+1}=Az_k+Bu_k``, ``x̂=Cz_k``.

    Parameters
    ----------
    state_matrix : numpy.ndarray
        ``A`` of shape ``(N, N)``.
    input_matrix : numpy.ndarray
        ``B`` of shape ``(N, m)``.
    output_matrix : numpy.ndarray
        ``C`` of shape ``(n, N)``.
    dictionary : KoopmanDictionary
        The observable dictionary used for the lift.
    fit_residual : float
        Root-mean-square one-step lift residual on the training snapshots.
    """

    state_matrix: FloatArray
    input_matrix: FloatArray
    output_matrix: FloatArray
    dictionary: KoopmanObservables
    fit_residual: float

    @property
    def lift_dim(self) -> int:
        """The lifted-state dimension ``N``."""
        return int(self.state_matrix.shape[0])

    @property
    def input_dim(self) -> int:
        """The control dimension ``m``."""
        return int(self.input_matrix.shape[1])

    @property
    def state_dim(self) -> int:
        """The original state dimension ``n``."""
        return int(self.output_matrix.shape[0])

    def lift(self, state: FloatArray) -> FloatArray:
        """Lift a single state ``(n,)`` to its observable ``z = ψ(x)`` ``(N,)``.

        Parameters
        ----------
        state : numpy.ndarray
            The state vector of shape ``(n,)``.

        Returns
        -------
        numpy.ndarray
            The lifted observable ``z = ψ(x)`` of shape ``(N,)``.

        Raises
        ------
        ValueError
            If ``state`` is not a finite vector of length ``state_dim``.
        """
        vector = _validate_vector(state, name="state")
        if vector.shape[0] != self.state_dim:
            raise ValueError(
                f"state must have {self.state_dim} entries, got {vector.shape[0]}"
            )
        return cast("FloatArray", self.dictionary.lift(vector[None, :])[0])

    def predict(
        self, initial_state: FloatArray, input_sequence: FloatArray
    ) -> FloatArray:
        """Roll the linear predictor forward over an input sequence.

        Parameters
        ----------
        initial_state : numpy.ndarray
            The initial state ``x_0`` of shape ``(n,)``.
        input_sequence : numpy.ndarray
            The control sequence of shape ``(T, m)``.

        Returns
        -------
        numpy.ndarray
            Predicted states ``x̂_0 … x̂_T`` of shape ``(T + 1, n)``; the first
            row is the reconstruction ``C ψ(x_0)``. An empty input sequence
            ``(0, m)`` yields the single-row reconstruction.

        Raises
        ------
        ValueError
            If ``input_sequence`` is not a finite ``(T, m)`` array.
        """
        raw = np.asarray(input_sequence)
        if raw.ndim == 2 and raw.shape == (0, self.input_dim):
            reconstruction = self.output_matrix @ self.lift(initial_state)
            return np.ascontiguousarray(reconstruction[None, :], dtype=np.float64)
        inputs = _validate_matrix(input_sequence, name="input_sequence")
        if inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"input_sequence must have {self.input_dim} columns, "
                f"got {inputs.shape[1]}"
            )
        z = self.lift(initial_state)
        horizon = inputs.shape[0]
        states = np.empty((horizon + 1, self.state_dim), dtype=np.float64)
        states[0] = self.output_matrix @ z
        for step in range(horizon):
            z = self.state_matrix @ z + self.input_matrix @ inputs[step]
            states[step + 1] = self.output_matrix @ z
        return states


def fit_koopman_predictor(
    states: FloatArray,
    next_states: FloatArray,
    inputs: FloatArray,
    *,
    dictionary: KoopmanObservables,
    regularisation: float = 1.0e-8,
) -> KoopmanPredictor:
    """Fit an EDMD-with-control linear predictor from snapshot triples.

    Parameters
    ----------
    states : numpy.ndarray
        Snapshot states ``x_i`` of shape ``(K, n)``.
    next_states : numpy.ndarray
        Successor states ``y_i = f(x_i, u_i)`` of shape ``(K, n)``.
    inputs : numpy.ndarray
        Applied controls ``u_i`` of shape ``(K, m)``.
    dictionary : KoopmanDictionary
        The observable dictionary defining the lift ``ψ``.
    regularisation : float
        Tikhonov ridge ``ρ ≥ 0`` added to the normal-equation Gram matrices for
        a well-posed solve.

    Returns
    -------
    KoopmanPredictor
        The fitted predictor with matrices ``(A, B, C)`` and the RMS one-step
        lift residual.

    Raises
    ------
    ValueError
        If the snapshot shapes are inconsistent or the dictionary state
        dimension does not match the data.
    """
    state_matrix = _validate_matrix(states, name="states")
    next_matrix = _validate_matrix(next_states, name="next_states")
    input_matrix = _validate_matrix(inputs, name="inputs")
    regulariser = _validate_non_negative_real(regularisation, name="regularisation")
    if state_matrix.shape != next_matrix.shape:
        raise ValueError(
            f"states {state_matrix.shape} and next_states {next_matrix.shape} "
            "must have the same shape"
        )
    if input_matrix.shape[0] != state_matrix.shape[0]:
        raise ValueError(
            f"inputs must have {state_matrix.shape[0]} rows, "
            f"got {input_matrix.shape[0]}"
        )
    if state_matrix.shape[1] != dictionary.state_dim:
        raise ValueError(
            f"states must have {dictionary.state_dim} columns to match the "
            f"dictionary, got {state_matrix.shape[1]}"
        )

    x_lift = dictionary.lift(state_matrix)
    y_lift = dictionary.lift(next_matrix)
    a, b, c = _edmd_solve(x_lift, input_matrix, y_lift, state_matrix, regulariser)
    predicted_lift = x_lift @ a.T + input_matrix @ b.T
    residual = float(
        np.sqrt(np.mean((predicted_lift - y_lift) ** 2)) if y_lift.size else 0.0
    )
    return KoopmanPredictor(
        state_matrix=a,
        input_matrix=b,
        output_matrix=c,
        dictionary=dictionary,
        fit_residual=residual,
    )
