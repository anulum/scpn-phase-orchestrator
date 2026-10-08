# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — C15_sec ethical cost term
#
# Implements the ethical Lagrangian from R5 Insight 19:
# L_ethical = U_total + w_c15 · C15_sec
# C15_sec = (1 - J_sec) + κ · Φ_ethics
#
# J_sec = α·R + β·K + γ·Q - ν·S_dev  (SEC functional)
# Φ_ethics = Σ max(0, g_k)²           (CBF constraint penalties)
#
# Grounded in: Harsanyi aggregation, MacAskill ECW,
# Lyapunov/CBF safety, Wiener cybernetic ethics.

"""Ethical-cost diagnostic term for SEC and CBF-style constraint penalties.

The module computes the C15_sec term from coherence, spectral connectivity,
coupling density, phase dispersion, and squared control-barrier violations.
It is a numeric diagnostic used by the SSGF cost surface, not an autonomous
policy authority or clinical decision surface. Rust and Python paths expose the
same result fields so callers can audit SEC score, ethics penalty, total term,
and violation count independently.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from numbers import Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._array_types import require_real_values
from scpn_phase_orchestrator.coupling.spectral import fiedler_value
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

__all__ = ["EthicalCost", "compute_ethical_cost"]

FloatArray: TypeAlias = NDArray[np.float64]
EthicalKernel: TypeAlias = Callable[
    [
        FloatArray,
        FloatArray,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    tuple[float, float, float, int],
]

try:
    _kernel = import_module("spo_kernel")
    _rust_ethical_cost: EthicalKernel | None = cast(
        "EthicalKernel | None", getattr(_kernel, "compute_ethical_cost_rust", None)
    )
except ImportError:
    _rust_ethical_cost = None

_HAS_RUST = _rust_ethical_cost is not None


@dataclass
class EthicalCost:
    """Report the numerical score and weighted constraint penalties.

    Attributes
    ----------
    J_sec : float
        Weighted coherence, connectivity, density and phase-dispersion score.
        Finite signed weights and self-loops can put it outside ``[0, 1]``.
    phi_ethics : float
        ``kappa`` times the sum of squared positive constraint residuals.
    c15_sec : float
        ``1 - J_sec + phi_ethics``; this diagnostic is not a safety guarantee.
    constraints_violated : int
        Number of strictly positive residuals, independent of ``kappa``.
    """

    J_sec: float
    phi_ethics: float
    c15_sec: float
    constraints_violated: int


def _finite_cost(j: float, phi: float, c15: float, nv: int) -> EthicalCost:
    """Refuse an unrepresentable diagnostic from either compute backend."""
    if not all(np.isfinite(value) for value in (j, phi, c15)):
        raise ValueError("ethical cost arithmetic must remain finite")
    return EthicalCost(
        J_sec=j,
        phi_ethics=phi,
        c15_sec=c15,
        constraints_violated=nv,
    )


def _validated_inputs(phases: object, knm: object) -> tuple[FloatArray, FloatArray]:
    """Return finite phases and a matching square coupling matrix, else raise.

    Source types are checked before float conversion so boolean, text,
    complex and temporal aliases cannot become admissible measurements.
    Plain real numeric object arrays retain their ordinary numeric meaning.
    """
    try:
        require_real_values(phases, name="phases", allow_object=True)
        require_real_values(knm, name="knm", allow_object=True)
        phase_array = np.asarray(phases, dtype=np.float64)
        knm_array = np.asarray(knm, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("phases and knm must contain plain real numbers") from exc
    if phase_array.ndim != 1:
        raise ValueError("phases must be a one-dimensional vector")
    n = phase_array.shape[0]
    if knm_array.shape != (n, n):
        raise ValueError(
            f"knm must have shape ({n}, {n}) to match phases, got {knm_array.shape}"
        )
    if not np.all(np.isfinite(phase_array)):
        raise ValueError("phases must contain only finite values")
    if not np.all(np.isfinite(knm_array)):
        raise ValueError("knm must contain only finite values")
    return phase_array, knm_array


def _finite_parameter(value: object, *, name: str) -> float:
    """Extract a representable real parameter without source-type coercion."""
    if isinstance(
        value, (bool, np.bool_, np.datetime64, np.timedelta64)
    ) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    try:
        scalar = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    return scalar


def compute_ethical_cost(
    phases: FloatArray,
    knm: FloatArray,
    *,
    alpha_R: float = 0.4,
    beta_K: float = 0.3,
    gamma_Q: float = 0.2,
    nu_S: float = 0.1,
    kappa: float = 1.0,
    R_min: float = 0.2,
    connectivity_min: float = 0.1,
    max_coupling: float = 5.0,
) -> EthicalCost:
    """Compute C15_sec ethical cost term.

    J_sec = α·R + β·K_norm + γ·Q - ν·S_dev
    where:
      R = Kuramoto order parameter (coherence)
      K_norm = λ₂(L) / N (unit-complete-graph normalization)
      Q = count_nonzero(knm) / (N * (N - 1)), or zero for N < 2
      S_dev = std(phases, ddof=0) / π (raw, not wrapped, phase dispersion)

    The graph uses reciprocal magnitude averages and excludes self-loops.
    Density counts every exactly nonzero matrix entry, including signed,
    arbitrarily small and diagonal entries. Inputs are never modified.

    Φ_ethics = Σ max(0, g_k)² where g_k are CBF constraint violations:
      g_1: R_min - R                   (non-harm: minimum coherence)
      g_2: connectivity_min - λ₂       (Wiener: maintain connectivity)
      g_3: max(K_ij) - max_coupling    (boundary: coupling limits)

    Parameters
    ----------
    phases : FloatArray
        Finite real oscillator phases in radians, shape ``(N,)``. They need
        not be wrapped; boolean, text, complex and temporal aliases are refused.
    knm : FloatArray
        Finite real coupling matrix ``K_nm``, shape ``(N, N)``. Signed,
        asymmetric and diagonal weights are supported.
    alpha_R : float
        Order-parameter cost weight.
    beta_K : float
        Coupling-cost weight.
    gamma_Q : float
        Quality-cost weight.
    nu_S : float
        Raw phase-dispersion cost weight.
    kappa : float
        Multiplier applied once to the squared positive residual sum.
    R_min : float
        Minimum order-parameter target.
    connectivity_min : float
        Minimum algebraic connectivity.
    max_coupling : float
        Threshold for the largest positive coupling. If there is no positive
        entry, the coupling residual is zero regardless of this threshold.

    Returns
    -------
    EthicalCost
        C15_sec ethical cost term.

    Raises
    ------
    ValueError
        If ``phases`` is not a finite 1-D vector, ``knm`` is not a finite
        square matrix matching it, a weight or threshold is not finite,
        or constraint or cost arithmetic cannot remain finite. Scalar weights
        and thresholds must be non-boolean real numbers; signed finite values
        remain admissible. Empty inputs still validate these parameters.
    """
    phases, knm = _validated_inputs(phases, knm)
    alpha_R, beta_K, gamma_Q, nu_S, kappa, R_min, connectivity_min, max_coupling = (
        _finite_parameter(value, name=name)
        for name, value in (
            ("alpha_R", alpha_R),
            ("beta_K", beta_K),
            ("gamma_Q", gamma_Q),
            ("nu_S", nu_S),
            ("kappa", kappa),
            ("R_min", R_min),
            ("connectivity_min", connectivity_min),
            ("max_coupling", max_coupling),
        )
    )
    n = len(phases)
    if n == 0:
        return EthicalCost(
            J_sec=0.0, phi_ethics=0.0, c15_sec=1.0, constraints_violated=0
        )

    coupling_violation = float(np.max(knm)) - max_coupling if np.any(knm > 0) else 0.0
    if not np.isfinite(coupling_violation) or coupling_violation > np.sqrt(
        np.finfo(float).max
    ):
        raise ValueError("ethical constraint arithmetic must remain finite")

    if _rust_ethical_cost is not None:
        p: FloatArray = np.require(phases, dtype=np.float64, requirements=["C", "A"])
        k: FloatArray = np.require(
            knm.ravel(), dtype=np.float64, requirements=["C", "A"]
        )
        j, phi, c15, nv = _rust_ethical_cost(
            p,
            k,
            n,
            alpha_R,
            beta_K,
            gamma_Q,
            nu_S,
            kappa,
            R_min,
            connectivity_min,
            max_coupling,
        )
        return _finite_cost(j, phi, c15, nv)

    try:
        with np.errstate(over="raise", invalid="raise"):
            R, _ = compute_order_parameter(phases)
            lam2 = fiedler_value(knm)
            S_dev = float(np.std(phases)) / np.pi
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ValueError("ethical SEC arithmetic must remain finite") from exc
    lam2_max = float(n)  # max λ₂ for complete graph with unit weights
    K_norm = lam2 / lam2_max if lam2_max > 0 else 0.0

    n_nonzero = np.count_nonzero(knm)
    n_possible = n * (n - 1)
    Q = n_nonzero / n_possible if n_possible > 0 else 0.0

    J_sec = alpha_R * R + beta_K * K_norm + gamma_Q * Q - nu_S * S_dev

    # CBF constraint violations
    g = [
        R_min - R,
        connectivity_min - lam2,
        coupling_violation,
    ]
    if not all(np.isfinite(value) for value in g) or any(
        value > np.sqrt(np.finfo(float).max) for value in g
    ):
        raise ValueError("ethical constraint arithmetic must remain finite")
    violations = [max(0.0, gi) ** 2 for gi in g]
    phi_ethics = kappa * sum(violations)
    n_violated = sum(1 for gi in g if gi > 0)

    c15_sec = (1.0 - J_sec) + phi_ethics

    return _finite_cost(J_sec, phi_ethics, c15_sec, n_violated)
