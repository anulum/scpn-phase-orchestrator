# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based invariant tests (UPDE engines)

"""Property-based invariants for the UPDE integrator family.

Two Phase-7 invariants from the SPO backlog are covered here:

* Index permutation equivariance — for any permutation π,
  ``step(π·phases, π·omegas, π·knm, π·alpha) == π · step(phases,
  omegas, knm, alpha)``. Kuramoto dynamics are label-free, so this is
  a structural invariant of the integrator implementation.
* Dense ↔ sparse CSR parity — for arbitrary sparsity patterns,
  ``UPDEEngine.step`` and ``SparseUPDEEngine.step`` must agree to
  ~1e-12; the sparse path is just CSR dispatch of the same
  arithmetic. The existing point-sample test is strengthened here to
  a Hypothesis property over random phases, frequencies, and
  sparsity masks.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine

TWO_PI = 2.0 * np.pi


def _dense_to_csr(
    knm: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert (K, α) to CSR row_ptr, col_indices, knm_values, alpha_values."""
    n = knm.shape[0]
    row_ptr = [0]
    col_indices: list[int] = []
    knm_values: list[float] = []
    alpha_values: list[float] = []
    for i in range(n):
        for j in range(n):
            if knm[i, j] != 0.0:
                col_indices.append(j)
                knm_values.append(float(knm[i, j]))
                alpha_values.append(float(alpha[i, j]))
        row_ptr.append(len(col_indices))
    return (
        np.array(row_ptr, dtype=np.uint64),
        np.array(col_indices, dtype=np.uint64),
        np.array(knm_values, dtype=np.float64),
        np.array(alpha_values, dtype=np.float64),
    )


# ---------------------------------------------------------------------
# Index permutation equivariance
# ---------------------------------------------------------------------


@given(
    n=st.integers(min_value=2, max_value=16),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_upde_index_permutation_equivariance(n: int, seed: int) -> None:
    """Step is equivariant under oscillator relabeling: for any
    permutation π, π(step(x)) == step(π(x)).
    """
    rng = np.random.default_rng(seed)
    dt = 0.01
    phases = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)
    omegas = rng.normal(loc=1.0, scale=0.1, size=n).astype(np.float64)
    knm = rng.normal(scale=0.3, size=(n, n)).astype(np.float64)
    np.fill_diagonal(knm, 0.0)
    alpha = rng.normal(scale=0.1, size=(n, n)).astype(np.float64)
    np.fill_diagonal(alpha, 0.0)

    perm = rng.permutation(n)

    engine = UPDEEngine(n_oscillators=n, dt=dt, method="euler")
    step_then_perm = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)[perm]

    # Apply permutation to all inputs
    p_phases = phases[perm]
    p_omegas = omegas[perm]
    p_knm = knm[np.ix_(perm, perm)]
    p_alpha = alpha[np.ix_(perm, perm)]

    engine2 = UPDEEngine(n_oscillators=n, dt=dt, method="euler")
    perm_then_step = engine2.step(p_phases, p_omegas, p_knm, 0.0, 0.0, p_alpha)

    # Phases wrap to [0, 2π); compare modulo the torus.
    diff = ((step_then_perm - perm_then_step + np.pi) % TWO_PI) - np.pi
    np.testing.assert_allclose(diff, np.zeros_like(diff), atol=1e-10)


@given(
    n=st.integers(min_value=3, max_value=12),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=20, deadline=None)
def test_upde_identity_permutation_is_fixed(n: int, seed: int) -> None:
    """Identity permutation leaves step() output unchanged bit-for-bit."""
    rng = np.random.default_rng(seed)
    dt = 0.01
    phases = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)
    omegas = rng.normal(size=n).astype(np.float64)
    knm = rng.normal(scale=0.2, size=(n, n)).astype(np.float64)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n), dtype=np.float64)

    engine = UPDEEngine(n_oscillators=n, dt=dt, method="euler")
    out = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    identity = np.arange(n)
    out_perm = engine.step(
        phases[identity],
        omegas[identity],
        knm[np.ix_(identity, identity)],
        0.0,
        0.0,
        alpha[np.ix_(identity, identity)],
    )
    np.testing.assert_array_equal(out, out_perm)


# ---------------------------------------------------------------------
# Dense ↔ sparse parity
# ---------------------------------------------------------------------


@given(
    n=st.integers(min_value=3, max_value=12),
    sparsity=st.floats(min_value=0.1, max_value=0.9),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_dense_sparse_euler_parity(n: int, sparsity: float, seed: int) -> None:
    """Dense and CSR-sparse Euler steps agree for arbitrary sparsity."""
    rng = np.random.default_rng(seed)
    dt = 0.01
    phases = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)
    omegas = rng.normal(loc=1.0, scale=0.2, size=n).astype(np.float64)

    knm = rng.normal(scale=0.3, size=(n, n)).astype(np.float64)
    mask = rng.uniform(size=(n, n)) > sparsity
    knm = knm * mask
    np.fill_diagonal(knm, 0.0)

    alpha = rng.normal(scale=0.05, size=(n, n)).astype(np.float64) * mask
    np.fill_diagonal(alpha, 0.0)

    engine_dense = UPDEEngine(n_oscillators=n, dt=dt, method="euler")
    engine_sparse = SparseUPDEEngine(n_oscillators=n, dt=dt, method="euler")

    p_dense = engine_dense.step(phases, omegas, knm, 0.0, 0.0, alpha)

    row_ptr, col_indices, knm_values, alpha_values = _dense_to_csr(knm, alpha)
    p_sparse = engine_sparse.step(
        phases, omegas, row_ptr, col_indices, knm_values, 0.0, 0.0, alpha_values
    )
    np.testing.assert_allclose(p_dense, p_sparse, atol=1e-12)


@given(
    n=st.integers(min_value=4, max_value=10),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_dense_sparse_rk4_parity(n: int, seed: int) -> None:
    """Dense and CSR-sparse RK4 steps agree for arbitrary inputs."""
    rng = np.random.default_rng(seed)
    dt = 0.01
    phases = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)
    omegas = rng.normal(loc=1.0, scale=0.2, size=n).astype(np.float64)
    knm = rng.normal(scale=0.25, size=(n, n)).astype(np.float64)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n), dtype=np.float64)

    engine_dense = UPDEEngine(n_oscillators=n, dt=dt, method="rk4")
    engine_sparse = SparseUPDEEngine(n_oscillators=n, dt=dt, method="rk4")

    p_dense = engine_dense.step(phases, omegas, knm, 0.0, 0.0, alpha)
    row_ptr, col_indices, knm_values, alpha_values = _dense_to_csr(knm, alpha)
    p_sparse = engine_sparse.step(
        phases, omegas, row_ptr, col_indices, knm_values, 0.0, 0.0, alpha_values
    )
    np.testing.assert_allclose(p_dense, p_sparse, atol=1e-12)
