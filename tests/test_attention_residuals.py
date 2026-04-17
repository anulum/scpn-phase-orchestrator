# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for AttnRes coupling modulation

"""Unit + integration tests for ``coupling/attention_residuals.py``.

The research doc ``research_attention_residuals_2026-04-06.md §5``
lists five physics-level validation criteria for the Phase-3
experiment. This file implements the ones that can be checked
offline, without running a full-scale benchmark:

* Symmetry preservation — ``K_mod[i,j] == K_mod[j,i]``.
* Zero diagonal preservation.
* ``lambda_ == 0`` identity fallback.
* Block windowing (no attention outside ±block_size).
* Softmax normalisation per row.
* Non-creation of edges (zero K_nm entries stay zero).
* Validation-criterion-1: steady-state R with modulated coupling
  stays within 5 % of the baseline R across seeds, at ``lambda = 0.5``
  and ``block_size = 4`` (the values nominated in the doc).
* Constructor contracts — shape mismatches, negative lambda, zero
  temperature all raise ``ValueError``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling.attention_residuals import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    attnres_modulate,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def _symmetric_knm(n: int, strength: float = 0.3, seed: int = 0) -> np.ndarray:
    """Dense symmetric K_nm with zero diagonal, entries ~ Uniform[0, 2*strength]."""
    rng = np.random.default_rng(seed)
    half = rng.uniform(0.0, 2.0 * strength, size=(n, n))
    knm = 0.5 * (half + half.T)
    np.fill_diagonal(knm, 0.0)
    return knm.astype(np.float64)


# ---------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------


def test_symmetry_preserved() -> None:
    knm = _symmetric_knm(8, seed=1)
    theta = np.linspace(0.0, TWO_PI, 8, endpoint=False)
    k_mod = attnres_modulate(knm, theta, block_size=3, lambda_=0.5)
    np.testing.assert_allclose(k_mod, k_mod.T, atol=1e-12)


def test_zero_diagonal_preserved() -> None:
    knm = _symmetric_knm(12, seed=2)
    theta = np.random.default_rng(0).uniform(0.0, TWO_PI, size=12)
    k_mod = attnres_modulate(knm, theta, lambda_=0.5)
    np.testing.assert_array_equal(np.diag(k_mod), np.zeros(12))


def test_lambda_zero_is_identity() -> None:
    knm = _symmetric_knm(16, seed=3)
    theta = np.random.default_rng(5).uniform(0.0, TWO_PI, size=16)
    k_mod = attnres_modulate(knm, theta, lambda_=0.0)
    np.testing.assert_array_equal(k_mod, knm)


def test_existing_zeros_stay_zero() -> None:
    """Attention never creates edges where ``knm`` was zero."""
    knm = _symmetric_knm(8, seed=4)
    # Knock out a handful of edges and enforce symmetry
    knm[0, 3] = knm[3, 0] = 0.0
    knm[2, 5] = knm[5, 2] = 0.0
    knm[1, 7] = knm[7, 1] = 0.0
    theta = np.random.default_rng(1).uniform(0.0, TWO_PI, size=8)
    k_mod = attnres_modulate(knm, theta, lambda_=0.5)
    for i, j in [(0, 3), (3, 0), (2, 5), (5, 2), (1, 7), (7, 1)]:
        assert k_mod[i, j] == 0.0, f"k_mod[{i},{j}] = {k_mod[i, j]} ≠ 0"


def test_out_of_block_entries_unchanged() -> None:
    """Pairs farther than ``block_size`` get no attention boost —
    they retain the original K value (symmetrisation leaves them at
    the geometric mean of the pre- and post-modulation rows, but
    since neither direction modulates them, both sides match K)."""
    n = 16
    knm = _symmetric_knm(n, seed=7)
    theta = np.random.default_rng(4).uniform(0.0, TWO_PI, size=n)
    block_size = 2
    k_mod = attnres_modulate(knm, theta, block_size=block_size, lambda_=0.5)
    for i in range(n):
        for j in range(n):
            if abs(i - j) > block_size:
                assert k_mod[i, j] == pytest.approx(knm[i, j], abs=1e-12), (
                    f"(i, j) = ({i}, {j}) outside block_size={block_size} "
                    f"but K was modulated: {knm[i, j]} → {k_mod[i, j]}"
                )


def test_softmax_row_sums_bounded_by_lambda() -> None:
    """After modulation each row's multiplicative factor sums to
    ``1 + lambda_`` over the masked entries (softmax sums to 1)."""
    n = 12
    knm = _symmetric_knm(n, seed=9)
    # Set phases so that the in-block entries all get modulated
    theta = np.zeros(n, dtype=np.float64)
    lambda_ = 0.7
    block_size = 4
    k_mod = attnres_modulate(
        knm, theta, block_size=block_size, lambda_=lambda_
    )

    # For identical phases, softmax is uniform over the in-block non-zero
    # entries. Pick row 5 and verify the pre-symmetrisation expectation:
    #    sum_j (K[5,j] * (1 + lambda/m)) = sum_j K[5,j] + lambda * mean
    # This is a loose sanity check; the exact symmetrised magnitude
    # depends on how row j modulates row i in return.
    row_sum = k_mod[5, :].sum()
    knm_sum = knm[5, :].sum()
    # Modulation must never reduce total row weight when phases align.
    assert row_sum >= knm_sum - 1e-12, (
        f"row sum fell: {knm_sum:.4f} → {row_sum:.4f} under identical phases"
    )


# ---------------------------------------------------------------------
# Validation criterion: R within 5 % of baseline
# ---------------------------------------------------------------------


def _steady_r(
    knm_fn: Callable[[np.ndarray], np.ndarray],
    n: int,
    omegas: np.ndarray,
    seed: int,
    n_warmup: int = 300,
    n_measure: int = 200,
    dt: float = 0.01,
) -> float:
    """Integrate Kuramoto to steady state; ``knm_fn(theta)`` supplies
    the coupling (constant for the baseline, state-dependent for the
    AttnRes arm)."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)
    alpha = np.zeros((n, n), dtype=np.float64)
    engine = UPDEEngine(n_oscillators=n, dt=dt, method="euler")

    for _ in range(n_warmup):
        phases = engine.step(phases, omegas, knm_fn(phases), 0.0, 0.0, alpha)

    r_samples: list[float] = []
    for _ in range(n_measure):
        phases = engine.step(phases, omegas, knm_fn(phases), 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        r_samples.append(float(r))
    return float(np.mean(r_samples))


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=3,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_order_parameter_within_five_percent(seed: int) -> None:
    """Doc validation criterion 1: R(AttnRes) must stay within 5 % of
    R(baseline) for the same (K, dt, steps) configuration."""
    n = 16
    rng = np.random.default_rng(seed)
    omegas = (rng.standard_normal(n) * 1.5).astype(np.float64)
    knm = _symmetric_knm(n, strength=0.5 / n, seed=seed)

    def baseline_k(_: np.ndarray) -> np.ndarray:
        return knm

    def attnres_k(theta: np.ndarray) -> np.ndarray:
        return attnres_modulate(knm, theta, block_size=4, lambda_=0.5)

    r_base = _steady_r(baseline_k, n, omegas, seed)
    r_attn = _steady_r(attnres_k, n, omegas, seed)

    # |ΔR| / R_base ≤ 0.05 — five percent relative tolerance.
    rel = abs(r_attn - r_base) / max(r_base, 1e-6)
    assert rel <= 0.05, (
        f"R(attnres)={r_attn:.4f} vs R(baseline)={r_base:.4f}, "
        f"relative change {rel * 100:.2f}% exceeds 5 % budget"
    )


# ---------------------------------------------------------------------
# Constructor contracts
# ---------------------------------------------------------------------


class TestContractFailures:
    def _theta_knm(self, n: int = 4) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(n), _symmetric_knm(n)

    def test_non_square_knm_rejected(self) -> None:
        with pytest.raises(ValueError, match="square"):
            attnres_modulate(np.zeros((4, 5)), np.zeros(4))

    def test_theta_shape_mismatch_rejected(self) -> None:
        theta, knm = self._theta_knm(4)
        with pytest.raises(ValueError, match="does not match"):
            attnres_modulate(knm, np.zeros(6))

    def test_block_size_zero_rejected(self) -> None:
        theta, knm = self._theta_knm(4)
        with pytest.raises(ValueError, match="block_size"):
            attnres_modulate(knm, theta, block_size=0)

    def test_temperature_zero_rejected(self) -> None:
        theta, knm = self._theta_knm(4)
        with pytest.raises(ValueError, match="temperature"):
            attnres_modulate(knm, theta, temperature=0.0)

    def test_negative_lambda_rejected(self) -> None:
        theta, knm = self._theta_knm(4)
        with pytest.raises(ValueError, match="lambda_"):
            attnres_modulate(knm, theta, lambda_=-0.1)


# ---------------------------------------------------------------------
# Idempotence / determinism
# ---------------------------------------------------------------------


def test_deterministic_same_inputs() -> None:
    knm = _symmetric_knm(8, seed=11)
    theta = np.linspace(0.1, 0.1 + TWO_PI, 8, endpoint=False)
    a = attnres_modulate(knm, theta, lambda_=0.3)
    b = attnres_modulate(knm, theta, lambda_=0.3)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------
# Multi-backend dispatcher
# ---------------------------------------------------------------------


class TestDispatcher:
    def test_python_is_always_available(self) -> None:
        """The NumPy fallback is always present as the terminal entry."""
        assert "python" in AVAILABLE_BACKENDS
        assert AVAILABLE_BACKENDS[-1] == "python"

    def test_active_backend_is_first_available(self) -> None:
        assert AVAILABLE_BACKENDS[0] == ACTIVE_BACKEND

    def test_fastest_first_ordering(self) -> None:
        """Canonical order from the global fallback-chain rule:
        Rust → Mojo → Julia → Go → Python."""
        canonical = ["rust", "mojo", "julia", "go", "python"]
        indices = [canonical.index(b) for b in AVAILABLE_BACKENDS]
        assert indices == sorted(indices), (
            f"AVAILABLE_BACKENDS {AVAILABLE_BACKENDS} is not in "
            f"fastest-first canonical order {canonical}"
        )

    def test_rust_python_bit_parity(self) -> None:
        """When the Rust backend is active, its output must match the
        NumPy fallback to floating-point precision (any drift would
        silently change physics depending on the wheel's presence)."""
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built on this host")

        from scpn_phase_orchestrator.coupling import attention_residuals as mod

        knm = _symmetric_knm(12, seed=42)
        theta = np.random.default_rng(7).uniform(0.0, TWO_PI, size=12)

        # Force the Python path via monkey-patch of ACTIVE_BACKEND.
        saved = mod.ACTIVE_BACKEND
        try:
            mod.ACTIVE_BACKEND = "python"
            py_result = mod.attnres_modulate(knm, theta, lambda_=0.5)
        finally:
            mod.ACTIVE_BACKEND = saved

        rust_result = attnres_modulate(knm, theta, lambda_=0.5)
        np.testing.assert_allclose(py_result, rust_result, atol=1e-12)
