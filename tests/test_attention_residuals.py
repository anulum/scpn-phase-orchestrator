# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for full multi-head AttnRes

"""Unit + integration tests for the multi-head AttnRes modulation
(``coupling/attention_residuals.py``).

Covers every structural and physics-level invariant from the research
doc ``research_attention_residuals_2026-04-06.md §5``:

* Symmetry preservation — ``K_mod[i, j] == K_mod[j, i]``.
* Zero diagonal preservation.
* ``lambda_ == 0`` identity fallback.
* Block windowing (no attention outside ±``block_size``).
* Non-creation of edges (zero ``K_nm`` entries stay zero).
* Default projections load and are reproducible from ``seed``.
* Per-head structure — shapes of ``W_Q/W_K/W_V/W_O`` are validated.
* ``R`` within 5 % of baseline for a small ``lambda_`` and the
  default projections.
* Constructor contracts — shape mismatches, non-divisible ``n_heads``,
  negative lambda, zero temperature.
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
    PHASE_EMBED_DIM,
    attnres_modulate,
    default_projections,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def _symmetric_knm(n: int, strength: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = rng.uniform(0.0, 2.0 * strength, size=(n, n))
    knm = 0.5 * (half + half.T)
    np.fill_diagonal(knm, 0.0)
    return knm.astype(np.float64)


# ---------------------------------------------------------------------
# default_projections
# ---------------------------------------------------------------------


class TestDefaultProjections:
    def test_shapes(self) -> None:
        w_q, w_k, w_v, w_o = default_projections(n_heads=4, seed=0)
        d_model = PHASE_EMBED_DIM
        d_head = d_model // 4
        assert w_q.shape == (4, d_model, d_head)
        assert w_k.shape == (4, d_model, d_head)
        assert w_v.shape == (4, d_model, d_head)
        assert w_o.shape == (4 * d_head, d_model)

    def test_seed_reproducible(self) -> None:
        a = default_projections(n_heads=4, seed=42)
        b = default_projections(n_heads=4, seed=42)
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)

    def test_different_seeds_differ(self) -> None:
        a = default_projections(n_heads=4, seed=0)
        b = default_projections(n_heads=4, seed=1)
        # At least one of the four should differ — they're independently seeded.
        assert any(not np.array_equal(x, y) for x, y in zip(a, b, strict=True))

    def test_non_divisible_rejected(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            default_projections(n_heads=3, d_model=8)


# ---------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------


def test_symmetry_preserved() -> None:
    knm = _symmetric_knm(8, seed=1)
    theta = np.linspace(0.0, TWO_PI, 8, endpoint=False)
    k_mod = attnres_modulate(knm, theta, lambda_=0.5)
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
    knm = _symmetric_knm(8, seed=4)
    # Knock out symmetric pairs
    knm[0, 3] = knm[3, 0] = 0.0
    knm[2, 5] = knm[5, 2] = 0.0
    theta = np.random.default_rng(1).uniform(0.0, TWO_PI, size=8)
    k_mod = attnres_modulate(knm, theta, lambda_=0.5)
    for i, j in [(0, 3), (3, 0), (2, 5), (5, 2)]:
        assert k_mod[i, j] == 0.0


def test_block_size_restricts_attention() -> None:
    """With block_size = 2, pairs with |i - j| > 2 keep original K."""
    n = 12
    knm = _symmetric_knm(n, seed=7)
    theta = np.random.default_rng(4).uniform(0.0, TWO_PI, size=n)
    k_mod = attnres_modulate(knm, theta, block_size=2, lambda_=0.5)
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 2:
                assert k_mod[i, j] == pytest.approx(knm[i, j], abs=1e-12), (
                    f"out-of-block ({i}, {j}) was modulated"
                )


def test_full_attention_default() -> None:
    """Default block_size=None → full-N attention. Distant pairs DO get
    modulated (unlike block_size=4 where they would not)."""
    n = 16
    knm = _symmetric_knm(n, strength=0.1, seed=99)
    theta = np.random.default_rng(13).uniform(0.0, TWO_PI, size=n)
    k_full = attnres_modulate(knm, theta, lambda_=0.5)
    k_block = attnres_modulate(knm, theta, block_size=2, lambda_=0.5)
    # The two results must differ for at least one distant pair.
    far_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 2:
                far_mask[i, j] = True
    assert np.any(np.abs(k_full - k_block)[far_mask] > 1e-8)


# ---------------------------------------------------------------------
# Validation criterion — R within 5 % of baseline
# ---------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=3,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_order_parameter_within_five_percent(seed: int) -> None:
    """Steady-state R under AttnRes must stay within 5 % of the baseline
    un-modulated R at small ``lambda_`` for a supercritical coupling."""
    n = 16
    dt = 0.01
    n_warmup = 300
    n_measure = 200
    lambda_ = 0.1  # small modulation so the physics stays dominated
    # by the base coupling — doc §5 explicitly uses small λ here.

    rng = np.random.default_rng(seed)
    omegas = (rng.standard_normal(n) * 0.5).astype(np.float64)
    knm = _symmetric_knm(n, strength=5.0 / n, seed=seed)
    alpha = np.zeros((n, n), dtype=np.float64)

    def _r(knm_fn: Callable[[np.ndarray], np.ndarray]) -> float:
        phases = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)
        engine = UPDEEngine(n_oscillators=n, dt=dt, method="euler")
        for _ in range(n_warmup):
            phases = engine.step(phases, omegas, knm_fn(phases), 0.0, 0.0, alpha)
        rs: list[float] = []
        for _ in range(n_measure):
            phases = engine.step(phases, omegas, knm_fn(phases), 0.0, 0.0, alpha)
            r, _ = compute_order_parameter(phases)
            rs.append(float(r))
        return float(np.mean(rs))

    r_base = _r(lambda _theta: knm)
    r_attn = _r(lambda theta: attnres_modulate(knm, theta, lambda_=lambda_))

    rel = abs(r_attn - r_base) / max(r_base, 1e-6)
    assert rel <= 0.05, (
        f"R(attnres)={r_attn:.4f} vs R(baseline)={r_base:.4f}, "
        f"relative change {rel * 100:.2f}% exceeds 5 % budget"
    )


# ---------------------------------------------------------------------
# Constructor contracts
# ---------------------------------------------------------------------


class TestContractFailures:
    def test_non_square_knm_rejected(self) -> None:
        with pytest.raises(ValueError, match="square"):
            attnres_modulate(np.zeros((4, 5)), np.zeros(4))

    def test_theta_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            attnres_modulate(_symmetric_knm(4), np.zeros(6))

    def test_block_size_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="block_size"):
            attnres_modulate(_symmetric_knm(4), np.zeros(4), block_size=0)

    def test_temperature_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            attnres_modulate(_symmetric_knm(4), np.zeros(4), temperature=0.0)

    def test_negative_lambda_rejected(self) -> None:
        with pytest.raises(ValueError, match="lambda_"):
            attnres_modulate(_symmetric_knm(4), np.zeros(4), lambda_=-0.1)


# ---------------------------------------------------------------------
# Idempotence
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
        assert "python" in AVAILABLE_BACKENDS
        assert AVAILABLE_BACKENDS[-1] == "python"

    def test_active_backend_is_first_available(self) -> None:
        assert AVAILABLE_BACKENDS[0] == ACTIVE_BACKEND

    def test_fastest_first_ordering(self) -> None:
        canonical = ["rust", "mojo", "julia", "go", "python"]
        indices = [canonical.index(b) for b in AVAILABLE_BACKENDS]
        assert indices == sorted(indices)
