# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mutation killer tests

"""Tests written specifically to kill mutants that survived mutation testing.
Each test targets a specific line/operator that mutmut can flip.

Source: Kaggle mutation testing run (2026-03-28), mutmut 2.4.5.
Survivors: order_params.py (16), numerics.py (5).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.numerics import IntegrationConfig, check_stability
from scpn_phase_orchestrator.upde.order_params import (
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
)

TWO_PI = 2.0 * np.pi


# ── order_params.py: compute_order_parameter ─────────────────────────────


class TestOrderParameterMutationKillers:
    """Kill mutants in compute_order_parameter lines 19-35."""

    def test_empty_returns_zero_zero(self) -> None:
        """Line 24-25: empty → (0.0, 0.0) exactly."""
        R, psi = compute_order_parameter(np.array([]))
        assert R == 0.0
        assert psi == 0.0

    def test_single_phase_r_one(self) -> None:
        """Single oscillator → R = 1.0 exactly."""
        R, _ = compute_order_parameter(np.array([1.5]))
        assert abs(R - 1.0) < 1e-12

    def test_opposite_phases_r_zero(self) -> None:
        """Two phases π apart → R = 0."""
        R, _ = compute_order_parameter(np.array([0.0, np.pi]))
        assert R < 1e-12

    def test_identical_phases_r_one(self) -> None:
        """All same phase → R = 1."""
        R, _ = compute_order_parameter(np.array([2.0, 2.0, 2.0]))
        assert abs(R - 1.0) < 1e-12

    def test_psi_equals_phase_for_single(self) -> None:
        """Single oscillator: psi = theta (mod 2π)."""
        theta = 3.7
        _, psi = compute_order_parameter(np.array([theta]))
        assert abs(psi - (theta % TWO_PI)) < 1e-10

    def test_psi_in_0_2pi(self) -> None:
        """psi must be in [0, 2π)."""
        for seed in range(10):
            rng = np.random.default_rng(seed)
            phases = rng.uniform(0, TWO_PI, 8)
            _, psi = compute_order_parameter(phases)
            assert 0.0 <= psi < TWO_PI + 1e-10

    def test_r_bounded_0_1(self) -> None:
        """R ∈ [0, 1] for any input."""
        for seed in range(10):
            rng = np.random.default_rng(seed)
            phases = rng.uniform(0, TWO_PI, 20)
            R, _ = compute_order_parameter(phases)
            assert 0.0 <= R <= 1.0 + 1e-12

    def test_known_three_phases(self) -> None:
        """Three phases at 0, 2π/3, 4π/3 → R ≈ 0."""
        phases = np.array([0.0, TWO_PI / 3, 2 * TWO_PI / 3])
        R, _ = compute_order_parameter(phases)
        assert R < 0.01

    def test_two_close_phases_high_r(self) -> None:
        """Two phases 0.01 apart → R ≈ 1."""
        R, _ = compute_order_parameter(np.array([1.0, 1.01]))
        assert R > 0.99

    def test_exp_ij_not_exp_j(self) -> None:
        """Verify the imaginary unit is used (mutant: 1j → 1)."""
        # If mutant replaces 1j with 1, exp(1*theta) is real → R = |mean(exp(theta))|
        # which diverges from the correct circular mean
        phases = np.array([0.0, np.pi / 2])
        R_correct, _ = compute_order_parameter(phases)
        # R should be cos(π/4) ≈ 0.707, not something wildly different
        assert abs(R_correct - np.sqrt(2) / 2) < 0.01


# ── order_params.py: compute_plv ─────────────────────────────────────────


class TestPLVMutationKillers:
    """Kill mutants in compute_plv lines 38-61."""

    def test_empty_a_returns_zero(self) -> None:
        assert compute_plv(np.array([]), np.array([1.0])) == 0.0

    def test_empty_b_returns_zero(self) -> None:
        assert compute_plv(np.array([1.0]), np.array([])) == 0.0

    def test_both_empty_returns_zero(self) -> None:
        assert compute_plv(np.array([]), np.array([])) == 0.0

    def test_size_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            compute_plv(np.array([1.0, 2.0]), np.array([1.0]))

    def test_identical_phases_plv_one(self) -> None:
        a = np.array([0.5, 1.0, 1.5])
        assert abs(compute_plv(a, a) - 1.0) < 1e-12

    def test_opposite_phases_plv_one(self) -> None:
        """Constant offset π → PLV = 1 (locked but anti-phase)."""
        a = np.array([0.0, 0.5, 1.0, 1.5])
        b = a + np.pi
        plv = compute_plv(a, b)
        assert abs(plv - 1.0) < 1e-10

    def test_random_unlocked_plv_low(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.uniform(0, TWO_PI, 200)
        b = rng.uniform(0, TWO_PI, 200)
        plv = compute_plv(a, b)
        assert plv < 0.3

    def test_plv_bounded_0_1(self) -> None:
        for seed in range(5):
            rng = np.random.default_rng(seed)
            a = rng.uniform(0, TWO_PI, 50)
            b = rng.uniform(0, TWO_PI, 50)
            plv = compute_plv(a, b)
            assert 0.0 <= plv <= 1.0 + 1e-12


# ── order_params.py: compute_layer_coherence ──────────────────────────────


class TestLayerCoherenceMutationKillers:
    """Kill mutants in compute_layer_coherence lines 64-75."""

    def test_empty_mask_returns_zero(self) -> None:
        phases = np.array([1.0, 2.0, 3.0])
        mask = np.array([False, False, False])
        assert compute_layer_coherence(phases, mask) == 0.0

    def test_full_mask_equals_global_r(self) -> None:
        phases = np.array([1.0, 1.0, 1.0])
        mask = np.array([True, True, True])
        R_layer = compute_layer_coherence(phases, mask)
        R_global, _ = compute_order_parameter(phases)
        assert abs(R_layer - R_global) < 1e-12

    def test_single_oscillator_r_one(self) -> None:
        phases = np.array([0.0, 1.0, 2.0])
        mask = np.array([False, True, False])
        assert abs(compute_layer_coherence(phases, mask) - 1.0) < 1e-12

    def test_subset_r_bounded(self) -> None:
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, 10)
        mask = np.array([True] * 5 + [False] * 5)
        R = compute_layer_coherence(phases, mask)
        assert 0.0 <= R <= 1.0 + 1e-12


# ── numerics.py: IntegrationConfig + check_stability ─────────────────────


class TestNumericsMutationKillers:
    """Kill mutants in IntegrationConfig defaults and check_stability."""

    def test_config_default_substeps(self) -> None:
        cfg = IntegrationConfig(dt=0.01)
        assert cfg.substeps == 1

    def test_config_default_method(self) -> None:
        cfg = IntegrationConfig(dt=0.01)
        assert cfg.method == "euler"

    def test_config_default_max_dt(self) -> None:
        cfg = IntegrationConfig(dt=0.01)
        assert cfg.max_dt == 0.01

    def test_config_default_atol(self) -> None:
        cfg = IntegrationConfig(dt=0.01)
        assert cfg.atol == 1e-6

    def test_config_default_rtol(self) -> None:
        cfg = IntegrationConfig(dt=0.01)
        assert cfg.rtol == 1e-3

    def test_stability_zero_deriv_true(self) -> None:
        assert check_stability(999.0, 0.0, 0.0) is True

    def test_stability_exact_pi_false(self) -> None:
        """dt * deriv = π → not strictly less → False."""
        assert check_stability(1.0, math.pi, 0.0) is False

    def test_stability_just_below_pi_true(self) -> None:
        assert check_stability(1.0, math.pi - 0.001, 0.0) is True

    def test_stability_sum_matters(self) -> None:
        """Both omega and coupling contribute to max_deriv."""
        assert check_stability(0.5, 4.0, 4.0) is False  # 0.5 * 8 = 4 > π
        assert check_stability(0.1, 3.0, 3.0) is True  # 0.1 * 6 = 0.6 < π

    def test_stability_addition_not_max(self) -> None:
        """Mutant: max_omega + max_coupling → max(max_omega, max_coupling)."""
        # If + mutated to max(), 0.1 * max(5,5) = 0.5 < π → True
        # But 0.1 * (5+5) = 1.0 < π → also True. Need case where they differ.
        # dt * (a+b) >= π but dt * max(a,b) < π
        dt = 1.0
        a, b = 2.0, 2.0
        # 1.0 * (2+2) = 4 > π → False
        # 1.0 * max(2,2) = 2 < π → True (if mutated)
        assert check_stability(dt, a, b) is False
