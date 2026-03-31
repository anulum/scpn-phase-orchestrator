# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE numerics tests

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.numerics import IntegrationConfig, check_stability

# ---------------------------------------------------------------------------
# IntegrationConfig: defaults and constraints
# ---------------------------------------------------------------------------


class TestIntegrationConfig:
    """Verify IntegrationConfig defaults, immutability, and that parameters
    are consistent with solver requirements."""

    def test_defaults_match_safe_euler(self):
        cfg = IntegrationConfig(dt=0.001)
        assert cfg.substeps == 1
        assert cfg.method == "euler"
        assert cfg.max_dt == 0.01
        assert cfg.atol == 1e-6
        assert cfg.rtol == 1e-3

    def test_custom_rk4_config(self):
        cfg = IntegrationConfig(dt=0.005, substeps=4, method="rk4", max_dt=0.1)
        assert cfg.dt == 0.005
        assert cfg.substeps == 4
        assert cfg.method == "rk4"
        assert cfg.max_dt == 0.1

    def test_frozen_immutability(self):
        cfg = IntegrationConfig(dt=0.001)
        with pytest.raises(AttributeError):
            cfg.dt = 0.1

    def test_dt_less_than_max_dt_is_normal(self):
        """Typical usage: dt < max_dt. Config must accept this."""
        cfg = IntegrationConfig(dt=0.001, max_dt=0.01)
        assert cfg.dt < cfg.max_dt


# ---------------------------------------------------------------------------
# CFL stability check: the core numerical safety contract
# ---------------------------------------------------------------------------


class TestStabilityCheck:
    """Verify the CFL-like stability bound: dt * (max_omega + max_coupling) < π.
    This is the fundamental safety check that prevents the Euler integrator
    from producing nonsensical phase jumps (> half-cycle per step)."""

    def test_stable_small_dt(self):
        """Well within the stability region."""
        assert check_stability(dt=0.001, max_omega=2.0, max_coupling=1.0) is True

    def test_unstable_large_dt(self):
        """Far outside stability region: dt * deriv = 2 * 10 = 20 >> π."""
        assert check_stability(dt=2.0, max_omega=5.0, max_coupling=5.0) is False

    def test_zero_derivatives_always_stable(self):
        """No dynamics → no instability, regardless of dt."""
        assert check_stability(dt=1e6, max_omega=0.0, max_coupling=0.0) is True

    def test_boundary_just_below_pi(self):
        """dt * deriv just below π → stable."""
        # deriv = 3.14, dt = 0.999 → product = 3.13686 < π = 3.14159...
        assert check_stability(dt=0.999, max_omega=1.57, max_coupling=1.57) is True

    def test_boundary_just_above_pi(self):
        """dt * deriv just above π → unstable."""
        # deriv = 3.15, dt = 1.0 → product = 3.15 > π
        assert check_stability(dt=1.0, max_omega=1.58, max_coupling=1.57) is False

    def test_exact_pi_boundary(self):
        """dt * deriv = π exactly → unstable (strict inequality)."""
        # Construct exact boundary: dt=1, omega+coupling = π
        assert (
            check_stability(dt=1.0, max_omega=math.pi / 2, max_coupling=math.pi / 2)
            is False
        )

    def test_coupling_only_no_omega(self):
        """Pure coupling (ω=0) can still cause instability."""
        assert check_stability(dt=1.0, max_omega=0.0, max_coupling=10.0) is False
        assert check_stability(dt=0.1, max_omega=0.0, max_coupling=10.0) is True

    def test_omega_only_no_coupling(self):
        """Pure natural frequency (K=0) can still cause instability."""
        assert check_stability(dt=0.5, max_omega=10.0, max_coupling=0.0) is False
        assert check_stability(dt=0.1, max_omega=10.0, max_coupling=0.0) is True

    def test_stability_agrees_with_actual_euler_step(self):
        """Cross-validate: when check_stability says stable, an actual Euler step
        must not produce a phase change exceeding π (half-cycle)."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        dt = 0.01
        max_omega = 10.0
        max_coupling = 5.0
        assert check_stability(dt, max_omega, max_coupling) is True

        eng = UPDEEngine(2, dt=dt)
        phases = np.array([0.0, np.pi])
        omegas = np.array([max_omega, -max_omega])
        knm = np.array([[0.0, max_coupling], [max_coupling, 0.0]])
        alpha = np.zeros((2, 2))

        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        phase_change = np.abs(result - phases)
        assert np.all(phase_change < math.pi), (
            f"Stable dt produced phase change > π: {phase_change}"
        )

    def test_instability_produces_large_phase_jump(self):
        """Cross-validate: when check_stability says unstable, the Euler step
        should produce a phase change at or above π."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        dt = 1.0
        max_omega = 5.0
        max_coupling = 5.0
        assert check_stability(dt, max_omega, max_coupling) is False

        eng = UPDEEngine(2, dt=dt)
        phases = np.array([0.0, np.pi])
        omegas = np.array([max_omega, -max_omega])
        knm = np.array([[0.0, max_coupling], [max_coupling, 0.0]])
        alpha = np.zeros((2, 2))

        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        # With unstable dt, the raw phase change (before wrapping) exceeds π
        raw_change = np.abs(result - phases)
        max_change = np.max(raw_change)
        assert max_change > 1.0, (
            f"Unstable dt should produce large phase jumps, got max {max_change:.3f}"
        )


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
