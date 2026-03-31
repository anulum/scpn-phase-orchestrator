# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stuart-Landau coupling tests

"""Tests for amplitude coupling extensions in CouplingBuilder
and their wiring into the Stuart-Landau simulation pipeline."""

from __future__ import annotations

import time

import numpy as np

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder


# ---------------------------------------------------------------------------
# K_nm^r: amplitude coupling matrix contracts
# ---------------------------------------------------------------------------


class TestAmplitudeCouplingContracts:
    """Verify that build_with_amplitude produces K_nm^r matrices satisfying
    all mathematical contracts: symmetry, zero diagonal, non-negativity,
    distance-dependent decay."""

    def test_knm_r_symmetric(self):
        cs = CouplingBuilder().build_with_amplitude(8, 0.5, 0.3, 0.3, 0.3)
        np.testing.assert_allclose(cs.knm_r, cs.knm_r.T, atol=1e-14)

    def test_knm_r_zero_diagonal(self):
        """No self-coupling: K_ii^r = 0 for all i."""
        cs = CouplingBuilder().build_with_amplitude(8, 0.5, 0.3, 0.3, 0.3)
        np.testing.assert_allclose(np.diag(cs.knm_r), 0.0)

    def test_knm_r_nonnegative(self):
        cs = CouplingBuilder().build_with_amplitude(8, 0.5, 0.3, 0.3, 0.3)
        assert np.all(cs.knm_r >= 0.0)

    def test_knm_r_decays_with_distance(self):
        """K_{0,1}^r > K_{0,4}^r: closer oscillators couple more strongly."""
        cs = CouplingBuilder().build_with_amplitude(8, 0.5, 0.3, 0.5, 0.3)
        assert cs.knm_r[0, 1] >= cs.knm_r[0, 4], (
            f"K_01^r={cs.knm_r[0,1]:.4f} should >= K_04^r={cs.knm_r[0,4]:.4f}"
        )

    def test_knm_r_independent_of_phase_coupling(self):
        """Phase coupling (K_nm) and amplitude coupling (K_nm^r) should
        have different magnitudes when configured differently."""
        cs = CouplingBuilder().build_with_amplitude(4, 0.5, 0.3, 0.8, 0.1)
        assert not np.allclose(cs.knm, cs.knm_r)

    def test_phase_coupling_unchanged_by_amplitude(self):
        """Adding amplitude coupling must not alter the phase coupling matrix."""
        builder = CouplingBuilder()
        phase_only = builder.build(4, 0.5, 0.3)
        with_amp = builder.build_with_amplitude(4, 0.5, 0.3, 0.3, 0.3)
        np.testing.assert_allclose(phase_only.knm, with_amp.knm)

    def test_default_build_has_no_knm_r(self):
        """Standard build (Kuramoto-only) must have knm_r = None."""
        cs = CouplingBuilder().build(4, 0.5, 0.3)
        assert cs.knm_r is None

    def test_stronger_amp_coupling_larger_knm_r(self):
        """Higher amp_coupling_strength → larger K_nm^r values."""
        cs_weak = CouplingBuilder().build_with_amplitude(4, 0.5, 0.3, 0.1, 0.3)
        cs_strong = CouplingBuilder().build_with_amplitude(4, 0.5, 0.3, 1.0, 0.3)
        assert np.mean(cs_strong.knm_r) > np.mean(cs_weak.knm_r), (
            "Stronger amp coupling should produce larger K_nm^r"
        )


# ---------------------------------------------------------------------------
# Template switching with amplitude coupling
# ---------------------------------------------------------------------------


class TestTemplateSwitchWithAmplitude:
    """Verify that switching K_nm templates preserves K_nm^r (amplitude
    coupling topology is fixed by the binding spec, not by templates)."""

    def test_switch_preserves_knm_r(self):
        builder = CouplingBuilder()
        cs = builder.build_with_amplitude(4, 0.5, 0.3, 0.3, 0.3)
        templates = {"custom": np.eye(4)}
        switched = builder.switch_template(cs, "custom", templates)
        assert switched.knm_r is not None
        np.testing.assert_allclose(switched.knm_r, cs.knm_r)

    def test_switch_changes_phase_coupling_only(self):
        builder = CouplingBuilder()
        cs = builder.build_with_amplitude(4, 0.5, 0.3, 0.3, 0.3)
        new_knm = np.full((4, 4), 0.1)
        np.fill_diagonal(new_knm, 0.0)
        templates = {"strong": new_knm}
        switched = builder.switch_template(cs, "strong", templates)
        # Phase coupling changed
        assert not np.allclose(switched.knm, cs.knm)
        # Amplitude coupling unchanged
        np.testing.assert_allclose(switched.knm_r, cs.knm_r)


# ---------------------------------------------------------------------------
# Pipeline wiring: CouplingBuilder → StuartLandauEngine → order parameter
# ---------------------------------------------------------------------------


class TestStuartLandauCouplingPipeline:
    """End-to-end: build coupling → feed into SL engine → compute R.
    Proves the amplitude coupling matrix is actually used in simulation."""

    def test_amplitude_coupling_affects_dynamics(self):
        """SL simulation with K_nm^r must produce different trajectories
        than without (amplitude coupling changes amplitude dynamics)."""
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        n = 8
        builder = CouplingBuilder()
        cs_no_amp = builder.build(n, 0.5, 0.3)
        cs_with_amp = builder.build_with_amplitude(n, 0.5, 0.3, 0.5, 0.3)

        rng = np.random.default_rng(42)
        phases0 = rng.uniform(0, 2 * np.pi, n)
        amps0 = np.full(n, 0.5)
        mu = np.full(n, 0.3)
        omegas = np.ones(n)
        alpha = np.zeros((n, n))

        # Run without amplitude coupling
        eng1 = StuartLandauEngine(n, dt=0.01)
        state1 = np.concatenate([phases0, amps0])
        for _ in range(200):
            state1 = eng1.step(
                state1, omegas, mu, cs_no_amp.knm,
                np.zeros((n, n)),  # no knm_r
                0.0, 0.0, alpha,
            )

        # Run with amplitude coupling
        eng2 = StuartLandauEngine(n, dt=0.01)
        state2 = np.concatenate([phases0, amps0])
        for _ in range(200):
            state2 = eng2.step(
                state2, omegas, mu, cs_with_amp.knm,
                cs_with_amp.knm_r,
                0.0, 0.0, alpha,
            )

        # Amplitude coupling should produce different final amplitudes
        amps1 = state1[n:]
        amps2 = state2[n:]
        assert not np.allclose(amps1, amps2, atol=1e-4), (
            "Amplitude coupling must change SL dynamics"
        )

        # Both should produce valid R
        r1, _ = compute_order_parameter(state1[:n])
        r2, _ = compute_order_parameter(state2[:n])
        assert 0.0 <= r1 <= 1.0
        assert 0.0 <= r2 <= 1.0

    def test_coupling_build_performance(self):
        """CouplingBuilder.build_with_amplitude for N=100 must complete
        in under 50ms (regression guard)."""
        builder = CouplingBuilder()
        t0 = time.perf_counter()
        for _ in range(10):
            builder.build_with_amplitude(100, 0.5, 0.3, 0.3, 0.3)
        elapsed = (time.perf_counter() - t0) / 10
        assert elapsed < 0.05, f"build_with_amplitude(100) took {elapsed*1000:.1f}ms, limit 50ms"
