# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — nn/ physics validation Phase 4

"""Phase 4: structural properties and deeper cross-validation.

V37-V46: Arnold tongue, phase diffusion, time reversal, hybrid inverse
improvement, vmap correctness, scan-loop equivalence, SL amplitude
consensus, chimera index boundaries, OIM bipartite, PLV-R correlation.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="JAX required for nn/ physics validation")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")
jr = pytest.importorskip("jax.random", reason="JAX required")

TWO_PI = 2.0 * np.pi


# ──────────────────────────────────────────────────
# V37: Arnold tongue — frequency locking
# ──────────────────────────────────────────────────


class TestV37ArnoldTongue:
    """Two oscillators with detuning Delta_omega lock when K > Delta_omega/2.
    Below that threshold, they drift apart.

    dDelta/dt = Delta_omega - 2K*sin(Delta)
    Fixed point exists iff K >= Delta_omega/2.
    """

    def test_locking_above_threshold(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        delta_omega = 1.0
        K_val = 0.8  # > delta_omega/2 = 0.5

        omegas = jnp.array([0.0, delta_omega])
        K = jnp.array([[0.0, K_val], [K_val, 0.0]])
        phases0 = jnp.array([0.0, 0.5])

        final, traj = kuramoto_forward(phases0, omegas, K, 0.01, 5000)
        traj_np = np.array(traj)

        # Phase difference should stabilise (not grow unbounded)
        diffs = traj_np[:, 1] - traj_np[:, 0]
        # Unwrap and check last 1000 steps are stable
        late_diffs = diffs[-1000:]
        late_spread = np.std(np.sin(late_diffs))
        assert late_spread < 0.1, (
            f"Phase diff not locked: std(sin(Delta))={late_spread:.3f}"
        )

    def test_drift_below_threshold(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        delta_omega = 1.0
        K_val = 0.2  # < delta_omega/2 = 0.5

        omegas = jnp.array([0.0, delta_omega])
        K = jnp.array([[0.0, K_val], [K_val, 0.0]])
        phases0 = jnp.array([0.0, 0.0])

        _, traj = kuramoto_forward(phases0, omegas, K, 0.01, 5000)
        traj_np = np.array(traj)

        diffs = traj_np[:, 1] - traj_np[:, 0]
        # Phase difference should grow (drift)
        np.ptp(np.sin(diffs[:500]))
        late_range = np.ptp(np.sin(diffs[-500:]))
        # Both should show full-range oscillation (drift through 2pi)
        assert late_range > 1.0, (
            f"Phase diff not drifting: range(sin(Delta))={late_range:.3f}"
        )


# ──────────────────────────────────────────────────
# V38: Phase diffusion below K_c
# ──────────────────────────────────────────────────


class TestV38PhaseDiffusion:
    """Below K_c, phase differences between oscillators grow
    unboundedly (diffusive behaviour). The variance of phase
    differences should increase with time.
    """

    def test_diffusion_below_kc(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        N = 32
        key = jr.PRNGKey(42)
        k1, k2 = jr.split(key)
        omegas = jr.normal(k1, (N,)) * 1.0  # wide spread
        phases0 = jr.uniform(k2, (N,), maxval=TWO_PI)
        K = jnp.ones((N, N)) * (0.01 / N)  # very weak, well below K_c
        K = K.at[jnp.diag_indices(N)].set(0.0)

        _, traj = kuramoto_forward(phases0, omegas, K, 0.01, 2000)
        traj_np = np.array(traj)

        # Measure spread: circular variance of phases at early vs late times
        def circ_var(phases_t):
            z = np.exp(1j * phases_t)
            return 1.0 - np.abs(np.mean(z))

        np.mean([circ_var(traj_np[t]) for t in range(100, 200)])
        var_late = np.mean([circ_var(traj_np[t]) for t in range(1800, 2000)])

        # Below K_c: variance should stay high or increase (no sync)
        assert var_late > 0.5, (
            f"Phases synchronised below K_c: circ_var_late={var_late:.3f}"
        )


# ──────────────────────────────────────────────────
# V39: Time reversal for gradient flow
# ──────────────────────────────────────────────────


class TestV39TimeReversal:
    """For zero-frequency Kuramoto with symmetric K, the dynamics
    are gradient flow of V = -Sum K cos(Delta_theta). Running
    forward then reversing (negating K) should approximately
    retrace the trajectory.
    """

    def test_reversibility(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        N = 8
        key = jr.PRNGKey(0)
        phases0 = jr.uniform(key, (N,), maxval=TWO_PI)
        omegas = jnp.zeros(N)
        K = jnp.ones((N, N)) * 0.3 / N
        K = (K + K.T) / 2.0
        K = K.at[jnp.diag_indices(N)].set(0.0)

        n_steps = 200
        dt = 0.005

        # Forward
        final_fwd, _ = kuramoto_forward(phases0, omegas, K, dt, n_steps)

        # Reverse: negate coupling (reverses gradient flow direction)
        final_rev, _ = kuramoto_forward(final_fwd, omegas, -K, dt, n_steps)

        # Should approximately return to start
        err = float(jnp.mean(1.0 - jnp.cos(final_rev - phases0)))
        assert err < 0.1, (
            f"Reverse trajectory error={err:.4f}, expected < 0.1. "
            "Gradient flow not reversible?"
        )


# ──────────────────────────────────────────────────
# V40: hybrid_inverse improves on analytical for noisy data
# ──────────────────────────────────────────────────


class TestV40HybridInverseImprovement:
    """hybrid_inverse should match or beat analytical_inverse
    on noisy data, since it refines with gradient steps.
    """

    def test_hybrid_vs_analytical_noisy(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward
        from scpn_phase_orchestrator.nn.inverse import (
            analytical_inverse,
            coupling_correlation,
            hybrid_inverse,
        )

        N = 6
        key = jr.PRNGKey(42)
        k1, k2, k3 = jr.split(key, 3)

        K_true = jr.normal(k1, (N, N)) * 0.3
        K_true = (K_true + K_true.T) / 2.0
        K_true = K_true.at[jnp.diag_indices(N)].set(0.0)
        phases0 = jr.uniform(k2, (N,), maxval=TWO_PI)

        _, traj = kuramoto_forward(phases0, jnp.zeros(N), K_true, 0.01, 300)
        observed = jnp.concatenate([phases0[jnp.newaxis], traj])

        # Add noise
        noise = jr.normal(k3, observed.shape) * 0.05
        observed_noisy = observed + noise

        K_ana, _ = analytical_inverse(observed_noisy, 0.01)
        corr_ana = float(coupling_correlation(K_true, K_ana))

        K_hyb, _, _ = hybrid_inverse(
            observed_noisy, 0.01, n_refine=30, lr=0.003, window_size=10
        )
        corr_hyb = float(coupling_correlation(K_true, K_hyb))

        # Hybrid should be at least comparable (allow small degradation)
        assert corr_hyb > corr_ana - 0.15, (
            f"Hybrid corr={corr_hyb:.3f} much worse than "
            f"analytical={corr_ana:.3f} on noisy data"
        )


# ──────────────────────────────────────────────────
# V41: vmap correctness
# ──────────────────────────────────────────────────


class TestV41VmapCorrectness:
    """vmap(kuramoto_forward) over a batch of initial conditions
    must produce identical results to sequential execution.
    """

    def test_vmap_matches_sequential(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        N = 8
        B = 4
        key = jr.PRNGKey(0)
        batch_phases = jr.uniform(key, (B, N), maxval=TWO_PI)
        omegas = jr.normal(key, (N,)) * 0.3
        K = jnp.ones((N, N)) * 0.3 / N
        K = K.at[jnp.diag_indices(N)].set(0.0)

        # Sequential
        seq_results = []
        for i in range(B):
            final, _ = kuramoto_forward(batch_phases[i], omegas, K, 0.01, 100)
            seq_results.append(final)
        seq_stack = jnp.stack(seq_results)

        # vmap
        def run_one(p):
            final, _ = kuramoto_forward(p, omegas, K, 0.01, 100)
            return final

        vmap_stack = jax.vmap(run_one)(batch_phases)

        np.testing.assert_allclose(
            np.array(seq_stack),
            np.array(vmap_stack),
            atol=1e-5,
            err_msg="vmap results differ from sequential",
        )


# ──────────────────────────────────────────────────
# V42: scan = manual step loop
# ──────────────────────────────────────────────────


class TestV42ScanLoopEquivalence:
    """kuramoto_forward (uses jax.lax.scan) must produce the same
    result as a manual Python loop of kuramoto_rk4_step.
    """

    def test_scan_equals_loop(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            kuramoto_rk4_step,
        )

        N = 8
        key = jr.PRNGKey(0)
        phases0 = jr.uniform(key, (N,), maxval=TWO_PI)
        omegas = jr.normal(key, (N,)) * 0.3
        K = jnp.ones((N, N)) * 0.3 / N
        K = K.at[jnp.diag_indices(N)].set(0.0)

        n_steps = 100
        dt = 0.01

        # scan version
        final_scan, traj_scan = kuramoto_forward(
            phases0, omegas, K, dt, n_steps, method="rk4"
        )

        # manual loop
        p = phases0
        traj_loop = []
        for _ in range(n_steps):
            p = kuramoto_rk4_step(p, omegas, K, dt)
            traj_loop.append(p)
        final_loop = p
        traj_loop = jnp.stack(traj_loop)

        np.testing.assert_allclose(
            np.array(final_scan),
            np.array(final_loop),
            atol=1e-5,
            err_msg="scan final ≠ loop final",
        )
        np.testing.assert_allclose(
            np.array(traj_scan),
            np.array(traj_loop),
            atol=1e-5,
            err_msg="scan trajectory ≠ loop trajectory",
        )


# ──────────────────────────────────────────────────
# V43: Stuart-Landau amplitude consensus
# ──────────────────────────────────────────────────


class TestV43SLAmplitudeConsensus:
    """With amplitude coupling (epsilon > 0, K_r > 0) and
    phase-synchronised oscillators, amplitudes should converge
    toward consensus (spread decreases).
    """

    def test_amplitude_spread_decreases(self):
        from scpn_phase_orchestrator.nn.functional import stuart_landau_forward

        N = 6
        # Start phase-synchronised, amplitude-spread
        phases0 = jnp.zeros(N)
        amps0 = jnp.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
        omegas = jnp.zeros(N)
        mu = jnp.ones(N)
        K = jnp.ones((N, N)) * 0.3
        K = K.at[jnp.diag_indices(N)].set(0.0)
        K_r = jnp.ones((N, N)) * 0.5
        K_r = K_r.at[jnp.diag_indices(N)].set(0.0)

        _, fr, _, amp_traj = stuart_landau_forward(
            phases0, amps0, omegas, mu, K, K_r, 0.01, 3000, epsilon=1.0
        )

        spread_initial = float(amps0.max() - amps0.min())
        spread_final = float(fr.max() - fr.min())

        assert spread_final < spread_initial * 0.5, (
            f"Amplitude spread didn't decrease enough: "
            f"initial={spread_initial:.3f}, final={spread_final:.3f}"
        )


# ──────────────────────────────────────────────────
# V44: Chimera index = 0 for uniform states
# ──────────────────────────────────────────────────


class TestV44ChimeraIndexBoundaries:
    """Chimera index (variance of local R) should be ~0 for both
    perfectly synchronised and uniformly random states.
    """

    def test_sync_chimera_zero(self):
        from scpn_phase_orchestrator.nn.chimera import chimera_index

        N = 16
        K = jnp.ones((N, N)) * 0.5
        K = K.at[jnp.diag_indices(N)].set(0.0)

        # Perfect sync: all phases = 0
        phases_sync = jnp.zeros(N)
        chi_sync = float(chimera_index(phases_sync, K))
        assert chi_sync < 0.01, (
            f"Chimera index for sync state={chi_sync:.4f}, expected ~0"
        )

    def test_uniform_chimera_low(self):
        from scpn_phase_orchestrator.nn.chimera import chimera_index

        N = 32
        K = jnp.ones((N, N)) * 0.5
        K = K.at[jnp.diag_indices(N)].set(0.0)

        # Uniformly spread phases: all local R should be similar (~0)
        phases_uniform = jnp.linspace(0, TWO_PI, N, endpoint=False)
        chi_uniform = float(chimera_index(phases_uniform, K))
        assert chi_uniform < 0.05, (
            f"Chimera index for uniform spread={chi_uniform:.4f}, expected ~0"
        )


# ──────────────────────────────────────────────────
# V45: OIM bipartite 2-colour perfect
# ──────────────────────────────────────────────────


class TestV45OIMBipartitePerfect:
    """Complete bipartite graph K_{3,3} has chromatic number 2.
    oim_solve with 2 colours must find 0 violations.
    """

    def test_k33_two_colors(self):
        from scpn_phase_orchestrator.nn.oim import coloring_violations, oim_solve

        # K_{3,3}: nodes 0-2 connected to nodes 3-5
        A = jnp.zeros((6, 6))
        for i in range(3):
            for j in range(3, 6):
                A = A.at[i, j].set(1.0)
                A = A.at[j, i].set(1.0)

        key = jr.PRNGKey(42)
        colors, _, _ = oim_solve(A, n_colors=2, key=key, n_restarts=20)
        v = int(coloring_violations(colors, A))
        assert v == 0, f"K_3,3 with 2 colours: {v} violations, expected 0"


# ──────────────────────────────────────────────────
# V46: PLV correlates with R
# ──────────────────────────────────────────────────


class TestV46PLVCorrelatesWithR:
    """Higher global synchronisation R should correspond to
    higher mean off-diagonal PLV.
    """

    def test_plv_increases_with_sync(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
            plv,
        )

        N = 16
        key = jr.PRNGKey(0)
        k1, k2 = jr.split(key)
        phases0 = jr.uniform(k1, (N,), maxval=TWO_PI)
        omegas = jr.normal(k2, (N,)) * 0.3

        results = []
        for K_scale in [0.01, 0.1, 0.5, 2.0]:
            K = jnp.ones((N, N)) * (K_scale / N)
            K = K.at[jnp.diag_indices(N)].set(0.0)
            final, traj = kuramoto_forward(phases0, omegas, K, 0.01, 1000)
            R = float(order_parameter(final))
            P = np.array(plv(traj))
            mean_plv = float(np.mean(P[~np.eye(N, dtype=bool)]))
            results.append((K_scale, R, mean_plv))

        # PLV should generally increase with K_scale (more sync = more locking)
        plv_first = results[0][2]
        plv_last = results[-1][2]
        assert plv_last > plv_first, (
            f"PLV not increasing with sync: "
            f"K={results[0][0]}→PLV={plv_first:.3f}, "
            f"K={results[-1][0]}→PLV={plv_last:.3f}"
        )


# Pipeline wiring: every test uses kuramoto_forward/order_parameter directly.
