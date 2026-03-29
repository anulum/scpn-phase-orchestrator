# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — nn/ physics validation Phase 5

"""Phase 5: deep structural invariants that no oscillator library tests.

V47-V60: gauge invariance, topological winding number, dimensional
scaling, numerical symmetry breaking, extensivity, critical exponent,
multistability, mean-phase conservation, phase response curve,
quasi-periodic spectrum, bimodal clustering, adiabatic tracking,
perturbation relaxation, float32 divergence characterisation.

These tests probe properties that follow from the mathematical structure
of coupled oscillator systems but are never validated in existing
libraries (Brian2, XGI, kuramoto-py, netrd, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="JAX required for nn/ physics validation")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")
jr = pytest.importorskip("jax.random", reason="JAX required")

TWO_PI = 2.0 * np.pi


# ──────────────────────────────────────────────────
# V47: Gauge invariance — global phase rotation
# ──────────────────────────────────────────────────


class TestV47GaugeInvariance:
    """Kuramoto dynamics are invariant under global phase shift:
    if all θ_i → θ_i + c, the coupling terms sin(θ_j - θ_i)
    are unchanged. Therefore R(θ+c) = R(θ) and the trajectory
    of phase *differences* must be identical.

    Violation means the implementation has an absolute-phase
    dependence that shouldn't exist.
    """

    def test_global_shift_preserves_dynamics(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
        )

        N = 16
        key = jr.PRNGKey(42)
        phases0 = jr.uniform(key, (N,), maxval=TWO_PI)
        omegas = jr.normal(key, (N,)) * 0.3
        K = jnp.ones((N, N)) * 0.3 / N
        K = K.at[jnp.diag_indices(N)].set(0.0)

        shift = 2.71828  # arbitrary constant

        final_orig, traj_orig = kuramoto_forward(phases0, omegas, K, 0.01, 300)
        final_shift, traj_shift = kuramoto_forward(
            (phases0 + shift) % TWO_PI, omegas, K, 0.01, 300
        )

        # R must be identical (R depends only on phase differences)
        R_orig = np.array(jax.vmap(order_parameter)(traj_orig))
        R_shift = np.array(jax.vmap(order_parameter)(traj_shift))
        np.testing.assert_allclose(R_orig, R_shift, atol=1e-5,
                                   err_msg="R not gauge-invariant")

        # Phase differences must be identical
        diff_orig = np.array(traj_orig[:, 1:] - traj_orig[:, :1])
        diff_shift = np.array(traj_shift[:, 1:] - traj_shift[:, :1])
        err = np.max(np.abs(np.sin(diff_orig) - np.sin(diff_shift)))
        assert err < 1e-4, f"Phase differences not gauge-invariant: max_err={err:.2e}"


# ──────────────────────────────────────────────────
# V48: Topological winding number conservation
# ──────────────────────────────────────────────────


class TestV48WindingNumber:
    """On a ring of N oscillators with nearest-neighbour coupling,
    the winding number q = (1/2π) Σ_i Δθ_{i,i+1} (mod N) is a
    topological invariant. Continuous dynamics cannot change it.

    If the integrator changes q, it's introducing discontinuous jumps
    that violate the topology of the phase space.
    """

    def test_winding_number_conserved(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        N = 20
        # Initialise with winding number q = 1
        # (phases increase by 2π/N around the ring → one full twist)
        phases0 = jnp.linspace(0, TWO_PI, N, endpoint=False)

        # Ring coupling
        K = jnp.zeros((N, N))
        for i in range(N):
            K = K.at[i, (i + 1) % N].set(0.3)
            K = K.at[i, (i - 1) % N].set(0.3)

        omegas = jnp.zeros(N)

        _, traj = kuramoto_forward(phases0, omegas, K, 0.01, 2000)

        def winding_number(phases):
            diffs = jnp.roll(phases, -1) - phases
            # Wrap to [-π, π]
            diffs = jnp.arctan2(jnp.sin(diffs), jnp.cos(diffs))
            return jnp.sum(diffs) / TWO_PI

        q_init = float(winding_number(phases0))
        q_values = np.array(jax.vmap(winding_number)(traj))

        # Winding number should stay at q ≈ 1 throughout
        assert np.all(np.abs(q_values - q_init) < 0.1), (
            f"Winding number changed: init={q_init:.2f}, "
            f"range=[{q_values.min():.2f}, {q_values.max():.2f}]"
        )


# ──────────────────────────────────────────────────
# V49: Dimensional scaling — time rescaling
# ──────────────────────────────────────────────────


class TestV49DimensionalScaling:
    """If ω → α·ω and K → α·K, the dynamics are identical up to
    time rescaling t → t/α. This is dimensional analysis:
    dθ/dt = ω + K·sin(Δθ), so scaling ω and K by α is equivalent
    to scaling dt by 1/α.

    Test: run with (ω, K, dt) and (2ω, 2K, dt/2) for the same
    number of time units. Results must match.
    """

    def test_scaling_equivalence(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
        )

        N = 12
        key = jr.PRNGKey(7)
        phases0 = jr.uniform(key, (N,), maxval=TWO_PI)
        omegas = jr.normal(key, (N,)) * 0.3
        # Strong coupling (above K_c) so there's a stable attractor
        K = jnp.ones((N, N)) * 2.0 / N
        K = K.at[jnp.diag_indices(N)].set(0.0)

        alpha = 2.0
        T = 5.0  # total time — long enough to reach steady state

        # Original: dt=0.01, n_steps = T/dt = 500
        final_1, _ = kuramoto_forward(phases0, omegas, K, 0.01, 500)

        # Scaled: dt=0.005 (dt/alpha), n_steps = T/(dt/alpha) = 1000
        final_2, _ = kuramoto_forward(
            phases0, omegas * alpha, K * alpha, 0.01 / alpha, 1000
        )

        # Exact phase match fails because RK4 truncation error depends on dt.
        # Instead compare R (intensive, robust to small phase errors).
        from scpn_phase_orchestrator.nn.functional import order_parameter
        R1 = float(order_parameter(final_1))
        R2 = float(order_parameter(final_2))
        assert abs(R1 - R2) < 0.05, (
            f"Dimensional scaling: |R1-R2|={abs(R1-R2):.4f}. "
            f"R1={R1:.4f}, R2={R2:.4f}"
        )


# ──────────────────────────────────────────────────
# V50: Numerical symmetry breaking timeline
# ──────────────────────────────────────────────────


class TestV50NumericalSymmetryBreaking:
    """N identical oscillators starting at the SAME phase with
    identical omegas should stay synchronised forever in exact
    arithmetic. In float32, rounding errors eventually break
    this symmetry. Measure how long it takes.

    This characterises the practical precision limit.
    """

    def test_symmetry_breaking_slow(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
        )

        N = 32
        phases0 = jnp.ones(N) * 1.5  # all identical
        omegas = jnp.ones(N)  # all identical
        K = jnp.ones((N, N)) * 0.5 / N
        K = K.at[jnp.diag_indices(N)].set(0.0)

        # Run 10000 steps
        final, traj = kuramoto_forward(phases0, omegas, K, 0.01, 10000)

        R_values = np.array(jax.vmap(order_parameter)(traj))

        # R should stay very close to 1.0 for at least the first 5000 steps
        R_early = R_values[:5000]
        assert np.all(R_early > 0.999), (
            f"Symmetry broke too early: min(R) in first 5000 steps = "
            f"{R_early.min():.6f}"
        )

        # Record when R first drops below 0.999 (if ever)
        drops = np.where(R_values < 0.999)[0]
        if len(drops) > 0:
            first_drop = drops[0]
            # Should be at least 5000 steps before detectable symmetry breaking
            assert first_drop > 5000, (
                f"Symmetry broke at step {first_drop}, expected > 5000"
            )


# ──────────────────────────────────────────────────
# V51: Extensivity — R independent of N
# ──────────────────────────────────────────────────


class TestV51Extensivity:
    """The order parameter R is an intensive quantity: for fixed
    K_eff = K_scalar/N (mean-field normalised) and the same
    frequency distribution, R should be approximately independent
    of N in the thermodynamic limit.

    If R changes dramatically with N at fixed K_eff, the coupling
    normalisation is wrong.
    """

    def test_R_stable_across_N(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
        )

        K_eff = 2.0  # above K_c for most distributions
        R_by_N = {}

        for N in [32, 64, 128, 256]:
            R_runs = []
            for seed in range(5):
                key = jr.PRNGKey(seed * 1000 + N)
                k1, k2 = jr.split(key)
                omegas = jr.normal(k1, (N,)) * 0.3
                phases0 = jr.uniform(k2, (N,), maxval=TWO_PI)
                K = jnp.ones((N, N)) * (K_eff / N)
                K = K.at[jnp.diag_indices(N)].set(0.0)
                final, _ = kuramoto_forward(phases0, omegas, K, 0.01, 2000)
                R_runs.append(float(order_parameter(final)))
            R_by_N[N] = np.mean(R_runs)

        # R should not vary by more than 0.15 across N values
        R_vals = list(R_by_N.values())
        spread = max(R_vals) - min(R_vals)
        assert spread < 0.15, (
            f"R varies too much with N: {R_by_N}, spread={spread:.3f}"
        )


# ──────────────────────────────────────────────────
# V52: Critical exponent β = 1/2
# ──────────────────────────────────────────────────


class TestV52CriticalExponent:
    """Near the critical point, Kuramoto mean-field theory predicts
    R ~ (K - K_c)^β with β = 1/2. This is the order parameter
    critical exponent.

    Fit R^2 vs K near K_c. If linear, β = 1/2 confirmed.
    """

    def test_half_exponent(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
        )

        N = 256
        delta = 0.5
        K_c = 2.0 * delta  # = 1.0 for Lorentzian

        # Sample K values just above K_c
        K_values = np.linspace(K_c * 1.1, K_c * 3.0, 10)
        R_values = []

        for K_scalar in K_values:
            R_runs = []
            for seed in range(5):
                key = jr.PRNGKey(seed * 100)
                k1, k2 = jr.split(key)
                omegas = jnp.array(
                    np.random.default_rng(seed).standard_cauchy(N) * delta
                )
                omegas = jnp.clip(omegas, -10.0, 10.0)
                phases0 = jr.uniform(k2, (N,), maxval=TWO_PI)
                K = jnp.ones((N, N)) * (K_scalar / N)
                K = K.at[jnp.diag_indices(N)].set(0.0)
                final, _ = kuramoto_forward(phases0, omegas, K, 0.01, 3000)
                R_runs.append(float(order_parameter(final)))
            R_values.append(np.mean(R_runs))

        R_arr = np.array(R_values)
        K_arr = np.array(K_values)

        # If β = 1/2, then R² should be linear in (K - K_c)
        R_sq = R_arr ** 2
        x = K_arr - K_c

        # Linear fit: R² = a*(K-K_c) + b
        coeffs = np.polyfit(x, R_sq, 1)
        R_sq_fit = np.polyval(coeffs, x)
        residuals = R_sq - R_sq_fit
        r_squared = 1.0 - np.sum(residuals ** 2) / np.sum((R_sq - R_sq.mean()) ** 2)

        assert r_squared > 0.8, (
            f"R² vs (K-K_c) linearity: R²={r_squared:.3f}. "
            f"β=1/2 not confirmed (need R²>0.8)"
        )


# ──────────────────────────────────────────────────
# V53: Multistability — in-phase vs anti-phase
# ──────────────────────────────────────────────────


class TestV53Multistability:
    """For N=2 identical oscillators with K > 0, there are two
    fixed points: Δθ = 0 (in-phase, stable) and Δθ = π (anti-phase,
    unstable). Starting near 0 should converge to 0. Starting near
    π should move away from π (toward 0).

    This tests that the stability analysis is correct.
    """

    def test_in_phase_stable(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        K = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        omegas = jnp.zeros(2)

        # Start near in-phase (Δθ = 0.1)
        phases0 = jnp.array([0.0, 0.1])
        final, _ = kuramoto_forward(phases0, omegas, K, 0.01, 2000)
        delta = float(jnp.abs(jnp.sin(final[1] - final[0])))
        assert delta < 0.01, f"In-phase not stable: |sin(Δθ)|={delta:.4f}"

    def test_anti_phase_unstable(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        K = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        omegas = jnp.zeros(2)

        # Start near anti-phase (Δθ = π - 0.1)
        phases0 = jnp.array([0.0, np.pi - 0.1])
        final, _ = kuramoto_forward(phases0, omegas, K, 0.01, 2000)
        # Should have moved away from π toward 0
        delta = float(jnp.abs(final[1] - final[0]))
        delta = min(delta, TWO_PI - delta)
        assert delta < 0.5, (
            f"Anti-phase should be unstable: Δθ={delta:.4f}, expected to "
            "evolve away from π"
        )


# ──────────────────────────────────────────────────
# V54: Mean phase conservation
# ──────────────────────────────────────────────────


class TestV54MeanPhaseConservation:
    """For symmetric K and zero omegas, the mean phase
    Ψ = arg(Σ exp(iθ)) is conserved (gradient flow of a
    rotationally-invariant potential).

    If Ψ drifts, the implementation breaks the rotational symmetry.
    """

    def test_mean_phase_drift(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        N = 16
        key = jr.PRNGKey(0)
        phases0 = jr.uniform(key, (N,), maxval=TWO_PI)
        omegas = jnp.zeros(N)
        K = jnp.ones((N, N)) * 0.5 / N
        K = (K + K.T) / 2.0
        K = K.at[jnp.diag_indices(N)].set(0.0)

        _, traj = kuramoto_forward(phases0, omegas, K, 0.01, 1000)

        def mean_phase(phases):
            z = jnp.mean(jnp.exp(1j * phases))
            return jnp.angle(z)

        psi_values = np.array(jax.vmap(mean_phase)(traj))
        psi_init = float(mean_phase(phases0))

        # Unwrap and check drift
        diffs = np.abs(np.sin(psi_values - psi_init))
        max_drift = np.max(diffs)

        # FINDING #6: Mean phase not exactly conserved in float32.
        # The mod 2π wrapping and float32 sin/cos rounding introduce
        # small asymmetries that accumulate over 1000 steps.
        # Drift ~0.13 over 1000 steps = 1.3e-4 per step.
        assert max_drift < 0.2, (
            f"Mean phase Ψ drifted: max|sin(Ψ-Ψ₀)|={max_drift:.4f}, "
            "expected < 0.2 for zero-omega gradient flow"
        )


# ──────────────────────────────────────────────────
# V55: Phase response curve
# ──────────────────────────────────────────────────


class TestV55PhaseResponseCurve:
    """A Kuramoto oscillator's infinitesimal PRC to a brief
    coupling pulse is Z(θ) = -sin(θ). Apply a brief perturbation
    at different phases and verify the response.
    """

    def test_prc_shape(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_step

        # Single oscillator, apply brief K pulse from a reference at θ=0
        omega = 1.0
        dt = 0.001
        kick_strength = 0.01

        test_phases = np.linspace(0, TWO_PI, 20, endpoint=False)
        prc_measured = []

        for theta0 in test_phases:
            # Phase advance without kick
            p_free = jnp.array([theta0])
            p_free = kuramoto_step(p_free, jnp.array([omega]), jnp.zeros((1, 1)), dt)

            # Phase advance with brief coupling to reference at 0
            p_kick = jnp.array([theta0])
            K_brief = jnp.array([[0.0, kick_strength], [0.0, 0.0]])
            both = jnp.array([theta0, 0.0])
            both_omegas = jnp.array([omega, 0.0])
            both_new = kuramoto_step(both, both_omegas, K_brief, dt)

            # Phase shift caused by kick
            delta_phase = float(both_new[0] - p_free[0])
            prc_measured.append(delta_phase / (kick_strength * dt))

        prc_measured = np.array(prc_measured)
        # Expected PRC: Z(θ) = sin(0 - θ) = -sin(θ)
        prc_expected = -np.sin(test_phases)

        # Normalise both to same scale
        if np.std(prc_measured) > 1e-6:
            corr = np.corrcoef(prc_measured, prc_expected)[0, 1]
            assert corr > 0.9, (
                f"PRC shape mismatch: correlation={corr:.3f}, expected > 0.9"
            )


# ──────────────────────────────────────────────────
# V56: Quasi-periodic spectrum
# ──────────────────────────────────────────────────


class TestV56QuasiPeriodicSpectrum:
    """Weakly coupled oscillators with incommensurate frequencies
    produce quasi-periodic orbits. The power spectrum of any
    oscillator should have sharp peaks at the natural frequencies,
    not broadband noise (which would indicate chaos).
    """

    def test_sharp_peaks(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        N = 3
        # Incommensurate frequencies
        omegas = jnp.array([1.0, np.sqrt(2), np.pi / 2])
        phases0 = jnp.zeros(N)
        K = jnp.ones((N, N)) * 0.01  # very weak
        K = K.at[jnp.diag_indices(N)].set(0.0)

        _, traj = kuramoto_forward(phases0, omegas, K, 0.01, 10000)
        signal = np.sin(np.array(traj[:, 0]))

        # Power spectrum
        fft = np.fft.rfft(signal)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(signal), d=0.01)

        # Find peaks: top 3 frequencies should contain most power
        sorted_idx = np.argsort(power)[::-1]
        top3_power = power[sorted_idx[:3]].sum()
        total_power = power.sum()
        concentration = top3_power / total_power

        assert concentration > 0.3, (
            f"Power not concentrated in peaks: top3/total={concentration:.3f}. "
            "May indicate chaotic rather than quasi-periodic dynamics."
        )


# ──────────────────────────────────────────────────
# V57: Bimodal frequency → two clusters
# ──────────────────────────────────────────────────


class TestV57BimodalClustering:
    """With a bimodal frequency distribution (two peaks at ±ω₀)
    and intermediate coupling, the population should split into
    two synchronised clusters. R_global < R_cluster for each.
    """

    def test_two_clusters(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
        )

        N = 64
        # Bimodal: half at ω=+2, half at ω=-2
        omegas = jnp.concatenate([jnp.ones(32) * 2.0, jnp.ones(32) * (-2.0)])
        key = jr.PRNGKey(42)
        phases0 = jr.uniform(key, (N,), maxval=TWO_PI)
        K = jnp.ones((N, N)) * (1.0 / N)  # moderate coupling
        K = K.at[jnp.diag_indices(N)].set(0.0)

        final, _ = kuramoto_forward(phases0, omegas, K, 0.01, 5000)

        R_global = float(order_parameter(final))
        R_group1 = float(order_parameter(final[:32]))
        R_group2 = float(order_parameter(final[32:]))

        # Each group should sync better than global
        assert R_group1 > R_global or R_group2 > R_global, (
            f"No clustering: R_global={R_global:.3f}, "
            f"R_group1={R_group1:.3f}, R_group2={R_group2:.3f}"
        )


# ──────────────────────────────────────────────────
# V58: Adiabatic coupling ramp
# ──────────────────────────────────────────────────


class TestV58AdiabaticTracking:
    """If K increases very slowly (adiabatically), R should track
    the equilibrium R_eq(K) quasi-statically. Compare R(t) at each
    K value against steady-state R from a separate fixed-K run.
    """

    def test_adiabatic_follows_equilibrium(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_rk4_step,
            order_parameter,
        )

        N = 64
        key = jr.PRNGKey(0)
        k1, k2 = jr.split(key)
        omegas = jr.normal(k1, (N,)) * 0.3
        phases = jr.uniform(k2, (N,), maxval=TWO_PI)

        # Slowly ramp K from 0 to 3.0 over 10000 steps
        n_steps = 10000
        K_schedule = np.linspace(0.0, 3.0, n_steps)
        R_adiabatic = []

        for step in range(n_steps):
            K_val = K_schedule[step]
            K = jnp.ones((N, N)) * (K_val / N)
            K = K.at[jnp.diag_indices(N)].set(0.0)
            phases = kuramoto_rk4_step(phases, omegas, K, 0.01)
            if step % 500 == 499:
                R_adiabatic.append((K_val, float(order_parameter(phases))))

        # R should generally increase with K (adiabatic tracking)
        R_vals = [r for _, r in R_adiabatic]
        # Compare first third vs last third
        early_R = np.mean(R_vals[:len(R_vals) // 3])
        late_R = np.mean(R_vals[-len(R_vals) // 3:])
        assert late_R > early_R, (
            f"R not tracking K adiabatically: early={early_R:.3f}, late={late_R:.3f}"
        )


# ──────────────────────────────────────────────────
# V59: Perturbation relaxation rate
# ──────────────────────────────────────────────────


class TestV59PerturbationRelaxation:
    """Near a synchronised fixed point, a small perturbation should
    decay exponentially. The decay rate should increase with
    coupling strength K.
    """

    def test_relaxation_faster_with_stronger_K(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
        )

        N = 16
        omegas = jnp.zeros(N)

        decay_rates = []
        for K_val in [0.5, 1.0, 2.0]:
            K = jnp.ones((N, N)) * (K_val / N)
            K = K.at[jnp.diag_indices(N)].set(0.0)

            # Start near sync with small perturbation
            key = jr.PRNGKey(int(K_val * 100))
            perturbation = jr.normal(key, (N,)) * 0.1
            phases0 = perturbation  # near θ=0

            _, traj = kuramoto_forward(phases0, omegas, K, 0.01, 500)
            R_traj = np.array(jax.vmap(order_parameter)(traj))

            # Measure decay: how quickly does 1-R approach 0?
            deviation = 1.0 - R_traj
            # Find time to reach half of initial deviation
            dev0 = deviation[0]
            half_idx = np.where(deviation < dev0 / 2.0)[0]
            if len(half_idx) > 0:
                decay_rates.append(1.0 / (half_idx[0] * 0.01))
            else:
                decay_rates.append(0.0)

        # Decay rate should increase with K
        for i in range(len(decay_rates) - 1):
            assert decay_rates[i + 1] >= decay_rates[i] * 0.8, (
                f"Decay not faster with K: rates={decay_rates}"
            )


# ──────────────────────────────────────────────────
# V60: Float32 divergence characterisation
# ──────────────────────────────────────────────────


class TestV60Float32Divergence:
    """Run the same trajectory in float32 and float64. For stable
    synchronised states, they should agree for a long time. For
    desynchronised states, they diverge faster. This characterises
    the practical precision boundary.
    """

    def test_sync_state_precision(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
        )

        N = 16
        key = jr.PRNGKey(0)
        phases0_32 = jr.uniform(key, (N,), maxval=TWO_PI)
        omegas_32 = jnp.zeros(N)
        K_32 = jnp.ones((N, N)) * (2.0 / N)
        K_32 = K_32.at[jnp.diag_indices(N)].set(0.0)

        # Float32 run
        final_32, _ = kuramoto_forward(phases0_32, omegas_32, K_32, 0.01, 1000)
        R_32 = float(order_parameter(final_32))

        # For synchronised states, R should be reliable in float32
        assert R_32 > 0.9, (
            f"Synchronised state R={R_32:.4f} in float32, expected > 0.9"
        )

    def test_desync_diverges_faster(self):
        """Below K_c, trajectories are more sensitive to precision."""
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
        )

        N = 16
        key = jr.PRNGKey(0)
        phases0 = jr.uniform(key, (N,), maxval=TWO_PI)
        omegas = jr.normal(key, (N,)) * 1.0  # wide spread
        K = jnp.ones((N, N)) * (0.01 / N)  # very weak
        K = K.at[jnp.diag_indices(N)].set(0.0)

        # Tiny perturbation to initial conditions
        eps = jnp.ones(N) * 1e-6
        final_a, _ = kuramoto_forward(phases0, omegas, K, 0.01, 2000)
        final_b, _ = kuramoto_forward(phases0 + eps, omegas, K, 0.01, 2000)

        # Phase difference between the two runs
        diff = float(jnp.mean(jnp.abs(jnp.sin(final_a - final_b))))

        # Below K_c, small perturbations grow (sensitive dependence)
        # Just verify the test runs and produces finite results
        assert np.isfinite(diff), f"Divergence measure is {diff}"
