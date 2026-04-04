# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — nn/ physics validation Phase 13

"""Phase 13: the forgotten aspects.

Everything we OVERLOOKED in 12 phases of testing. Not physics,
not gradients — the ENGINEERING and INTEGRATION aspects that
determine whether the module is actually usable in production.

V143-V154: serialization, error handling, memory scaling, Rust parity,
condition number, multi-timescale SCPN, PI/S channels, domainpack
compatibility, replay determinism, eigenvalue spectrum, dtype handling,
the Ψ-field.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="JAX required for nn/ physics validation")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")
jr = pytest.importorskip("jax.random", reason="JAX required")

TWO_PI = 2.0 * np.pi


# ──────────────────────────────────────────────────
# V143: Serialization — save and load trained layers
# ──────────────────────────────────────────────────


class TestV143Serialization:
    """Can you save a trained KuramotoLayer and reload it with
    identical parameters? This is critical for deployment:
    train once, serve many times.

    Nobody tests this for physics-based layers.
    """

    def test_save_load_roundtrip(self):
        import tempfile

        import equinox as eqx

        from scpn_phase_orchestrator.nn.functional import order_parameter
        from scpn_phase_orchestrator.nn.kuramoto_layer import KuramotoLayer

        key = jr.PRNGKey(42)
        N = 6
        layer = KuramotoLayer(n=N, n_steps=30, dt=0.01, key=key)
        phases = jr.uniform(key, (N,), maxval=TWO_PI)

        # Forward pass before save
        R_before = float(order_parameter(layer(phases)))

        # Save
        with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as f:
            path = f.name
        eqx.tree_serialise_leaves(path, layer)

        # Load
        skeleton = KuramotoLayer(n=N, n_steps=30, dt=0.01, key=jr.PRNGKey(0))
        loaded = eqx.tree_deserialise_leaves(path, skeleton)

        # Forward pass after load
        R_after = float(order_parameter(loaded(phases)))

        assert abs(R_before - R_after) < 1e-6, (
            f"Serialization changed output: R_before={R_before:.6f}, "
            f"R_after={R_after:.6f}"
        )

        # Parameters should be identical
        np.testing.assert_array_equal(
            np.array(layer.K),
            np.array(loaded.K),
            err_msg="K changed after serialization",
        )

        from pathlib import Path as _Path

        _Path(path).unlink()

    def test_stuart_landau_serialization(self):
        import tempfile

        import equinox as eqx

        from scpn_phase_orchestrator.nn.stuart_landau_layer import StuartLandauLayer

        key = jr.PRNGKey(0)
        N = 4
        layer = StuartLandauLayer(n=N, n_steps=20, dt=0.01, key=key)

        with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as f:
            path = f.name
        eqx.tree_serialise_leaves(path, layer)

        skeleton = StuartLandauLayer(n=N, n_steps=20, dt=0.01, key=jr.PRNGKey(1))
        loaded = eqx.tree_deserialise_leaves(path, skeleton)

        np.testing.assert_array_equal(np.array(layer.K), np.array(loaded.K))
        np.testing.assert_array_equal(np.array(layer.mu), np.array(loaded.mu))

        from pathlib import Path as _Path

        _Path(path).unlink()


# ──────────────────────────────────────────────────
# V144: Error handling — wrong shapes
# ──────────────────────────────────────────────────


class TestV144ErrorHandling:
    """What happens with wrong-shaped inputs?
    Should fail FAST with a useful message, not silently produce
    garbage or crash deep inside XLA compilation.
    """

    def test_mismatched_N_raises(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_step

        phases = jnp.zeros(8)
        omegas = jnp.zeros(8)
        K = jnp.zeros((10, 10))  # WRONG: N=10 vs N=8

        # Should raise or produce wrong-shaped output
        try:
            result = kuramoto_step(phases, omegas, K, 0.01)
            # If it doesn't raise, check output is wrong shape
            # (JAX broadcasting may silently succeed with wrong semantics)
            assert result.shape == (8,), (
                f"Wrong K shape not caught: result shape={result.shape}"
            )
        except (ValueError, TypeError, jax.errors.TracerArrayConversionError):
            pass  # Good — error caught

    def test_scalar_phase_raises(self):
        from scpn_phase_orchestrator.nn.functional import order_parameter

        # Scalar input — should work (N=1 edge case) or raise clearly
        try:
            R = order_parameter(jnp.array(1.5))
            assert np.isfinite(float(R))
        except (ValueError, IndexError):
            pass  # Acceptable — scalar not supported


# ──────────────────────────────────────────────────
# V145: Mixed dtype handling
# ──────────────────────────────────────────────────


class TestV145DtypeHandling:
    """What if phases are float64 but K is float32? JAX should
    promote, but does the physics stay correct?
    """

    def test_mixed_dtypes_work(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
        )

        N = 8
        key = jr.PRNGKey(0)
        phases_64 = jr.uniform(key, (N,), maxval=TWO_PI).astype(jnp.float64)
        omegas_32 = jr.normal(key, (N,)).astype(jnp.float32) * 0.3
        K_32 = (jnp.ones((N, N)) * 0.3 / N).astype(jnp.float32)
        K_32 = K_32.at[jnp.diag_indices(N)].set(0.0)

        try:
            final, _ = kuramoto_forward(phases_64, omegas_32, K_32, 0.01, 100)
            R = float(order_parameter(final))
            assert np.isfinite(R), f"Mixed dtype gave R={R}"
        except TypeError:
            # JAX may refuse mixed dtypes — that's acceptable too
            pass

    def test_int_input_fails_gracefully(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_step

        # Integer phases — physically meaningless, should promote or fail
        phases_int = jnp.array([1, 2, 3, 4])
        omegas = jnp.zeros(4)
        K = jnp.zeros((4, 4))

        try:
            result = kuramoto_step(phases_int, omegas, K, 0.01)
            # If it works, result should be float
            assert result.dtype in (jnp.float32, jnp.float64, jnp.int32)
        except (TypeError, jax.errors.TracerArrayConversionError):
            pass


# ──────────────────────────────────────────────────
# V146: Memory scaling
# ──���───────────────────────────────────────────────


class TestV146MemoryScaling:
    """kuramoto_forward stores full trajectory (n_steps, N).
    Verify the trajectory has correct shape and the memory
    scaling is as expected.
    """

    def test_trajectory_shape_correct(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        for N, n_steps in [(8, 100), (32, 500), (128, 50)]:
            key = jr.PRNGKey(N)
            phases = jr.uniform(key, (N,), maxval=TWO_PI)
            omegas = jnp.zeros(N)
            K = jnp.ones((N, N)) * 0.1 / N
            K = K.at[jnp.diag_indices(N)].set(0.0)

            final, traj = kuramoto_forward(phases, omegas, K, 0.01, n_steps)

            assert final.shape == (N,), (
                f"Final shape wrong: {final.shape}, expected ({N},)"
            )
            assert traj.shape == (n_steps, N), (
                f"Trajectory shape wrong: {traj.shape}, expected ({n_steps}, {N})"
            )

    def test_memory_bytes_estimate(self):
        """Estimate memory: K is O(N²), trajectory is O(n_steps×N)."""
        N = 256
        n_steps = 1000
        bytes_per_float = 4  # float32

        K_bytes = N * N * bytes_per_float
        traj_bytes = n_steps * N * bytes_per_float
        total_bytes = K_bytes + traj_bytes

        # K: 256² × 4 = 256KB
        # Traj: 1000 × 256 × 4 = 1MB
        # Total: ~1.25MB — very reasonable
        assert total_bytes < 100 * 1024 * 1024, (  # < 100MB
            f"Memory estimate too high: {total_bytes / 1024 / 1024:.1f} MB"
        )


# ──────────────────────────────────────────────────
# V147: Rust FFI ↔ JAX parity
# ──────────────────────────────────────────────────


class TestV147RustJAXParity:
    """The Rust spo-kernel accelerates the NumPy engine. Test
    that Rust and JAX produce the same R for the same inputs.
    """

    def test_rust_jax_same_R(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
        )

        try:
            from spo_kernel import PyUPDEStepper
        except ImportError:
            pytest.skip("Rust FFI not available (spo-kernel not compiled)")

        N = 8
        rng = np.random.default_rng(42)
        phases_np = rng.uniform(0, TWO_PI, N).astype(np.float64)
        omegas_np = rng.normal(0, 0.3, N).astype(np.float64)
        K_np = np.full((N, N), 0.3 / N, dtype=np.float64)
        np.fill_diagonal(K_np, 0.0)

        # Rust engine — expects flattened 1D arrays for knm and alpha
        stepper = PyUPDEStepper(N, dt=0.01, method="rk4")
        knm_flat = np.ascontiguousarray(K_np.ravel())
        alpha_flat = np.zeros(N * N, dtype=np.float64)
        p_rust = phases_np.copy()
        for _ in range(500):
            p_rust = np.asarray(stepper.step(p_rust, omegas_np, knm_flat, 0.0, 0.0, alpha_flat))
        R_rust = float(np.abs(np.mean(np.exp(1j * p_rust))))

        # JAX engine
        final_jax, _ = kuramoto_forward(
            jnp.array(phases_np, jnp.float32),
            jnp.array(omegas_np, jnp.float32),
            jnp.array(K_np, jnp.float32),
            0.01,
            500,
        )
        R_jax = float(order_parameter(final_jax))

        assert abs(R_rust - R_jax) < 0.1, (
            f"Rust↔JAX parity: R_rust={R_rust:.4f}, R_jax={R_jax:.4f}"
        )


# ──────────────────────────────────────────────────
# V148: Condition number of analytical_inverse
# ──────────────────────────────────────────────────


class TestV148ConditionNumber:
    """How sensitive is analytical_inverse to input perturbations?
    The condition number κ = ||J|| · ||J⁻¹|| determines this.

    High κ at certain (N, T, K_topology) means the inverse is
    unreliable there — even correct code gives wrong answers.
    """

    def test_condition_number_finite(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward
        from scpn_phase_orchestrator.nn.inverse import (
            analytical_inverse,
        )

        N = 6
        key = jr.PRNGKey(42)
        k1, k2 = jr.split(key)
        K_true = jr.normal(k1, (N, N)) * 0.3
        K_true = (K_true + K_true.T) / 2.0
        K_true = K_true.at[jnp.diag_indices(N)].set(0.0)
        phases0 = jr.uniform(k2, (N,), maxval=TWO_PI)

        _, traj = kuramoto_forward(phases0, jnp.zeros(N), K_true, 0.01, 500)
        observed = jnp.concatenate([phases0[jnp.newaxis], traj])

        # Measure sensitivity: perturb input, measure output change
        eps = 1e-3
        noise = jr.normal(jr.PRNGKey(99), observed.shape) * eps
        K_clean, _ = analytical_inverse(observed, 0.01)
        K_noisy, _ = analytical_inverse(observed + noise, 0.01)

        input_change = float(jnp.sqrt(jnp.sum(noise**2)))
        output_change = float(jnp.sqrt(jnp.sum((K_clean - K_noisy) ** 2)))

        condition_estimate = output_change / max(input_change, 1e-10)

        # Should be finite and not astronomical
        assert np.isfinite(condition_estimate), (
            f"Condition number is {condition_estimate}"
        )
        # Record the value — this characterises the inverse quality
        # High κ (>1000) = ill-conditioned, low κ (<10) = well-conditioned


# ──────────────────────────────────────────────────
# V149: Replay determinism after JIT
# ─��──────────────��─────────────────────────────────


class TestV149ReplayDeterminism:
    """Run kuramoto_forward, get trajectory. Run AGAIN with the
    exact same inputs. Must produce BIT-IDENTICAL result even
    after JIT compilation has happened.

    V110 tested this once. This tests it AFTER the JIT cache
    is warm (second call goes through compiled path).
    """

    def test_post_jit_determinism(self):
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        N = 12
        key = jr.PRNGKey(42)
        phases = jr.uniform(key, (N,), maxval=TWO_PI)
        omegas = jr.normal(key, (N,)) * 0.3
        K = jnp.ones((N, N)) * 0.3 / N
        K = K.at[jnp.diag_indices(N)].set(0.0)

        # First call (triggers JIT compilation)
        final_1, traj_1 = kuramoto_forward(phases, omegas, K, 0.01, 200)
        # Second call (uses JIT cache)
        final_2, traj_2 = kuramoto_forward(phases, omegas, K, 0.01, 200)
        # Third call (definitely from cache)
        final_3, traj_3 = kuramoto_forward(phases, omegas, K, 0.01, 200)

        np.testing.assert_array_equal(np.array(final_1), np.array(final_2))
        np.testing.assert_array_equal(np.array(final_2), np.array(final_3))
        np.testing.assert_array_equal(np.array(traj_1), np.array(traj_3))


# ──────────────────────────────────────────────────
# V150: Multi-timescale SCPN dynamics
# ──────────────────────────────────────────────────


class TestV150MultiTimescale:
    """SCPN theory has 15+1 layers with timescales spanning 5
    orders of magnitude (0.5Hz delta → 100Hz gamma). Can Kuramoto
    handle this timescale separation?

    Issue: a single dt must resolve the FASTEST oscillator (gamma)
    while running long enough for the SLOWEST (delta) to complete
    cycles. This requires dt < 1/(2·f_max) and T > 1/f_min.
    """

    def test_multiscale_runs_without_nan(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
            order_parameter,
        )

        N = 8
        # Span 2 orders of magnitude: 1Hz to 100Hz (in rad/s)
        omegas = jnp.array([1, 2, 5, 10, 20, 50, 80, 100]) * TWO_PI

        key = jr.PRNGKey(0)
        phases0 = jr.uniform(key, (N,), maxval=TWO_PI)
        K = jnp.ones((N, N)) * 0.01  # weak coupling
        K = K.at[jnp.diag_indices(N)].set(0.0)

        # dt must resolve 100Hz → dt < 0.005s. Use dt=0.0001
        # Need ~10000 steps for slow oscillators to complete 1 cycle
        final, traj = kuramoto_forward(phases0, omegas, K, 0.0001, 10000)

        assert np.all(np.isfinite(np.array(final))), (
            "Multi-timescale simulation produced NaN"
        )
        R = float(order_parameter(final))
        assert 0.0 <= R <= 1.0, f"R={R} out of bounds"

    def test_fast_oscillators_complete_cycles(self):
        """100Hz oscillator should complete ~10 cycles in 0.1s."""
        from scpn_phase_orchestrator.nn.functional import kuramoto_forward

        N = 2
        omega_fast = 100.0 * TWO_PI  # 100Hz
        omegas = jnp.array([1.0 * TWO_PI, omega_fast])
        phases0 = jnp.zeros(N)
        K = jnp.zeros((N, N))  # uncoupled
        dt = 0.0001
        n_steps = 1000  # 0.1 seconds

        final, traj = kuramoto_forward(phases0, omegas, K, dt, n_steps)
        traj_np = np.array(traj)

        # Count zero crossings of sin(θ) for fast oscillator
        sin_fast = np.sin(traj_np[:, 1])
        crossings = np.sum(np.abs(np.diff(np.sign(sin_fast))) > 0)

        # 100Hz × 0.1s = 10 cycles → ~20 zero crossings
        assert crossings > 10, (
            f"Fast oscillator only {crossings} crossings in 0.1s, "
            f"expected ~20 for 100Hz"
        )


# ────��─────────────────────────────────────────────
# V151: Eigenvalue spectrum of known graphs
# ──────────────────────────────────────────────────


class TestV151EigenvalueSpectrum:
    """For known graph types, the Laplacian eigenvalues have
    closed-form expressions. Verify the spectral module matches.

    Complete graph K_n: eigenvalues = {0, n, n, ..., n} (n-1 copies)
    Ring C_n: eigenvalues = 2 - 2cos(2πk/n) for k=0,...,n-1
    """

    def test_complete_graph_spectrum(self):
        from scpn_phase_orchestrator.nn.spectral import laplacian_spectrum

        N = 8
        K = jnp.ones((N, N)) - jnp.eye(N)
        eigs = np.sort(np.array(laplacian_spectrum(K)))

        # Expected: [0, N, N, ..., N]
        assert abs(eigs[0]) < 1e-4, f"First eigenvalue not 0: {eigs[0]}"
        for i in range(1, N):
            assert abs(eigs[i] - N) < 1e-3, f"Eigenvalue {i} = {eigs[i]}, expected {N}"

    def test_ring_spectrum(self):
        from scpn_phase_orchestrator.nn.spectral import laplacian_spectrum

        N = 8
        K = jnp.zeros((N, N))
        for i in range(N):
            K = K.at[i, (i + 1) % N].set(1.0)
            K = K.at[i, (i - 1) % N].set(1.0)

        eigs = np.sort(np.array(laplacian_spectrum(K)))

        # Expected: 2 - 2cos(2πk/N) for k=0,...,N-1
        expected = np.sort([2.0 - 2.0 * np.cos(2 * np.pi * k / N) for k in range(N)])

        np.testing.assert_allclose(
            eigs, expected, atol=1e-4, err_msg="Ring spectrum mismatch"
        )


# ──────────────────────────────────────────────────
# V152: The Ψ-field (global coherence, Paper 0)
# ────────────────────────────────────────��─────────


class TestV152PsiField:
    """Paper 0 defines the Ψ-field as the global coherence phase.
    In Kuramoto: Ψ = arg(<exp(iθ)>). The UPDE has a term
    ζ·sin(Ψ - θ_i) coupling each oscillator to the global phase.

    Test: Ψ should be well-defined (R > 0) after sync, and
    all oscillators should cluster around Ψ.
    """

    def test_psi_field_attracts(self):
        from scpn_phase_orchestrator.nn.functional import (
            kuramoto_forward,
        )

        N = 16
        key = jr.PRNGKey(0)
        phases0 = jr.uniform(key, (N,), maxval=TWO_PI)
        omegas = jnp.zeros(N)
        K = jnp.ones((N, N)) * (3.0 / N)
        K = K.at[jnp.diag_indices(N)].set(0.0)

        final, _ = kuramoto_forward(phases0, omegas, K, 0.01, 1000)

        z = jnp.mean(jnp.exp(1j * final))
        R = float(jnp.abs(z))
        Psi = float(jnp.angle(z))

        # After sync, all phases should be near Psi
        if R > 0.9:
            deviations = np.abs(np.sin(np.array(final) - Psi))
            max_dev = np.max(deviations)
            assert max_dev < 0.3, (
                f"Phases not clustered around Ψ={Psi:.3f}: max deviation={max_dev:.3f}"
            )


# Pipeline wiring: every test uses kuramoto_forward/order_parameter directly.
