# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Rust/Python parity and performance benchmarks
#
# Verifies that:
# 1. Rust FFI and Python fallback produce identical results
# 2. Performance budgets are documented for both paths
# 3. Rust path is faster than Python for N >= threshold

from __future__ import annotations

import time

import numpy as np
import pytest

from scpn_phase_orchestrator._compat import HAS_RUST

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Parity: Rust and Python must agree to machine precision
# ---------------------------------------------------------------------------


class TestRustPythonParity:
    """When Rust FFI is available, Rust and Python paths must produce
    identical numerical results for the same inputs."""

    def _python_engine(self, n, dt=0.01, method="euler"):
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(n, dt=dt, method=method)
        eng._rust = None  # force Python path
        return eng

    def _default_engine(self, n, dt=0.01, method="euler"):
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        return UPDEEngine(n, dt=dt, method=method)

    def test_euler_step_parity(self):
        """Euler step: Python and default (Rust if available) must agree."""
        n = 8
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(0.5, 2.0, n)
        knm = rng.uniform(0.1, 0.5, (n, n))
        knm = (knm + knm.T) / 2
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        py_result = self._python_engine(n).step(phases, omegas, knm, 0.0, 0.0, alpha)
        default_result = self._default_engine(n).step(
            phases, omegas, knm, 0.0, 0.0, alpha
        )
        np.testing.assert_allclose(
            py_result,
            default_result,
            atol=1e-10,
            err_msg="Python/Rust Euler step disagreement",
        )

    def test_order_parameter_parity(self):
        """compute_order_parameter: Python and Rust must agree."""
        import scpn_phase_orchestrator.upde.order_params as op_mod

        phases = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

        # Python path
        saved = op_mod._HAS_RUST
        op_mod._HAS_RUST = False
        r_py, psi_py = op_mod.compute_order_parameter(phases)
        op_mod._HAS_RUST = saved

        # Default path (Rust if available)
        r_def, psi_def = op_mod.compute_order_parameter(phases)

        np.testing.assert_allclose(r_py, r_def, atol=1e-10)
        np.testing.assert_allclose(psi_py, psi_def, atol=1e-10)

    def test_coupling_build_parity(self):
        """CouplingBuilder: Python and Rust paths produce same K_nm."""
        import scpn_phase_orchestrator.coupling.knm as knm_mod

        saved = knm_mod._HAS_RUST
        knm_mod._HAS_RUST = False
        cs_py = knm_mod.CouplingBuilder().build(8, 0.5, 0.3)
        knm_mod._HAS_RUST = saved

        cs_def = knm_mod.CouplingBuilder().build(8, 0.5, 0.3)

        np.testing.assert_allclose(
            cs_py.knm,
            cs_def.knm,
            atol=1e-10,
            err_msg="Python/Rust CouplingBuilder disagreement",
        )


# ---------------------------------------------------------------------------
# Performance: document wall-clock budgets for both paths
# ---------------------------------------------------------------------------


class TestPerformanceBudgets:
    """Document and enforce performance budgets. These are regression
    guards — they catch accidental slowdowns, not absolute targets."""

    def _time_fn(self, fn, n_warmup=3, n_measure=20):
        for _ in range(n_warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(n_measure):
            fn()
        return (time.perf_counter() - t0) / n_measure

    def test_python_engine_step_n64(self):
        """Python Euler step(N=64): budget < 2ms."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(64, dt=0.01)
        eng._rust = None
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, 64)
        omegas = np.ones(64)
        knm = 0.3 * np.ones((64, 64))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((64, 64))

        elapsed = self._time_fn(
            lambda: eng.step(phases, omegas, knm, 0.0, 0.0, alpha),
        )
        print(f"  Python step(64): {elapsed * 1000:.2f}ms")
        assert elapsed < 0.002, f"Python step(64) = {elapsed * 1000:.2f}ms > 2ms"

    @pytest.mark.skipif(not HAS_RUST, reason="Rust FFI not available")
    def test_rust_engine_step_n64(self):
        """Rust Euler step(N=64): budget < 0.5ms (must be faster than Python)."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(64, dt=0.01)
        assert eng._rust is not None, "Rust FFI should be active"
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, 64)
        omegas = np.ones(64)
        knm = 0.3 * np.ones((64, 64))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((64, 64))

        elapsed = self._time_fn(
            lambda: eng.step(phases, omegas, knm, 0.0, 0.0, alpha),
        )
        print(f"  Rust step(64): {elapsed * 1000:.2f}ms")
        assert elapsed < 0.0005, f"Rust step(64) = {elapsed * 1000:.2f}ms > 0.5ms"

    def test_python_order_parameter_n256(self):
        """Python order_parameter(N=256): budget < 200μs."""
        import scpn_phase_orchestrator.upde.order_params as op_mod

        saved = op_mod._HAS_RUST
        op_mod._HAS_RUST = False
        phases = np.random.default_rng(0).uniform(0, TWO_PI, 256)

        elapsed = self._time_fn(
            lambda: op_mod.compute_order_parameter(phases),
        )
        op_mod._HAS_RUST = saved
        print(f"  Python order_param(256): {elapsed * 1e6:.0f}μs")
        assert elapsed < 0.0002, (
            f"Python order_param(256) = {elapsed * 1e6:.0f}μs > 200μs"
        )

    @pytest.mark.skipif(not HAS_RUST, reason="Rust FFI not available")
    def test_rust_order_parameter_n256(self):
        """Rust order_parameter(N=256): budget < 50μs."""
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        phases = np.random.default_rng(0).uniform(0, TWO_PI, 256)
        elapsed = self._time_fn(
            lambda: compute_order_parameter(phases),
        )
        print(f"  Rust order_param(256): {elapsed * 1e6:.0f}μs")
        assert elapsed < 0.00005, (
            f"Rust order_param(256) = {elapsed * 1e6:.0f}μs > 50μs"
        )

    @pytest.mark.skipif(not HAS_RUST, reason="Rust FFI not available")
    def test_rust_faster_than_python_step(self):
        """Rust must be at least 2× faster than Python for N=64."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 64
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        eng_py = UPDEEngine(n, dt=0.01)
        eng_py._rust = None
        t_py = self._time_fn(
            lambda: eng_py.step(phases, omegas, knm, 0.0, 0.0, alpha),
        )

        eng_rs = UPDEEngine(n, dt=0.01)
        t_rs = self._time_fn(
            lambda: eng_rs.step(phases, omegas, knm, 0.0, 0.0, alpha),
        )

        speedup = t_py / max(t_rs, 1e-9)
        print(
            f"  Rust speedup: {speedup:.1f}× "
            f"(Python={t_py * 1000:.2f}ms, Rust={t_rs * 1000:.2f}ms)"
        )
        assert speedup > 2.0, f"Rust should be ≥2× faster: speedup={speedup:.1f}×"


# ---------------------------------------------------------------------------
# Full pipeline with Rust path
# ---------------------------------------------------------------------------


class TestRustPipelineIntegration:
    """Verify Rust FFI integrates into the full SPO pipeline."""

    def test_full_loop_with_default_backend(self):
        """Coupling → Engine(default backend) → R → Regime → Policy.
        If Rust is available, this exercises the Rust hot path."""
        from scpn_phase_orchestrator.coupling import CouplingBuilder
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 16
        cs = CouplingBuilder().build(n, 0.5, 0.3)
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)

        for _ in range(200):
            phases = eng.step(
                phases,
                omegas,
                cs.knm,
                0.0,
                0.0,
                cs.alpha,
            )

        r, psi = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

        state = UPDEState(
            layers=[LayerState(R=r, psi=psi)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=r,
            regime_id="nominal",
        )
        mgr = RegimeManager(cooldown_steps=0)
        policy = SupervisorPolicy(mgr)
        actions = policy.decide(state, BoundaryState())
        assert isinstance(actions, list)
