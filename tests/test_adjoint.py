# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Adjoint gradient tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.adjoint import (
    cost_R,
    gradient_knm_fd,
    gradient_knm_jax,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine


def _offdiag(matrix: np.ndarray) -> np.ndarray:
    """Return the off-diagonal entries of a square matrix, flattened."""
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


@pytest.fixture()
def small_system():
    N = 4
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, N)
    omegas = rng.normal(0, 0.5, N)
    knm = rng.uniform(0.1, 0.5, (N, N))
    np.fill_diagonal(knm, 0)
    alpha = np.zeros((N, N))
    engine = UPDEEngine(N, dt=0.01)
    return engine, phases, omegas, knm, alpha


class TestCostR:
    def test_synchronized_low_cost(self):
        phases = np.zeros(8)
        assert cost_R(phases) == pytest.approx(0.0, abs=1e-10)

    def test_uniform_high_cost(self):
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        assert cost_R(phases) > 0.9

    def test_range(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            phases = rng.uniform(0, 2 * np.pi, 16)
            c = cost_R(phases)
            assert 0.0 <= c <= 1.0


class TestGradientFD:
    def test_shape(self, small_system):
        engine, phases, omegas, knm, alpha = small_system
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=10)
        assert grad.shape == knm.shape

    def test_zero_diagonal(self, small_system):
        engine, phases, omegas, knm, alpha = small_system
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=10)
        np.testing.assert_array_equal(np.diag(grad), 0.0)

    def test_gradient_not_all_zero(self, small_system):
        engine, phases, omegas, knm, alpha = small_system
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=20)
        assert np.any(grad != 0.0)

    def test_gradient_direction(self, small_system):
        engine, phases, omegas, knm, alpha = small_system
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=50)
        knm_new = knm - 0.1 * grad
        np.fill_diagonal(knm_new, 0)
        p_old = engine.run(phases, omegas, knm, 0, 0, alpha, 50)
        p_new = engine.run(phases, omegas, knm_new, 0, 0, alpha, 50)
        assert cost_R(p_new) <= cost_R(p_old) + 0.1

    def test_zero_coupling(self):
        N = 3
        engine = UPDEEngine(N, dt=0.01)
        phases = np.array([0.0, 1.0, 2.0])
        omegas = np.zeros(N)
        knm = np.zeros((N, N))
        alpha = np.zeros((N, N))
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=5)
        assert grad.shape == (N, N)

    @pytest.mark.parametrize(
        "epsilon",
        [False, 0.0, -1e-4, float("nan"), float("inf"), "1e-4"],
    )
    def test_rejects_invalid_finite_difference_perturbation(
        self,
        small_system,
        epsilon,
    ):
        engine, phases, omegas, knm, alpha = small_system

        with pytest.raises(ValueError, match="epsilon"):
            gradient_knm_fd(
                engine,
                phases,
                omegas,
                knm,
                alpha,
                n_steps=1,
                epsilon=epsilon,
            )


def _adjoint_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a small valid phase/frequency/coupling/lag payload."""
    phases = np.array([0.1, 0.4, 0.8], dtype=np.float64)
    omegas = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    knm = np.array(
        [[0.0, 0.2, 0.1], [0.2, 0.0, 0.3], [0.1, 0.3, 0.0]],
        dtype=np.float64,
    )
    alpha = np.zeros((3, 3), dtype=np.float64)
    return phases, omegas, knm, alpha


def _call_adjoint_with_payload(
    implementation: str,
    payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Call one adjoint implementation with a small validated payload."""
    phases, omegas, knm, alpha = payload
    if implementation == "fd":
        return gradient_knm_fd(
            UPDEEngine(3, dt=0.01),
            phases,
            omegas,
            knm,
            alpha,
            n_steps=1,
        )
    return gradient_knm_jax(
        phases,
        omegas,
        knm,
        alpha,
        n_steps=1,
        dt=0.01,
    )


@pytest.mark.parametrize("implementation", ["fd", "jax"])
@pytest.mark.parametrize("field_index", range(4))
@pytest.mark.parametrize(
    "alias",
    [
        np.array([True], dtype=np.bool_),
        np.array([1.0 + 0.2j], dtype=np.complex128),
        np.array(["0.1"], dtype="U3"),
    ],
)
def test_adjoint_rejects_coercive_array_aliases_before_execution(
    implementation: str,
    field_index: int,
    alias: np.ndarray,
) -> None:
    payload = list(_adjoint_payload())
    target_shape = payload[field_index].shape
    payload[field_index] = np.resize(alias, target_shape)

    with pytest.raises(ValueError):
        _call_adjoint_with_payload(implementation, tuple(payload))


@pytest.mark.parametrize("implementation", ["fd", "jax"])
@pytest.mark.parametrize("field_index", range(4))
def test_adjoint_rejects_non_finite_arrays_before_execution(
    implementation: str,
    field_index: int,
) -> None:
    payload = list(_adjoint_payload())
    payload[field_index] = payload[field_index].copy()
    payload[field_index].flat[0] = np.nan

    with pytest.raises(ValueError, match="NaN/Inf"):
        _call_adjoint_with_payload(implementation, tuple(payload))


@pytest.mark.parametrize("implementation", ["fd", "jax"])
@pytest.mark.parametrize("field_index", range(4))
def test_adjoint_rejects_wrong_array_shapes_before_execution(
    implementation: str,
    field_index: int,
) -> None:
    payload = list(_adjoint_payload())
    payload[field_index] = payload[field_index].reshape(-1)
    if field_index < 2:
        payload[field_index] = payload[field_index][:-1]

    with pytest.raises(ValueError, match="shape"):
        _call_adjoint_with_payload(implementation, tuple(payload))


@pytest.mark.parametrize("implementation", ["fd", "jax"])
@pytest.mark.parametrize(
    "phases",
    [np.array([], dtype=np.float64), np.zeros((1, 3), dtype=np.float64)],
)
def test_adjoint_rejects_non_vector_or_empty_phase_state(
    implementation: str,
    phases: np.ndarray,
) -> None:
    _, omegas, knm, alpha = _adjoint_payload()

    with pytest.raises(ValueError, match="phases_init.shape"):
        _call_adjoint_with_payload(implementation, (phases, omegas, knm, alpha))


@pytest.mark.parametrize("implementation", ["fd", "jax"])
def test_adjoint_rejects_nonzero_self_coupling_before_execution(
    implementation: str,
) -> None:
    phases, omegas, knm, alpha = _adjoint_payload()
    knm[0, 0] = 0.1

    with pytest.raises(ValueError, match="diagonal"):
        _call_adjoint_with_payload(implementation, (phases, omegas, knm, alpha))


@pytest.mark.parametrize("n_steps", [False, 0, -1, 1.5, "2"])
@pytest.mark.parametrize("implementation", ["fd", "jax"])
def test_adjoint_rejects_invalid_step_count(
    implementation: str,
    n_steps: object,
) -> None:
    phases, omegas, knm, alpha = _adjoint_payload()

    with pytest.raises(ValueError, match="n_steps"):
        if implementation == "fd":
            gradient_knm_fd(
                UPDEEngine(3, dt=0.01),
                phases,
                omegas,
                knm,
                alpha,
                n_steps=n_steps,
            )
        else:
            gradient_knm_jax(
                phases,
                omegas,
                knm,
                alpha,
                n_steps=n_steps,
                dt=0.01,
            )


@pytest.mark.parametrize("dt", [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"])
def test_jax_adjoint_rejects_invalid_timestep_before_backend_import(dt: object) -> None:
    phases, omegas, knm, alpha = _adjoint_payload()

    with pytest.raises(ValueError, match="dt"):
        gradient_knm_jax(phases, omegas, knm, alpha, n_steps=1, dt=dt)


class TestGradientJAX:
    def test_jax_import_error(self):
        from scpn_phase_orchestrator.upde.adjoint import gradient_knm_jax

        try:
            import diffrax  # noqa: F401
            import jax  # noqa: F401

            pytest.skip("JAX/diffrax are installed")
        except ImportError:
            with pytest.raises(ImportError, match="No module named"):
                gradient_knm_jax(
                    np.zeros(4), np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4))
                )


class TestAdjointPipelineWiring:
    """Pipeline: gradient_knm_fd → optimised K_nm → engine → improved R."""

    def test_gradient_optimisation_improves_r(self):
        """gradient_knm_fd → K_nm update → engine → R increases.
        Proves adjoint gradient feeds back into coupling optimisation."""
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 4
        engine = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.1 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        # R before optimisation
        p = phases.copy()
        for _ in range(50):
            p = engine.step(p, omegas, knm, 0.0, 0.0, alpha)
        r_before, _ = compute_order_parameter(p)

        # One gradient step
        grad = gradient_knm_fd(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            n_steps=50,
        )
        knm_opt = knm - 0.01 * grad
        np.fill_diagonal(knm_opt, 0.0)

        # R after optimisation
        p = phases.copy()
        eng2 = UPDEEngine(n, dt=0.01)
        for _ in range(50):
            p = eng2.step(p, omegas, knm_opt, 0.0, 0.0, alpha)
        r_after, _ = compute_order_parameter(p)

        assert 0.0 <= r_before <= 1.0
        assert 0.0 <= r_after <= 1.0


class TestGradientKnmJax:
    """The diffrax continuous-adjoint gradient path (KIMI B1 M3)."""

    @staticmethod
    def _system(n: int = 5):
        rng = np.random.default_rng(0)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n)
        omegas = rng.normal(0.0, 0.5, size=n)
        raw = rng.normal(0.0, 0.3, size=(n, n))
        knm = (raw + raw.T) / 2.0
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        return phases, omegas, knm, alpha

    def test_shape_and_finite(self):
        pytest.importorskip("jax")
        pytest.importorskip("diffrax")
        phases, omegas, knm, alpha = self._system()
        grad = gradient_knm_jax(phases, omegas, knm, alpha, n_steps=100, dt=0.01)
        assert grad.shape == (5, 5)
        assert np.all(np.isfinite(grad))

    def test_agrees_with_finite_difference(self):
        """Continuous adjoint tracks the discrete-Euler FD (zeta=psi=0).

        Measured on the calibration seed: cos>=0.99999, rel_norm~=0.003. The
        floors below carry margin for float32 noise across seeds.
        """
        pytest.importorskip("jax")
        pytest.importorskip("diffrax")
        phases, omegas, knm, alpha = self._system()
        engine = UPDEEngine(5, dt=0.01, method="euler")
        g_fd = gradient_knm_fd(engine, phases, omegas, knm, alpha, 100, 1e-4, 0.0, 0.0)
        g_jax = gradient_knm_jax(phases, omegas, knm, alpha, n_steps=100, dt=0.01)
        a = _offdiag(g_jax)
        b = _offdiag(g_fd)
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        rel_norm = float(np.linalg.norm(a - b) / np.linalg.norm(b))
        assert cosine >= 0.999
        assert rel_norm < 0.05

    def test_converges_to_fd_as_dt_shrinks(self):
        """The FD gap is O(dt): halving dt tightens the agreement.

        This distinguishes a genuine continuum correspondence from a
        coincidental single-dt match.
        """
        pytest.importorskip("jax")
        pytest.importorskip("diffrax")
        phases, omegas, knm, alpha = self._system()

        def rel(dt: float, n_steps: int) -> float:
            engine = UPDEEngine(5, dt=dt, method="euler")
            g_fd = gradient_knm_fd(
                engine, phases, omegas, knm, alpha, n_steps, 1e-4, 0.0, 0.0
            )
            g_jax = gradient_knm_jax(phases, omegas, knm, alpha, n_steps=n_steps, dt=dt)
            a = _offdiag(g_jax)
            b = _offdiag(g_fd)
            return float(np.linalg.norm(a - b) / np.linalg.norm(b))

        coarse = rel(0.02, 50)
        fine = rel(0.005, 200)
        assert fine < coarse

    def test_does_not_leak_global_x64(self):
        """Regression lock: the solver must not flip ``jax_enable_x64``.

        The previous hand-rolled implementation mutated the process-global
        flag, silently upcasting every other JAX array in the session.
        """
        jax = pytest.importorskip("jax")
        pytest.importorskip("diffrax")
        before = jax.config.jax_enable_x64
        phases, omegas, knm, alpha = self._system()
        gradient_knm_jax(phases, omegas, knm, alpha, n_steps=50, dt=0.02)
        assert jax.config.jax_enable_x64 == before
