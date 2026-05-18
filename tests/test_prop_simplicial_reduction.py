# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based simplicial reduction proofs

"""Hypothesis-driven invariant proofs for the simplicial Kuramoto model.

The key mathematical identity: σ₂=0 reduces to standard pairwise Kuramoto.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import simplicial as simplicial_mod
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n)) * strength
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestSimplicialReductionInvariants:
    @given(
        n=st.integers(min_value=3, max_value=8),
        n_steps=st.integers(min_value=2, max_value=20),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_sigma2_zero_matches_upde(self, n: int, n_steps: int, seed: int) -> None:
        """σ₂=0 must produce identical output to standard Kuramoto."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n, seed=seed)
        alpha = np.zeros((n, n))
        upde = UPDEEngine(n, dt=0.01)
        simp = SimplicialEngine(n, dt=0.01, sigma2=0.0)
        out_upde = upde.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=n_steps)
        out_simp = simp.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=n_steps)
        np.testing.assert_allclose(out_simp, out_upde, atol=1e-10)

    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=40, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_output_finite(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n, seed=seed)
        alpha = np.zeros((n, n))
        simp = SimplicialEngine(n, dt=0.01, sigma2=1.0)
        out = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(out))

    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=40, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_output_length_n(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n, seed=seed)
        alpha = np.zeros((n, n))
        simp = SimplicialEngine(n, dt=0.01, sigma2=0.5)
        out = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert len(out) == n

    @pytest.mark.parametrize("sigma2", [0.01, 0.1, 0.5, 1.0, 5.0])
    def test_nonzero_sigma2_differs(self, sigma2: float) -> None:
        """σ₂ ≠ 0 → different from standard Kuramoto."""
        n = 5
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n)
        alpha = np.zeros((n, n))
        upde = UPDEEngine(n, dt=0.01)
        simp = SimplicialEngine(n, dt=0.01, sigma2=sigma2)
        out_upde = upde.step(phases, omegas, knm, 0.0, 0.0, alpha)
        out_simp = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert not np.allclose(out_simp, out_upde, atol=1e-6)


class TestSimplicialEdgeCases:
    """Edge and error paths a prior audit flagged as missing from the property
    suite — the hypothesis cases exercise the happy path; these cover
    degenerate inputs, long-run behaviour and cross-path parity."""

    def test_zero_coupling_reduces_to_drift(self) -> None:
        """K ≡ 0, σ₂=0 → phases advance only by ω·dt."""
        n = 4
        dt = 0.01
        phases = np.zeros(n)
        omegas = np.array([0.5, 1.0, 1.5, 2.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        simp = SimplicialEngine(n, dt=dt, sigma2=0.0)
        out = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(out, dt * omegas, atol=1e-12)

    def test_antisymmetric_alpha_breaks_sigma_reduction(self) -> None:
        """With σ₂=0 and a non-trivial α, SimplicialEngine must still
        match UPDEEngine — the reduction identity does not depend on α."""
        n = 4
        dt = 0.01
        rng = np.random.default_rng(2026)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = _connected_knm(n, seed=7)
        alpha = np.full((n, n), 0.15)
        np.fill_diagonal(alpha, 0.0)
        upde = UPDEEngine(n, dt=dt)
        simp = SimplicialEngine(n, dt=dt, sigma2=0.0)
        out_upde = upde.step(phases, omegas, knm, 0.0, 0.0, alpha)
        out_simp = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(out_simp, out_upde, atol=1e-10)

    def test_long_run_bounded(self) -> None:
        """After 500 steps with moderate σ₂, phases must stay in [0, 2π)."""
        n = 6
        dt = 0.01
        rng = np.random.default_rng(55)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(0.5, 1.5, n)
        knm = _connected_knm(n, seed=55)
        alpha = np.zeros((n, n))
        simp = SimplicialEngine(n, dt=dt, sigma2=0.3)
        for _ in range(500):
            phases = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(phases >= 0.0)
        assert np.all(phases < TWO_PI)
        assert np.all(np.isfinite(phases))

    def test_external_drive_attractor(self) -> None:
        """ζ > 0, Ψ=π/2, ω=0, K=0 → all phases pulled toward π/2.

        Starting away from both Ψ and Ψ + π ensures sin(Ψ − θ) > 0 so the
        drive term is non-trivial — exactly Ψ or Ψ ± π is an unstable or
        neutral fixed point of the attractor.
        """
        n = 3
        dt = 0.01
        phases = np.array([0.1, 0.2, 0.3])
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        simp = SimplicialEngine(n, dt=dt, sigma2=0.0)
        zeta = 0.5
        psi = np.pi / 2.0
        for _ in range(3000):
            phases = simp.step(phases, omegas, knm, zeta, psi, alpha)
        np.testing.assert_allclose(phases, np.full(n, psi), atol=0.05)

    def test_minimum_n_is_three_for_higher_order_term(self) -> None:
        """Simplicial k=3 coupling requires ≥3 oscillators; N=2 should
        not crash — the σ₂ term silently contributes zero."""
        n = 2
        dt = 0.01
        phases = np.array([0.1, 0.5])
        omegas = np.ones(n)
        knm = np.array([[0.0, 0.3], [0.3, 0.0]])
        alpha = np.zeros((n, n))
        simp = SimplicialEngine(n, dt=dt, sigma2=0.5)
        out = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(out))
        assert len(out) == n

    @pytest.mark.parametrize(
        "sigma2",
        [-1.0, False, float("nan"), float("inf"), "0.1"],
    )
    def test_constructor_rejects_invalid_sigma2(self, sigma2: Any) -> None:
        with pytest.raises(ValueError, match="sigma2 must be non-negative"):
            SimplicialEngine(4, dt=0.01, sigma2=sigma2)

    @pytest.mark.parametrize(
        "sigma2",
        [False, -0.1, float("nan"), float("inf"), "0.1"],
    )
    def test_sigma2_setter_rejects_invalid_values(self, sigma2: Any) -> None:
        engine = SimplicialEngine(4, dt=0.01)
        with pytest.raises(ValueError, match="sigma2 must be non-negative"):
            engine.sigma2 = sigma2

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("zeta", False, "zeta"),
            ("zeta", complex(1.0, 2.0), "zeta"),
            ("zeta", "0.5", "zeta"),
            ("psi", False, "psi"),
            ("psi", complex(0.0, 1.0), "psi"),
            ("psi", object(), "psi"),
        ],
    )
    def test_run_rejects_non_real_zeta_psi(
        self,
        field: str,
        value: Any,
        match: str,
    ) -> None:
        engine = SimplicialEngine(n_oscillators=4, dt=0.01)
        phases = np.zeros(4)
        omegas = np.ones(4)
        knm = np.zeros((4, 4))
        alpha = np.zeros((4, 4))
        kwargs = {"zeta": 0.0, "psi": 0.0}
        kwargs[field] = value

        with pytest.raises(ValueError, match=match):
            engine.run(phases, omegas, knm, kwargs["zeta"], kwargs["psi"], alpha, 1)

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("phases", np.array([0.0, np.nan, 0.0, 0.0])),
            ("phases", np.array([True, False, False, True], dtype=np.bool_)),
            ("omegas", np.array([0.0, np.inf, 0.0, 0.0])),
            (
                "knm",
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, np.nan, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            ),
            (
                "alpha",
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, np.inf, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            ),
        ],
    )
    def test_run_rejects_malformed_state_arrays(
        self,
        field: str,
        bad_value: Any,
    ) -> None:
        engine = SimplicialEngine(4, dt=0.01)
        base_phases = np.zeros(4)
        base_omegas = np.ones(4)
        base_knm = np.zeros((4, 4))
        base_alpha = np.zeros((4, 4))
        base_phases[0] = 0.0
        args = {
            "phases": base_phases,
            "omegas": base_omegas,
            "knm": base_knm,
            "alpha": base_alpha,
        }
        args[field] = bad_value
        with pytest.raises(ValueError):
            engine.run(
                args["phases"],
                args["omegas"],
                args["knm"],
                0.0,
                0.0,
                args["alpha"],
                n_steps=1,
            )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("phases", np.zeros(5)),
            ("omegas", np.zeros(5)),
            ("knm", np.zeros((4, 3))),
            ("alpha", np.zeros((3, 4))),
        ],
    )
    def test_run_rejects_shape_mismatch(self, field: str, bad_value: Any) -> None:
        engine = SimplicialEngine(4, dt=0.01)
        base_phases = np.zeros(4)
        base_omegas = np.ones(4)
        base_knm = np.zeros((4, 4))
        base_alpha = np.zeros((4, 4))
        args = {
            "phases": base_phases,
            "omegas": base_omegas,
            "knm": base_knm,
            "alpha": base_alpha,
        }
        args[field] = bad_value
        with pytest.raises(ValueError, match="shape"):
            engine.run(
                args["phases"],
                args["omegas"],
                args["knm"],
                0.0,
                0.0,
                args["alpha"],
                n_steps=1,
            )

    def test_run_zero_steps_is_identity(self) -> None:
        """n_steps=0 returns a copy of input phases without changing state."""
        engine = SimplicialEngine(4, dt=0.01)
        phases = np.array([0.1, 0.4, 0.7, 1.0], dtype=np.float64)
        out = engine.run(
            phases,
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.zeros((4, 4)),
            0.0,
            0.0,
            np.zeros((4, 4)),
            n_steps=0,
        )
        assert np.array_equal(out, phases)
        assert out is not phases

    def test_backend_invalid_output_shape_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Backend output must be validated for shape and finite values."""
        engine = SimplicialEngine(4, dt=0.01)
        phases = np.zeros(4)
        omegas = np.ones(4)
        knm = np.zeros((4, 4))
        alpha = np.zeros((4, 4))

        def invalid_backend(*_args: Any, **_kwargs: Any) -> np.ndarray:
            return np.array([0.1, 0.2], dtype=np.float64)

        monkeypatch.setattr(simplicial_mod, "_dispatch", lambda: invalid_backend)
        monkeypatch.setattr(simplicial_mod, "_python_run", lambda *args: phases.copy())

        with pytest.raises(ValueError, match="backend output"):
            engine.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=1)

    def test_dispatcher_fallback_on_backend_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dispatcher should fall back to python when active backend throws."""
        engine = SimplicialEngine(4, dt=0.01)
        phases = np.array([0.2, 0.4, 0.6, 0.8])
        omegas = np.array([0.1, 0.2, 0.3, 0.4])
        knm = np.ones((4, 4))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((4, 4))
        expected = phases + 0.01 * omegas
        expected = np.mod(expected, TWO_PI)

        def raising_backend(*_args: Any, **_kwargs: Any) -> np.ndarray:
            raise RuntimeError("backend unavailable")

        def fast_python_run(
            _phases: np.ndarray,
            _omegas: np.ndarray,
            _knm: np.ndarray,
            _alpha: np.ndarray,
            _n: int,
            _zeta: float,
            _psi: float,
            _sigma2: float,
            _dt: float,
            _n_steps: int,
        ) -> np.ndarray:
            return expected

        monkeypatch.setattr(simplicial_mod, "_dispatch", lambda: raising_backend)
        monkeypatch.setattr(simplicial_mod, "_python_run", fast_python_run)
        out = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=1)
        np.testing.assert_allclose(out, expected)


# Pipeline wiring: simplicial reduction covers the identity σ₂=0 ⇔ pairwise
# Kuramoto. The hypothesis cases scan the small-N regime; the edge cases
# above pin drift-only reduction, long-run boundedness, external-drive
# attractor and the degenerate N=2 path.
