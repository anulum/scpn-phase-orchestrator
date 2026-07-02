# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge decomposition dispatch and validation tests

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.coupling.hodge as hodge
from scpn_phase_orchestrator.coupling import hodge_decomposition
from scpn_phase_orchestrator.coupling.hodge import HodgeResult


@pytest.fixture(name="python_backend")
def _python_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hodge, "_dispatch", lambda: None)


def _matrix_reference(
    knm_flat: np.ndarray,
    phases: np.ndarray,
    n: int,
    edges_flat: np.ndarray,
    n_edges: int,
    tris_flat: np.ndarray,
    n_tris: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A faithful backend stand-in delegating to the NumPy reference."""
    k = np.asarray(knm_flat, dtype=np.float64).reshape(n, n)
    return hodge._python_decomposition(k, np.asarray(phases, dtype=np.float64))


class TestHodgeDecompositionSemantics:
    def test_components_reconstruct_flow(self, python_backend: None) -> None:
        phases = np.array([0.1, 0.9, -0.6], dtype=np.float64)
        knm = np.array(
            [[0.0, 0.3, -0.8], [0.1, 0.0, 0.5], [-0.4, -0.2, 0.0]],
            dtype=np.float64,
        )
        res = hodge_decomposition(knm, phases)
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-12
        )

    def test_filled_complete_graph_has_zero_harmonic(
        self, python_backend: None
    ) -> None:
        phases = np.array([0.4, -0.1, 1.0], dtype=np.float64)
        knm = np.array(
            [[0.0, 0.5, 1.0], [0.5, 0.0, -0.4], [1.0, -0.4, 0.0]],
            dtype=np.float64,
        )
        res = hodge_decomposition(knm, phases)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-10)
        assert res.betti_one == 0

    def test_components_orthogonal_and_antisymmetric(
        self, python_backend: None
    ) -> None:
        rng = np.random.default_rng(4)
        knm = rng.standard_normal((6, 6))
        phases = rng.uniform(0, 2 * np.pi, 6)
        res = hodge_decomposition(knm, phases)
        upper = np.triu_indices(6, k=1)
        for a, b in (
            (res.gradient, res.curl),
            (res.gradient, res.harmonic),
            (res.curl, res.harmonic),
        ):
            assert abs(float(np.sum(a[upper] * b[upper]))) < 1e-10
        for comp in (res.gradient, res.curl, res.harmonic):
            np.testing.assert_allclose(comp, -comp.T, atol=1e-12)

    def test_hodge_result_is_dataclass(self, python_backend: None) -> None:
        res = hodge_decomposition(np.array([[0.0]]), np.array([0.5]))
        assert isinstance(res, HodgeResult)
        assert res.gradient.shape == (1, 1)
        assert res.flow.shape == (1, 1)
        assert res.potential.shape == (1,)


class TestHodgeBackendDispatch:
    def test_python_backend_is_selected_when_dispatch_returns_none(
        self,
        python_backend: None,
    ) -> None:
        phases = np.array([0.0, np.pi / 2], dtype=np.float64)
        knm = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float64)
        res = hodge_decomposition(knm, phases)
        assert np.isfinite(res.gradient).all()
        assert np.isfinite(res.curl).all()
        assert np.isfinite(res.harmonic).all()

    def test_active_backend_result_is_forwarded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = 0

        def fake_backend(*args: object) -> tuple:
            nonlocal calls
            calls += 1
            return _matrix_reference(*args)  # type: ignore[arg-type]

        monkeypatch.setattr(hodge, "_dispatch", lambda: fake_backend)

        phases = np.array([0.3, -0.2])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        res = hodge_decomposition(knm, phases)

        assert calls == 1
        expected = 1.0 * np.sin(phases[1] - phases[0])
        assert res.flow[0, 1] == pytest.approx(expected)
        np.testing.assert_allclose(res.gradient, res.flow, atol=1e-12)
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-12)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-12)

    def test_invalid_backend_result_falls_back_to_python_reference(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_backend(*args: object) -> tuple:
            n = int(args[2])  # type: ignore[arg-type]
            ones = np.full((n, n), 7.0, dtype=np.float64)
            return ones, ones.copy(), ones.copy()

        monkeypatch.setattr(hodge, "_dispatch", lambda: fake_backend)

        phases = np.array([0.3, -0.2])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        res = hodge_decomposition(knm, phases)

        # Backend output mismatches the reference → Python fallback used.
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, res.flow, atol=1e-12
        )
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-12)


class TestHodgeValidation:
    def test_empty_phases_produces_empty_results(
        self,
        python_backend: None,
    ) -> None:
        res = hodge_decomposition(
            np.zeros((0, 0), dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        assert res.gradient.size == 0
        assert res.curl.size == 0
        assert res.harmonic.size == 0
        assert isinstance(res.gradient, np.ndarray)

    def test_non_finite_phase_values_raise(self, python_backend: None) -> None:
        phases = np.array([0.0, np.nan], dtype=np.float64)
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="phases must contain only finite values"):
            hodge_decomposition(knm, phases)

    def test_non_finite_coupling_values_raise(self, python_backend: None) -> None:
        phases = np.array([0.0, 1.0], dtype=np.float64)
        knm = np.array([[0.0, np.inf], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="knm must contain only finite values"):
            hodge_decomposition(knm, phases)

    def test_invalid_phase_shape_is_rejected(self, python_backend: None) -> None:
        phases = np.array([[0.0, 1.0]], dtype=np.float64)
        knm = np.eye(2, dtype=np.float64)
        with pytest.raises(
            ValueError, match="phases must be a finite 1-D phase vector"
        ):
            hodge_decomposition(knm, phases)

    def test_shape_mismatch_between_knm_and_phases_is_rejected(
        self,
        python_backend: None,
    ) -> None:
        phases = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        knm = np.eye(2, dtype=np.float64)
        with pytest.raises(ValueError, match=r"does not match \(3, 3\)"):
            hodge_decomposition(knm, phases)

    def test_boolean_phases_rejected(self, python_backend: None) -> None:
        phases = np.array([True, False], dtype=np.bool_)
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="phases must not contain boolean values"):
            hodge_decomposition(knm, phases)

    def test_empty_boolean_phase_dtype_rejected(self, python_backend: None) -> None:
        phases = np.array([], dtype=np.bool_)
        knm = np.zeros((0, 0), dtype=np.float64)
        with pytest.raises(ValueError, match="phases must not contain boolean values"):
            hodge_decomposition(knm, phases)

    def test_numpy_boolean_phase_alias_rejected(self, python_backend: None) -> None:
        phases = np.array([0.0, np.bool_(True)], dtype=object)
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="phases must not contain boolean values"):
            hodge_decomposition(knm, phases)

    def test_non_numeric_phase_object_array_rejected(
        self, python_backend: None
    ) -> None:
        phases = np.array(["bad"], dtype=object)
        knm = np.zeros((1, 1), dtype=np.float64)
        with pytest.raises(ValueError, match="phases must be a finite 1-D"):
            hodge_decomposition(knm, phases)

    def test_boolean_coupling_matrix_rejected(self, python_backend: None) -> None:
        phases = np.array([0.0, 1.0], dtype=np.float64)
        knm = np.array([[False, True], [True, False]], dtype=np.bool_)
        with pytest.raises(ValueError, match="knm must not contain boolean values"):
            hodge_decomposition(knm, phases)

    def test_empty_boolean_coupling_dtype_rejected(self, python_backend: None) -> None:
        knm = np.zeros((0, 0), dtype=np.bool_)
        with pytest.raises(ValueError, match="knm must not contain boolean values"):
            hodge._validate_coupling_matrix(knm, expected_n=0)

    def test_numpy_boolean_coupling_alias_rejected(self, python_backend: None) -> None:
        phases = np.array([0.0, 1.0], dtype=np.float64)
        knm = np.array([[0.0, np.bool_(True)], [1.0, 0.0]], dtype=object)
        with pytest.raises(ValueError, match="knm must not contain boolean values"):
            hodge_decomposition(knm, phases)

    def test_non_numeric_coupling_object_array_rejected(
        self, python_backend: None
    ) -> None:
        phases = np.array([0.0], dtype=np.float64)
        knm = np.array([["bad"]], dtype=object)
        with pytest.raises(ValueError, match="knm must be a finite square matrix"):
            hodge_decomposition(knm, phases)

    def test_empty_pseudoinverse_application_returns_empty_vector(self) -> None:
        result = hodge._psd_pinv_apply(
            np.zeros((0, 0), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
        np.testing.assert_array_equal(result, np.zeros(0, dtype=np.float64))

    def test_backend_output_shape_is_validated(self) -> None:
        with pytest.raises(ValueError, match="invalid shape"):
            hodge._normalise_backend_output(
                (
                    np.zeros((1, 1), dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                ),
                expected_n=2,
            )

    def test_backend_output_finiteness_is_validated(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            hodge._normalise_backend_output(
                (
                    np.full((2, 2), np.nan, dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                ),
                expected_n=2,
            )


def test_large_phase_values_preserve_phase_shift_invariance(
    python_backend: None,
) -> None:
    knm = np.array(
        [[0.0, 0.3, -0.2], [0.3, 0.0, 0.8], [-0.2, 0.8, 0.0]],
        dtype=np.float64,
    )
    base = hodge_decomposition(knm, np.array([0.1, -0.2, 0.3]))
    large = hodge_decomposition(knm, np.array([1e8, -2e8, 3e8]) + 0.1)
    # sin(θ_j − θ_i) is bounded; components stay finite under huge phases.
    assert np.all(np.isfinite(large.gradient))
    assert np.all(np.isfinite(large.curl))
    assert np.all(np.isfinite(large.harmonic))
    np.testing.assert_allclose(
        large.gradient + large.curl + large.harmonic, large.flow, atol=1e-10
    )
    assert base.betti_one == large.betti_one


class TestDispatchFallbackChain:
    def test_dispatch_returns_python_when_every_backend_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fail() -> None:
            raise ImportError("backend unavailable")

        monkeypatch.setattr(hodge, "_BACKEND_CACHE", {})
        monkeypatch.setattr(hodge, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(hodge, "AVAILABLE_BACKENDS", ["go"])
        monkeypatch.setattr(hodge, "_LOADERS", {"rust": _fail, "go": _fail})

        assert hodge._dispatch() is None

    def test_dispatch_falls_back_to_next_backend_when_active_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"rust": 0, "go": 0}

        def _fail_rust():
            calls["rust"] += 1
            raise ImportError("rust unavailable")

        def _ok_go():
            calls["go"] += 1

            def _backend(*args: object) -> tuple:
                n = int(args[2])  # type: ignore[arg-type]
                z = np.zeros((n, n), dtype=np.float64)
                return z, z.copy(), z.copy()

            return _backend

        monkeypatch.setattr(hodge, "_BACKEND_CACHE", {})
        monkeypatch.setattr(hodge, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(hodge, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(hodge, "_LOADERS", {"rust": _fail_rust, "go": _ok_go})

        backend = hodge._dispatch()
        assert backend is not None
        got = backend(
            np.zeros(4, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            2,
            np.zeros(0, dtype=np.int64),
            0,
            np.zeros(0, dtype=np.int64),
            0,
        )
        np.testing.assert_allclose(got[0], np.zeros((2, 2), dtype=np.float64))
        assert calls == {"rust": 1, "go": 1}

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"go": 0}

        def _ok_go():
            calls["go"] += 1

            def _backend(*args: object) -> tuple:
                n = int(args[2])  # type: ignore[arg-type]
                z = np.zeros((n, n), dtype=np.float64)
                return z, z.copy(), z.copy()

            return _backend

        monkeypatch.setattr(hodge, "_BACKEND_CACHE", {})
        monkeypatch.setattr(hodge, "ACTIVE_BACKEND", "go")
        monkeypatch.setattr(hodge, "AVAILABLE_BACKENDS", ["go", "python"])
        monkeypatch.setattr(hodge, "_LOADERS", {"go": _ok_go})

        hodge._dispatch()
        hodge._dispatch()

        assert calls["go"] == 1
