# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge decomposition tests

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.coupling.hodge as hodge
from scpn_phase_orchestrator.coupling import hodge_decomposition
from scpn_phase_orchestrator.coupling.hodge import HodgeResult


@pytest.fixture(name="python_backend")
def _python_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hodge, "_dispatch", lambda: None)


class TestHodgeDecompositionSemantics:
    def test_symmetric_matrix_has_no_curl_component(self, python_backend: None) -> None:
        phases = np.array([0.4, -0.1, 1.0], dtype=np.float64)
        knm = np.array(
            [
                [0.0, 0.5, 1.0],
                [0.5, 0.0, -0.4],
                [1.0, -0.4, 0.0],
            ],
            dtype=np.float64,
        )

        result = hodge_decomposition(knm, phases)

        assert np.allclose(result.curl, 0.0)
        assert np.allclose(result.harmonic, 0.0, atol=1e-12)

    def test_antisymmetric_matrix_has_no_gradient_component(
        self,
        python_backend: None,
    ) -> None:
        phases = np.array([0.2, 0.8, 1.5], dtype=np.float64)
        knm = np.array(
            [
                [0.0, 0.7, -0.4],
                [-0.7, 0.0, 1.2],
                [0.4, -1.2, 0.0],
            ],
            dtype=np.float64,
        )

        result = hodge_decomposition(knm, phases)

        assert np.allclose(result.gradient, 0.0, atol=1e-12)

    def test_python_reference_matches_manual_split(self, python_backend: None) -> None:
        phases = np.array([0.1, 0.9, -0.6], dtype=np.float64)
        knm = np.array(
            [
                [0.0, 0.3, -0.8],
                [0.1, 0.0, 0.5],
                [-0.4, -0.2, 0.0],
            ],
            dtype=np.float64,
        )

        result = hodge_decomposition(knm, phases)

        diff = phases[np.newaxis, :] - phases[:, np.newaxis]
        cos_diff = np.cos(diff)
        expected_total = np.sum(knm * cos_diff, axis=1)
        expected_gradient = np.sum(0.5 * (knm + knm.T) * cos_diff, axis=1)
        expected_curl = np.sum(0.5 * (knm - knm.T) * cos_diff, axis=1)
        expected_harmonic = expected_total - expected_gradient - expected_curl

        np.testing.assert_allclose(result.gradient, expected_gradient)
        np.testing.assert_allclose(result.curl, expected_curl)
        np.testing.assert_allclose(result.harmonic, expected_harmonic)

    def test_hodge_result_is_dataclass(self, python_backend: None) -> None:
        phases = np.array([0.5], dtype=np.float64)
        knm = np.array([[0.0]], dtype=np.float64)
        result = hodge_decomposition(knm, phases)

        assert isinstance(result, HodgeResult)
        assert result.gradient.shape == (1,)
        assert result.curl.shape == (1,)
        assert result.harmonic.shape == (1,)


class TestHodgeBackendDispatch:
    def test_python_backend_is_selected_when_dispatch_returns_none(
        self,
        python_backend: None,
    ) -> None:
        phases = np.array([0.0, np.pi / 2], dtype=np.float64)
        knm = np.array(
            [
                [0.0, 2.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )

        result = hodge_decomposition(knm, phases)

        assert np.isfinite(result.gradient).all()
        assert np.isfinite(result.curl).all()
        assert np.isfinite(result.harmonic).all()

    def test_active_backend_result_is_forwarded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = 0

        def fake_backend(knm_flat: np.ndarray, phases: np.ndarray, n: int) -> tuple:
            nonlocal calls
            calls += 1
            k = knm_flat.reshape(n, n)
            diff = phases[np.newaxis, :] - phases[:, np.newaxis]
            cos_diff = np.cos(diff)
            gradient = np.sum(0.5 * (k + k.T) * cos_diff, axis=1)
            curl = np.sum(0.5 * (k - k.T) * cos_diff, axis=1)
            harmonic = np.sum(k * cos_diff, axis=1) - gradient - curl
            return gradient, curl, harmonic

        monkeypatch.setattr(hodge, "_dispatch", lambda: fake_backend)

        phases = np.array([0.3, -0.2])
        result = hodge_decomposition(
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            phases,
        )

        assert calls == 1
        np.testing.assert_allclose(result.gradient, np.cos(phases[::-1] - phases))
        np.testing.assert_allclose(result.curl, np.array([0.0, 0.0]))
        np.testing.assert_allclose(result.harmonic, np.array([0.0, 0.0]))

    def test_invalid_backend_result_falls_back_to_python_reference(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_backend(knm_flat: np.ndarray, phases: np.ndarray, n: int) -> tuple:
            return (
                np.full(n, 1.0, dtype=np.float64),
                np.full(n, 2.0, dtype=np.float64),
                np.full(n, 3.0, dtype=np.float64),
            )

        monkeypatch.setattr(hodge, "_dispatch", lambda: fake_backend)

        phases = np.array([0.3, -0.2])
        result = hodge_decomposition(
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            phases,
        )

        expected = np.array([np.cos(-0.5), np.cos(0.5)])
        np.testing.assert_allclose(result.gradient, expected)
        np.testing.assert_allclose(result.curl, np.array([0.0, 0.0]))
        np.testing.assert_allclose(result.harmonic, np.array([0.0, 0.0]))


class TestHodgeValidation:
    def test_empty_phases_produces_empty_results(
        self,
        python_backend: None,
    ) -> None:
        result = hodge_decomposition(
            np.zeros((0, 0), dtype=np.float64),
            np.array([], dtype=np.float64),
        )

        assert result.gradient.size == 0
        assert result.curl.size == 0
        assert result.harmonic.size == 0
        assert isinstance(result.gradient, np.ndarray)

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

    def test_invalid_phase_shape_is_rejected(
        self,
        python_backend: None,
    ) -> None:
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

    def test_boolean_coupling_matrix_rejected(self, python_backend: None) -> None:
        phases = np.array([0.0, 1.0], dtype=np.float64)
        knm = np.array([[False, True], [True, False]], dtype=np.bool_)

        with pytest.raises(ValueError, match="knm must not contain boolean values"):
            hodge_decomposition(knm, phases)


def test_large_phase_values_do_not_change_symmetry_identity(
    python_backend: None,
) -> None:
    knm = np.array(
        [
            [0.0, 0.3, -0.2],
            [0.3, 0.0, 0.8],
            [-0.2, 0.8, 0.0],
        ],
        dtype=np.float64,
    )
    phases = np.array([1e8, -2e8, 3e8], dtype=np.float64)

    result = hodge_decomposition(knm, phases)
    diff = phases[np.newaxis, :] - phases[:, np.newaxis]
    cos_diff = np.cos(diff)
    expected_gradient = np.sum(0.5 * (knm + knm.T) * cos_diff, axis=1)
    expected_curl = np.sum(0.5 * (knm - knm.T) * cos_diff, axis=1)
    expected_total = np.sum(knm * cos_diff, axis=1)
    expected_harmonic = expected_total - expected_gradient - expected_curl

    np.testing.assert_allclose(result.gradient, expected_gradient)
    np.testing.assert_allclose(result.curl, expected_curl)
    np.testing.assert_allclose(result.harmonic, expected_harmonic)
    assert np.all(np.isfinite(result.gradient))
    assert np.all(np.isfinite(result.curl))
    assert np.all(np.isfinite(result.harmonic))


class TestDispatchFallbackChain:
    def test_dispatch_falls_back_to_next_backend_when_active_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"rust": 0, "go": 0}

        def _fail_rust():
            calls["rust"] += 1
            raise ImportError("rust unavailable")

        def _ok_go():
            calls["go"] += 1
            return lambda knm_flat, phases, n: (
                np.ones(n, dtype=np.float64),
                np.zeros(n, dtype=np.float64),
                np.zeros(n, dtype=np.float64),
            )

        monkeypatch.setattr(hodge, "_BACKEND_CACHE", {})
        monkeypatch.setattr(hodge, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(hodge, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(hodge, "_LOADERS", {"rust": _fail_rust, "go": _ok_go})

        backend = hodge._dispatch()
        assert backend is not None
        got = backend(np.zeros(4, dtype=np.float64), np.zeros(2, dtype=np.float64), 2)
        np.testing.assert_allclose(got[0], np.ones(2, dtype=np.float64))
        assert calls == {"rust": 1, "go": 1}

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"go": 0}

        def _ok_go():
            calls["go"] += 1
            return lambda knm_flat, phases, n: (
                np.zeros(n, dtype=np.float64),
                np.zeros(n, dtype=np.float64),
                np.zeros(n, dtype=np.float64),
            )

        monkeypatch.setattr(hodge, "_BACKEND_CACHE", {})
        monkeypatch.setattr(hodge, "ACTIVE_BACKEND", "go")
        monkeypatch.setattr(hodge, "AVAILABLE_BACKENDS", ["go", "python"])
        monkeypatch.setattr(hodge, "_LOADERS", {"go": _ok_go})

        hodge._dispatch()
        hodge._dispatch()

        assert calls["go"] == 1
