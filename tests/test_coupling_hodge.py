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

        np.testing.assert_array_equal(result.gradient, np.array([1.0, 1.0]))
        np.testing.assert_array_equal(result.curl, np.array([2.0, 2.0]))
        np.testing.assert_array_equal(result.harmonic, np.array([3.0, 3.0]))


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
