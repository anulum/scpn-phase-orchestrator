# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — NPE tests

from __future__ import annotations

from typing import Any, get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import npe as npe_module
from scpn_phase_orchestrator.monitor.npe import compute_npe, phase_distance_matrix
from tests.typing_contracts import assert_precise_ndarray_hint


class TestPhaseDistanceMatrix:
    def test_public_array_contracts_are_parameterised(self):
        hints = (
            get_type_hints(phase_distance_matrix)["phases"],
            get_type_hints(phase_distance_matrix)["return"],
            get_type_hints(compute_npe)["phases"],
        )

        for hint in hints:
            assert_precise_ndarray_hint(hint)
            assert "float64" in str(hint)

    def test_symmetric(self):
        phases = np.array([0.0, 1.0, 2.0])
        D = phase_distance_matrix(phases)
        np.testing.assert_allclose(D, D.T)

    def test_diagonal_zero(self):
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        D = phase_distance_matrix(phases)
        np.testing.assert_allclose(np.diag(D), 0.0)

    def test_wrapping(self):
        phases = np.array([0.1, 2 * np.pi - 0.1])
        D = phase_distance_matrix(phases)
        assert D[0, 1] < 0.3

    def test_max_distance_is_pi(self):
        phases = np.array([0.0, np.pi])
        D = phase_distance_matrix(phases)
        assert abs(D[0, 1] - np.pi) < 1e-10

    @pytest.mark.parametrize(
        ("phases", "match"),
        [
            (np.zeros((3, 1), dtype=np.float64), "phases shape"),
            (np.array([0.0, np.nan], dtype=np.float64), "phases"),
            (np.array([0.0, np.inf], dtype=np.float64), "phases"),
            (np.array([True, False]), "phases"),
        ],
    )
    def test_rejects_invalid_phase_buffers(
        self,
        phases: np.ndarray,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            phase_distance_matrix(phases)


class TestNPE:
    def test_synchronized_low_npe(self):
        phases = np.zeros(10)
        npe = compute_npe(phases)
        assert npe == 0.0

    def test_spread_high_npe(self):
        phases = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        npe = compute_npe(phases)
        assert npe > 0.5

    def test_single_oscillator(self):
        assert compute_npe(np.array([1.0])) == 0.0

    def test_empty(self):
        assert compute_npe(np.array([])) == 0.0

    def test_two_in_phase(self):
        npe = compute_npe(np.array([0.0, 0.0]))
        assert npe == 0.0

    def test_two_anti_phase(self):
        npe = compute_npe(np.array([0.0, np.pi]))
        assert npe == 0.0  # only one lifetime → entropy 0

    def test_range_zero_one(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            phases = rng.uniform(0, 2 * np.pi, 16)
            npe = compute_npe(phases)
            assert 0.0 <= npe <= 1.0

    def test_zero_radius_forces_zero_npe(self):
        phases = np.linspace(0.0, np.pi, 12)
        npe = compute_npe(phases, max_radius=0.0)
        assert npe == 0.0

    def test_more_sync_lower_npe(self):
        rng = np.random.default_rng(42)
        spread = rng.uniform(0, 2 * np.pi, 20)
        tight = rng.normal(1.0, 0.1, 20) % (2 * np.pi)
        npe_spread = compute_npe(spread)
        npe_tight = compute_npe(tight)
        assert npe_tight < npe_spread

    def test_custom_max_radius(self):
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        _npe_full = compute_npe(phases)
        npe_half = compute_npe(phases, max_radius=0.5)
        # Restricting radius may change result
        assert isinstance(npe_half, float)

    @pytest.mark.parametrize(
        ("phases", "match"),
        [
            (np.zeros((3, 1), dtype=np.float64), "phases shape"),
            (np.array([0.0, np.nan], dtype=np.float64), "phases"),
            (np.array([0.0, np.inf], dtype=np.float64), "phases"),
            (np.array([True, False]), "phases"),
        ],
    )
    def test_rejects_invalid_phase_buffers(
        self,
        phases: np.ndarray,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            compute_npe(phases)

    @pytest.mark.parametrize("max_radius", [False, np.nan, np.inf, "1.0", -0.1])
    def test_rejects_invalid_max_radius(self, max_radius: Any) -> None:
        with pytest.raises(ValueError, match="max_radius"):
            compute_npe(np.array([0.0, 1.0, 2.0]), max_radius=max_radius)

    def test_accepts_array_like_phase_buffer(self) -> None:
        npe = compute_npe([0.0, 1.0, 2.0, 3.0], max_radius=1.5)

        assert 0.0 <= npe <= 1.0


class TestNPEPipelineWiring:
    """Pipeline: engine phases → NPE → topological disorder measure."""

    def test_engine_synced_phases_low_npe(self):
        """UPDEEngine with strong coupling → synchronised → NPE low.
        Proves NPE measures disorder from engine output."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.zeros(n)
        knm = 2.0 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        npe = compute_npe(phases)
        assert npe < 0.3, f"Synced engine phases → NPE should be low, got {npe}"


class TestNPERustDispatch:
    def test_phase_distance_uses_backend_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[np.ndarray] = []

        def _fake_pdm(phases: np.ndarray) -> np.ndarray:
            calls.append(phases)
            return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)

        monkeypatch.setattr(
            npe_module,
            "_dispatch",
            lambda fn_name: _fake_pdm if fn_name == "phase_distance_matrix" else None,
        )
        D = phase_distance_matrix(np.array([0.0, 1.0], dtype=np.float64))
        np.testing.assert_allclose(D, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)
        assert len(calls) == 1

    def test_phase_distance_falls_back_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_pdm(_phases: np.ndarray) -> np.ndarray:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            npe_module,
            "_dispatch",
            lambda fn_name: (
                _raising_pdm if fn_name == "phase_distance_matrix" else None
            ),
        )
        phases = np.array([0.1, 2 * np.pi - 0.1], dtype=np.float64)
        D = phase_distance_matrix(phases)
        assert D.shape == (2, 2)
        assert D[0, 1] < 0.3

    def test_compute_npe_uses_backend_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[np.ndarray, float]] = []

        def _fake_npe(phases: np.ndarray, radius: float) -> float:
            calls.append((phases, radius))
            return 0.55

        monkeypatch.setattr(
            npe_module,
            "_dispatch",
            lambda fn_name: _fake_npe if fn_name == "compute_npe" else None,
        )
        npe = compute_npe(np.array([0.0, 1.0, 2.0], dtype=np.float64), max_radius=1.5)
        assert npe == pytest.approx(0.55, abs=1e-12)
        assert len(calls) == 1

    def test_compute_npe_falls_back_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_npe(_phases: np.ndarray, _radius: float) -> float:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            npe_module,
            "_dispatch",
            lambda fn_name: _raising_npe if fn_name == "compute_npe" else None,
        )
        npe = compute_npe(np.array([0.0, 1.0, 2.0], dtype=np.float64), max_radius=1.5)
        assert 0.0 <= npe <= 1.0
