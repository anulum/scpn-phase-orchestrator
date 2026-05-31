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
            (np.array([0.0, np.bool_(True)], dtype=object), "phases"),
            (np.array([0.0 + 0.0j, 1.0 + 0.5j]), "real-valued"),
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
            (np.array([0.0, np.bool_(True)], dtype=object), "phases"),
            (np.array([0.0 + 0.0j, 1.0 + 0.5j]), "real-valued"),
        ],
    )
    def test_rejects_invalid_phase_buffers(
        self,
        phases: np.ndarray,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            compute_npe(phases)

    @pytest.mark.parametrize(
        "max_radius",
        [False, np.bool_(False), np.nan, np.inf, "1.0", 0.5 + 0.0j, -0.1, np.pi + 1e-6],
    )
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
        phases = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        radius = 1.5
        expected = npe_module._compute_npe_reference(phases, radius)

        def _fake_npe(phases: np.ndarray, radius: float) -> float:
            calls.append((phases, radius))
            return expected

        monkeypatch.setattr(
            npe_module,
            "_dispatch",
            lambda fn_name: _fake_npe if fn_name == "compute_npe" else None,
        )
        npe = compute_npe(phases, max_radius=radius)
        assert npe == pytest.approx(expected, abs=1e-12)
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

    def test_dispatch_falls_back_to_python_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = npe_module.ACTIVE_BACKEND
        previous_available = list(npe_module.AVAILABLE_BACKENDS)
        previous_loader = npe_module._LOADERS["go"]
        npe_module.ACTIVE_BACKEND = "go"
        npe_module.AVAILABLE_BACKENDS = ["go", "python"]
        npe_module._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            npe_module._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            fn = npe_module._dispatch("compute_npe")
        finally:
            npe_module.ACTIVE_BACKEND = previous_backend
            npe_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(npe_module._LOADERS, "go", previous_loader)
            npe_module._BACKEND_CACHE.clear()

        assert fn is None

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = npe_module.ACTIVE_BACKEND
        previous_available = list(npe_module.AVAILABLE_BACKENDS)
        previous_loader = npe_module._LOADERS["go"]
        npe_module.ACTIVE_BACKEND = "go"
        npe_module.AVAILABLE_BACKENDS = ["go", "python"]
        npe_module._BACKEND_CACHE.clear()
        call_count = 0

        def fake_compute_npe(_phases: np.ndarray, _radius: float) -> float:
            return 0.0

        def loader() -> dict[str, object]:
            nonlocal call_count
            call_count += 1
            return {"compute_npe": fake_compute_npe}

        monkeypatch.setitem(npe_module._LOADERS, "go", loader)
        try:
            fn1 = npe_module._dispatch("compute_npe")
            fn2 = npe_module._dispatch("compute_npe")
        finally:
            npe_module.ACTIVE_BACKEND = previous_backend
            npe_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(npe_module._LOADERS, "go", previous_loader)
            npe_module._BACKEND_CACHE.clear()

        assert fn1 is fake_compute_npe
        assert fn2 is fake_compute_npe
        assert call_count == 1


# Salvaged module-specific behavioural contracts from deleted sprint file.
class TestNPEPhysicsContracts:
    """Verify NPE (normalised phase entropy) satisfies
    information-theoretic bounds."""

    def test_identical_phases_zero_entropy(self):
        """All phases identical → no disorder → NPE = 0."""
        assert compute_npe(np.array([1.0, 1.0, 1.0])) == 0.0

    def test_two_phases_zero(self):
        assert compute_npe(np.array([0.0, 1.0])) == 0.0

    def test_uniform_phases_high_entropy(self):
        """Uniformly spread phases → maximum disorder → NPE near 1."""
        phases = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        npe = compute_npe(phases)
        assert npe > 0.8, f"Uniform phases should give NPE near 1, got {npe:.3f}"

    def test_npe_bounded_zero_to_one(self):
        """NPE must always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            phases = rng.uniform(0, 2 * np.pi, rng.integers(3, 100))
            npe = compute_npe(phases)
            assert 0.0 <= npe <= 1.0 + 1e-10, f"NPE={npe} outside [0,1]"


# ---------------------------------------------------------------------------
# PGBO: phase-geometry-binding observer
# ---------------------------------------------------------------------------


class TestNPEBoundaryHardening:
    def test_phase_distance_rejects_mixed_boolean_aliases(self) -> None:
        with pytest.raises(ValueError, match="phases"):
            phase_distance_matrix(np.array([0.0, True], dtype=object))

    @pytest.mark.parametrize(
        "backend_output",
        [
            np.array([0.0, 1.0, 1.0], dtype=np.float64),
            np.array([[0.0, np.nan], [np.nan, 0.0]], dtype=np.float64),
            np.array([[0.0, 4.0], [4.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.1], [0.2, 0.0]], dtype=np.float64),
            np.array([[0.1, 0.2], [0.2, 0.0]], dtype=np.float64),
            np.array([[False, True], [True, False]], dtype=np.bool_),
            np.array([[0.0, np.bool_(True)], [1.0, 0.0]], dtype=object),
        ],
    )
    def test_invalid_backend_matrix_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_output: np.ndarray,
    ) -> None:
        previous_backend = npe_module.ACTIVE_BACKEND
        previous_available = list(npe_module.AVAILABLE_BACKENDS)
        previous_loader = npe_module._LOADERS["go"]

        def fake_phase_distance_matrix(*_args: object, **_kwargs: object) -> np.ndarray:
            return backend_output

        npe_module.ACTIVE_BACKEND = "go"
        npe_module.AVAILABLE_BACKENDS = ["go", "python"]
        npe_module._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            npe_module._LOADERS,
            "go",
            lambda: {"phase_distance_matrix": fake_phase_distance_matrix},
        )
        try:
            distances = phase_distance_matrix(np.array([0.0, np.pi], dtype=np.float64))
        finally:
            npe_module.ACTIVE_BACKEND = previous_backend
            npe_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(npe_module._LOADERS, "go", previous_loader)
            npe_module._BACKEND_CACHE.clear()

        np.testing.assert_allclose(distances, [[0.0, np.pi], [np.pi, 0.0]])

    @pytest.mark.parametrize("backend_value", [-0.1, 1.1, np.nan, np.inf, True])
    def test_invalid_backend_score_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_value: Any,
    ) -> None:
        previous_backend = npe_module.ACTIVE_BACKEND
        previous_available = list(npe_module.AVAILABLE_BACKENDS)
        previous_loader = npe_module._LOADERS["go"]
        phases = np.array([0.0, 0.5, 1.5], dtype=np.float64)

        def fake_compute_npe(*_args: object, **_kwargs: object) -> Any:
            return backend_value

        npe_module.ACTIVE_BACKEND = "python"
        expected = compute_npe(phases)
        npe_module.ACTIVE_BACKEND = "go"
        npe_module.AVAILABLE_BACKENDS = ["go", "python"]
        npe_module._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            npe_module._LOADERS,
            "go",
            lambda: {"compute_npe": fake_compute_npe},
        )
        try:
            got = compute_npe(phases)
        finally:
            npe_module.ACTIVE_BACKEND = previous_backend
            npe_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(npe_module._LOADERS, "go", previous_loader)
            npe_module._BACKEND_CACHE.clear()

        assert got == expected
