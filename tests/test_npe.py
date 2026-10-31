# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — NPE tests

from __future__ import annotations

import math
from typing import Any, get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import npe as npe_module
from scpn_phase_orchestrator.monitor.npe import compute_npe, phase_distance_matrix
from tests._runtime_owners import owner_params
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
            (np.array([0.0, 1.0 + 0.5j], dtype=object), "real-valued"),
            (np.array(["0.0", "1.0"]), "numeric-string"),
            (np.array([0.0, "1.0"], dtype=object), "numeric-string"),
            (np.array(["not-number"]), "float array"),
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
            (np.array([0.0, 1.0 + 0.5j], dtype=object), "real-valued"),
            (np.array(["0.0", "1.0"]), "numeric-string"),
            (np.array([0.0, "1.0"], dtype=object), "numeric-string"),
            (np.array(["not-number"]), "float array"),
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


def _circular_distance(a: float, b: float) -> float:
    """Return the shortest arc between two angles."""
    gap = abs(a - b) % (2 * math.pi)
    return min(gap, 2 * math.pi - gap)


def _oracle_npe(phases: list[float], radius: float) -> float:
    """Return the H0 persistent entropy from Prim's minimum spanning tree.

    Single-linkage H0 lifetimes are the MST edge weights; edges longer than
    ``radius`` never merge within the filtration and are dropped.
    """
    n = len(phases)
    in_tree = [False] * n
    best = [math.inf] * n
    best[0] = 0.0
    lifetimes = []
    for _ in range(n):
        u = min((i for i in range(n) if not in_tree[i]), key=lambda i: best[i])
        if any(in_tree):
            lifetimes.append(best[u])
        in_tree[u] = True
        for v in range(n):
            if not in_tree[v]:
                best[v] = min(best[v], _circular_distance(phases[u], phases[v]))
    kept = [life for life in lifetimes if life <= radius]
    total = math.fsum(kept)
    if not kept or total < 1e-15:
        return 0.0
    probs = [life / total for life in kept if life > 0]
    if len(probs) < 2:
        return 0.0
    return -math.fsum(p * math.log(p) for p in probs) / math.log(len(probs))


class TestNPERustDispatch:
    @pytest.mark.parametrize(
        "owner", owner_params(["python", "rust", "mojo", "julia", "go"])
    )
    def test_phase_distance_uses_backend_when_available(self, owner: str) -> None:
        """Each named owner returns the circular distances of the oracle."""
        if owner not in npe_module.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        assert npe_module.npe_backend("phase_distance_matrix", owner) == owner
        phases = np.array([0.1, 2 * np.pi - 0.1, 3.0, -2.5, 6.0])
        got = phase_distance_matrix(phases, backend=owner)
        expected = [
            [_circular_distance(a, b) for b in phases.tolist()] for a in phases.tolist()
        ]
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)

    def test_phase_distance_falls_back_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_pdm(_phases: np.ndarray) -> np.ndarray:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            npe_module,
            "_dispatch",
            lambda fn_name, _backend="auto": (
                ("go", _raising_pdm)
                if fn_name == "phase_distance_matrix"
                else ("python", None)
            ),
        )
        phases = np.array([0.1, 2 * np.pi - 0.1], dtype=np.float64)
        D = phase_distance_matrix(phases)
        assert D.shape == (2, 2)
        assert D[0, 1] < 0.3

    @pytest.mark.parametrize(
        "owner", owner_params(["python", "rust", "mojo", "julia", "go"])
    )
    def test_compute_npe_uses_backend_when_available(self, owner: str) -> None:
        """Each named owner returns the Prim-MST persistent entropy of the oracle."""
        if owner not in npe_module.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        assert npe_module.npe_backend("compute_npe", owner) == owner
        rng = np.random.default_rng(41)
        for radius in (np.pi, 1.0, 0.3):
            phases = rng.uniform(0.0, 2 * np.pi, 9)
            got = compute_npe(phases, max_radius=radius, backend=owner)
            # The Mojo text bridge carries about 1e-11; the module admits 1e-9.
            assert got == pytest.approx(
                _oracle_npe(phases.tolist(), radius),
                abs=1e-9 if owner == "mojo" else 1e-12,
            )

    def test_named_owner_refusals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Absent, failing or incomplete named owners and unknown names refuse."""

        def broken_loader() -> dict[str, object]:
            """Fail like a missing shared library."""
            raise OSError("libnpe.so missing")

        def crash(*_args: object) -> object:
            """Fail like a crashed kernel."""
            raise RuntimeError("kernel crashed")

        phases = np.array([0.0, 1.0, 2.0])
        monkeypatch.setattr(npe_module, "_BACKEND_CACHE", {})
        monkeypatch.setitem(npe_module._LOADERS, "go", broken_loader)
        with pytest.raises(RuntimeError, match="NPE backend 'go' unavailable"):
            compute_npe(phases, backend="go")
        monkeypatch.setitem(
            npe_module._LOADERS, "go", lambda: {"phase_distance_matrix": crash}
        )
        npe_module._BACKEND_CACHE.clear()
        with pytest.raises(RuntimeError, match="NPE backend 'go' failed"):
            phase_distance_matrix(phases, backend="go")
        with pytest.raises(RuntimeError, match="NPE backend 'go' lacks compute_npe"):
            compute_npe(phases, backend="go")
        monkeypatch.setitem(npe_module._LOADERS, "go", lambda: {"compute_npe": crash})
        npe_module._BACKEND_CACHE.clear()
        with pytest.raises(RuntimeError, match="NPE backend 'go' failed"):
            compute_npe(phases, backend="go")
        with pytest.raises(ValueError, match="unknown NPE backend"):
            phase_distance_matrix(phases, backend="cuda")
        with pytest.raises(ValueError, match="unknown NPE operation"):
            npe_module.npe_backend("betti")

    def test_compute_npe_falls_back_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_npe(_phases: np.ndarray, _radius: float) -> float:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            npe_module,
            "_dispatch",
            lambda fn_name, _backend="auto": (
                ("go", _raising_npe) if fn_name == "compute_npe" else ("python", None)
            ),
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

        assert fn == ("python", None)

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

        assert fn1 == ("go", fake_compute_npe)
        assert fn2 == ("go", fake_compute_npe)
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
            np.array([["0.0", str(np.pi)], [str(np.pi), "0.0"]]),
            np.array([[0.0, 1.0j], [1.0j, 0.0]], dtype=object),
            np.array([["not-number", "bad"], ["bad", "not-number"]]),
        ],
    )
    def test_invalid_backend_matrix_fails_closed(
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
            with pytest.raises(ValueError):
                phase_distance_matrix(np.array([0.0, np.pi], dtype=np.float64))
        finally:
            npe_module.ACTIVE_BACKEND = previous_backend
            npe_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(npe_module._LOADERS, "go", previous_loader)
            npe_module._BACKEND_CACHE.clear()

    def test_load_julia_fns_exports_bridge_callables(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(npe_module, "require_juliacall_main", lambda: None)

        loaded = npe_module._load_julia_fns()

        assert callable(loaded["phase_distance_matrix"])
        assert callable(loaded["compute_npe"])

    def test_dispatch_skips_backend_missing_requested_callable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = npe_module.ACTIVE_BACKEND
        previous_available = list(npe_module.AVAILABLE_BACKENDS)
        previous_loader = npe_module._LOADERS["go"]

        npe_module.ACTIVE_BACKEND = "go"
        npe_module.AVAILABLE_BACKENDS = ["go"]
        npe_module._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            npe_module._LOADERS,
            "go",
            lambda: {"phase_distance_matrix": lambda _phases: np.zeros((1, 1))},
        )
        try:
            assert npe_module._dispatch("compute_npe") == ("python", None)
        finally:
            npe_module.ACTIVE_BACKEND = previous_backend
            npe_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(npe_module._LOADERS, "go", previous_loader)
            npe_module._BACKEND_CACHE.clear()

    def test_distance_matrix_validator_rejects_wrong_expected_shape(self) -> None:
        with pytest.raises(ValueError, match="expected output shape"):
            npe_module._validate_distance_matrix(
                np.zeros((2, 2), dtype=np.float64),
                n_phases=2,
                expected=np.zeros((3,), dtype=np.float64),
            )

    def test_private_npe_distance_entropy_degenerate_cases(self) -> None:
        assert (
            npe_module._npe_from_distance_matrix(
                np.zeros((1, 1), dtype=np.float64),
                np.pi,
            )
            == 0.0
        )
        assert (
            npe_module._npe_from_distance_matrix(
                np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64),
                0.5,
            )
            == 0.0
        )
        assert (
            npe_module._npe_from_distance_matrix(
                np.array([[0.0, 1.0e-20], [1.0e-20, 0.0]], dtype=np.float64),
                np.pi,
            )
            == 0.0
        )

    def test_private_npe_distance_entropy_rejects_zero_max_entropy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_log = np.log

        def fake_log(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return original_log(value)
            return 0.0

        monkeypatch.setattr(npe_module.np, "log", fake_log)

        assert (
            npe_module._npe_from_distance_matrix(
                np.array(
                    [
                        [0.0, 0.25, 0.5],
                        [0.25, 0.0, 0.75],
                        [0.5, 0.75, 0.0],
                    ],
                    dtype=np.float64,
                ),
                np.pi,
            )
            == 0.0
        )

    @pytest.mark.parametrize("backend_value", [-0.1, 1.1, np.nan, np.inf, True])
    def test_invalid_backend_score_fails_closed(
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

        npe_module.ACTIVE_BACKEND = "go"
        npe_module.AVAILABLE_BACKENDS = ["go", "python"]
        npe_module._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            npe_module._LOADERS,
            "go",
            lambda: {"compute_npe": fake_compute_npe},
        )
        try:
            with pytest.raises(ValueError):
                compute_npe(phases)
        finally:
            npe_module.ACTIVE_BACKEND = previous_backend
            npe_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(npe_module._LOADERS, "go", previous_loader)
            npe_module._BACKEND_CACHE.clear()
