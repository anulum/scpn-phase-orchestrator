# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for hypergraph Kuramoto

"""Cross-backend parity for ``HypergraphEngine.run``.

The five backends share the same sincos-expansion semantics on
the alpha-zero branch and the same direct-``sin(diff)`` form on
the alpha-nonzero branch, so bit-exact parity (``0.0``) is
expected across Rust / Julia / Go / Python. Mojo drifts only by
its subprocess text-round-trip epsilon.
"""

from __future__ import annotations

import contextlib
import math
import sys
import types
from collections.abc import Callable, Iterator
from typing import NoReturn, TypeAlias, cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde import _hypergraph_mojo
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _hypergraph_validation as hypergraph_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._hypergraph_go import (
    hypergraph_run_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._hypergraph_julia import (
    hypergraph_run_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._hypergraph_mojo import (
    hypergraph_run_mojo,
)
from scpn_phase_orchestrator.upde import hypergraph as h_mod
from scpn_phase_orchestrator.upde.hypergraph import HypergraphEngine

TWO_PI = 2.0 * math.pi
TOL = 1e-12
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BackendFn: TypeAlias = Callable[
    [
        FloatArray,
        FloatArray,
        int,
        IntArray,
        IntArray,
        FloatArray,
        FloatArray,
        FloatArray,
        float,
        float,
        float,
        int,
    ],
    FloatArray,
]
BackendArgs: TypeAlias = tuple[
    FloatArray,
    FloatArray,
    int,
    IntArray,
    IntArray,
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    float,
    float,
    int,
]
DIRECT_BACKENDS: tuple[BackendFn, ...] = cast(
    tuple[BackendFn, ...],
    (hypergraph_run_go, hypergraph_run_julia, hypergraph_run_mojo),
)


def test__hypergraph_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(hypergraph_validation.validate_hypergraph_inputs)
    assert callable(hypergraph_validation.validate_hypergraph_output)


@contextlib.contextmanager
def _force_backend(name: str) -> Iterator[None]:
    prev = h_mod.ACTIVE_BACKEND
    h_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        h_mod.ACTIVE_BACKEND = prev


def _problem(seed: int, n: int = 6) -> tuple[FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, TWO_PI, n)
    omega = rng.normal(0.5, 0.2, n)
    knm = rng.uniform(0, 0.3, (n, n))
    np.fill_diagonal(knm, 0.0)
    return theta, omega, knm


def _run_backend(
    backend: str,
    seed: int,
    n: int = 6,
    n_steps: int = 20,
    *,
    alpha: FloatArray | None = None,
    zeta: float = 0.0,
    psi: float = 0.0,
) -> FloatArray:
    if backend not in h_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    theta, omega, knm = _problem(seed, n)
    eng = HypergraphEngine(n, 0.01)
    eng.add_edge((0, 1, 2), strength=0.4)
    eng.add_edge((1, 3, 4, 5), strength=0.25)
    with _force_backend(backend):
        return eng.run(
            theta,
            omega,
            n_steps=n_steps,
            pairwise_knm=knm,
            alpha=alpha,
            zeta=zeta,
            psi=psi,
        )


class TestBackendParity:
    def test_rust_matches_python(self) -> None:
        ref = _run_backend("python", 0)
        got = _run_backend("rust", 0)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia_matches_python(self) -> None:
        ref = _run_backend("python", 1)
        got = _run_backend("julia", 1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go_matches_python(self) -> None:
        ref = _run_backend("python", 2)
        got = _run_backend("go", 2)
        assert np.max(np.abs(got - ref)) < TOL

    def test_mojo_matches_python(self) -> None:
        ref = _run_backend("python", 3, n=6)
        got = _run_backend("mojo", 3, n=6)
        assert np.max(np.abs(got - ref)) < 1e-10


class TestAlphaNonZero:
    """The alpha-nonzero branch uses the direct ``sin(diff)`` form."""

    def _run(self, backend: str, seed: int) -> FloatArray:
        n = 6  # ``_run_backend`` seeds edges up to index 5
        rng = np.random.default_rng(seed + 100)
        theta, omega, knm = _problem(seed, n)
        alpha = rng.uniform(-0.3, 0.3, (n, n))
        np.fill_diagonal(alpha, 0.0)
        return _run_backend(
            backend,
            seed,
            n=n,
            alpha=alpha,
            zeta=0.5,
            psi=1.2,
        )

    def test_rust_alpha(self) -> None:
        ref = self._run("python", 5)
        got = self._run("rust", 5)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go_alpha(self) -> None:
        ref = self._run("python", 6)
        got = self._run("go", 6)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia_alpha(self) -> None:
        ref = self._run("python", 7)
        got = self._run("julia", 7)
        assert np.max(np.abs(got - ref)) < TOL


class TestNoPairwise:
    """Hypergraph-only coupling (no pairwise K) must still agree."""

    def _run(self, backend: str, seed: int) -> FloatArray:
        if backend not in h_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"backend {backend!r} unavailable")
        n = 5
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, TWO_PI, n)
        omega = rng.normal(0.5, 0.2, n)
        eng = HypergraphEngine(n, 0.01)
        eng.add_edge((0, 1, 2), strength=0.5)
        eng.add_edge((2, 3, 4), strength=0.3)
        with _force_backend(backend):
            return eng.run(theta, omega, n_steps=30)

    def test_rust_vs_python(self) -> None:
        ref = self._run("python", 9)
        got = self._run("rust", 9)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go_vs_python(self) -> None:
        ref = self._run("python", 10)
        got = self._run("go", 10)
        assert np.max(np.abs(got - ref)) < TOL


class TestHypothesisParity:
    @given(
        n=st.integers(min_value=6, max_value=10),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rust_hypothesis(self, n: int, seed: int) -> None:
        if "rust" not in h_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        ref = _run_backend("python", seed, n=n)
        got = _run_backend("rust", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL

    @given(
        n=st.integers(min_value=6, max_value=10),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(self, n: int, seed: int) -> None:
        if "go" not in h_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        ref = _run_backend("python", seed, n=n)
        got = _run_backend("go", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL


class TestDispatchFallbackChain:
    def test_load_julia_fn_returns_bridge_when_runtime_gate_passes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def runtime_gate() -> object:
            return object()

        monkeypatch.setattr(h_mod, "require_juliacall_main", runtime_gate)

        assert h_mod._load_julia_fn() is hypergraph_run_julia

    def test_dispatch_falls_back_to_python_when_loader_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = h_mod.ACTIVE_BACKEND
        previous_available = list(h_mod.AVAILABLE_BACKENDS)
        previous_loader = h_mod._LOADERS["go"]
        h_mod.ACTIVE_BACKEND = "go"
        h_mod.AVAILABLE_BACKENDS = ["go", "python"]
        h_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            h_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend = h_mod._dispatch()
        finally:
            h_mod.ACTIVE_BACKEND = previous_backend
            h_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(h_mod._LOADERS, "go", previous_loader)
            h_mod._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_returns_python_when_every_optional_backend_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = h_mod.ACTIVE_BACKEND
        previous_available = list(h_mod.AVAILABLE_BACKENDS)
        previous_loader = h_mod._LOADERS["go"]
        h_mod.ACTIVE_BACKEND = "go"
        h_mod.AVAILABLE_BACKENDS = ["go"]
        h_mod._BACKEND_CACHE.clear()

        def unavailable_loader() -> BackendFn:
            raise ImportError("go backend unavailable")

        monkeypatch.setitem(h_mod._LOADERS, "go", unavailable_loader)
        try:
            backend = h_mod._dispatch()
        finally:
            h_mod.ACTIVE_BACKEND = previous_backend
            h_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(h_mod._LOADERS, "go", previous_loader)
            h_mod._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_uses_cached_loader_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = h_mod.ACTIVE_BACKEND
        previous_available = list(h_mod.AVAILABLE_BACKENDS)
        previous_loader = h_mod._LOADERS["go"]
        h_mod.ACTIVE_BACKEND = "go"
        h_mod.AVAILABLE_BACKENDS = ["go", "python"]
        h_mod._BACKEND_CACHE.clear()
        call_count = 0

        def fake_backend(
            phases: np.ndarray,
            _omegas: np.ndarray,
            _n: int,
            _edge_nodes: np.ndarray,
            _edge_offsets: np.ndarray,
            _edge_strengths: np.ndarray,
            _knm_flat: np.ndarray,
            _alpha_flat: np.ndarray,
            _zeta: float,
            _psi: float,
            _dt: float,
            _n_steps: int,
        ) -> np.ndarray:
            return np.asarray(phases, dtype=np.float64)

        def loader() -> BackendFn:
            nonlocal call_count
            call_count += 1
            return fake_backend

        monkeypatch.setitem(h_mod._LOADERS, "go", loader)
        try:
            b1 = h_mod._dispatch()
            b2 = h_mod._dispatch()
        finally:
            h_mod.ACTIVE_BACKEND = previous_backend
            h_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(h_mod._LOADERS, "go", previous_loader)
            h_mod._BACKEND_CACHE.clear()

        assert b1 is fake_backend
        assert b2 is fake_backend
        assert call_count == 1


def _direct_payload(n: int = 6) -> BackendArgs:
    phases = np.linspace(0.1, 1.1, n, dtype=np.float64)
    omegas = np.linspace(0.2, 0.7, n, dtype=np.float64)
    edge_nodes = np.array([0, 1, 2, 1, 3, 4, 5], dtype=np.int64)
    edge_offsets = np.array([0, 3], dtype=np.int64)
    edge_strengths = np.array([0.4, 0.25], dtype=np.float64)
    knm = np.full((n, n), 0.02, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n), dtype=np.float64)
    return (
        phases,
        omegas,
        n,
        edge_nodes,
        edge_offsets,
        edge_strengths,
        knm.ravel(),
        alpha.ravel(),
        0.1,
        0.3,
        0.01,
        4,
    )


class TestPublicBackendOutputContracts:
    """Public dispatchers validate optional backend output before publication."""

    def test_run_rejects_uncoercible_phase_payload(self) -> None:
        class BadArray:
            def __array__(self, dtype: object | None = None) -> NoReturn:
                raise TypeError("cannot coerce")

        engine = HypergraphEngine(2, 0.01)

        with pytest.raises(ValueError, match="phases must be a finite float array"):
            engine.run(
                cast(FloatArray, BadArray()),
                np.array([0.3, 0.4], dtype=np.float64),
                n_steps=1,
            )

    @pytest.mark.parametrize(
        "backend_output",
        [
            np.array([-0.1, 0.2], dtype=np.float64),
            np.array([0.1, TWO_PI + 1e-6], dtype=np.float64),
        ],
    )
    def test_run_rejects_optional_backend_output_outside_torus(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_output: FloatArray,
    ) -> None:
        def malformed_backend(
            _phases: FloatArray,
            _omegas: FloatArray,
            _n: int,
            _edge_nodes: IntArray,
            _edge_offsets: IntArray,
            _edge_strengths: FloatArray,
            _knm_flat: FloatArray,
            _alpha_flat: FloatArray,
            _zeta: float,
            _psi: float,
            _dt: float,
            _n_steps: int,
        ) -> FloatArray:
            return backend_output

        monkeypatch.setattr(h_mod, "_dispatch", lambda: malformed_backend)
        engine = HypergraphEngine(2, 0.01)
        engine.add_edge((0, 1), strength=0.4)

        with pytest.raises(ValueError, match=r"\[0, 2\*pi\)"):
            engine.run(
                np.array([0.1, 0.2], dtype=np.float64),
                np.array([0.3, 0.4], dtype=np.float64),
                n_steps=1,
            )

    def test_rust_wrapper_rejects_optional_backend_output_outside_torus(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_module = types.ModuleType("spo_kernel")

        def hypergraph_run_rust(
            _phases: FloatArray,
            _omegas: FloatArray,
            n: int,
            _edge_nodes: IntArray,
            _edge_offsets: IntArray,
            _edge_strengths: FloatArray,
            _knm_flat: FloatArray,
            _alpha_flat: FloatArray,
            _zeta: float,
            _psi: float,
            _dt: float,
            _n_steps: int,
        ) -> list[float]:
            return [0.1] * (int(n) - 1) + [TWO_PI + 1e-6]

        fake_module.__dict__["hypergraph_run_rust"] = hypergraph_run_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_module)
        rust_backend = h_mod._load_rust_fn()

        with pytest.raises(ValueError, match=r"\[0, 2\*pi\)"):
            rust_backend(*_direct_payload())


class TestDirectMojoBoundaryContracts:
    """Direct Mojo subprocess parsing validates raw stdout before publication."""

    @pytest.mark.parametrize(
        ("stdout", "match"),
        [
            ("", "Mojo HGRUN returned 0 lines, expected 6"),
            ("0.1\n0.2\n0.3\n0.4\n0.5\n0.6\n0.7\n", "expected 6"),
            ("0.1\n0.2\n\n0.4\n0.5\n0.6\n0.7\n", "expected 6"),
            ("0.1\n0.2\n0.3\nbad\n0.5\n0.6\n", "finite phases"),
            ("0.1\n0.2\n0.3\nnan\n0.5\n0.6\n", "finite phases"),
            ("0.1\n0.2\n0.3\n7.0\n0.5\n0.6\n", "finite phases"),
        ],
    )
    def test_mojo_runner_rejects_malformed_raw_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(_hypergraph_mojo, "_ensure_exe", lambda: "hypergraph")

        class Proc:
            def __init__(self, stdout_value: str) -> None:
                self.returncode = 0
                self.stdout = stdout_value
                self.stderr = ""

        def fake_run(*_args: object, **_kwargs: object) -> Proc:
            return Proc(stdout)

        monkeypatch.setattr(
            cast(object, _hypergraph_mojo.__dict__["subprocess"]),
            "run",
            fake_run,
        )

        with pytest.raises(ValueError, match=match):
            _hypergraph_mojo.hypergraph_run_mojo(*_direct_payload())


class TestDirectBackendBoundaryContracts:
    """Direct Go/Julia/Mojo bridges reject invalid hypergraph payloads early."""

    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    @pytest.mark.parametrize(
        "index,replacement",
        [
            (0, np.array([0.0, np.nan], dtype=np.float64)),
            (1, np.array([0.0, 1.0 + 0.1j], dtype=np.complex128)),
            (2, True),
            (3, np.array([0, 1, 6], dtype=np.int64)),
            (4, np.array([1], dtype=np.int64)),
            (5, np.array([0.2], dtype=np.float64)),
            (6, np.ones(7, dtype=np.float64)),
            (7, np.eye(6, dtype=np.float64).ravel()),
            (8, float("nan")),
            (9, float("inf")),
            (10, 0.0),
            (11, -1),
        ],
    )
    def test_validation_precedes_runtime_load(
        self,
        backend: BackendFn,
        index: int,
        replacement: object,
    ) -> None:
        args = list(_direct_payload())
        args[index] = replacement
        with pytest.raises(ValueError):
            backend(*cast(BackendArgs, tuple(args)))

    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    def test_rejects_duplicate_nodes_in_one_hyperedge(
        self,
        backend: BackendFn,
    ) -> None:
        args = list(_direct_payload())
        args[3] = np.array([0, 1, 1], dtype=np.int64)
        args[4] = np.array([0], dtype=np.int64)
        args[5] = np.array([0.4], dtype=np.float64)

        with pytest.raises(ValueError, match="repeat nodes"):
            backend(*cast(BackendArgs, tuple(args)))

    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    def test_zero_step_normalises_without_optional_runtime(
        self,
        backend: BackendFn,
    ) -> None:
        args = list(_direct_payload())
        raw_phases = np.array([-0.25, 0.0, TWO_PI + 0.2, 9.0, 1.5, 2.0])
        args[0] = raw_phases
        args[11] = 0

        got = backend(*cast(BackendArgs, tuple(args)))

        np.testing.assert_allclose(got, np.mod(raw_phases, TWO_PI))
