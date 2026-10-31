# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for torus integrator

"""Cross-backend parity for ``TorusEngine.run``.

Tolerances
----------
The alpha-zero branch uses the sincos expansion and agrees
bit-exactly on Rust / Julia / Go / Python. The alpha-nonzero
branch routes through ``atan2`` per step, which introduces
identity-wise ``~1e-15`` drift across backends — still tight
enough for sub-``1e-12`` agreement after 50 steps.
"""

from __future__ import annotations

import cmath
import contextlib
import math
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _geometric_julia,
    _geometric_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._geometric_go import (
    torus_run_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._geometric_julia import (
    torus_run_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._geometric_mojo import (
    torus_run_mojo,
)
from scpn_phase_orchestrator.upde import (
    _geometric_validation as geometric_validation,
)
from scpn_phase_orchestrator.upde import geometric as g_mod
from scpn_phase_orchestrator.upde.geometric import TorusEngine

TWO_PI = 2.0 * math.pi
TOL_EXPANSION = 1e-12
TOL_ATAN2 = 1e-10
DirectBackend = Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        float,
        float,
        float,
        int,
    ],
    np.ndarray,
]
DIRECT_BACKENDS = (torus_run_go, torus_run_julia, torus_run_mojo)


def test__geometric_validation_helper_is_directly_linked_to_backend_tests() -> None:
    """The shared validation helper module is exercised directly by these tests."""
    assert callable(geometric_validation.validate_torus_inputs)
    assert callable(geometric_validation.validate_torus_output)


@contextlib.contextmanager
def _force_backend(name: str) -> Iterator[None]:
    prev = g_mod.ACTIVE_BACKEND
    g_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        g_mod.ACTIVE_BACKEND = prev


_FloatArray = NDArray[np.float64]


def _problem(
    seed: int, n: int = 6, alpha_nonzero: bool = False
) -> tuple[_FloatArray, _FloatArray, _FloatArray, _FloatArray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, TWO_PI, n)
    omegas = rng.normal(1.0, 0.2, n)
    knm = rng.uniform(0, 0.3, (n, n))
    np.fill_diagonal(knm, 0.0)
    if alpha_nonzero:
        alpha = rng.uniform(-0.3, 0.3, (n, n))
        np.fill_diagonal(alpha, 0.0)
    else:
        alpha = np.zeros((n, n))
    return theta, omegas, knm, alpha


def _direct_payload(
    n: int = 5,
) -> tuple[
    _FloatArray, _FloatArray, _FloatArray, _FloatArray, int, float, float, float, int
]:
    phases, omegas, knm, alpha = _problem(11, n=n, alpha_nonzero=True)
    return phases, omegas, knm.ravel(), alpha.ravel(), n, 0.2, 0.7, 0.01, 4


def _run_backend(
    backend: str,
    seed: int,
    n: int = 6,
    n_steps: int = 50,
    zeta: float = 0.0,
    psi: float = 0.0,
    alpha_nonzero: bool = False,
) -> _FloatArray:
    if backend not in g_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    theta, omegas, knm, alpha = _problem(seed, n, alpha_nonzero)
    eng = TorusEngine(n, 0.01)
    with _force_backend(backend):
        return eng.run(theta, omegas, knm, zeta, psi, alpha, n_steps=n_steps)


class TestDirectMojoBoundaryContracts:
    """The direct Mojo runner validates its raw output."""

    @pytest.mark.parametrize(
        ("stdout", "match"),
        [
            ("", "Mojo TORUS returned 0 lines, expected 5"),
            ("0.1\n0.2\n0.3\n0.4\n0.5\n0.6\n", "expected 5"),
            ("0.1\n0.2\n\n0.4\n0.5\n0.6\n", "expected 5"),
            ("0.1\n0.2\nbad\n0.4\n0.5\n", "finite phases"),
            ("0.1\n0.2\nnan\n0.4\n0.5\n", "finite phases"),
            ("0.1\n0.2\n7.0\n0.4\n0.5\n", "finite phases"),
        ],
    )
    def test_mojo_runner_rejects_malformed_raw_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        match: str,
    ) -> None:
        """Malformed Mojo executable output is refused."""
        monkeypatch.setattr(_geometric_mojo, "_ensure_exe", lambda: "geometric")
        # The runner calls run through the shared subprocess module, so
        # patching that module patches what the runner resolves.
        assert vars(_geometric_mojo)["subprocess"] is subprocess
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *_args, **_kwargs: type(
                "Proc",
                (),
                {"returncode": 0, "stdout": stdout, "stderr": ""},
            )(),
        )

        with pytest.raises(ValueError, match=match):
            _geometric_mojo.torus_run_mojo(*_direct_payload())


class TestDirectBackendBoundaryContracts:
    """Direct Go/Julia/Mojo adapters validate inputs before loading runtimes."""

    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    @pytest.mark.parametrize(
        ("index", "replacement"),
        [
            (0, lambda payload: payload[0].reshape(1, -1)),
            (0, lambda payload: payload[0].astype(bool)),
            (0, lambda payload: payload[0].astype(np.complex128) + 1j),
            (0, lambda payload: np.array([np.nan, *payload[0][1:]])),
            (1, lambda payload: payload[1][:-1]),
            (1, lambda payload: payload[1].astype(bool)),
            (1, lambda payload: np.array([np.inf, *payload[1][1:]])),
            (2, lambda payload: payload[2][:-1]),
            (2, lambda payload: payload[2].astype(bool)),
            (2, lambda payload: payload[2].astype(np.complex128) + 1j),
            (3, lambda payload: payload[3][:-1]),
            (3, lambda payload: payload[3].astype(bool)),
            (3, lambda payload: np.full_like(payload[3], np.nan)),
            (4, lambda payload: True),
            (4, lambda payload: 0),
            (4, lambda payload: payload[4] + 1),
            (5, lambda payload: True),
            (5, lambda payload: float("nan")),
            (6, lambda payload: float("inf")),
            (7, lambda payload: 0.0),
            (7, lambda payload: True),
            (8, lambda payload: -1),
            (8, lambda payload: True),
        ],
    )
    def test_invalid_inputs_fail_before_optional_runtime_loading(
        self,
        backend: DirectBackend,
        index: int,
        replacement: Callable[[tuple[object, ...]], object],
    ) -> None:
        """Direct Go/Julia/Mojo wrappers share the torus-run contract."""

        # The payload is deliberately corrupted, so the backend is called
        # through an unchecked alias.
        backend_unchecked: Callable[..., object] = backend
        payload = list(_direct_payload())
        payload[index] = replacement(tuple(payload))
        with pytest.raises((TypeError, ValueError)):
            backend_unchecked(*payload)

    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    @pytest.mark.parametrize(
        ("index", "replacement"),
        [
            (0, lambda payload: np.array(["0.1"] * payload[4], dtype=object)),
            (1, lambda payload: np.array(["1.0"] * payload[4], dtype=object)),
            (
                2,
                lambda payload: np.array(
                    ["0.0"] * (payload[4] * payload[4]), dtype=object
                ),
            ),
            (
                3,
                lambda payload: np.array(
                    ["0.0"] * (payload[4] * payload[4]), dtype=object
                ),
            ),
            (4, lambda _payload: "5"),
            (5, lambda _payload: "0.2"),
            (6, lambda _payload: "0.7"),
            (7, lambda _payload: "0.01"),
            (8, lambda _payload: "4"),
        ],
    )
    def test_numeric_string_aliases_fail_before_optional_runtime_loading(
        self,
        backend: DirectBackend,
        index: int,
        replacement: Callable[[tuple[object, ...]], object],
    ) -> None:
        """Reject direct Go/Julia/Mojo numeric-string aliases before loading."""
        # The payload is deliberately corrupted, so the backend is called
        # through an unchecked alias.
        backend_unchecked: Callable[..., object] = backend
        payload = list(_direct_payload())
        payload[index] = replacement(tuple(payload))

        with pytest.raises(ValueError, match="numeric-string"):
            backend_unchecked(*payload)

    def test_validation_rejects_numeric_string_backend_output(self) -> None:
        """Reject raw numeric-string phase output before float coercion."""
        with pytest.raises(ValueError, match="numeric-string"):
            geometric_validation.validate_torus_output(
                np.array(["0.1", "0.2", "0.3"], dtype=object),
                n=3,
            )

    def test_validation_preserves_real_numeric_object_output(self) -> None:
        """Preserve valid real numeric object values during normalisation."""
        result = geometric_validation.validate_torus_output(
            np.array([np.float64(0.1), np.int64(1)], dtype=object),
            n=2,
        )

        assert result.dtype == np.float64
        assert result.tolist() == [0.1, 1.0]

    def test_numeric_string_helper_rejects_blank_and_bad_array_aliases(self) -> None:
        """Classify only coercible string scalars as numeric-string aliases."""

        class BadArray:
            """Array-like object that refuses NumPy conversion."""

            def __array__(self) -> object:
                """Raise during NumPy conversion."""
                raise TypeError("no array conversion")

        assert not geometric_validation._is_numeric_string_alias("   ")
        assert geometric_validation._contains_numeric_string_alias("0.25")
        assert not geometric_validation._contains_numeric_string_alias(BadArray())

    @pytest.mark.parametrize(
        ("index", "replacement", "exc_type", "match"),
        [
            (
                0,
                lambda _payload: np.array(["bad", "bad", "bad", "bad", "bad"]),
                TypeError,
                "phases must be numeric",
            ),
            (
                0,
                lambda _payload: np.array([], dtype=np.float64),
                ValueError,
                "phases must contain at least one oscillator",
            ),
            (4, lambda _payload: 1.5, TypeError, "n must be an integer"),
            (5, lambda _payload: object(), TypeError, "zeta must be a real scalar"),
        ],
    )
    def test_validation_rejects_non_alias_malformed_inputs(
        self,
        index: int,
        replacement: Callable[[tuple[object, ...]], object],
        exc_type: type[Exception],
        match: str,
    ) -> None:
        """Keep non-alias validation branches explicit after alias hardening."""
        # The payload is deliberately corrupted, so the validator is called
        # through an unchecked alias.
        validate_unchecked: Callable[..., object] = (
            geometric_validation.validate_torus_inputs
        )
        payload = list(_direct_payload())
        payload[index] = replacement(tuple(payload))

        with pytest.raises(exc_type, match=match):
            validate_unchecked(*payload)

    def test_julia_loader_returns_cached_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Return the cached Julia module without probing the runtime."""
        cached = object()
        monkeypatch.setattr(_geometric_julia, "_JULIA_MODULE", cached)

        assert _geometric_julia._ensure() is cached

    def test_julia_loader_rejects_missing_side_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Reject missing Julia side files after runtime acquisition."""
        missing_file = tmp_path / "missing_geometric.jl"
        monkeypatch.setattr(_geometric_julia, "_JULIA_MODULE", None)
        monkeypatch.setattr(_geometric_julia, "_JULIA_FILE", missing_file)
        monkeypatch.setattr(_geometric_julia, "require_julia_main", lambda: object())

        with pytest.raises(ImportError, match="julia side-file not found"):
            _geometric_julia._ensure()

    def test_julia_loader_includes_side_file_and_caches_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Include the Julia side file and cache the exported module handle."""

        class FakeJuliaMain:
            """Minimal Julia runtime stand-in with an exported module."""

            GeometricJL = object()

            def __init__(self) -> None:
                self.included: str | None = None

            def include(self, path: str) -> None:
                """Record the included Julia source path."""
                self.included = path

        side_file = tmp_path / "geometric.jl"
        side_file.write_text("module GeometricJL\nend\n", encoding="utf-8")
        fake_main = FakeJuliaMain()
        monkeypatch.setattr(_geometric_julia, "_JULIA_MODULE", None)
        monkeypatch.setattr(_geometric_julia, "_JULIA_FILE", side_file)
        monkeypatch.setattr(
            _geometric_julia,
            "require_julia_main",
            lambda: fake_main,
        )

        assert _geometric_julia._ensure() is fake_main.GeometricJL
        assert fake_main.included == str(side_file)

    def test_julia_output_validation_rejects_raw_numeric_string_aliases(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject Julia return values that arrive as numeric-string aliases."""

        class FakeJuliaModule:
            """Minimal Julia module stand-in returning malformed torus output."""

            def torus_run(self, *_args: object) -> np.ndarray:
                """Return a raw numeric-string vector."""
                return np.array(["0.1", "0.2", "0.3", "0.4", "0.5"], dtype=object)

        monkeypatch.setattr(_geometric_julia, "_ensure", FakeJuliaModule)

        with pytest.raises(ValueError, match="numeric-string"):
            _geometric_julia.torus_run_julia(*_direct_payload())

    @pytest.mark.parametrize(
        ("raw_result", "match"),
        [
            (np.array([False] * 5), "boolean"),
            (np.array([0.1 + 0.2j] * 5), "complex"),
        ],
    )
    def test_julia_output_validation_rejects_raw_coercive_aliases(
        self,
        monkeypatch: pytest.MonkeyPatch,
        raw_result: np.ndarray,
        match: str,
    ) -> None:
        """Reject Julia source types before compatibility coercion."""

        class FakeJuliaModule:
            """Minimal Julia module stand-in returning coercive torus output."""

            def torus_run(self, *_args: object) -> np.ndarray:
                """Return the configured raw backend payload."""
                return raw_result

        monkeypatch.setattr(_geometric_julia, "_ensure", FakeJuliaModule)

        with pytest.raises(TypeError, match=match):
            _geometric_julia.torus_run_julia(*_direct_payload())

    def test_julia_output_validation_accepts_numeric_vectors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Return validated Julia phase vectors after compatibility coercion."""

        class FakeJuliaModule:
            """Minimal Julia module stand-in returning valid torus output."""

            def torus_run(self, *_args: object) -> list[float]:
                """Return a valid finite torus phase vector."""
                return [0.1, 0.2, 0.3, 0.4, 0.5]

        monkeypatch.setattr(_geometric_julia, "_ensure", FakeJuliaModule)

        np.testing.assert_allclose(
            _geometric_julia.torus_run_julia(*_direct_payload()),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
        )

    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    def test_zero_steps_normalises_phases_without_optional_runtime(
        self,
        backend: DirectBackend,
    ) -> None:
        """Zero steps only wraps the phases, without loading a runtime."""
        _phases, omegas, knm, alpha, n, zeta, psi, dt, _n_steps = _direct_payload()
        phases = np.array([-0.25, 0.0, TWO_PI + 0.5, 8.0, -TWO_PI])
        np.testing.assert_allclose(
            backend(phases, omegas, knm, alpha, n, zeta, psi, dt, 0),
            phases % TWO_PI,
            atol=0.0,
        )


class TestAlphaZero:
    """Owners agree with the NumPy law without phase lags."""

    def test_rust(self) -> None:
        """Rust agrees with the NumPy law."""
        ref = _run_backend("python", 0)
        got = _run_backend("rust", 0)
        assert np.max(np.abs(got - ref)) < TOL_EXPANSION

    def test_julia(self) -> None:
        """Julia agrees with the NumPy law."""
        ref = _run_backend("python", 1)
        got = _run_backend("julia", 1)
        assert np.max(np.abs(got - ref)) < TOL_EXPANSION

    def test_go(self) -> None:
        """Go agrees with the NumPy law."""
        ref = _run_backend("python", 2)
        got = _run_backend("go", 2)
        assert np.max(np.abs(got - ref)) < TOL_EXPANSION

    def test_mojo(self) -> None:
        """Mojo agrees with the NumPy law."""
        ref = _run_backend("python", 3, n=5)
        got = _run_backend("mojo", 3, n=5)
        assert np.max(np.abs(got - ref)) < TOL_ATAN2

    def test_mojo_wraps_negative_zero_roundoff_to_zero(self) -> None:
        """Mojo wraps a negative round-off phase to zero."""
        if "mojo" not in g_mod.AVAILABLE_BACKENDS:
            pytest.skip("backend 'mojo' unavailable")
        phases = np.array([0.5, 2.0, 5.0])
        omegas = np.array([1.0, -2.0, 10.0])
        zeros = np.zeros(9)

        got = _geometric_mojo.torus_run_mojo(
            phases,
            omegas,
            zeros,
            zeros,
            3,
            0.0,
            0.0,
            0.005,
            200,
        )
        expected = (phases + omegas) % TWO_PI
        circular_error = np.abs(got - expected)
        circular_error = np.minimum(circular_error, TWO_PI - circular_error)

        assert np.all((got >= 0.0) & (got < TWO_PI))
        np.testing.assert_allclose(circular_error, 0.0, atol=TOL_ATAN2)


class TestAlphaNonZero:
    """Owners agree with the NumPy law with phase lags."""

    def test_rust(self) -> None:
        """Rust agrees with the NumPy law."""
        ref = _run_backend("python", 4, alpha_nonzero=True, zeta=0.5, psi=1.2)
        got = _run_backend("rust", 4, alpha_nonzero=True, zeta=0.5, psi=1.2)
        assert np.max(np.abs(got - ref)) < TOL_ATAN2

    def test_julia(self) -> None:
        """Julia agrees with the NumPy law."""
        ref = _run_backend("python", 5, alpha_nonzero=True, zeta=0.5, psi=1.2)
        got = _run_backend("julia", 5, alpha_nonzero=True, zeta=0.5, psi=1.2)
        assert np.max(np.abs(got - ref)) < TOL_ATAN2

    def test_go(self) -> None:
        """Go agrees with the NumPy law."""
        ref = _run_backend("python", 6, alpha_nonzero=True, zeta=0.5, psi=1.2)
        got = _run_backend("go", 6, alpha_nonzero=True, zeta=0.5, psi=1.2)
        assert np.max(np.abs(got - ref)) < TOL_ATAN2


class TestHypothesisParity:
    """Seeded random problems agree across owners."""

    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rust_hypothesis(self, n: int, seed: int) -> None:
        """Rust agrees with the NumPy law on random problems."""
        if "rust" not in g_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        ref = _run_backend("python", seed, n=n)
        got = _run_backend("rust", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL_EXPANSION

    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(self, n: int, seed: int) -> None:
        """Go agrees with the NumPy law on random problems."""
        if "go" not in g_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        ref = _run_backend("python", seed, n=n)
        got = _run_backend("go", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL_EXPANSION


def _oracle_torus(
    phases: list[float],
    omegas: list[float],
    knm: list[list[float]],
    alpha: list[list[float]],
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> list[float]:
    """Rotate unit phasors by the Kuramoto rate with ``cmath``, in plain loops."""
    n = len(phases)
    z = [cmath.exp(1j * theta) for theta in phases]
    for _ in range(n_steps):
        theta = [cmath.phase(zi) for zi in z]
        rates = [
            omegas[i]
            + math.fsum(
                knm[i][j] * math.sin(theta[j] - theta[i] - alpha[i][j])
                for j in range(n)
            )
            + zeta * math.sin(psi - theta[i])
            for i in range(n)
        ]
        z = [zi * cmath.exp(1j * rate * dt) for zi, rate in zip(z, rates, strict=True)]
        z = [zi / abs(zi) for zi in z]
    return [cmath.phase(zi) % TWO_PI for zi in z]


def _circular_gap(got: np.ndarray, expected: list[float]) -> float:
    """Return the largest wrapped angular distance between two phase vectors."""
    diff = (np.asarray(got) - np.asarray(expected) + np.pi) % TWO_PI - np.pi
    return float(np.max(np.abs(diff)))


class TestBackendLoaderContracts:
    """The real loaders return kernels that integrate the torus law."""

    @staticmethod
    def _case(alpha_nonzero: bool) -> tuple[np.ndarray, ...]:
        """Return phases, frequencies and flat coupling and lag matrices."""
        rng = np.random.default_rng(11)
        n = 5
        phases = rng.uniform(0.0, TWO_PI, n)
        omegas = rng.normal(1.0, 0.4, n)
        knm = rng.uniform(0.0, 0.8, (n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = rng.uniform(-0.3, 0.3, (n, n)) if alpha_nonzero else np.zeros((n, n))
        return phases, omegas, knm, alpha

    def test_rust_loader_returns_float64_torus_phases(self) -> None:
        """The compiled Rust loader integrates the law on contiguous float64 input."""
        if "rust" not in g_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        kernel = g_mod._load_rust_fn()
        for alpha_nonzero in (False, True):
            phases, omegas, knm, alpha = self._case(alpha_nonzero)
            got = kernel(
                phases[::-1][::-1],
                omegas,
                knm.ravel(),
                alpha.ravel(),
                5,
                0.2,
                1.1,
                0.01,
                40,
            )
            assert got.dtype == np.float64
            assert got.shape == (5,)
            expected = _oracle_torus(
                phases.tolist(),
                omegas.tolist(),
                knm.tolist(),
                alpha.tolist(),
                0.2,
                1.1,
                0.01,
                40,
            )
            assert _circular_gap(got, expected) < TOL_ATAN2

    @pytest.mark.parametrize("owner", ["mojo", "julia", "go"])
    def test_optional_backend_loaders_return_callable_numeric_kernels(
        self, owner: str
    ) -> None:
        """Each installed optional loader returns a kernel that follows the law."""
        if owner not in g_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        loader = {
            "mojo": g_mod._load_mojo_fn,
            "julia": g_mod._load_julia_fn,
            "go": g_mod._load_go_fn,
        }[owner]
        kernel = loader()
        for alpha_nonzero in (False, True):
            phases, omegas, knm, alpha = self._case(alpha_nonzero)
            got = np.asarray(
                kernel(
                    phases, omegas, knm.ravel(), alpha.ravel(), 5, 0.2, 1.1, 0.01, 40
                )
            )
            expected = _oracle_torus(
                phases.tolist(),
                omegas.tolist(),
                knm.tolist(),
                alpha.tolist(),
                0.2,
                1.1,
                0.01,
                40,
            )
            assert _circular_gap(got, expected) < TOL_ATAN2


class TestDispatchFallbackChain:
    """Automatic dispatch falls through failing loaders."""

    def test_dispatch_falls_back_to_python_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing loader resolves to the NumPy law."""
        previous_backend = g_mod.ACTIVE_BACKEND
        previous_available = list(g_mod.AVAILABLE_BACKENDS)
        previous_loader = g_mod._LOADERS["go"]
        g_mod.ACTIVE_BACKEND = "go"
        g_mod.AVAILABLE_BACKENDS = ["go", "python"]
        g_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            g_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend = g_mod._dispatch()
        finally:
            g_mod.ACTIVE_BACKEND = previous_backend
            g_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(g_mod._LOADERS, "go", previous_loader)
            g_mod._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated dispatch loads an owner once."""
        previous_backend = g_mod.ACTIVE_BACKEND
        previous_available = list(g_mod.AVAILABLE_BACKENDS)
        previous_loader = g_mod._LOADERS["go"]
        g_mod.ACTIVE_BACKEND = "go"
        g_mod.AVAILABLE_BACKENDS = ["go", "python"]
        g_mod._BACKEND_CACHE.clear()
        call_count = 0

        def fake_backend(
            phases: np.ndarray,
            _omegas: np.ndarray,
            _knm_flat: np.ndarray,
            _alpha_flat: np.ndarray,
            _n: int,
            _zeta: float,
            _psi: float,
            _dt: float,
            _n_steps: int,
        ) -> np.ndarray:
            return np.asarray(phases, dtype=np.float64)

        def loader() -> Callable[..., np.ndarray]:
            nonlocal call_count
            call_count += 1
            return fake_backend

        monkeypatch.setitem(g_mod._LOADERS, "go", loader)
        try:
            b1 = g_mod._dispatch()
            b2 = g_mod._dispatch()
        finally:
            g_mod.ACTIVE_BACKEND = previous_backend
            g_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(g_mod._LOADERS, "go", previous_loader)
            g_mod._BACKEND_CACHE.clear()

        assert b1 is fake_backend
        assert b2 is fake_backend
        assert call_count == 1
