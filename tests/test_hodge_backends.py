# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for the Hodge decomposition

"""Cross-backend parity for :func:`hodge_decomposition`.

All four accelerated backends must match the NumPy reference on the
gradient, curl, and harmonic flow matrices within 1e-10 (Rust / Julia /
Go) or 1e-8 (Mojo, subprocess text round-trip). Direct backend adapters
validate the coupling, phase, edge, and triangle inputs before any
runtime is loaded.
"""

from __future__ import annotations

import cProfile
import subprocess
import sys
import types
from collections.abc import Callable
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling import (
    _hodge_validation as hodge_validation,
)
from scpn_phase_orchestrator.coupling import hodge as h_mod
from scpn_phase_orchestrator.coupling.hodge import (
    AVAILABLE_BACKENDS,
    HodgeResult,
    hodge_decomposition,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling import (
    _hodge_go,
    _hodge_julia,
    _hodge_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_go import (
    hodge_decomposition_go,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_julia import (
    hodge_decomposition_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_mojo import (
    hodge_decomposition_mojo,
)
from tests.typing_contracts import assert_precise_ndarray_hint

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

TWO_PI = 2.0 * np.pi
HodgeDirectBackend = Callable[
    [object, object, object, object, object, object, object],
    tuple[FloatArray, FloatArray, FloatArray],
]

# Complete graph K3: edges (0,1),(0,2),(1,2) and the single 2-simplex.
_K3_EDGES = np.array([0, 1, 0, 2, 1, 2], dtype=np.int64)
_K3_TRIS = np.array([0, 1, 2], dtype=np.int64)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(hodge_decomposition_go, id="go"),
        pytest.param(hodge_decomposition_julia, id="julia"),
        pytest.param(hodge_decomposition_mojo, id="mojo"),
    ],
)
@pytest.mark.parametrize(
    "edges,n_edges,triangles,n_triangles,match",
    [
        (np.array([1, 0]), 1, np.empty(0, dtype=np.int64), 0, "ascending"),
        (np.array([0, 1, 0, 1]), 2, np.empty(0, dtype=np.int64), 0, "unique"),
        (_K3_EDGES, 3, np.array([0, 2, 1]), 1, "ascending"),
        (_K3_EDGES, 3, np.array([1, 0, 2]), 1, "ascending"),
        (_K3_EDGES, 3, np.array([0, 1, 2, 0, 1, 2]), 2, "unique"),
        (np.array([0, 1, 1, 2]), 2, _K3_TRIS, 1, "missing edge"),
    ],
    ids=[
        "reversed-edge",
        "duplicate-edge",
        "reversed-face",
        "reversed-first-face-edge",
        "duplicate-face",
        "missing-face-edge",
    ],
)
def test_direct_adapters_refuse_invalid_simplicial_complex(
    backend: HodgeDirectBackend,
    edges: IntArray,
    n_edges: int,
    triangles: IntArray,
    n_triangles: int,
    match: str,
) -> None:
    """Real direct adapters refuse invalid simplices before loading a native owner."""
    coupling = (np.ones((3, 3)) - np.eye(3)).ravel()
    phases = np.array([0.0, 1.0, 2.3])
    with pytest.raises(ValueError, match=match):
        backend(coupling, phases, 3, edges, n_edges, triangles, n_triangles)


def test__hodge_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(hodge_validation.validate_hodge_backend_inputs)


def _reference(knm: FloatArray, phases: FloatArray) -> HodgeResult:
    """Compute the real public NumPy comparison without global mutation."""
    return hodge_decomposition(knm, phases, backend="python")


def _problem(seed: int, n: int = 16) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    k = rng.uniform(-1, 1, (n, n))
    np.fill_diagonal(k, 0.0)
    phases = rng.uniform(0, TWO_PI, n)
    return k, phases


def _zeros_int() -> IntArray:
    return np.zeros(0, dtype=np.int64)


def _mojo_proc(stdout: str) -> object:
    return type("Proc", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()


class _FakeJuliaHodgeModule:
    """Minimal Julia module stand-in for direct bridge output-boundary tests."""

    def __init__(self, output: tuple[object, object, object]) -> None:
        self._output = output

    def hodge_decomposition(self, *_args: object) -> tuple[object, object, object]:
        """Return the configured backend payload."""
        return self._output


class TestDirectBackendBoundaryContracts:
    """Direct optional Hodge backends validate before runtime loading."""

    @pytest.mark.parametrize(
        "backend",
        [
            hodge_decomposition_go,
            hodge_decomposition_julia,
            hodge_decomposition_mojo,
        ],
    )
    @pytest.mark.parametrize(
        ("knm_flat", "phases", "n", "match"),
        [
            (np.array([True, False, False, True]), np.zeros(2), 2, "knm_flat"),
            (np.array([0.0, np.nan, 0.0, 0.0]), np.zeros(2), 2, "finite"),
            (np.array([0.0, 1.0 + 0.0j, 0.0, 0.0]), np.zeros(2), 2, "real-valued"),
            (np.zeros((2, 2)), np.zeros(2), 2, "knm_flat"),
            (np.zeros(3), np.zeros(2), 2, "n\\*n"),
            (np.zeros(4), np.array([True, False]), 2, "phases"),
            (np.zeros(4), np.array([0.0, np.inf]), 2, "finite"),
            (np.zeros(4), np.array([0.0, 1.0 + 0.0j]), 2, "real-valued"),
            (np.zeros(4), np.array([[0.0, 1.0]]), 2, "one-dimensional"),
            (np.zeros(4), np.zeros(1), 2, "phases length"),
            (np.zeros(4), np.zeros(2), True, "n"),
            (np.zeros(4), np.zeros(2), -1, "n"),
        ],
    )
    def test_validation_precedes_runtime_load(
        self,
        backend: HodgeDirectBackend,
        knm_flat: object,
        phases: object,
        n: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            backend(knm_flat, phases, n, _zeros_int(), 0, _zeros_int(), 0)

    @pytest.mark.parametrize(
        "backend",
        [
            hodge_decomposition_go,
            hodge_decomposition_julia,
            hodge_decomposition_mojo,
        ],
    )
    @pytest.mark.parametrize(
        ("edges", "n_edges", "tris", "n_tris", "match"),
        [
            (np.array([0, 1, 0, 9], dtype=np.int64), 2, _zeros_int(), 0, r"\[0, 2\)"),
            (np.array([0, 1], dtype=np.int64), 2, _zeros_int(), 0, "edges_flat length"),
            (np.array([True, False], dtype=object), 1, _zeros_int(), 0, "boolean"),
            (
                np.array([0, 1], dtype=np.int64),
                1,
                np.array([0, 1, 9], dtype=np.int64),
                1,
                r"\[0, 2\)",
            ),
            (
                np.array([0, 1], dtype=np.int64),
                1,
                np.array([0, 1], dtype=np.int64),
                1,
                "tris_flat length",
            ),
        ],
    )
    def test_simplex_validation_rejects_malformed_complex(
        self,
        backend: HodgeDirectBackend,
        edges: object,
        n_edges: int,
        tris: object,
        n_tris: int,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            backend(np.zeros(4), np.zeros(2), 2, edges, n_edges, tris, n_tris)

    @pytest.mark.parametrize(
        "backend",
        [
            hodge_decomposition_go,
            hodge_decomposition_julia,
            hodge_decomposition_mojo,
        ],
    )
    def test_empty_hodge_returns_empty_components_before_runtime_load(
        self,
        backend: HodgeDirectBackend,
    ) -> None:
        gradient, curl, harmonic = backend(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            0,
            _zeros_int(),
            0,
            _zeros_int(),
            0,
        )
        assert gradient.dtype == np.float64
        assert curl.dtype == np.float64
        assert harmonic.dtype == np.float64
        assert gradient.shape == curl.shape == harmonic.shape == (0, 0)

    def test_direct_julia_rejects_boolean_output_alias(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Julia outputs are validated before object bools can widen to floats."""
        bad_gradient = np.array([0.0, np.bool_(True), -1.0, 0.0], dtype=object)
        zero = np.zeros(4, dtype=np.float64)
        monkeypatch.setattr(
            _hodge_julia,
            "_ensure",
            lambda: _FakeJuliaHodgeModule((bad_gradient, zero, zero.copy())),
        )

        with pytest.raises(ValueError, match="finite real-valued"):
            hodge_decomposition_julia(
                np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.4], dtype=np.float64),
                2,
                np.array([0, 1], dtype=np.int64),
                1,
                _zeros_int(),
                0,
            )

    @pytest.mark.parametrize(
        ("backend", "patch_target"),
        [
            (hodge_decomposition_go, _hodge_go),
            (hodge_decomposition_julia, _hodge_julia),
            (hodge_decomposition_mojo, _hodge_mojo),
        ],
    )
    @pytest.mark.parametrize(
        "args",
        [
            (
                np.array(["0.0", "1.0", "-1.0", "0.0"], dtype=object),
                np.zeros(2, dtype=np.float64),
                2,
                np.array([0, 1], dtype=np.int64),
                1,
                _zeros_int(),
                0,
            ),
            (
                np.zeros(4, dtype=np.float64),
                np.array(["0.0", "0.4"], dtype=object),
                2,
                np.array([0, 1], dtype=np.int64),
                1,
                _zeros_int(),
                0,
            ),
            (
                np.zeros(4, dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                2,
                np.array(["0", "1"], dtype=object),
                1,
                _zeros_int(),
                0,
            ),
        ],
    )
    def test_direct_backends_reject_numeric_string_inputs_before_runtime_load(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend: HodgeDirectBackend,
        patch_target: object,
        args: tuple[object, object, object, object, object, object, object],
    ) -> None:
        """Stringified numeric direct inputs fail before loading any runtime."""

        def forbidden_loader() -> object:
            raise AssertionError("runtime loader must not be called")

        if patch_target is _hodge_go:
            monkeypatch.setattr(_hodge_go, "_load_lib", forbidden_loader)
        elif patch_target is _hodge_julia:
            monkeypatch.setattr(_hodge_julia, "_ensure", forbidden_loader)
        else:
            monkeypatch.setattr(_hodge_mojo, "_ensure_exe", forbidden_loader)

        with pytest.raises(ValueError, match="numeric-string"):
            backend(*args)

    def test_direct_julia_rejects_numeric_string_output_alias(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Julia outputs reject stringified numerics before float coercion."""
        bad_gradient = np.array(["0.0", "1.0", "-1.0", "0.0"], dtype=object)
        zero = np.zeros(4, dtype=np.float64)
        monkeypatch.setattr(
            _hodge_julia,
            "_ensure",
            lambda: _FakeJuliaHodgeModule((bad_gradient, zero, zero.copy())),
        )

        with pytest.raises(ValueError, match="numeric-string"):
            hodge_decomposition_julia(
                np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.4], dtype=np.float64),
                2,
                np.array([0, 1], dtype=np.int64),
                1,
                _zeros_int(),
                0,
            )


class TestDirectMojoBoundaryContracts:
    """Direct Mojo Hodge adapter rejects malformed backend stdout."""

    @pytest.mark.parametrize(
        ("stdout", "match"),
        [
            ("", "Mojo HODGE returned 0 lines, expected 27"),
            ("\n".join(str(i) for i in range(28)) + "\n", "expected 27"),
            ("\n".join(["1"] * 13 + [""] + ["1"] * 13) + "\n", "finite gradient"),
            ("\n".join(["1"] * 13 + ["bad"] + ["1"] * 13) + "\n", "finite gradient"),
            ("\n".join(["1"] * 13 + ["nan"] + ["1"] * 13) + "\n", "finite gradient"),
            ("\n".join(["1"] * 13 + ["inf"] + ["1"] * 13) + "\n", "finite gradient"),
        ],
    )
    def test_mojo_runner_rejects_malformed_raw_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        match: str,
    ) -> None:
        knm = (np.ones((3, 3)) - np.eye(3)).ravel()
        phases = np.array([0.0, 1.0, 2.3], dtype=np.float64)
        monkeypatch.setattr(_hodge_mojo, "_ensure_exe", lambda: "hodge")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *_args, **_kwargs: _mojo_proc(stdout),
        )

        with pytest.raises(ValueError, match=match):
            _hodge_mojo.hodge_decomposition_mojo(
                knm, phases, 3, _K3_EDGES, 3, _K3_TRIS, 1
            )


@pytest.mark.native_runtime
class TestRustParity:
    @given(
        # Keep n <= 16: the dense QR eigensolver in the Rust backend is
        # bounded but can exceed the 60 s test budget for n > 16 under load.
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        knm, phases = _problem(seed, n)
        ref = _reference(knm, phases)
        got = hodge_decomposition(knm, phases, backend="rust")
        np.testing.assert_allclose(got.gradient, ref.gradient, atol=1e-10)
        np.testing.assert_allclose(got.curl, ref.curl, atol=1e-10)
        np.testing.assert_allclose(got.harmonic, ref.harmonic, atol=1e-10)


@pytest.mark.native_runtime
class TestJuliaParity:
    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, seed: int) -> None:
        knm, phases = _problem(seed)
        ref = _reference(knm, phases)
        got = hodge_decomposition(knm, phases, backend="julia")
        np.testing.assert_allclose(got.gradient, ref.gradient, atol=1e-10)
        np.testing.assert_allclose(got.curl, ref.curl, atol=1e-10)
        np.testing.assert_allclose(got.harmonic, ref.harmonic, atol=1e-10)


@pytest.mark.native_runtime
class TestGoParity:
    @given(
        # Keep n <= 16: the Go backend's dense eigensolver does not reliably
        # finish inside the 60 s test budget for n > 16.
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        knm, phases = _problem(seed, n)
        ref = _reference(knm, phases)
        got = hodge_decomposition(knm, phases, backend="go")
        np.testing.assert_allclose(got.gradient, ref.gradient, atol=1e-10)
        np.testing.assert_allclose(got.curl, ref.curl, atol=1e-10)
        np.testing.assert_allclose(got.harmonic, ref.harmonic, atol=1e-10)


@pytest.mark.native_runtime
class TestMojoParity:
    @pytest.mark.parametrize("seed", [0, 77])
    def test_matches_python(self, seed: int) -> None:
        knm, phases = _problem(seed)
        ref = _reference(knm, phases)
        got = hodge_decomposition(knm, phases, backend="mojo")
        np.testing.assert_allclose(got.gradient, ref.gradient, atol=1e-8)
        np.testing.assert_allclose(got.curl, ref.curl, atol=1e-8)
        np.testing.assert_allclose(got.harmonic, ref.harmonic, atol=1e-8)


class TestCrossBackendConsistency:
    def test_all_backends_agree(self) -> None:
        knm, phases = _problem(2026, n=12)
        ref = _reference(knm, phases)
        tolerances = {
            "rust": 1e-10,
            "julia": 1e-10,
            "go": 1e-10,
            "mojo": 1e-8,
            "python": 0.0,
        }
        for backend in AVAILABLE_BACKENDS:
            got = hodge_decomposition(knm, phases, backend=backend)
            atol = tolerances[backend]
            np.testing.assert_allclose(got.gradient, ref.gradient, atol=atol)
            np.testing.assert_allclose(got.curl, ref.curl, atol=atol)
            np.testing.assert_allclose(got.harmonic, ref.harmonic, atol=atol)
            assert got.betti_one == ref.betti_one


class TestBackendTypingContracts:
    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (hodge_decomposition_go, "go"),
            (hodge_decomposition_julia, "julia"),
            (hodge_decomposition_mojo, "mojo"),
        ],
    )
    def test_backend_annotations_use_float64_ndarray(
        self, fn: HodgeDirectBackend, label: str
    ) -> None:
        hints = get_type_hints(fn)
        for name in ("knm_flat", "phases", "return"):
            text = str(hints[name])
            assert_precise_ndarray_hint(hints[name], context=f"{label}:{name}")
            assert "numpy.float64" in text, f"{label}:{name} missing float64 annotation"


class TestBackendLoaderDispatch:
    @pytest.mark.native_runtime
    def test_rust_loader_wraps_spo_kernel_flattened_arrays(self) -> None:
        """The native kernel receives noncontiguous public arrays correctly."""
        knm = np.array([[0.0, -0.25], [0.5, 0.0]], dtype=np.float64).T
        phases = np.array([0.1, 9.0, 0.4, 9.0], dtype=np.float64)[::2]
        measurement = cProfile.Profile()
        measurement.enable()
        try:
            result = hodge_decomposition(knm, phases, backend="rust")
        finally:
            measurement.disable()
        assert (
            sum(
                entry.callcount
                for entry in measurement.getstats()
                if isinstance(entry.code, str)
                and "hodge_decomposition_rust>" in entry.code
            )
            == 1
        )
        flow = 0.125 * np.sin(0.3)
        np.testing.assert_allclose(
            result.gradient, [[0.0, flow], [-flow, 0.0]], atol=1e-12
        )
        np.testing.assert_allclose(result.curl, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-12)

    def test_rust_loader_rejects_boolean_output_alias(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_rust(*_args: object) -> tuple[object, object, object]:
            bad_gradient = np.array([0.0, np.bool_(True), -1.0, 0.0], dtype=object)
            zero = np.zeros(4, dtype=np.float64)
            return bad_gradient, zero, zero.copy()

        fake_module = types.ModuleType("spo_kernel")
        monkeypatch.setattr(
            fake_module, "hodge_decomposition_rust", fake_rust, raising=False
        )
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_module)

        backend = h_mod._load_rust_fn()

        with pytest.raises(ValueError, match="finite real-valued"):
            backend(
                np.array([[0.0, 0.5], [-0.25, 0.0]], dtype=np.float64),
                np.array([0.1, 0.4], dtype=np.float64),
                2,
                _K3_EDGES[:2],
                1,
                _zeros_int(),
                0,
            )

    @pytest.mark.native_runtime
    def test_julia_loader_requires_juliacall_then_returns_backend(self) -> None:
        """The public Julia owner resolves and decomposes a nonzero edge."""
        result = hodge_decomposition(
            np.array([[0.0, 0.5], [-0.25, 0.0]]),
            np.array([0.1, 0.4]),
            backend="julia",
        )
        assert result.gradient[0, 1] == pytest.approx(0.125 * np.sin(0.3), abs=1e-12)
        np.testing.assert_allclose(result.curl, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-12)

    @pytest.mark.native_runtime
    def test_mojo_loader_runs_preflight_and_returns_backend(self) -> None:
        """The compiled Mojo owner executes through the public selector."""
        result = hodge_decomposition(
            np.array([[0.0, 0.5], [-0.25, 0.0]]),
            np.array([0.1, 0.4]),
            backend="mojo",
        )
        assert result.gradient[0, 1] == pytest.approx(0.125 * np.sin(0.3), abs=1e-10)
        np.testing.assert_allclose(result.curl, 0.0, atol=1e-10)
        np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-10)

    @pytest.mark.native_runtime
    def test_go_loader_invokes_shared_object_loader(self) -> None:
        """The actual Go shared object executes through the public selector."""
        result = hodge_decomposition(
            np.array([[0.0, 0.5], [-0.25, 0.0]]),
            np.array([0.1, 0.4]),
            backend="go",
        )
        assert result.gradient[0, 1] == pytest.approx(0.125 * np.sin(0.3), abs=1e-12)
        np.testing.assert_allclose(result.curl, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.harmonic, 0.0, atol=1e-12)


class TestBackendResolution:
    @pytest.mark.native_runtime
    def test_resolve_backends_chooses_first_available_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []
        real_go_loader = h_mod._load_go_fn

        def _fail() -> h_mod.HodgeBackend:
            calls.append("fail")
            raise RuntimeError("backend unavailable")

        def _go() -> h_mod.HodgeBackend:
            calls.append("go")
            return real_go_loader()

        monkeypatch.setitem(h_mod._LOADERS, "rust", _fail)
        monkeypatch.setitem(h_mod._LOADERS, "mojo", _fail)
        monkeypatch.setitem(h_mod._LOADERS, "julia", _fail)
        monkeypatch.setitem(h_mod._LOADERS, "go", _go)

        active, available = h_mod._resolve_backends()
        assert active == "go"
        assert available == ["go", "python"]
        assert calls == ["fail", "fail", "fail", "go"]
        monkeypatch.setattr(h_mod, "ACTIVE_BACKEND", active)
        monkeypatch.setattr(h_mod, "AVAILABLE_BACKENDS", available)
        monkeypatch.setattr(h_mod, "_BACKEND_CACHE", {})
        result = hodge_decomposition(
            np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0.0, 0.4])
        )
        assert result.gradient[0, 1] == pytest.approx(np.sin(0.4), abs=1e-12)

    def test_resolve_backends_falls_back_to_python_when_all_backends_fail(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        def _fail() -> h_mod.HodgeBackend:
            calls.append("fail")
            raise RuntimeError("backend unavailable")

        monkeypatch.setitem(h_mod._LOADERS, "rust", _fail)
        monkeypatch.setitem(h_mod._LOADERS, "mojo", _fail)
        monkeypatch.setitem(h_mod._LOADERS, "julia", _fail)
        monkeypatch.setitem(h_mod._LOADERS, "go", _fail)

        active, available = h_mod._resolve_backends()

        assert active == "python"
        assert available == ["python"]
        assert calls == ["fail", "fail", "fail", "fail"]
        monkeypatch.setattr(h_mod, "ACTIVE_BACKEND", active)
        monkeypatch.setattr(h_mod, "AVAILABLE_BACKENDS", available)
        monkeypatch.setattr(h_mod, "_BACKEND_CACHE", {})
        result = hodge_decomposition(
            np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0.0, 0.4])
        )
        assert result.gradient[0, 1] == pytest.approx(np.sin(0.4), abs=1e-12)
