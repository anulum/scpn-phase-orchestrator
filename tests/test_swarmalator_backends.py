# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for swarmalator stepper

"""Cross-backend parity for :class:`SwarmalatorEngine`.step.

All four non-Python backends must match the Python reference on
the same input within 1e-12 (Rust / Julia / Go) or 1e-9 (Mojo).
All five backends use the canonical O'Keeffe-Hong-Strogatz
inverse-distance repulsion ``b / (|dx|^2 + eps)`` (scalar
``b / (d2 + eps)`` multiplying the separation vector).
"""

from __future__ import annotations

import ctypes
import math
import sys
import types
from collections.abc import Callable
from typing import TypeAlias, cast, get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _swarmalator_go,
    _swarmalator_julia,
    _swarmalator_mojo,
)
from scpn_phase_orchestrator.upde import (
    _swarmalator_validation as swarmalator_validation,
)
from scpn_phase_orchestrator.upde import swarmalator as sw_mod
from scpn_phase_orchestrator.upde.swarmalator import (
    AVAILABLE_BACKENDS,
    SwarmalatorEngine,
)
from tests.typing_contracts import assert_precise_ndarray_hint

swarmalator_step_go = _swarmalator_go.swarmalator_step_go
swarmalator_step_julia = _swarmalator_julia.swarmalator_step_julia
swarmalator_step_mojo = _swarmalator_mojo.swarmalator_step_mojo

FloatArray: TypeAlias = NDArray[np.float64]
BackendArgs: TypeAlias = tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    int,
    int,
    float,
    float,
    float,
    float,
    float,
]
BackendFn: TypeAlias = Callable[
    [
        FloatArray,
        FloatArray,
        FloatArray,
        int,
        int,
        float,
        float,
        float,
        float,
        float,
    ],
    tuple[FloatArray, FloatArray],
]

TWO_PI = 2.0 * np.pi


def test__swarmalator_validation_helper_is_directly_linked_to_backend_tests() -> None:
    args = _valid_direct_args()

    validated = swarmalator_validation.validate_swarmalator_inputs(*args)
    positions, phases = swarmalator_validation.validate_swarmalator_output(
        args[0],
        args[1],
        n=2,
        dim=2,
    )

    assert validated[3] == 2
    assert validated[4] == 2
    np.testing.assert_array_equal(positions, args[0])
    np.testing.assert_array_equal(phases, args[1])


def _valid_direct_args() -> BackendArgs:
    return (
        np.array([[0.0, 0.5], [1.0, -0.25]], dtype=np.float64),
        np.array([0.2, 0.4], dtype=np.float64),
        np.array([0.1, -0.2], dtype=np.float64),
        2,
        2,
        1.0,
        1.0,
        0.8,
        1.2,
        0.01,
    )


def _with_direct_arg(index: int, value: object) -> tuple[object, ...]:
    args = list(_valid_direct_args())
    args[index] = value
    return tuple(args)


class _UnarrayableProbe:
    def __array__(self, _dtype: object | None = None) -> object:
        raise TypeError("cannot expose array")


class _FakeGoSwarmalatorLib:
    def __init__(
        self,
        positions: tuple[float, ...],
        phases: tuple[float, ...],
        rc: int = 0,
    ) -> None:
        self.positions = positions
        self.phases = phases
        self.rc = rc

    def SwarmalatorStep(self, *_args: object) -> int:
        pos_ref = cast(ctypes.Array[ctypes.c_double], _args[-2])
        phase_ref = cast(ctypes.Array[ctypes.c_double], _args[-1])
        for index, value in enumerate(self.positions):
            pos_ref[index] = value
        for index, value in enumerate(self.phases):
            phase_ref[index] = value
        return self.rc


class _FakeJuliaSwarmalatorModule:
    def __init__(
        self,
        positions: tuple[float, ...],
        phases: tuple[float, ...],
    ) -> None:
        self.positions = positions
        self.phases = phases

    def swarmalator_step(self, *_args: object) -> tuple[FloatArray, FloatArray]:
        return (
            np.array(self.positions, dtype=np.float64),
            np.array(self.phases, dtype=np.float64),
        )


def _call_direct_backend(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
    positions: tuple[float, ...],
    phases: tuple[float, ...],
) -> tuple[FloatArray, FloatArray]:
    if module is _swarmalator_go:
        monkeypatch.setattr(
            _swarmalator_go,
            "_load_lib",
            lambda: _FakeGoSwarmalatorLib(positions, phases),
        )
        return _swarmalator_go.swarmalator_step_go(*_valid_direct_args())
    if module is _swarmalator_julia:
        monkeypatch.setattr(
            _swarmalator_julia,
            "_ensure",
            lambda: _FakeJuliaSwarmalatorModule(positions, phases),
        )
        return _swarmalator_julia.swarmalator_step_julia(*_valid_direct_args())

    monkeypatch.setattr(_swarmalator_mojo, "_ensure_exe", lambda: "swarmalator")
    stdout = "".join(f"{value}\n" for value in (*positions, *phases))
    monkeypatch.setattr(
        cast(object, _swarmalator_mojo.__dict__["subprocess"]),
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0,
            stdout=stdout,
            stderr="",
        ),
    )
    return _swarmalator_mojo.swarmalator_step_mojo(*_valid_direct_args())


def _force(backend: str) -> str:
    prev = sw_mod.ACTIVE_BACKEND
    sw_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    sw_mod.ACTIVE_BACKEND = prev


def _problem(
    seed: int, n: int = 16, dim: int = 2
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
]:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, dim))
    phases = rng.uniform(0, TWO_PI, n)
    omegas = rng.normal(0.5, 0.2, n)
    return pos, phases, omegas


def _direct_payload(n: int = 3, dim: int = 2) -> BackendArgs:
    pos, phases, omegas = _problem(19, n=n, dim=dim)
    return pos, phases, omegas, n, dim, 1.0, 1.0, 0.8, 1.2, 0.01


def _reference_step(
    pos: FloatArray,
    phases: FloatArray,
    omegas: FloatArray,
    n: int,
    dim: int,
) -> tuple[FloatArray, FloatArray]:
    prev = _force("python")
    try:
        eng = SwarmalatorEngine(n, dim, 0.01)
        return eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
    finally:
        _reset(prev)


def _oracle_swarmalator(
    pos: list[list[float]],
    phases: list[float],
    omegas: list[float],
    a: float,
    b: float,
    j: float,
    k: float,
    dt: float,
) -> tuple[list[list[float]], list[float]]:
    """One O'Keeffe–Hong–Strogatz Euler step written as plain loops."""
    eps = 1e-6
    n = len(phases)
    new_pos, new_phases = [], []
    for i in range(n):
        velocity = [0.0] * len(pos[i])
        phase_terms = []
        for m in range(n):
            sep = [x - y for x, y in zip(pos[m], pos[i], strict=True)]
            d2 = math.fsum(c * c for c in sep)
            attract = (a + j * math.cos(phases[m] - phases[i])) / math.sqrt(d2 + eps)
            repulse = b / (d2 + eps)
            velocity = [
                v + c * (attract - repulse) for v, c in zip(velocity, sep, strict=True)
            ]
            weight = 0.0 if m == i else 1 / math.sqrt(d2 + eps)
            phase_terms.append(math.sin(phases[m] - phases[i]) * weight)
        new_pos.append([x + dt * v / n for x, v in zip(pos[i], velocity, strict=True)])
        rate = omegas[i] + k * math.fsum(phase_terms) / n
        new_phases.append((phases[i] + dt * rate) % (2 * math.pi))
    return new_pos, new_phases


def _assert_kernel_follows_law(owner: str, kernel: Callable[..., object]) -> None:
    """Check one step of a real swarmalator kernel in 2-D and 3-D."""
    rng = np.random.default_rng(109)
    for dim in (2, 3):
        pos = rng.uniform(-1.5, 1.5, (6, dim))
        phases = rng.uniform(0.0, 2 * np.pi, 6)
        omegas = rng.normal(0.5, 0.2, 6)
        got_pos, got_phases = cast(
            "tuple[np.ndarray, np.ndarray]",
            kernel(pos, phases, omegas, 6, dim, 1.0, 0.7, 0.9, -0.4, 0.02),
        )
        exp_pos, exp_phases = _oracle_swarmalator(
            pos.tolist(), phases.tolist(), omegas.tolist(), 1.0, 0.7, 0.9, -0.4, 0.02
        )
        atol = 1e-9 if owner == "mojo" else 1e-12
        np.testing.assert_allclose(np.reshape(got_pos, (6, dim)), exp_pos, atol=atol)
        gap = (np.asarray(got_phases) - exp_phases + np.pi) % (2 * np.pi) - np.pi
        assert float(np.max(np.abs(gap))) < atol


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=2, max_value=24),
        dim=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_step(self, n: int, dim: int, seed: int) -> None:
        pos, phases, omegas = _problem(seed, n, dim)
        ref_p, ref_ph = _reference_step(pos, phases, omegas, n, dim)
        prev = _force("rust")
        try:
            eng = SwarmalatorEngine(n, dim, 0.01)
            p, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
        finally:
            _reset(prev)
        np.testing.assert_allclose(p, ref_p, atol=1e-12)
        np.testing.assert_allclose(ph, ref_ph, atol=1e-12)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_step(self, seed: int) -> None:
        pos, phases, omegas = _problem(seed)
        ref_p, ref_ph = _reference_step(pos, phases, omegas, 16, 2)
        prev = _force("julia")
        try:
            eng = SwarmalatorEngine(16, 2, 0.01)
            p, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
        finally:
            _reset(prev)
        np.testing.assert_allclose(p, ref_p, atol=1e-12)
        np.testing.assert_allclose(ph, ref_ph, atol=1e-12)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=2, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_step(self, n: int, seed: int) -> None:
        pos, phases, omegas = _problem(seed, n, 2)
        ref_p, ref_ph = _reference_step(pos, phases, omegas, n, 2)
        prev = _force("go")
        try:
            eng = SwarmalatorEngine(n, 2, 0.01)
            p, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
        finally:
            _reset(prev)
        np.testing.assert_allclose(p, ref_p, atol=1e-12)
        np.testing.assert_allclose(ph, ref_ph, atol=1e-12)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_step(self, seed: int) -> None:
        pos, phases, omegas = _problem(seed)
        ref_p, ref_ph = _reference_step(pos, phases, omegas, 16, 2)
        prev = _force("mojo")
        try:
            eng = SwarmalatorEngine(16, 2, 0.01)
            p, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
        finally:
            _reset(prev)
        np.testing.assert_allclose(p, ref_p, atol=1e-9)
        np.testing.assert_allclose(ph, ref_ph, atol=1e-9)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        pos, phases, omegas = _problem(2026, 20, 2)
        ref_p, ref_ph = _reference_step(pos, phases, omegas, 20, 2)
        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-9,
            "python": 0.0,
        }
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                eng = SwarmalatorEngine(20, 2, 0.01)
                p, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
            finally:
                _reset(prev)
            atol = tolerances[backend]
            np.testing.assert_allclose(p, ref_p, atol=atol)
            np.testing.assert_allclose(ph, ref_ph, atol=atol)


class TestBackendTypingContracts:
    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (swarmalator_step_go, "go"),
            (swarmalator_step_julia, "julia"),
            (swarmalator_step_mojo, "mojo"),
        ],
    )
    def test_backend_annotations_use_float64_ndarray(
        self,
        fn: Callable[..., object],
        label: str,
    ) -> None:
        hints = get_type_hints(fn)
        for name in ("pos", "phases", "omegas", "return"):
            text = str(hints[name])
            assert_precise_ndarray_hint(
                hints[name],
                context=f"{label}:{name}",
            )
            assert "numpy.float64" in text, f"{label}:{name} missing float64 annotation"


class TestDirectSwarmalatorBoundaryContracts:
    def test_direct_numeric_string_probe_ignores_unarrayable_value(self) -> None:
        assert (
            swarmalator_validation._contains_numeric_string_alias(_UnarrayableProbe())
            is False
        )

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (
                np.array([["0.0", "0.5"], ["1.0", "-0.25"]], dtype=object),
                "numeric-string",
            ),
            (
                np.array(["0.2", "0.4"], dtype=object),
                "numeric-string",
            ),
        ],
    )
    def test_direct_validation_output_rejects_numeric_string_aliases(
        self,
        value: object,
        match: str,
    ) -> None:
        positions = (
            value
            if np.asarray(value).ndim == 2
            else np.array([[0.0, 0.5], [1.0, -0.25]], dtype=np.float64)
        )
        phases = value if np.asarray(value).ndim == 1 else np.array([0.2, 0.4])

        with pytest.raises(ValueError, match=match):
            swarmalator_validation.validate_swarmalator_output(
                positions,
                phases,
                n=2,
                dim=2,
            )

    @pytest.mark.parametrize(
        ("module", "fn_name", "loader_name"),
        [
            (_swarmalator_go, "swarmalator_step_go", "_load_lib"),
            (_swarmalator_julia, "swarmalator_step_julia", "_ensure"),
            (_swarmalator_mojo, "swarmalator_step_mojo", "_ensure_exe"),
        ],
    )
    @pytest.mark.parametrize(
        ("args", "match"),
        [
            (_with_direct_arg(0, np.array([[True, False], [False, True]])), "pos"),
            (_with_direct_arg(0, np.zeros(3, dtype=np.float64)), "pos"),
            (
                _with_direct_arg(
                    0,
                    np.array([["0.0", "0.5"], ["1.0", "-0.25"]], dtype=object),
                ),
                "numeric-string",
            ),
            (
                _with_direct_arg(
                    0,
                    np.array([["bad", "0.5"], ["1.0", "-0.25"]], dtype=object),
                ),
                "pos",
            ),
            (_with_direct_arg(1, np.array([0.1 + 0.0j, 0.2])), "phases"),
            (_with_direct_arg(1, np.array(["0.2", "0.4"])), "numeric-string"),
            (_with_direct_arg(1, np.array(["bad", "worse"], dtype=object)), "phases"),
            (_with_direct_arg(1, np.array([[0.2, 0.4]])), "one-dimensional"),
            (_with_direct_arg(1, np.array([0.2])), "phases length"),
            (_with_direct_arg(2, np.array([True, False])), "omegas"),
            (
                _with_direct_arg(2, np.array([0.1, "-0.2"], dtype=object)),
                "numeric-string",
            ),
            (_with_direct_arg(2, np.array([0.1])), "omegas length"),
            (_with_direct_arg(3, True), "n"),
            (_with_direct_arg(3, "2"), "numeric-string"),
            (_with_direct_arg(3, 1.5), "n"),
            (_with_direct_arg(3, 0), "n"),
            (_with_direct_arg(4, True), "dim"),
            (_with_direct_arg(4, "2"), "numeric-string"),
            (_with_direct_arg(4, 0), "dim"),
            (_with_direct_arg(5, True), "a"),
            (_with_direct_arg(5, "1.0"), "numeric-string"),
            (_with_direct_arg(5, ""), "a"),
            (_with_direct_arg(5, "bad"), "a"),
            (_with_direct_arg(6, np.inf), "b"),
            (_with_direct_arg(6, "1.0"), "numeric-string"),
            (_with_direct_arg(7, object()), "j"),
            (_with_direct_arg(7, "0.8"), "numeric-string"),
            (_with_direct_arg(8, np.nan), "k"),
            (_with_direct_arg(8, "1.2"), "numeric-string"),
            (_with_direct_arg(9, 0.0), "dt"),
            (_with_direct_arg(9, -0.01), "dt"),
            (_with_direct_arg(9, "0.01"), "numeric-string"),
        ],
    )
    def test_direct_backend_rejects_invalid_inputs_before_runtime_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: types.ModuleType,
        fn_name: str,
        loader_name: str,
        args: tuple[object, ...],
        match: str,
    ) -> None:
        def forbidden_loader() -> None:
            raise AssertionError("runtime loader must not be called")

        monkeypatch.setattr(module, loader_name, forbidden_loader)

        with pytest.raises(ValueError, match=match):
            getattr(module, fn_name)(*args)

    @pytest.mark.parametrize(
        "module",
        [_swarmalator_go, _swarmalator_julia, _swarmalator_mojo],
    )
    def test_direct_backend_accepts_valid_position_phase_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: types.ModuleType,
    ) -> None:
        got_pos, got_phases = _call_direct_backend(
            module,
            monkeypatch,
            (0.1, 0.2, 0.3, 0.4),
            (0.5, 0.6),
        )

        np.testing.assert_allclose(
            got_pos,
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        )
        np.testing.assert_allclose(
            got_phases,
            np.array([0.5, 0.6], dtype=np.float64),
        )
        assert got_pos.dtype == np.float64
        assert got_phases.dtype == np.float64

    @pytest.mark.parametrize(
        "module",
        [_swarmalator_go, _swarmalator_julia, _swarmalator_mojo],
    )
    @pytest.mark.parametrize(
        ("positions", "phases", "match"),
        [
            ((0.1, np.nan, 0.3, 0.4), (0.5, 0.6), "positions"),
            ((0.1, 0.2, 0.3, 0.4), (-0.1, 0.6), "phases"),
            ((0.1, 0.2, 0.3, 0.4), (0.5, TWO_PI), "phases"),
        ],
    )
    def test_direct_backend_rejects_non_physical_outputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: types.ModuleType,
        positions: tuple[float, ...],
        phases: tuple[float, ...],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _call_direct_backend(module, monkeypatch, positions, phases)

    @pytest.mark.parametrize("module", [_swarmalator_julia, _swarmalator_mojo])
    @pytest.mark.parametrize(
        ("positions", "phases", "match"),
        [
            ((0.1, 0.2, 0.3), (0.5, 0.6), "positions|expected 6"),
            ((0.1, 0.2, 0.3, 0.4), (0.5,), "phases|expected 6"),
        ],
    )
    def test_direct_backend_rejects_wrong_output_cardinality(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: types.ModuleType,
        positions: tuple[float, ...],
        phases: tuple[float, ...],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _call_direct_backend(module, monkeypatch, positions, phases)

    def test_direct_julia_backend_rejects_numeric_string_output_before_coercion(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class RawJuliaSwarmalatorModule:
            def swarmalator_step(self, *_args: object) -> tuple[object, object]:
                return (
                    np.array(["0.1", "0.2", "0.3", "0.4"], dtype=object),
                    np.array([0.5, "0.6"], dtype=object),
                )

        monkeypatch.setattr(
            _swarmalator_julia,
            "_ensure",
            lambda: RawJuliaSwarmalatorModule(),
        )

        with pytest.raises(ValueError, match="numeric-string"):
            _swarmalator_julia.swarmalator_step_julia(*_valid_direct_args())


class TestBackendLoaderContracts:
    @pytest.mark.native_runtime
    def test_rust_loader_reshapes_flat_kernel_positions(self) -> None:
        """The compiled Rust loader returns positions shaped ``(n, dim)``."""
        if "rust" not in sw_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        _assert_kernel_follows_law("rust", sw_mod._load_rust_fn())

    @pytest.mark.parametrize(
        ("stdout", "match"),
        [
            ("", "Mojo STEP returned 0 lines, expected 9"),
            ("0\n1\n2\n3\n4\n5\n0.1\n0.2\n0.3\n0.4\n", "expected 9"),
            ("0\n1\n2\n3\n4\n\n0.1\n0.2\n0.3\n", "finite positions"),
            ("0\n1\n2\nnan\n4\n5\n0.1\n0.2\n0.3\n", "finite positions"),
            ("0\n1\n2\n3\n4\n5\nbad\n0.2\n0.3\n", "finite positions"),
            ("0\n1\n2\n3\n4\n5\n7.0\n0.2\n0.3\n", "finite positions"),
            ("0\n1\n2\n3\n4\n5\n0.1\ninf\n0.3\n", "finite positions"),
        ],
    )
    def test_mojo_runner_rejects_malformed_raw_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(_swarmalator_mojo, "_ensure_exe", lambda: "swarmalator")
        monkeypatch.setattr(
            cast(object, _swarmalator_mojo.__dict__["subprocess"]),
            "run",
            lambda *_args, **_kwargs: type(
                "Proc",
                (),
                {"returncode": 0, "stdout": stdout, "stderr": ""},
            )(),
        )

        with pytest.raises(ValueError, match=match):
            _swarmalator_mojo.swarmalator_step_mojo(*_direct_payload())

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("owner", ["mojo", "julia", "go"])
    def test_optional_backend_loaders_return_callable_numeric_kernels(
        self, owner: str
    ) -> None:
        """Each real optional loader returns a kernel that follows the law."""
        if owner not in sw_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        loader = {
            "mojo": sw_mod._load_mojo_fn,
            "julia": sw_mod._load_julia_fn,
            "go": sw_mod._load_go_fn,
        }[owner]
        _assert_kernel_follows_law(owner, loader())


class TestPublicBackendOutputContracts:
    """Public swarmalator dispatchers replay direct backend output contracts."""

    @pytest.mark.parametrize(
        ("constructor_args", "match"),
        [
            (("2", 2, 0.01), "numeric-string"),
            ((2, "2", 0.01), "numeric-string"),
            ((2, 2, "0.01"), "numeric-string"),
            ((0, 2, 0.01), "n_agents"),
            ((2, 2, True), "dt"),
            ((2, 2, 0.0), "dt"),
        ],
    )
    def test_constructor_rejects_numeric_string_aliases(
        self,
        constructor_args: tuple[object, object, object],
        match: str,
    ) -> None:
        # Malformed constructor arguments are passed on purpose, so the engine
        # constructor is bound loosely.
        make_engine: Callable[..., SwarmalatorEngine] = SwarmalatorEngine
        with pytest.raises(ValueError, match=match):
            make_engine(*constructor_args)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"a": "1.0"}, "numeric-string"),
            ({"b": "1.0"}, "numeric-string"),
            ({"j": "0.8"}, "numeric-string"),
            ({"k": "1.2"}, "numeric-string"),
        ],
    )
    def test_step_rejects_numeric_string_scalar_aliases(
        self,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        engine = SwarmalatorEngine(2, 2, 0.01)
        # Malformed scalar keywords are passed on purpose, so ``step`` is bound
        # loosely.
        step_unchecked: Callable[..., object] = engine.step

        with pytest.raises(ValueError, match=match):
            step_unchecked(
                np.array([[0.0, 0.5], [1.0, -0.25]], dtype=np.float64),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
                **kwargs,
            )

    @pytest.mark.parametrize(
        ("pos", "phases", "omegas", "match"),
        [
            (
                np.array([["0.0", "0.5"], ["1.0", "-0.25"]], dtype=object),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
                "numeric-string",
            ),
            (
                np.array([[0.0, 0.5], [1.0, -0.25]], dtype=np.float64),
                np.array(["0.2", "0.4"], dtype=object),
                np.array([0.1, -0.2], dtype=np.float64),
                "numeric-string",
            ),
            (
                np.array([[0.0, 0.5], [1.0, -0.25]], dtype=np.float64),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, "-0.2"], dtype=object),
                "numeric-string",
            ),
        ],
    )
    def test_step_rejects_numeric_string_state_aliases(
        self,
        pos: object,
        phases: object,
        omegas: object,
        match: str,
    ) -> None:
        engine = SwarmalatorEngine(2, 2, 0.01)
        # Malformed state arrays are passed on purpose, so ``step`` is bound loosely.
        step_unchecked: Callable[..., object] = engine.step

        with pytest.raises(ValueError, match=match):
            step_unchecked(pos, phases, omegas)

    def test_run_rejects_numeric_string_step_count(self) -> None:
        engine = SwarmalatorEngine(2, 2, 0.01)
        # A numeric-string step count is passed on purpose, so ``run`` is bound
        # loosely.
        run_unchecked: Callable[..., object] = engine.run
        string_steps: object = "2"

        with pytest.raises(ValueError, match="numeric-string"):
            run_unchecked(
                np.array([[0.0, 0.5], [1.0, -0.25]], dtype=np.float64),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
                n_steps=string_steps,
            )

    def test_order_parameter_rejects_numeric_string_phase_aliases(self) -> None:
        engine = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match="numeric-string"):
            engine.order_parameter(np.array(["0.2", "0.4"], dtype=object))

    def test_order_parameter_rejects_scalar_numeric_string_alias(self) -> None:
        engine = SwarmalatorEngine(2, 2, 0.01)
        # A scalar string is passed on purpose, so ``order_parameter`` is bound
        # loosely.
        order_parameter_unchecked: Callable[..., float] = engine.order_parameter
        string_phase: object = "0.2"

        with pytest.raises(ValueError, match="numeric-string"):
            order_parameter_unchecked(string_phase)

    def test_step_rejects_optional_backend_numeric_string_aliases(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def malformed_backend(
            _pos: FloatArray,
            _phases: FloatArray,
            _omegas: FloatArray,
            _n: int,
            _dim: int,
            _a: float,
            _b: float,
            _j: float,
            _k: float,
            _dt: float,
        ) -> tuple[object, object]:
            return (
                np.array([["0.1", "0.2"], ["0.3", "0.4"]], dtype=object),
                np.array([0.5, "0.6"], dtype=object),
            )

        monkeypatch.setattr(sw_mod, "_dispatch", lambda: malformed_backend)
        engine = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match="numeric-string"):
            engine.step(
                np.array([[0.0, 0.5], [1.0, -0.25]], dtype=np.float64),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
            )

    def test_step_rejects_boolean_coupling_alias(
        self,
    ) -> None:
        engine = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match="a must be finite real"):
            engine.step(
                np.array([[0.0, 0.5], [1.0, -0.25]], dtype=np.float64),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
                a=True,
            )

    def test_step_rejects_nonfinite_coupling(
        self,
    ) -> None:
        engine = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match="b must be finite real"):
            engine.step(
                np.array([[0.0, 0.5], [1.0, -0.25]], dtype=np.float64),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
                b=np.inf,
            )

    def test_step_rejects_boolean_state_array(self) -> None:
        engine = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match="pos must be real-valued, not boolean"):
            engine.step(
                np.array([[True, False], [False, True]]),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
            )

    def test_step_rejects_wrong_shape_state_array(self) -> None:
        engine = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match="pos shape"):
            engine.step(
                np.array([0.0, 0.5, 1.0], dtype=np.float64),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
            )

    def test_step_rejects_nonfinite_state_array(self) -> None:
        engine = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match="pos must contain only finite values"):
            engine.step(
                np.array([[0.0, np.inf], [1.0, -0.25]], dtype=np.float64),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
            )

    def test_step_rejects_complex_position_input(
        self,
    ) -> None:
        engine = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match="pos"):
            engine.step(
                np.array([[0.0 + 1.0j, 0.5], [1.0, -0.25]], dtype=np.complex128),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
            )

    def test_step_rejects_uncoercible_position_input(
        self,
    ) -> None:
        engine = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match="pos"):
            engine.step(
                np.array([[object(), 0.5], [1.0, -0.25]], dtype=object),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
            )

    def test_step_rejects_optional_backend_boolean_position_alias(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def malformed_backend(
            _pos: FloatArray,
            _phases: FloatArray,
            _omegas: FloatArray,
            _n: int,
            _dim: int,
            _a: float,
            _b: float,
            _j: float,
            _k: float,
            _dt: float,
        ) -> tuple[object, FloatArray]:
            return (
                np.array([[True, 0.2], [0.3, 0.4]], dtype=object),
                np.array([0.5, 0.6], dtype=np.float64),
            )

        monkeypatch.setattr(sw_mod, "_dispatch", lambda: malformed_backend)
        engine = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match="not boolean"):
            engine.step(
                np.array([[0.0, 0.5], [1.0, -0.25]], dtype=np.float64),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
            )

    def test_step_rejects_optional_backend_boolean_phase_alias(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def malformed_backend(
            _pos: FloatArray,
            _phases: FloatArray,
            _omegas: FloatArray,
            _n: int,
            _dim: int,
            _a: float,
            _b: float,
            _j: float,
            _k: float,
            _dt: float,
        ) -> tuple[FloatArray, object]:
            return (
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
                np.array([True, 0.6], dtype=object),
            )

        monkeypatch.setattr(sw_mod, "_dispatch", lambda: malformed_backend)
        engine = SwarmalatorEngine(2, 2, 0.01)

        with pytest.raises(ValueError, match="not boolean"):
            engine.step(
                np.array([[0.0, 0.5], [1.0, -0.25]], dtype=np.float64),
                np.array([0.2, 0.4], dtype=np.float64),
                np.array([0.1, -0.2], dtype=np.float64),
            )

    def test_rust_wrapper_rejects_boolean_phase_alias(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class PySwarmalatorStepper:
            def __init__(self, _n: int, _dim: int, _dt: float) -> None:
                pass

            def step(
                self,
                _pos: FloatArray,
                _phases: FloatArray,
                _omegas: FloatArray,
                _a: float,
                _b: float,
                _j: float,
                _k: float,
            ) -> tuple[FloatArray, object]:
                return (
                    np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
                    np.array([True, 0.6], dtype=object),
                )

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.__dict__["PySwarmalatorStepper"] = PySwarmalatorStepper
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        with pytest.raises(ValueError, match="not boolean"):
            sw_mod._load_rust_fn()(*_direct_payload(n=2, dim=2))


class TestDispatchFallbackChain:
    def test_dispatch_returns_python_when_every_native_loader_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = sw_mod.ACTIVE_BACKEND
        previous_available = list(sw_mod.AVAILABLE_BACKENDS)
        previous_go_loader = sw_mod._LOADERS["go"]
        previous_rust_loader = sw_mod._LOADERS["rust"]
        sw_mod.ACTIVE_BACKEND = "go"
        sw_mod.AVAILABLE_BACKENDS = ["rust"]
        sw_mod._BACKEND_CACHE.clear()

        def failing_loader() -> BackendFn:
            raise ImportError("native backend unavailable")

        monkeypatch.setitem(sw_mod._LOADERS, "go", failing_loader)
        monkeypatch.setitem(sw_mod._LOADERS, "rust", failing_loader)
        try:
            backend = sw_mod._dispatch()
        finally:
            sw_mod.ACTIVE_BACKEND = previous_backend
            sw_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(sw_mod._LOADERS, "go", previous_go_loader)
            monkeypatch.setitem(sw_mod._LOADERS, "rust", previous_rust_loader)
            sw_mod._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_falls_back_to_python_when_loader_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = sw_mod.ACTIVE_BACKEND
        previous_available = list(sw_mod.AVAILABLE_BACKENDS)
        previous_loader = sw_mod._LOADERS["go"]
        sw_mod.ACTIVE_BACKEND = "go"
        sw_mod.AVAILABLE_BACKENDS = ["go", "python"]
        sw_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            sw_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend = sw_mod._dispatch()
        finally:
            sw_mod.ACTIVE_BACKEND = previous_backend
            sw_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(sw_mod._LOADERS, "go", previous_loader)
            sw_mod._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_uses_cached_loader_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = sw_mod.ACTIVE_BACKEND
        previous_available = list(sw_mod.AVAILABLE_BACKENDS)
        previous_loader = sw_mod._LOADERS["go"]
        sw_mod.ACTIVE_BACKEND = "go"
        sw_mod.AVAILABLE_BACKENDS = ["go", "python"]
        sw_mod._BACKEND_CACHE.clear()
        call_count = 0

        def fake_backend(
            pos: FloatArray,
            phases: FloatArray,
            _omegas: FloatArray,
            _n: int,
            _dim: int,
            _a: float,
            _b: float,
            _j: float,
            _k: float,
            _dt: float,
        ) -> tuple[FloatArray, FloatArray]:
            return pos.copy(), phases.copy()

        def loader() -> BackendFn:
            nonlocal call_count
            call_count += 1
            return fake_backend

        monkeypatch.setitem(sw_mod._LOADERS, "go", loader)
        try:
            b1 = sw_mod._dispatch()
            b2 = sw_mod._dispatch()
        finally:
            sw_mod.ACTIVE_BACKEND = previous_backend
            sw_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(sw_mod._LOADERS, "go", previous_loader)
            sw_mod._BACKEND_CACHE.clear()

        assert b1 is fake_backend
        assert b2 is fake_backend
        assert call_count == 1
