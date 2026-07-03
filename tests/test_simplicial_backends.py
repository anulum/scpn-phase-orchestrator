# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for simplicial Kuramoto

"""Cross-backend parity for the ``simplicial_run`` kernel."""

from __future__ import annotations

import contextlib
import ctypes
import math
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any, TypeAlias, cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _simplicial_go,
    _simplicial_julia,
    _simplicial_mojo,
)
from scpn_phase_orchestrator.upde import (
    _simplicial_validation as simplicial_validation,
)
from scpn_phase_orchestrator.upde import simplicial as s_mod
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine

TWO_PI = 2.0 * math.pi
TOL = 1e-12
FloatArray: TypeAlias = NDArray[np.float64]
DirectArgs: TypeAlias = tuple[object, ...]
DirectRunner: TypeAlias = Callable[..., FloatArray]
DirectArgMutator: TypeAlias = Callable[[DirectArgs], DirectArgs]


def test__simplicial_validation_helper_is_directly_linked_to_backend_tests() -> None:
    args = _valid_direct_args()

    validated = simplicial_validation.validate_simplicial_inputs(*args)
    output = simplicial_validation.validate_simplicial_output(
        np.array([0.2, 0.4], dtype=np.float64),
        n=2,
    )

    assert validated[4] == 2
    assert validated[9] == 1
    np.testing.assert_array_equal(output, np.array([0.2, 0.4], dtype=np.float64))


@contextlib.contextmanager
def _force_backend(name: str) -> Iterator[None]:
    prev = s_mod.ACTIVE_BACKEND
    s_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        s_mod.ACTIVE_BACKEND = prev


def _problem(
    seed: int,
    n: int = 6,
    alpha_nonzero: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
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


def _run_backend(
    backend: str,
    seed: int,
    n: int = 6,
    n_steps: int = 20,
    sigma2: float = 0.5,
    zeta: float = 0.0,
    psi: float = 0.0,
    alpha_nonzero: bool = False,
) -> FloatArray:
    if backend not in s_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    theta, omegas, knm, alpha = _problem(seed, n, alpha_nonzero)
    eng = SimplicialEngine(n, 0.01, sigma2=sigma2)
    with _force_backend(backend):
        return eng.run(theta, omegas, knm, zeta, psi, alpha, n_steps=n_steps)


def _valid_direct_args() -> DirectArgs:
    return (
        np.array([0.1, 0.3], dtype=np.float64),
        np.array([0.2, -0.1], dtype=np.float64),
        np.array([0.0, 0.4, 0.2, 0.0], dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        2,
        0.0,
        0.0,
        0.5,
        0.01,
        1,
    )


def _call_direct_runner(runner: DirectRunner, args: DirectArgs) -> FloatArray:
    """Call a direct backend runner with deliberately dynamic test arguments."""
    return runner(*args)


class _FakeGoSimplicialLib:
    def __init__(self, output: tuple[float, ...], rc: int = 0) -> None:
        self.output = output
        self.rc = rc

    def SimplicialRun(self, *_args: object) -> int:
        output_ref = ctypes.cast(
            cast(Any, _args[-1]),
            ctypes.POINTER(ctypes.c_double),
        )
        for index, value in enumerate(self.output):
            output_ref[index] = value
        return self.rc


class _FakeJuliaSimplicialModule:
    def __init__(self, output: tuple[float, ...]) -> None:
        self.output = output

    def simplicial_run(self, *_args: object) -> np.ndarray:
        return np.asarray(self.output, dtype=np.float64)


class _AstypeFailure:
    """Array-like test double that exercises the state coercion error path."""

    dtype = np.dtype(np.float64)

    def astype(self, *_args: object, **_kwargs: object) -> FloatArray:
        """Raise the same exception family as a failed NumPy coercion."""
        raise TypeError("failed coercion")


def _mojo_proc_from_output(output: tuple[float, ...]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        returncode=0,
        stdout="\n".join(str(value) for value in output) + "\n",
        stderr="",
    )


def _mojo_subprocess_module() -> types.ModuleType:
    """Return the private subprocess module used by the Mojo shim."""
    return cast(types.ModuleType, cast(Any, _simplicial_mojo).subprocess)


def _install_direct_output(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    output: tuple[float, ...],
) -> None:
    if module is _simplicial_go:
        monkeypatch.setattr(
            _simplicial_go,
            "_load_lib",
            lambda: _FakeGoSimplicialLib(output),
        )
        return
    if module is _simplicial_julia:
        monkeypatch.setattr(
            _simplicial_julia,
            "_ensure",
            lambda: _FakeJuliaSimplicialModule(output),
        )
        return
    monkeypatch.setattr(_simplicial_mojo, "_ensure_exe", lambda: "simplicial")
    monkeypatch.setattr(
        _mojo_subprocess_module(),
        "run",
        lambda *_args, **_kwargs: _mojo_proc_from_output(output),
    )


class TestDirectSimplicialBoundaryContracts:
    @pytest.mark.parametrize(
        "module",
        [_simplicial_go, _simplicial_julia, _simplicial_mojo],
        ids=["go", "julia", "mojo"],
    )
    @pytest.mark.parametrize(
        ("mutator", "match"),
        [
            (
                lambda args: (
                    np.array([False, True]),
                    *args[1:],
                ),
                "phases",
            ),
            (
                lambda args: (
                    np.array([0.1], dtype=np.float64),
                    *args[1:],
                ),
                "phases",
            ),
            (
                lambda args: (
                    args[0],
                    np.array([1.0 + 0.0j, 1.0 + 0.1j]),
                    *args[2:],
                ),
                "omegas",
            ),
            (
                lambda args: (
                    *args[:2],
                    np.array([0.0, 0.2, 0.0], dtype=np.float64),
                    *args[3:],
                ),
                "knm_flat",
            ),
            (
                lambda args: (
                    *args[:2],
                    np.array([1.0, 0.2, 0.4, 0.0], dtype=np.float64),
                    *args[3:],
                ),
                "diagonal",
            ),
            (
                lambda args: (*args[:4], True, *args[5:]),
                "n",
            ),
            (
                lambda args: (*args[:7], -0.1, *args[8:]),
                "sigma2",
            ),
            (
                lambda args: (*args[:8], 0.0, args[9]),
                "dt",
            ),
            (
                lambda args: (*args[:9], -1),
                "n_steps",
            ),
        ],
    )
    def test_direct_backend_rejects_invalid_inputs_before_runtime_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: object,
        mutator: DirectArgMutator,
        match: str,
    ) -> None:
        args = mutator(_valid_direct_args())
        if module is _simplicial_go:
            monkeypatch.setattr(
                _simplicial_go,
                "_load_lib",
                lambda: pytest.fail("Go runtime must not be loaded"),
            )
            runner: DirectRunner = _simplicial_go.simplicial_run_go
        elif module is _simplicial_julia:
            monkeypatch.setattr(
                _simplicial_julia,
                "_ensure",
                lambda: pytest.fail("Julia runtime must not be loaded"),
            )
            runner = _simplicial_julia.simplicial_run_julia
        else:
            monkeypatch.setattr(
                _simplicial_mojo,
                "_ensure_exe",
                lambda: pytest.fail("Mojo runtime must not be loaded"),
            )
            runner = _simplicial_mojo.simplicial_run_mojo

        with pytest.raises(ValueError, match=match):
            runner(*args)

    @pytest.mark.parametrize(
        "module",
        [_simplicial_go, _simplicial_julia, _simplicial_mojo],
        ids=["go", "julia", "mojo"],
    )
    def test_direct_backend_zero_steps_returns_phase_copy_without_runtime_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: object,
    ) -> None:
        args = (*_valid_direct_args()[:9], 0)
        if module is _simplicial_go:
            monkeypatch.setattr(
                _simplicial_go,
                "_load_lib",
                lambda: pytest.fail("Go runtime must not be loaded"),
            )
            runner: DirectRunner = _simplicial_go.simplicial_run_go
        elif module is _simplicial_julia:
            monkeypatch.setattr(
                _simplicial_julia,
                "_ensure",
                lambda: pytest.fail("Julia runtime must not be loaded"),
            )
            runner = _simplicial_julia.simplicial_run_julia
        else:
            monkeypatch.setattr(
                _simplicial_mojo,
                "_ensure_exe",
                lambda: pytest.fail("Mojo runtime must not be loaded"),
            )
            runner = _simplicial_mojo.simplicial_run_mojo

        got = _call_direct_runner(runner, args)
        np.testing.assert_allclose(got, cast(FloatArray, args[0]), atol=0.0, rtol=0.0)
        assert got is not args[0]

    @pytest.mark.parametrize(
        "module",
        [_simplicial_go, _simplicial_julia, _simplicial_mojo],
        ids=["go", "julia", "mojo"],
    )
    def test_direct_backend_accepts_valid_torus_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: object,
    ) -> None:
        _install_direct_output(monkeypatch, module, (0.2, 0.4))

        if module is _simplicial_go:
            runner: DirectRunner = _simplicial_go.simplicial_run_go
        elif module is _simplicial_julia:
            runner = _simplicial_julia.simplicial_run_julia
        else:
            runner = _simplicial_mojo.simplicial_run_mojo
        got = _call_direct_runner(runner, _valid_direct_args())

        np.testing.assert_allclose(got, [0.2, 0.4], atol=1e-12)

    @pytest.mark.parametrize(
        "module",
        [_simplicial_go, _simplicial_julia, _simplicial_mojo],
        ids=["go", "julia", "mojo"],
    )
    @pytest.mark.parametrize("output", [(-0.1, 0.2), (0.1, TWO_PI)])
    def test_direct_backend_rejects_out_of_torus_outputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: object,
        output: tuple[float, float],
    ) -> None:
        _install_direct_output(monkeypatch, module, output)

        if module is _simplicial_go:
            runner: DirectRunner = _simplicial_go.simplicial_run_go
        elif module is _simplicial_julia:
            runner = _simplicial_julia.simplicial_run_julia
        else:
            runner = _simplicial_mojo.simplicial_run_mojo

        with pytest.raises(ValueError, match="phases"):
            _call_direct_runner(runner, _valid_direct_args())

    @pytest.mark.parametrize(
        "module",
        [_simplicial_julia, _simplicial_mojo],
        ids=["julia", "mojo"],
    )
    def test_direct_backend_rejects_wrong_output_length(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: object,
    ) -> None:
        _install_direct_output(monkeypatch, module, (0.2,))

        runner: DirectRunner = (
            _simplicial_julia.simplicial_run_julia
            if module is _simplicial_julia
            else _simplicial_mojo.simplicial_run_mojo
        )
        with pytest.raises(ValueError, match="length 2|expected 2"):
            _call_direct_runner(runner, _valid_direct_args())


class TestPublicSimplicialBoundaryContracts:
    """Public dispatcher contract coverage for optional simplicial backends."""

    @pytest.mark.parametrize("output", [(-0.1, 0.2), (0.1, TWO_PI)])
    def test_public_engine_rejects_out_of_torus_backend_outputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        output: tuple[float, float],
    ) -> None:
        """Reject optional backend phase vectors outside the public torus."""

        def fake_backend(*_args: object) -> np.ndarray:
            return np.asarray(output, dtype=np.float64)

        monkeypatch.setattr(s_mod, "_dispatch", lambda: fake_backend)
        phases, omegas, knm, alpha = _problem(17, n=2)
        engine = SimplicialEngine(2, 0.01, sigma2=0.4)

        with pytest.raises(ValueError, match=r"\[0, 2\*pi\)"):
            engine.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=1)

    def test_public_engine_rejects_negative_step_count(self) -> None:
        """Reject negative integration counts before backend dispatch."""
        phases, omegas, knm, alpha = _problem(19, n=2)
        engine = SimplicialEngine(2, 0.01, sigma2=0.4)

        with pytest.raises(ValueError, match="n_steps"):
            engine.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=-1)

    def test_public_engine_rejects_nonfinite_drive_control(self) -> None:
        """Reject nonfinite scalar controls before backend dispatch."""
        phases, omegas, knm, alpha = _problem(23, n=2)
        engine = SimplicialEngine(2, 0.01, sigma2=0.4)

        with pytest.raises(ValueError, match="zeta"):
            engine.run(phases, omegas, knm, float("inf"), 0.0, alpha, n_steps=1)

    def test_state_validator_wraps_failed_array_coercion(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Normalize low-level ndarray coercion failures to ``ValueError``."""

        def fake_asarray(_value: object) -> _AstypeFailure:
            return _AstypeFailure()

        monkeypatch.setattr(np, "asarray", fake_asarray)

        with pytest.raises(ValueError, match="finite float array"):
            s_mod._validate_state_array(object(), name="phases", shape=(1,))

    def test_rust_loader_rejects_out_of_torus_kernel_outputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject Rust kernel phase vectors outside the direct backend torus."""

        def simplicial_run_rust(*_args: object) -> list[float]:
            return [0.1, TWO_PI]

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.__dict__["simplicial_run_rust"] = simplicial_run_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        run = s_mod._load_rust_fn()

        with pytest.raises(ValueError, match=r"\[0, 2\*pi\)"):
            run(
                np.array([0.0, 0.1], dtype=np.float64),
                np.ones(2, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                2,
                0.0,
                0.0,
                0.4,
                0.01,
                1,
            )


class TestParityAlphaZero:
    def test_rust(self) -> None:
        ref = _run_backend("python", 0)
        got = _run_backend("rust", 0)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia(self) -> None:
        ref = _run_backend("python", 1)
        got = _run_backend("julia", 1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go(self) -> None:
        ref = _run_backend("python", 2)
        got = _run_backend("go", 2)
        assert np.max(np.abs(got - ref)) < TOL

    def test_mojo(self) -> None:
        ref = _run_backend("python", 3, n=5)
        got = _run_backend("mojo", 3, n=5)
        assert np.max(np.abs(got - ref)) < 1e-10


class TestParityAlphaNonZero:
    def test_rust(self) -> None:
        ref = _run_backend("python", 4, alpha_nonzero=True, zeta=0.3, psi=1.1)
        got = _run_backend("rust", 4, alpha_nonzero=True, zeta=0.3, psi=1.1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia(self) -> None:
        ref = _run_backend("python", 5, alpha_nonzero=True, zeta=0.3, psi=1.1)
        got = _run_backend("julia", 5, alpha_nonzero=True, zeta=0.3, psi=1.1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go(self) -> None:
        ref = _run_backend("python", 6, alpha_nonzero=True, zeta=0.3, psi=1.1)
        got = _run_backend("go", 6, alpha_nonzero=True, zeta=0.3, psi=1.1)
        assert np.max(np.abs(got - ref)) < TOL


class TestSigma2Zero:
    """σ₂ = 0 must take the pure-pairwise branch unchanged."""

    def test_rust_sigma2_zero(self) -> None:
        ref = _run_backend("python", 10, sigma2=0.0)
        got = _run_backend("rust", 10, sigma2=0.0)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go_sigma2_zero(self) -> None:
        ref = _run_backend("python", 11, sigma2=0.0)
        got = _run_backend("go", 11, sigma2=0.0)
        assert np.max(np.abs(got - ref)) < TOL


class TestHypothesisParity:
    @given(
        n=st.integers(min_value=3, max_value=8),
        sigma2=st.floats(min_value=0.0, max_value=1.5),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rust_hypothesis(self, n: int, sigma2: float, seed: int) -> None:
        if "rust" not in s_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        ref = _run_backend("python", seed, n=n, sigma2=sigma2)
        got = _run_backend("rust", seed, n=n, sigma2=sigma2)
        assert np.max(np.abs(got - ref)) < TOL

    @given(
        n=st.integers(min_value=3, max_value=8),
        sigma2=st.floats(min_value=0.0, max_value=1.5),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(self, n: int, sigma2: float, seed: int) -> None:
        if "go" not in s_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        ref = _run_backend("python", seed, n=n, sigma2=sigma2)
        got = _run_backend("go", seed, n=n, sigma2=sigma2)
        assert np.max(np.abs(got - ref)) < TOL


class TestDirectMojoBoundaryContracts:
    @pytest.mark.parametrize(
        ("stdout", "match"),
        [
            ("", "Mojo SIMP returned 0 lines, expected 2"),
            ("0.1\n0.2\n0.3\n", "Mojo SIMP returned 3 lines, expected 2"),
            ("0.1\n\n0.2\n", "Mojo SIMP returned 3 lines, expected 2"),
            ("0.1\nnot-a-number\n", "finite phases"),
            ("0.1\nnan\n", "finite phases"),
            ("0.1\n7.0\n", "finite phases"),
        ],
    )
    def test_mojo_runner_rejects_malformed_raw_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(_simplicial_mojo, "_ensure_exe", lambda: "simplicial")
        monkeypatch.setattr(
            _mojo_subprocess_module(),
            "run",
            lambda *_args, **_kwargs: types.SimpleNamespace(
                returncode=0,
                stdout=stdout,
                stderr="",
            ),
        )

        with pytest.raises(ValueError, match=match):
            _simplicial_mojo.simplicial_run_mojo(
                np.zeros(2, dtype=np.float64),
                np.ones(2, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                2,
                0.0,
                0.0,
                0.5,
                0.01,
                1,
            )


class TestOptionalLoaderSuccessPaths:
    def test_rust_loader_wraps_spo_kernel_function(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[object, ...]] = []

        def simplicial_run_rust(*args: object) -> list[float]:
            calls.append(args)
            return [0.1, 0.2, 0.3]

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.__dict__["simplicial_run_rust"] = simplicial_run_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        run = s_mod._load_rust_fn()
        out = run(
            np.array([0.0, 0.1, 0.2], dtype=np.float64),
            np.ones(3, dtype=np.float64),
            np.zeros(9, dtype=np.float64),
            np.zeros(9, dtype=np.float64),
            3,
            0.2,
            0.3,
            0.4,
            0.01,
            2,
        )
        np.testing.assert_allclose(out, [0.1, 0.2, 0.3], atol=1e-12)
        assert calls
        assert calls[0][4] == 3
        assert calls[0][9] == 2

    def test_mojo_loader_runs_availability_probe(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_mojo = types.ModuleType(
            "scpn_phase_orchestrator.experimental.accelerators.upde._simplicial_mojo"
        )
        probe_calls: list[bool] = []

        def _ensure_exe() -> None:
            probe_calls.append(True)

        def simplicial_run_mojo(*args: object) -> FloatArray:
            return np.array([0.4, 0.5], dtype=np.float64)

        fake_mojo.__dict__["_ensure_exe"] = _ensure_exe
        fake_mojo.__dict__["simplicial_run_mojo"] = simplicial_run_mojo
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.experimental.accelerators.upde._simplicial_mojo",
            fake_mojo,
        )

        run = s_mod._load_mojo_fn()
        np.testing.assert_allclose(run(), [0.4, 0.5], atol=1e-12)
        assert probe_calls == [True]

    def test_julia_loader_requires_juliacall_and_returns_runner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_juliacall = types.ModuleType("juliacall")
        fake_juliacall.__dict__["Main"] = object()
        fake_julia = types.ModuleType(
            "scpn_phase_orchestrator.experimental.accelerators.upde._simplicial_julia"
        )

        def simplicial_run_julia(*args: object) -> FloatArray:
            return np.array([0.6, 0.7], dtype=np.float64)

        fake_julia.__dict__["simplicial_run_julia"] = simplicial_run_julia
        monkeypatch.setitem(sys.modules, "juliacall", fake_juliacall)
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.experimental.accelerators.upde._simplicial_julia",
            fake_julia,
        )

        run = s_mod._load_julia_fn()
        np.testing.assert_allclose(run(), [0.6, 0.7], atol=1e-12)

    def test_go_loader_runs_shared_library_probe(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_go = types.ModuleType(
            "scpn_phase_orchestrator.experimental.accelerators.upde._simplicial_go"
        )
        probe_calls: list[bool] = []

        def _load_lib() -> None:
            probe_calls.append(True)

        def simplicial_run_go(*args: object) -> FloatArray:
            return np.array([0.8, 0.9], dtype=np.float64)

        fake_go.__dict__["_load_lib"] = _load_lib
        fake_go.__dict__["simplicial_run_go"] = simplicial_run_go
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.experimental.accelerators.upde._simplicial_go",
            fake_go,
        )

        run = s_mod._load_go_fn()
        np.testing.assert_allclose(run(), [0.8, 0.9], atol=1e-12)
        assert probe_calls == [True]


class TestDispatchFallbackChain:
    def test_dispatch_falls_back_to_python_when_loader_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = s_mod.ACTIVE_BACKEND
        previous_available = list(s_mod.AVAILABLE_BACKENDS)
        previous_loader = s_mod._LOADERS["go"]
        s_mod.ACTIVE_BACKEND = "go"
        s_mod.AVAILABLE_BACKENDS = ["go", "python"]
        s_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            s_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend = s_mod._dispatch()
        finally:
            s_mod.ACTIVE_BACKEND = previous_backend
            s_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(s_mod._LOADERS, "go", previous_loader)
            s_mod._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_returns_none_when_every_loader_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = s_mod.ACTIVE_BACKEND
        previous_available = list(s_mod.AVAILABLE_BACKENDS)
        previous_loader = s_mod._LOADERS["go"]
        s_mod.ACTIVE_BACKEND = "go"
        s_mod.AVAILABLE_BACKENDS = ["go"]
        s_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            s_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(OSError("go backend unavailable")),
        )
        try:
            backend = s_mod._dispatch()
        finally:
            s_mod.ACTIVE_BACKEND = previous_backend
            s_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(s_mod._LOADERS, "go", previous_loader)
            s_mod._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_uses_cached_loader_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = s_mod.ACTIVE_BACKEND
        previous_available = list(s_mod.AVAILABLE_BACKENDS)
        previous_loader = s_mod._LOADERS["go"]
        s_mod.ACTIVE_BACKEND = "go"
        s_mod.AVAILABLE_BACKENDS = ["go", "python"]
        s_mod._BACKEND_CACHE.clear()
        call_count = 0

        def fake_backend(
            phases: np.ndarray,
            _omegas: np.ndarray,
            _knm_flat: np.ndarray,
            _alpha_flat: np.ndarray,
            _n: int,
            _zeta: float,
            _psi: float,
            _sigma2: float,
            _dt: float,
            _n_steps: int,
        ) -> np.ndarray:
            return np.asarray(phases, dtype=np.float64)

        def loader() -> DirectRunner:
            nonlocal call_count
            call_count += 1
            return fake_backend

        monkeypatch.setitem(s_mod._LOADERS, "go", loader)
        try:
            b1 = s_mod._dispatch()
            b2 = s_mod._dispatch()
        finally:
            s_mod.ACTIVE_BACKEND = previous_backend
            s_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(s_mod._LOADERS, "go", previous_loader)
            s_mod._BACKEND_CACHE.clear()

        assert b1 is fake_backend
        assert b2 is fake_backend
        assert call_count == 1
