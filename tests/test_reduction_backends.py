# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for Ott-Antonsen reduction

"""Cross-backend parity for ``OttAntonsenReduction.run``.

The OA kernel is a scalar complex-ODE RK4 loop: five backends
should agree bit-for-bit modulo the subprocess text-round-trip
epsilon on Mojo.
"""

from __future__ import annotations

import cmath
import contextlib
import ctypes
import importlib
import math
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _reduction_go,
    _reduction_julia,
    _reduction_mojo,
)
from scpn_phase_orchestrator.upde import (
    _reduction_validation as reduction_validation,
)
from scpn_phase_orchestrator.upde import reduction as r_mod
from scpn_phase_orchestrator.upde.reduction import OAState, OttAntonsenReduction

TOL = 1e-12
OAOutput = tuple[float, float, float, float]
ReductionRunner = Callable[..., OAOutput]


def test__reduction_validation_helper_is_directly_linked_to_backend_tests() -> None:
    validated = reduction_validation.validate_oa_inputs(
        0.2,
        0.1,
        0.5,
        0.1,
        1.0,
        0.01,
        8,
    )
    output = reduction_validation.validate_oa_output(
        0.3,
        0.2,
        float(np.hypot(0.3, 0.2)),
        float(np.arctan2(0.2, 0.3)),
    )

    assert validated[6] == 8
    assert validated[5] == 0.01
    assert output[2] == float(np.hypot(0.3, 0.2))
    assert output[3] == float(np.arctan2(0.2, 0.3))


def test_reduction_validator_rejects_non_integer_step_count() -> None:
    with pytest.raises(TypeError, match="n_steps must be an integer"):
        reduction_validation.validate_oa_inputs(0.2, 0.1, 0.5, 0.1, 1.0, 0.01, True)


def test_reduction_validator_rejects_radius_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        reduction_validation.validate_oa_output(0.2, 0.0, 1.2, 0.0)


def test_reduction_validator_accepts_undefined_phase_at_zero_radius() -> None:
    output = reduction_validation.validate_oa_output(0.0, 0.0, 0.0, 2.5)

    assert output == (0.0, 0.0, 0.0, 2.5)


@contextlib.contextmanager
def _force_backend(name: str) -> Iterator[None]:
    prev = r_mod.ACTIVE_BACKEND
    r_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        r_mod.ACTIVE_BACKEND = prev


def _run(
    backend: str,
    *,
    z_re: float = 0.2,
    z_im: float = 0.1,
    omega_0: float = 0.5,
    delta: float = 0.1,
    K: float = 1.0,
    dt: float = 0.01,
    n_steps: int = 500,
) -> OAState:
    if backend not in r_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    red = OttAntonsenReduction(omega_0=omega_0, delta=delta, K=K, dt=dt)
    with _force_backend(backend):
        return red.run(complex(z_re, z_im), n_steps=n_steps)


def _mojo_proc(stdout: str) -> object:
    return type("Proc", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()


class _FakeGoReductionLib:
    def __init__(self, output: tuple[float, float, float, float], rc: int = 0) -> None:
        self.output = output
        self.rc = rc

    def OARun(self, *_args: object) -> int:
        output_refs = (
            ctypes.cast(cast(Any, arg), ctypes.POINTER(ctypes.c_double))
            for arg in _args[-4:]
        )
        for output_ref, value in zip(output_refs, self.output, strict=True):
            output_ref.contents.value = value
        return self.rc


class _FakeJuliaReductionModule:
    def __init__(self, output: tuple[float, float, float, float]) -> None:
        self.output = output

    def oa_run(self, *_args: object) -> tuple[float, float, float, float]:
        return self.output


def _install_reduction_output(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    output: tuple[float, float, float, float],
) -> None:
    if module is _reduction_go:
        monkeypatch.setattr(
            _reduction_go,
            "_load_lib",
            lambda: _FakeGoReductionLib(output),
        )
        return
    if module is _reduction_julia:
        monkeypatch.setattr(
            _reduction_julia,
            "_ensure",
            lambda: _FakeJuliaReductionModule(output),
        )
        return
    mojo_subprocess = cast(Any, _reduction_mojo).subprocess
    monkeypatch.setattr(_reduction_mojo, "_ensure_exe", lambda: "reduction")
    monkeypatch.setattr(
        mojo_subprocess,
        "run",
        lambda *_args, **_kwargs: _mojo_proc(
            "\n".join(str(value) for value in output) + "\n"
        ),
    )


class TestDirectReductionBoundaryContracts:
    @pytest.mark.parametrize(
        "module",
        [_reduction_go, _reduction_julia, _reduction_mojo],
        ids=["go", "julia", "mojo"],
    )
    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((True, 0.0, 0.5, 0.1, 1.0, 0.01, 8), "z_re must be"),
            ((0.8, 0.8, 0.5, 0.1, 1.0, 0.01, 8), "OA unit disk"),
            ((0.2, 0.1, float("nan"), 0.1, 1.0, 0.01, 8), "omega_0"),
            ((0.2, 0.1, 0.5, -0.1, 1.0, 0.01, 8), "delta"),
            ((0.2, 0.1, 0.5, 0.1, 1.0, 0.0, 8), "dt"),
            ((0.2, 0.1, 0.5, 0.1, 1.0, 0.01, 0), "n_steps"),
        ],
    )
    def test_direct_backend_rejects_invalid_inputs_before_runtime_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: object,
        args: tuple[object, ...],
        match: str,
    ) -> None:
        if module is _reduction_go:
            monkeypatch.setattr(
                _reduction_go,
                "_load_lib",
                lambda: pytest.fail("Go runtime must not be loaded"),
            )
            runner = _reduction_go.oa_run_go
        elif module is _reduction_julia:
            monkeypatch.setattr(
                _reduction_julia,
                "_ensure",
                lambda: pytest.fail("Julia runtime must not be loaded"),
            )
            runner = _reduction_julia.oa_run_julia
        else:
            monkeypatch.setattr(
                _reduction_mojo,
                "_ensure_exe",
                lambda: pytest.fail("Mojo runtime must not be loaded"),
            )
            runner = _reduction_mojo.oa_run_mojo

        runner = cast(ReductionRunner, runner)
        with pytest.raises((TypeError, ValueError), match=match):
            runner(*args)

    @pytest.mark.parametrize(
        "module",
        [_reduction_go, _reduction_julia, _reduction_mojo],
        ids=["go", "julia", "mojo"],
    )
    def test_direct_backend_accepts_valid_unit_disk_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: object,
    ) -> None:
        _install_reduction_output(monkeypatch, module, (0.2, 0.0, 0.2, 0.0))

        if module is _reduction_go:
            got = _reduction_go.oa_run_go(0.2, 0.0, 0.5, 0.1, 1.0, 0.01, 8)
        elif module is _reduction_julia:
            got = _reduction_julia.oa_run_julia(0.2, 0.0, 0.5, 0.1, 1.0, 0.01, 8)
        else:
            got = _reduction_mojo.oa_run_mojo(0.2, 0.0, 0.5, 0.1, 1.0, 0.01, 8)

        assert got == (0.2, 0.0, 0.2, 0.0)

    @pytest.mark.parametrize(
        "module",
        [_reduction_go, _reduction_julia, _reduction_mojo],
        ids=["go", "julia", "mojo"],
    )
    @pytest.mark.parametrize(
        ("output", "match"),
        [
            ((0.8, 0.8, 1.1313708498984762, 0.7853981633974483), "unit disk"),
            ((0.2, 0.0, 0.5, 0.0), "R must match"),
            ((0.0, 0.2, 0.2, 0.0), "psi must match"),
        ],
    )
    def test_direct_backend_rejects_non_physical_outputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: object,
        output: tuple[float, float, float, float],
        match: str,
    ) -> None:
        _install_reduction_output(monkeypatch, module, output)

        if module is _reduction_go:
            runner = _reduction_go.oa_run_go
        elif module is _reduction_julia:
            runner = _reduction_julia.oa_run_julia
        else:
            runner = _reduction_mojo.oa_run_mojo

        with pytest.raises(ValueError, match=match):
            runner(0.2, 0.0, 0.5, 0.1, 1.0, 0.01, 8)

    def test_direct_julia_backend_rejects_boolean_alias_outputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_reduction_output(monkeypatch, _reduction_julia, (True, 0.0, 1.0, 0.0))

        with pytest.raises(TypeError, match="backend z_re output"):
            _reduction_julia.oa_run_julia(0.2, 0.0, 0.5, 0.1, 1.0, 0.01, 8)


class TestDirectJuliaLoaderBoundaryContracts:
    def test_ensure_returns_cached_julia_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cached = object()
        monkeypatch.setattr(_reduction_julia, "_JULIA_MODULE", cached)
        monkeypatch.setattr(
            _reduction_julia,
            "require_julia_main",
            lambda: pytest.fail("cached module must avoid Julia runtime probing"),
        )

        assert _reduction_julia._ensure() is cached

    def test_ensure_rejects_missing_julia_side_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(_reduction_julia, "_JULIA_MODULE", None)
        monkeypatch.setattr(_reduction_julia, "_JULIA_FILE", tmp_path / "missing.jl")
        monkeypatch.setattr(_reduction_julia, "require_julia_main", lambda: object())

        with pytest.raises(ImportError, match="julia side-file not found"):
            _reduction_julia._ensure()

    def test_ensure_loads_existing_julia_side_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        side_file = tmp_path / "reduction.jl"
        side_file.write_text("module ReductionJL\nend\n", encoding="utf-8")
        reduction_module = object()
        includes: list[str] = []

        class FakeJuliaMain:
            ReductionJL = reduction_module

            @staticmethod
            def include(path: str) -> None:
                includes.append(path)

        monkeypatch.setattr(_reduction_julia, "_JULIA_MODULE", None)
        monkeypatch.setattr(_reduction_julia, "_JULIA_FILE", side_file)
        monkeypatch.setattr(
            _reduction_julia,
            "require_julia_main",
            lambda: FakeJuliaMain,
        )

        assert _reduction_julia._ensure() is reduction_module
        assert includes == [str(side_file)]


class TestPublicReductionOutputContracts:
    @pytest.mark.parametrize(
        ("output", "match"),
        [
            ((0.2, 0.0, 0.5, 0.0), "R must match"),
            ((0.0, 0.2, 0.2, 0.0), "psi must match"),
            ((True, 0.0, 1.0, 0.0), "backend z_re output"),
        ],
    )
    def test_public_run_rejects_invalid_optional_backend_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        output: tuple[object, object, object, object],
        match: str,
    ) -> None:
        def fake_backend(*_args: object) -> tuple[object, object, object, object]:
            return output

        reducer = OttAntonsenReduction(omega_0=0.0, delta=0.1, K=1.0, dt=0.01)
        monkeypatch.setattr(
            r_mod, "_dispatch", lambda _backend="auto": ("go", fake_backend)
        )

        with pytest.raises((TypeError, ValueError), match=match):
            reducer.run(complex(0.2, 0.0), n_steps=8)

    def test_public_run_rejects_invalid_step_count_before_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reducer = OttAntonsenReduction(omega_0=0.0, delta=0.1, K=1.0, dt=0.01)
        monkeypatch.setattr(
            r_mod,
            "_dispatch",
            lambda: pytest.fail("invalid n_steps must not reach backend dispatch"),
        )

        with pytest.raises(ValueError, match="n_steps must be a positive integer"):
            reducer.run(complex(0.2, 0.0), n_steps=False)

    @pytest.mark.parametrize("z0", ["0.2", complex(float("nan"), 0.0)])
    def test_public_run_rejects_non_complex_initial_state_before_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        z0: object,
    ) -> None:
        reducer = OttAntonsenReduction(omega_0=0.0, delta=0.1, K=1.0, dt=0.01)
        monkeypatch.setattr(
            r_mod,
            "_dispatch",
            lambda: pytest.fail("invalid z0 must not reach backend dispatch"),
        )

        with pytest.raises(ValueError, match="z0 must be a finite complex scalar"):
            reducer.run(cast(complex, z0), n_steps=8)

    def test_predict_from_oscillators_rejects_uncoercible_array_input(
        self,
    ) -> None:
        class BadArray:
            def __array__(self, _dtype: object | None = None) -> object:
                raise ValueError("cannot expose array")

        reducer = OttAntonsenReduction(omega_0=0.0, delta=0.1, K=1.0, dt=0.01)

        with pytest.raises(ValueError, match="finite one-dimensional array"):
            reducer.predict_from_oscillators(
                cast(NDArray[np.float64], BadArray()),
                K=1.0,
            )

    def test_subcritical_steady_state_uses_python_floor_when_rust_absent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reducer = OttAntonsenReduction(omega_0=0.0, delta=1.0, K=1.5, dt=0.01)
        monkeypatch.setattr(r_mod, "_HAS_RUST_SCALAR", False)

        assert reducer.steady_state_R() == 0.0

    @pytest.mark.parametrize(
        ("output", "match"),
        [
            ("0.5", "real scalar"),
            (True, "real scalar"),
            (float("nan"), "finite"),
            (-0.1, r"\[0, 1\]"),
            (1.1, r"\[0, 1\]"),
        ],
    )
    def test_steady_state_rejects_invalid_rust_scalar_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        output: object,
        match: str,
    ) -> None:
        reducer = OttAntonsenReduction(omega_0=0.0, delta=0.1, K=1.0, dt=0.01)
        monkeypatch.setattr(r_mod, "_HAS_RUST_SCALAR", True)
        monkeypatch.setattr(
            r_mod,
            "_rust_steady_state_r",
            lambda *_args: output,
            raising=False,
        )

        with pytest.raises((TypeError, ValueError), match=match):
            reducer.steady_state_R()


class TestDirectMojoBoundaryContracts:
    @pytest.mark.parametrize(
        ("stdout", "match"),
        [
            ("", "Mojo OARUN returned 0 lines, expected 4"),
            ("0.1\n0.2\n0.3\n0.4\n0.5\n", "expected 4"),
            ("0.1\n\n0.3\n0.4\n", "finite z_real"),
            ("0.1\nbad\n0.3\n0.4\n", "finite z_real"),
            ("0.1\nnan\n0.3\n0.4\n", "finite z_real"),
            ("0.1\n0.2\ninf\n0.4\n", "finite z_real"),
            ("0.1\n0.2\n0.3\n-inf\n", "finite z_real"),
        ],
    )
    def test_mojo_runner_rejects_malformed_raw_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        match: str,
    ) -> None:
        mojo_subprocess = cast(Any, _reduction_mojo).subprocess
        monkeypatch.setattr(_reduction_mojo, "_ensure_exe", lambda: "reduction")
        monkeypatch.setattr(
            mojo_subprocess,
            "run",
            lambda *_args, **_kwargs: _mojo_proc(stdout),
        )

        with pytest.raises(ValueError, match=match):
            _reduction_mojo.oa_run_mojo(0.2, 0.1, 0.5, 0.1, 1.0, 0.01, 8)


class TestBackendParity:
    def test_rust_matches_python(self) -> None:
        ref = _run("python")
        got = _run("rust")
        assert abs(got.z.real - ref.z.real) < TOL
        assert abs(got.z.imag - ref.z.imag) < TOL
        assert abs(got.R - ref.R) < TOL
        assert abs(got.psi - ref.psi) < TOL

    def test_julia_matches_python(self) -> None:
        ref = _run("python")
        got = _run("julia")
        assert abs(got.z.real - ref.z.real) < TOL
        assert abs(got.z.imag - ref.z.imag) < TOL

    def test_go_matches_python(self) -> None:
        ref = _run("python")
        got = _run("go")
        assert abs(got.z.real - ref.z.real) < TOL
        assert abs(got.z.imag - ref.z.imag) < TOL

    def test_mojo_matches_python(self) -> None:
        ref = _run("python")
        got = _run("mojo")
        # Text round-trip tolerance on scalar RK4 over 500 steps.
        assert abs(got.z.real - ref.z.real) < 1e-10
        assert abs(got.z.imag - ref.z.imag) < 1e-10


class TestSubcriticalParity:
    def test_rust_subcritical(self) -> None:
        ref = _run("python", delta=1.0, K=1.5, n_steps=1000)
        got = _run("rust", delta=1.0, K=1.5, n_steps=1000)
        assert abs(got.R - ref.R) < TOL

    def test_go_subcritical(self) -> None:
        ref = _run("python", delta=1.0, K=1.5, n_steps=1000)
        got = _run("go", delta=1.0, K=1.5, n_steps=1000)
        assert abs(got.R - ref.R) < TOL


class TestHypothesisParity:
    @given(
        delta=st.floats(min_value=0.05, max_value=0.5),
        K_ratio=st.floats(min_value=0.5, max_value=4.0),
        n_steps=st.integers(min_value=10, max_value=500),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rust_hypothesis(
        self,
        delta: float,
        K_ratio: float,
        n_steps: int,
    ) -> None:
        if "rust" not in r_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        K = K_ratio * delta
        ref = _run("python", delta=delta, K=K, n_steps=n_steps)
        got = _run("rust", delta=delta, K=K, n_steps=n_steps)
        assert abs(got.z.real - ref.z.real) < TOL
        assert abs(got.z.imag - ref.z.imag) < TOL

    @given(
        delta=st.floats(min_value=0.05, max_value=0.5),
        K_ratio=st.floats(min_value=0.5, max_value=4.0),
        n_steps=st.integers(min_value=10, max_value=500),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(
        self,
        delta: float,
        K_ratio: float,
        n_steps: int,
    ) -> None:
        if "go" not in r_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        K = K_ratio * delta
        ref = _run("python", delta=delta, K=K, n_steps=n_steps)
        got = _run("go", delta=delta, K=K, n_steps=n_steps)
        assert abs(got.z.real - ref.z.real) < TOL
        assert abs(got.z.imag - ref.z.imag) < TOL


def _analytic_oa_amplitude(r0: float, delta: float, k: float, t: float) -> float:
    """Return the OA amplitude ``R(t)`` from ``dR/dt = (K/2 − Δ)R − (K/2)R³``."""
    rate = k / 2 - delta
    r_star_sq = 1 - 2 * delta / k
    growth = (r_star_sq / r0**2 - 1) * math.exp(-2 * rate * t)
    return math.sqrt(r_star_sq / (1 + growth))


def _assert_loader_follows_oa(owner: str, loader: Callable[[], object]) -> None:
    """Run the runner a real loader returns against the analytic OA solution.

    With ``dz/dt = −(Δ + iω₀)z + (K/2)(1 − |z|²)z`` the phase turns at
    exactly ``−ω₀`` and the amplitude follows the logistic law, so
    ``z(t) = R(t)·exp(i(ψ₀ − ω₀ t))``.
    """
    if owner not in r_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"{owner} unavailable")
    run = cast("Callable[..., OAOutput]", loader())
    z0, omega_0, delta, k, dt, steps = cmath.rect(0.05, 0.4), 0.7, 0.1, 1.0, 0.01, 800
    z_re, z_im, r, psi = run(z0.real, z0.imag, omega_0, delta, k, dt, steps)
    t = dt * steps
    expected = cmath.rect(_analytic_oa_amplitude(0.05, delta, k, t), 0.4 - omega_0 * t)
    atol = 1e-8 if owner == "mojo" else 1e-9
    assert complex(z_re, z_im) == pytest.approx(expected, abs=atol)
    assert r == pytest.approx(abs(expected), abs=atol)
    assert cmath.rect(1.0, psi) == pytest.approx(
        cmath.rect(1.0, cmath.phase(expected)), abs=atol
    )


class TestOptionalLoaderSuccessPaths:
    @pytest.mark.native_runtime
    def test_rust_loader_wraps_spo_kernel_function(self) -> None:
        """The compiled Rust loader integrates the analytic OA solution."""
        _assert_loader_follows_oa("rust", r_mod._load_rust_fn)

    def test_rust_loader_rejects_inconsistent_backend_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def oa_run_rust(*_args: object) -> OAOutput:
            return 0.2, 0.0, 0.5, 0.0

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo_dynamic = cast(Any, fake_spo)
        fake_spo_dynamic.oa_run_rust = oa_run_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        run = r_mod._load_rust_fn()
        with pytest.raises(ValueError, match="R must match"):
            run(0.01, 0.02, 0.5, 0.1, 1.0, 0.01, 3)

    @pytest.mark.parametrize("value", ["0.01", True])
    def test_rust_loader_rejects_coercible_input_aliases_before_kernel_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: object,
    ) -> None:
        calls: list[tuple[object, ...]] = []

        def oa_run_rust(*args: object) -> OAOutput:
            calls.append(args)
            return 0.1, 0.0, 0.1, 0.0

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo_dynamic = cast(Any, fake_spo)
        fake_spo_dynamic.oa_run_rust = oa_run_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        run = r_mod._load_rust_fn()
        with pytest.raises(TypeError, match="dt must be a real scalar"):
            run(0.01, 0.02, 0.5, 0.1, 1.0, value, 3)
        assert calls == []

    @pytest.mark.native_runtime
    def test_mojo_loader_runs_availability_probe(self) -> None:
        """The real Mojo loader runs its probe and integrates the OA solution."""
        _assert_loader_follows_oa("mojo", r_mod._load_mojo_fn)

    @pytest.mark.native_runtime
    def test_julia_loader_requires_juliacall_and_returns_runner(self) -> None:
        """The real Julia loader returns a runner for the OA solution."""
        _assert_loader_follows_oa("julia", r_mod._load_julia_fn)

    @pytest.mark.native_runtime
    def test_go_loader_runs_shared_library_probe(self) -> None:
        """The real Go loader probes its library and integrates the OA solution."""
        _assert_loader_follows_oa("go", r_mod._load_go_fn)

    @pytest.mark.native_runtime
    def test_scalar_rust_helpers_feed_steady_state_and_fit(self) -> None:
        """The compiled scalar helpers give the analytic steady state and fit."""
        if not r_mod._HAS_RUST_SCALAR:
            pytest.skip("Rust scalar helpers not built")
        red = OttAntonsenReduction(
            omega_0=0.0, delta=0.2, K=1.0, dt=0.01, backend="rust"
        )
        assert red.steady_state_R() == pytest.approx(math.sqrt(1 - 0.4), abs=1e-15)
        assert (
            OttAntonsenReduction(0.0, 0.6, 1.0, backend="rust").steady_state_R() == 0.0
        )

        sample = np.array([0.9, 1.1, 1.4, 1.5, 2.0, 2.6, 3.0])
        omega_0, delta = sorted(sample.tolist())[3], (2.6 - 1.1) / 2
        state = red.predict_from_oscillators(sample, K=4.0)
        assert state.K_c == pytest.approx(2 * delta)
        assert (
            pytest.approx(_analytic_oa_amplitude(0.01, delta, 4.0, 10.0), abs=1e-8)
            == state.R
        )
        assert omega_0 == 1.5

    def test_module_import_detects_scalar_rust_helpers(self) -> None:
        """Import wires the compiled scalar helpers exactly when they are installed."""
        installed = importlib.util.find_spec("spo_kernel") is not None
        assert r_mod._HAS_RUST_SCALAR is installed
        if installed:
            kernel = importlib.import_module("spo_kernel")
            assert r_mod._rust_fit_lorentzian is kernel.fit_lorentzian_rust
            assert r_mod._rust_steady_state_r is kernel.steady_state_r_oa_rust


class TestDispatchFallbackChain:
    def test_dispatch_returns_python_floor_after_all_optional_loaders_fail(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = r_mod.ACTIVE_BACKEND
        previous_available = list(r_mod.AVAILABLE_BACKENDS)
        previous_loader = r_mod._LOADERS["go"]
        r_mod.ACTIVE_BACKEND = "go"
        r_mod.AVAILABLE_BACKENDS = ["go"]
        r_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            r_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend = r_mod._dispatch()
        finally:
            r_mod.ACTIVE_BACKEND = previous_backend
            r_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(r_mod._LOADERS, "go", previous_loader)
            r_mod._BACKEND_CACHE.clear()

        assert backend == ("python", None)

    def test_dispatch_falls_back_to_python_when_loader_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = r_mod.ACTIVE_BACKEND
        previous_available = list(r_mod.AVAILABLE_BACKENDS)
        previous_loader = r_mod._LOADERS["go"]
        r_mod.ACTIVE_BACKEND = "go"
        r_mod.AVAILABLE_BACKENDS = ["go", "python"]
        r_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            r_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend = r_mod._dispatch()
        finally:
            r_mod.ACTIVE_BACKEND = previous_backend
            r_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(r_mod._LOADERS, "go", previous_loader)
            r_mod._BACKEND_CACHE.clear()

        assert backend == ("python", None)

    def test_dispatch_uses_cached_loader_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        previous_backend = r_mod.ACTIVE_BACKEND
        previous_available = list(r_mod.AVAILABLE_BACKENDS)
        previous_loader = r_mod._LOADERS["go"]
        r_mod.ACTIVE_BACKEND = "go"
        r_mod.AVAILABLE_BACKENDS = ["go", "python"]
        r_mod._BACKEND_CACHE.clear()
        call_count = 0

        def fake_backend(
            z_re: float,
            z_im: float,
            _omega_0: float,
            _delta: float,
            _k_coupling: float,
            _dt: float,
            _n_steps: int,
        ) -> tuple[float, float, float, float]:
            return z_re, z_im, 0.0, 0.0

        def loader() -> ReductionRunner:
            nonlocal call_count
            call_count += 1
            return fake_backend

        monkeypatch.setitem(r_mod._LOADERS, "go", loader)
        try:
            b1 = r_mod._dispatch()
            b2 = r_mod._dispatch()
        finally:
            r_mod.ACTIVE_BACKEND = previous_backend
            r_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(r_mod._LOADERS, "go", previous_loader)
            r_mod._BACKEND_CACHE.clear()

        assert b1 == ("go", fake_backend)
        assert b2 == ("go", fake_backend)
        assert call_count == 1


class TestNamedReductionOwners:
    """``backend=`` selects the owner of the integration and of the scalars."""

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("owner", ["python", "rust", "mojo", "julia", "go"])
    def test_each_named_owner_runs_the_analytic_solution(self, owner: str) -> None:
        """``run`` on a named owner follows ``z(t) = R(t)·exp(i(ψ₀ − ω₀ t))``."""
        if owner not in r_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        assert r_mod.reduction_backend(owner) == owner
        red = OttAntonsenReduction(0.7, 0.1, 1.0, dt=0.01, backend=owner)
        state = red.run(cmath.rect(0.05, 0.4), 800)
        expected = cmath.rect(_analytic_oa_amplitude(0.05, 0.1, 1.0, 8.0), 0.4 - 5.6)
        assert state.z == pytest.approx(expected, abs=1e-8 if owner == "mojo" else 1e-9)

    def test_python_scalars_give_the_analytic_steady_state(self) -> None:
        """The Python owner computes ``√(1 − 2Δ/K)`` itself."""
        red = OttAntonsenReduction(0.0, 0.2, 1.0, backend="python")
        assert red.steady_state_R() == pytest.approx(math.sqrt(0.6), abs=1e-15)

    def test_named_owner_refusals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Absent named owners and unknown names refuse."""
        with pytest.raises(ValueError, match="unknown reduction backend"):
            OttAntonsenReduction(0.0, 0.1, 1.0, backend="cuda")
        monkeypatch.setattr(r_mod, "_BACKEND_CACHE", {})
        monkeypatch.setitem(
            r_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(OSError("libreduction.so missing")),
        )
        with pytest.raises(RuntimeError, match="'go' unavailable"):
            r_mod.reduction_backend("go")
        monkeypatch.setattr(r_mod, "_HAS_RUST_SCALAR", False)
        with pytest.raises(RuntimeError, match="'rust' unavailable"):
            OttAntonsenReduction(0.0, 0.1, 1.0, backend="rust").steady_state_R()
