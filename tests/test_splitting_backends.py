# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for Strang splitting

"""Cross-backend parity for ``SplittingEngine.run``."""

from __future__ import annotations

import contextlib
import cProfile
import ctypes
import math
import types
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _splitting_go,
    _splitting_julia,
    _splitting_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _splitting_validation as splitting_validation,
)
from scpn_phase_orchestrator.upde import splitting as sp_mod
from scpn_phase_orchestrator.upde.splitting import SplittingEngine
from tests.typing_contracts import assert_precise_ndarray_hint

splitting_run_go = _splitting_go.splitting_run_go
splitting_run_julia = _splitting_julia.splitting_run_julia
splitting_run_mojo = _splitting_mojo.splitting_run_mojo

TWO_PI = 2.0 * math.pi
TOL = 1e-12


def test__splitting_validation_helper_is_directly_linked_to_backend_tests() -> None:
    args = _valid_direct_args()

    validated = splitting_validation.validate_splitting_inputs(*args)
    output = splitting_validation.validate_splitting_output(
        np.array([0.2, 0.4], dtype=np.float64),
        n=2,
    )

    assert validated[4] == 2
    assert validated[8] == 1
    np.testing.assert_array_equal(output, np.array([0.2, 0.4], dtype=np.float64))


def _valid_direct_args():
    return (
        np.array([0.1, 0.3], dtype=np.float64),
        np.array([0.2, -0.1], dtype=np.float64),
        np.array([0.0, 0.4, 0.2, 0.0], dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        2,
        0.0,
        0.0,
        0.01,
        1,
    )


def _with_direct_arg(index: int, value):
    args = list(_valid_direct_args())
    args[index] = value
    return tuple(args)


class _ArrayProtocolFailure:
    def __array__(self, *_args, **_kwargs):
        raise TypeError("array protocol failed")


class _FakeGoSplittingLib:
    def __init__(self, output: tuple[float, ...], rc: int = 0):
        self.output = output
        self.rc = rc

    def SplittingRun(self, *_args):
        output_ref = ctypes.cast(_args[-1], ctypes.POINTER(ctypes.c_double))
        for index, value in enumerate(self.output):
            output_ref[index] = value
        return self.rc


class _FakeJuliaSplittingModule:
    def __init__(self, output: tuple[float, ...]):
        self.output = output

    def splitting_run(self, *_args):
        return np.array(self.output, dtype=np.float64)


def _call_direct_backend(module, monkeypatch, output: tuple[float, ...]):
    if module is _splitting_go:
        monkeypatch.setattr(
            _splitting_go,
            "_load_lib",
            lambda: _FakeGoSplittingLib(output),
        )
        return _splitting_go.splitting_run_go(*_valid_direct_args())
    if module is _splitting_julia:
        monkeypatch.setattr(
            _splitting_julia,
            "_ensure",
            lambda: _FakeJuliaSplittingModule(output),
        )
        return _splitting_julia.splitting_run_julia(*_valid_direct_args())

    monkeypatch.setattr(_splitting_mojo, "_ensure_exe", lambda: "splitting")
    monkeypatch.setattr(
        _splitting_mojo.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0,
            stdout="".join(f"{value}\n" for value in output),
            stderr="",
        ),
    )
    return _splitting_mojo.splitting_run_mojo(*_valid_direct_args())


@contextlib.contextmanager
def _force_backend(name: str):
    prev = sp_mod.ACTIVE_BACKEND
    sp_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        sp_mod.ACTIVE_BACKEND = prev


def _problem(seed: int, n: int = 6, alpha_nonzero: bool = False):
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
    zeta: float = 0.0,
    psi: float = 0.0,
    alpha_nonzero: bool = False,
):
    if backend not in sp_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    theta, omegas, knm, alpha = _problem(seed, n, alpha_nonzero)
    eng = SplittingEngine(n, 0.01)
    with _force_backend(backend):
        return eng.run(theta, omegas, knm, zeta, psi, alpha, n_steps=n_steps)


class TestAlphaZero:
    def test_rust(self):
        ref = _run_backend("python", 0)
        got = _run_backend("rust", 0)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia(self):
        ref = _run_backend("python", 1)
        got = _run_backend("julia", 1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go(self):
        ref = _run_backend("python", 2)
        got = _run_backend("go", 2)
        assert np.max(np.abs(got - ref)) < TOL

    def test_mojo(self):
        ref = _run_backend("python", 3, n=5)
        got = _run_backend("mojo", 3, n=5)
        assert np.max(np.abs(got - ref)) < 1e-10


class TestAlphaNonZero:
    def test_rust(self):
        ref = _run_backend("python", 4, alpha_nonzero=True, zeta=0.5, psi=1.1)
        got = _run_backend("rust", 4, alpha_nonzero=True, zeta=0.5, psi=1.1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia(self):
        ref = _run_backend("python", 5, alpha_nonzero=True, zeta=0.5, psi=1.1)
        got = _run_backend("julia", 5, alpha_nonzero=True, zeta=0.5, psi=1.1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go(self):
        ref = _run_backend("python", 6, alpha_nonzero=True, zeta=0.5, psi=1.1)
        got = _run_backend("go", 6, alpha_nonzero=True, zeta=0.5, psi=1.1)
        assert np.max(np.abs(got - ref)) < TOL


class TestHypothesisParity:
    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rust_hypothesis(self, n, seed):
        if "rust" not in sp_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        ref = _run_backend("python", seed, n=n)
        got = _run_backend("rust", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL

    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(self, n, seed):
        if "go" not in sp_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        ref = _run_backend("python", seed, n=n)
        got = _run_backend("go", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL


class TestBackendTypingContracts:
    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (splitting_run_go, "go"),
            (splitting_run_julia, "julia"),
            (splitting_run_mojo, "mojo"),
        ],
    )
    def test_backend_annotations_use_float64_ndarray(self, fn, label: str) -> None:
        hints = get_type_hints(fn)
        for name in ("phases", "omegas", "knm_flat", "alpha_flat", "return"):
            text = str(hints[name])
            assert_precise_ndarray_hint(
                hints[name],
                context=f"{label}:{name}",
            )
            assert "numpy.float64" in text, f"{label}:{name} missing float64 annotation"


class TestDirectSplittingBoundaryContracts:
    @pytest.mark.parametrize(
        ("module", "fn_name", "loader_name"),
        [
            (_splitting_go, "splitting_run_go", "_load_lib"),
            (_splitting_julia, "splitting_run_julia", "_ensure"),
            (_splitting_mojo, "splitting_run_mojo", "_ensure_exe"),
        ],
    )
    @pytest.mark.parametrize(
        ("args", "match"),
        [
            (_with_direct_arg(0, np.array([True, False])), "phases"),
            (_with_direct_arg(0, np.array([0.1])), "phases"),
            (
                _with_direct_arg(
                    0,
                    np.array(["0.1", "0.2"], dtype=object),
                ),
                "numeric-string",
            ),
            (_with_direct_arg(1, np.array([0.1 + 0.0j, 0.2])), "omegas"),
            (_with_direct_arg(1, np.array([0.1])), "omegas length"),
            (
                _with_direct_arg(
                    1,
                    np.array(["0.1", "0.2"], dtype=object),
                ),
                "numeric-string",
            ),
            (_with_direct_arg(2, np.zeros(3, dtype=np.float64)), "knm_flat"),
            (
                _with_direct_arg(
                    2,
                    np.array(["0.0", "0.4", "0.2", "0.0"], dtype=object),
                ),
                "numeric-string",
            ),
            (
                _with_direct_arg(
                    2,
                    np.array([1.0, 0.4, 0.2, 0.0], dtype=np.float64),
                ),
                "diagonal",
            ),
            (_with_direct_arg(3, np.zeros(3, dtype=np.float64)), "alpha_flat"),
            (
                _with_direct_arg(
                    3,
                    np.array(["0.0", "0.0", "0.0", "0.0"], dtype=object),
                ),
                "numeric-string",
            ),
            (_with_direct_arg(4, True), "n"),
            (_with_direct_arg(4, 1.5), "n"),
            (_with_direct_arg(5, True), "zeta"),
            (_with_direct_arg(5, "0.0"), "zeta"),
            (_with_direct_arg(5, np.nan), "zeta"),
            (_with_direct_arg(7, 0.0), "dt"),
            (_with_direct_arg(7, -0.01), "dt"),
            (_with_direct_arg(8, 0), "n_steps"),
        ],
    )
    def test_direct_backend_rejects_invalid_inputs_before_runtime_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module,
        fn_name: str,
        loader_name: str,
        args,
        match: str,
    ) -> None:
        def forbidden_loader():
            raise AssertionError("runtime loader must not be called")

        monkeypatch.setattr(module, loader_name, forbidden_loader)

        with pytest.raises(ValueError, match=match):
            getattr(module, fn_name)(*args)

    def test_direct_validation_output_rejects_numeric_string_aliases(self) -> None:
        with pytest.raises(ValueError, match="numeric-string"):
            splitting_validation.validate_splitting_output(
                np.array(["0.2", "0.4"], dtype=object),
                n=2,
            )

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            ("0.2", "numeric-string"),
            (np.array(["", "bad"], dtype=object), "numeric"),
            (np.zeros((1, 2), dtype=np.float64), "one-dimensional"),
            (np.array([np.nan, 0.2], dtype=np.float64), "finite"),
        ],
    )
    def test_direct_validation_output_defensive_failures(
        self,
        value,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            splitting_validation.validate_splitting_output(value, n=2)

    def test_direct_numeric_string_probe_ignores_unarrayable_value(self) -> None:
        assert not splitting_validation._contains_numeric_string_alias(
            _ArrayProtocolFailure(),
        )

    def test_direct_julia_backend_rejects_numeric_string_output_before_coercion(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        module = types.SimpleNamespace(
            splitting_run=lambda *_args: np.array(["0.2", "0.4"], dtype=object),
        )
        monkeypatch.setattr(_splitting_julia, "_ensure", lambda: module)

        with pytest.raises(ValueError, match="numeric-string"):
            _splitting_julia.splitting_run_julia(*_valid_direct_args())

    def test_direct_julia_loader_returns_cached_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cached = object()
        monkeypatch.setattr(_splitting_julia, "_JULIA_MODULE", cached)

        assert _splitting_julia._ensure() is cached

    def test_direct_julia_loader_rejects_missing_side_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        monkeypatch.setattr(_splitting_julia, "_JULIA_MODULE", None)
        monkeypatch.setattr(_splitting_julia, "_JULIA_FILE", tmp_path / "missing.jl")
        monkeypatch.setattr(_splitting_julia, "require_julia_main", lambda: object())

        with pytest.raises(ImportError, match="julia side-file not found"):
            _splitting_julia._ensure()

    def test_direct_julia_loader_includes_side_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        included: list[str] = []
        module = object()
        side_file = tmp_path / "splitting.jl"
        side_file.write_text("module SplittingJL end\n")
        fake_main = types.SimpleNamespace(
            include=lambda path: included.append(path),
            SplittingJL=module,
        )
        monkeypatch.setattr(_splitting_julia, "_JULIA_MODULE", None)
        monkeypatch.setattr(_splitting_julia, "_JULIA_FILE", side_file)
        monkeypatch.setattr(
            _splitting_julia,
            "require_julia_main",
            lambda: fake_main,
        )

        assert _splitting_julia._ensure() is module
        assert included == [str(side_file)]

    @pytest.mark.parametrize(
        "module",
        [_splitting_go, _splitting_julia, _splitting_mojo],
    )
    def test_direct_backend_accepts_valid_torus_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module,
    ) -> None:
        got = _call_direct_backend(module, monkeypatch, (0.2, 0.4))

        np.testing.assert_allclose(got, np.array([0.2, 0.4], dtype=np.float64))
        assert got.dtype == np.float64

    @pytest.mark.parametrize(
        "module",
        [_splitting_go, _splitting_julia, _splitting_mojo],
    )
    @pytest.mark.parametrize("output", [(-0.1, 0.2), (0.1, TWO_PI)])
    def test_direct_backend_rejects_out_of_torus_outputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module,
        output: tuple[float, ...],
    ) -> None:
        with pytest.raises(ValueError, match="phases"):
            _call_direct_backend(module, monkeypatch, output)

    @pytest.mark.parametrize("module", [_splitting_julia, _splitting_mojo])
    def test_direct_backend_rejects_wrong_output_length(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module,
    ) -> None:
        with pytest.raises(ValueError, match="length 2|expected 2"):
            _call_direct_backend(module, monkeypatch, (0.2,))


class TestDirectMojoBoundaryContracts:
    @pytest.mark.parametrize(
        ("stdout", "match"),
        [
            ("", "Mojo SPLIT returned 0 lines, expected 2"),
            ("0.1\n0.2\n0.3\n", "Mojo SPLIT returned 3 lines, expected 2"),
            ("0.1\n\n0.2\n", "Mojo SPLIT returned 3 lines, expected 2"),
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
        monkeypatch.setattr(_splitting_mojo, "_ensure_exe", lambda: "splitting")
        monkeypatch.setattr(
            _splitting_mojo.subprocess,
            "run",
            lambda *_args, **_kwargs: types.SimpleNamespace(
                returncode=0,
                stdout=stdout,
                stderr="",
            ),
        )

        with pytest.raises(ValueError, match=match):
            _splitting_mojo.splitting_run_mojo(
                np.zeros(2, dtype=np.float64),
                np.ones(2, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                2,
                0.0,
                0.0,
                0.01,
                1,
            )


def _oracle_strang(
    phases: list[float],
    omegas: list[float],
    knm: list[list[float]],
    alpha: list[list[float]],
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> list[float]:
    """Strang-split by plain loops: drift dt/2, RK4 coupling dt, drift dt/2."""
    two_pi = 2 * math.pi
    n = len(phases)

    def coupling(th: list[float]) -> list[float]:
        """Return Σⱼ K_ij sin(θⱼ − θᵢ − α_ij) + ζ sin(ψ − θᵢ)."""
        return [
            math.fsum(
                knm[i][j] * math.sin(th[j] - th[i] - alpha[i][j]) for j in range(n)
            )
            + zeta * math.sin(psi - th[i])
            for i in range(n)
        ]

    def shifted(base: list[float], slope: list[float], h: float) -> list[float]:
        """Return ``(base + h · slope) mod 2π``."""
        return [(b + h * k) % two_pi for b, k in zip(base, slope, strict=True)]

    p = list(phases)
    for _ in range(n_steps):
        p = shifted(p, omegas, dt / 2)
        k1 = coupling(p)
        k2 = coupling(shifted(p, k1, dt / 2))
        k3 = coupling(shifted(p, k2, dt / 2))
        k4 = coupling(shifted(p, k3, dt))
        p = [
            (p[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])) % two_pi
            for i in range(n)
        ]
        p = shifted(p, omegas, dt / 2)
    return p


def _strang_case(alpha_nonzero: bool) -> tuple[np.ndarray, ...]:
    """Return phases, frequencies, coupling and lags for a five-oscillator case."""
    rng = np.random.default_rng(107)
    n = 5
    knm = rng.uniform(0.0, 0.7, (n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(-0.2, 0.2, (n, n)) if alpha_nonzero else np.zeros((n, n))
    return rng.uniform(0.0, 2 * np.pi, n), rng.normal(1.0, 0.4, n), knm, alpha


def _assert_close_to_oracle(got: object, owner: str, alpha_nonzero: bool) -> None:
    """Compare a 25-step result with the oracle, wrapping the phase difference."""
    phases, omegas, knm, alpha = _strang_case(alpha_nonzero)
    expected = _oracle_strang(
        phases.tolist(),
        omegas.tolist(),
        knm.tolist(),
        alpha.tolist(),
        0.4,
        1.1,
        0.02,
        25,
    )
    gap = (np.asarray(got) - np.asarray(expected) + np.pi) % (2 * np.pi) - np.pi
    assert float(np.max(np.abs(gap))) < (1e-9 if owner == "mojo" else 1e-11)


class TestBackendLoaderDispatch:
    """The real loaders return kernels that follow the Strang law."""

    @pytest.mark.native_runtime
    def test_rust_loader_preserves_contiguous_float64_contract(self) -> None:
        """The compiled Rust loader accepts strided input and returns float64."""
        if "rust" not in sp_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        kernel = sp_mod._load_rust_fn()
        for alpha_nonzero in (False, True):
            phases, omegas, knm, alpha = _strang_case(alpha_nonzero)
            strided = np.repeat(phases, 2)[::2]
            assert not strided.flags.c_contiguous
            got = kernel(
                strided, omegas, knm.ravel(), alpha.ravel(), 5, 0.4, 1.1, 0.02, 25
            )
            assert np.asarray(got).dtype == np.float64
            _assert_close_to_oracle(got, "rust", alpha_nonzero)

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("owner", ["mojo", "julia", "go"])
    def test_toolchain_loaders_return_callable_with_dependencies_stubbed(
        self, owner: str
    ) -> None:
        """Each real optional loader returns a kernel that follows the law."""
        if owner not in sp_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        loader = {
            "mojo": sp_mod._load_mojo_fn,
            "julia": sp_mod._load_julia_fn,
            "go": sp_mod._load_go_fn,
        }[owner]
        kernel = loader()
        for alpha_nonzero in (False, True):
            phases, omegas, knm, alpha = _strang_case(alpha_nonzero)
            args = (knm.ravel(), alpha.ravel(), 5, 0.4, 1.1, 0.02, 25)
            _assert_close_to_oracle(kernel(phases, omegas, *args), owner, alpha_nonzero)


class TestBackendResolutionContracts:
    """Loading, resolution and selection with the real owners."""

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("owner", ["rust", "mojo", "julia", "go"])
    def test_load_backend_uses_cache(
        self, owner: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two loads on a cold cache run the real loader once."""
        if owner not in sp_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        monkeypatch.setattr(sp_mod, "_BACKEND_CACHE", {})
        profile = cProfile.Profile()
        profile.enable()
        try:
            first = sp_mod._load_backend(owner)
            second = sp_mod._load_backend(owner)
        finally:
            profile.disable()
        loads = sum(
            entry.callcount
            for entry in profile.getstats()
            if not isinstance(entry.code, str)
            and entry.code.co_name == f"_load_{owner}_fn"
        )
        assert loads == 1
        assert first is second

    def test_probe_marks_backend_failure_as_infinite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A loader that fails during probing is timed as infinitely slow."""

        def failing_backend() -> object:
            """Fail like a broken toolchain."""
            raise RuntimeError("backend unavailable")

        monkeypatch.setitem(sp_mod._LOADERS, "rust", failing_backend)
        monkeypatch.setattr(sp_mod, "_BACKEND_CACHE", {})

        assert sp_mod._splitting_probe_seconds("rust") == float("inf")

    def test_resolve_backends_falls_back_to_python_when_all_backends_fail_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With every native loader failing, resolution keeps Python only."""

        def fail_loader() -> object:
            """Fail like a broken toolchain."""
            raise RuntimeError("backend unavailable")

        for name in ("rust", "mojo", "julia", "go"):
            monkeypatch.setitem(sp_mod._LOADERS, name, fail_loader)
        monkeypatch.setattr(sp_mod, "_BACKEND_CACHE", {})

        active, available = sp_mod._resolve_backends()

        assert active == "python"
        assert available == ["python"]

    def test_resolve_backends_prefers_fastest_loaded_backend(self) -> None:
        """Resolution lists the working loaders and picks a finite, fastest probe."""
        loadable = []
        for name in ("rust", "mojo", "julia", "go"):
            try:
                sp_mod._LOADERS[name]()
            except (ImportError, RuntimeError, OSError, KeyError):
                continue
            loadable.append(name)

        active, available = sp_mod._resolve_backends()

        assert available == [*loadable, "python"]
        assert active in available
        assert sp_mod._splitting_probe_seconds(active) < float("inf")

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("owner", ["python", "rust", "mojo", "julia", "go"])
    def test_run_uses_selected_backend_for_positive_timestep(self, owner: str) -> None:
        """``SplittingEngine(backend=owner).run`` runs that owner on the law."""
        if owner not in sp_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        assert sp_mod.splitting_backend(owner) == owner
        for alpha_nonzero in (False, True):
            phases, omegas, knm, alpha = _strang_case(alpha_nonzero)
            eng = SplittingEngine(5, dt=0.02, backend=owner)
            got = eng.run(phases, omegas, knm, 0.4, 1.1, alpha, n_steps=25)
            _assert_close_to_oracle(got, owner, alpha_nonzero)

    def test_named_owner_refusals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown, absent, failing or negative-dt named owners refuse."""
        zero = np.zeros((2, 2))
        with pytest.raises(ValueError, match="unknown splitting backend"):
            SplittingEngine(2, 0.01, backend="cuda")
        with pytest.raises(ValueError, match="'go' needs a positive dt"):
            SplittingEngine(2, -0.01, backend="go").run(
                np.zeros(2), np.ones(2), zero, 0.0, 0.0, zero, n_steps=1
            )
        monkeypatch.setattr(sp_mod, "_BACKEND_CACHE", {})

        def crash(*_args: object) -> object:
            """Fail like a crashed kernel."""
            raise RuntimeError("kernel crashed")

        monkeypatch.setitem(
            sp_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(OSError("libsplitting.so missing")),
        )
        with pytest.raises(RuntimeError, match="'go' unavailable"):
            sp_mod.splitting_backend("go")
        monkeypatch.setitem(sp_mod._LOADERS, "go", lambda: crash)
        with pytest.raises(RuntimeError, match="'go' failed"):
            SplittingEngine(2, 0.01, backend="go").run(
                np.zeros(2), np.ones(2), zero, 0.0, 0.0, zero, n_steps=1
            )
        monkeypatch.setattr(sp_mod, "ACTIVE_BACKEND", "go")
        with pytest.raises(RuntimeError, match="kernel crashed"):
            SplittingEngine(2, 0.01).run(
                np.zeros(2), np.ones(2), zero, 0.0, 0.0, zero, n_steps=1
            )

    def test_run_uses_python_path_for_negative_timestep(self, monkeypatch):
        called = False

        def explode(*_args, **_kwargs):
            nonlocal called
            called = True
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        monkeypatch.setattr(sp_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(sp_mod, "_load_backend", lambda _name: explode)

        n = 4
        eng = SplittingEngine(n, dt=-0.01)
        phases = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)
        omegas = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        knm = np.full((n, n), 0.0, dtype=np.float64)
        np.fill_diagonal(knm, 0.0)

        out = eng.run(
            phases,
            omegas,
            knm,
            0.0,
            0.0,
            np.zeros((n, n), dtype=np.float64),
            n_steps=2,
        )
        assert not called
        expected = (phases - 0.02 * omegas) % (2.0 * np.pi)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, expected)
