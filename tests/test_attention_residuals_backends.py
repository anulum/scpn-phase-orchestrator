# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity tests for multi-head AttnRes

"""Per-backend parity tests for the multi-head AttnRes dispatcher.

Complements ``test_attention_residuals.py`` (which covers the
algorithm invariants through whatever backend is active) by
exercising each non-Python backend individually against the NumPy
reference. Any drift between backends is a silent physics bug — the
tests here guard against it.

Each backend is gated on its toolchain being present:

* Rust — always in a working SPO dev environment (built by maturin).
  Expect bit-exact parity (Python and Rust share identical f64
  arithmetic on the same hardware).
* Mojo — needs ``mojo/attnres_mojo`` compiled on disk. Parity is
  ~1e-13 (text round-trip rounding on the 17-digit payload).
* Julia — needs ``juliacall`` installed and ``julia/attnres.jl`` on
  disk. Bit-exact parity.
* Go — needs ``go/libattnres.so`` compiled. Bit-exact parity.

Tests that require a backend gate on ``pytest.skip`` when absent so
CI can run on hosts without the full toolchain matrix.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling import (
    _attnres_validation as attnres_validation,
)
from scpn_phase_orchestrator.coupling import (
    attention_residuals as attnres_mod,
)
from scpn_phase_orchestrator.coupling.attention_residuals import (
    AVAILABLE_BACKENDS,
    attnres_modulate,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling import (
    _attnres_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_go import (
    attnres_modulate_go,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_julia import (
    attnres_modulate_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_mojo import (
    attnres_modulate_mojo,
)
from tests.typing_contracts import assert_precise_ndarray_hint


def test__attnres_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(attnres_validation.validate_attnres_backend_inputs)
    assert callable(attnres_validation.validate_attnres_backend_output)


TWO_PI = 2.0 * np.pi
AttnResDirectBackend = Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        object,
        object,
        object,
        object,
        object,
    ],
    np.ndarray,
]


def _symmetric_knm(n: int, strength: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = rng.uniform(0.0, 2.0 * strength, size=(n, n))
    knm = 0.5 * (half + half.T)
    np.fill_diagonal(knm, 0.0)
    return knm.astype(np.float64)


def _force_backend(
    backend: str, knm: np.ndarray, theta: np.ndarray, **kw: object
) -> np.ndarray:
    saved = attnres_mod.ACTIVE_BACKEND
    try:
        attnres_mod.ACTIVE_BACKEND = backend
        out = attnres_modulate(knm, theta, **kw)
    finally:
        attnres_mod.ACTIVE_BACKEND = saved
    return np.asarray(out, dtype=np.float64)


def _python_reference(knm: np.ndarray, theta: np.ndarray, **kw: object) -> np.ndarray:
    return _force_backend("python", knm, theta, **kw)


def _direct_payload(
    n: int = 3,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    float,
    float,
]:
    knm = _symmetric_knm(n, seed=11).ravel()
    theta = np.linspace(0.0, TWO_PI, n, endpoint=False)
    w = np.zeros((1, 8, 8), dtype=np.float64).ravel()
    return knm, theta, w, w.copy(), w.copy(), w.copy(), n, 1, -1, 1.0, 0.25


def _mojo_proc(stdout: str) -> object:
    return type("Proc", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()


class TestDirectBackendBoundaryContracts:
    """Direct optional AttnRes backends validate before runtime loading."""

    @pytest.mark.parametrize(
        "backend",
        [
            attnres_modulate_go,
            attnres_modulate_julia,
            attnres_modulate_mojo,
        ],
    )
    @pytest.mark.parametrize(
        ("field", "replacement", "error", "match"),
        [
            ("knm", np.array([True] * 9), ValueError, "knm_flat"),
            ("knm", np.array([0.0, np.nan] + [0.0] * 7), ValueError, "finite"),
            ("knm", np.zeros((3, 3)), ValueError, "one-dimensional"),
            ("knm", np.zeros(8), ValueError, "n\\*n"),
            ("theta", np.array([0.0, np.inf, 1.0]), ValueError, "finite"),
            ("theta", np.array([0.0, 1.0 + 0.0j, 2.0]), ValueError, "real-valued"),
            ("theta", np.zeros(2), ValueError, "theta length"),
            ("w_q", np.array([True] * 64), ValueError, "w_q"),
            ("w_k", np.array([0.0, np.inf] + [0.0] * 62), ValueError, "finite"),
            ("w_v", np.zeros(63), ValueError, "w_q, w_k, and w_v"),
            ("w_o", np.zeros(63), ValueError, "w_o"),
            ("n", True, ValueError, "n"),
            ("n", -1, ValueError, "n"),
            ("n_heads", True, ValueError, "n_heads"),
            ("n_heads", 0, ValueError, "n_heads"),
            ("block_size", 0, ValueError, "block_size"),
            ("block_size", True, ValueError, "block_size"),
            ("temperature", 0.0, ValueError, "temperature"),
            ("temperature", np.inf, ValueError, "temperature"),
            ("lambda_", -0.1, ValueError, "lambda_"),
            ("lambda_", np.nan, ValueError, "lambda_"),
        ],
    )
    def test_validation_precedes_runtime_load(
        self,
        backend: AttnResDirectBackend,
        field: str,
        replacement: object,
        error: type[Exception],
        match: str,
    ) -> None:
        payload = list(_direct_payload())
        index = {
            "knm": 0,
            "theta": 1,
            "w_q": 2,
            "w_k": 3,
            "w_v": 4,
            "w_o": 5,
            "n": 6,
            "n_heads": 7,
            "block_size": 8,
            "temperature": 9,
            "lambda_": 10,
        }[field]
        payload[index] = replacement
        with pytest.raises(error, match=match):
            backend(*payload)

    @pytest.mark.parametrize(
        "backend",
        [
            attnres_modulate_go,
            attnres_modulate_julia,
            attnres_modulate_mojo,
        ],
    )
    def test_empty_attnres_returns_empty_vector_before_runtime_load(
        self,
        backend: AttnResDirectBackend,
    ) -> None:
        w = np.zeros(64, dtype=np.float64)
        out = backend(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            w,
            w.copy(),
            w.copy(),
            w.copy(),
            0,
            1,
            -1,
            1.0,
            0.25,
        )
        assert out.dtype == np.float64
        assert out.shape == (0,)


class TestDirectMojoBoundaryContracts:
    """Direct Mojo AttnRes adapter rejects malformed backend stdout."""

    @pytest.mark.parametrize(
        ("stdout", "match"),
        [
            ("", "Mojo returned 0 values, expected 9"),
            ("0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n", "expected 9"),
            ("0\n1\n2\n\n4\n5\n6\n7\n8\n", "finite modulated"),
            ("0\nbad\n2\n3\n4\n5\n6\n7\n8\n", "finite modulated"),
            ("0\nnan\n2\n3\n4\n5\n6\n7\n8\n", "finite modulated"),
            ("0\n1\n2\n3\ninf\n5\n6\n7\n8\n", "finite modulated"),
            ("0\n1\n2\n3\n4\n5\n6\n7\n-inf\n", "finite modulated"),
            ("1\n0\n0\n0\n0\n0\n0\n0\n0\n", "diagonal"),
            ("0\n1\n0\n0\n0\n0\n0\n0\n0\n", "symmetric"),
        ],
    )
    def test_mojo_runner_rejects_malformed_raw_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(_attnres_mojo, "_ensure_exe", lambda: "attnres")
        monkeypatch.setattr(
            _attnres_mojo.subprocess,
            "run",
            lambda *_args, **_kwargs: _mojo_proc(stdout),
        )

        with pytest.raises(ValueError, match=match):
            _attnres_mojo.attnres_modulate_mojo(*_direct_payload())

    def test_mojo_runner_rejects_output_that_creates_zero_edges(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_attnres_mojo, "_ensure_exe", lambda: "attnres")
        monkeypatch.setattr(
            _attnres_mojo.subprocess,
            "run",
            lambda *_args, **_kwargs: _mojo_proc(
                "0\n0.25\n0.5\n0.25\n0\n0\n0.5\n0\n0\n"
            ),
        )
        knm_flat = np.array(
            [
                0.0,
                0.25,
                0.0,
                0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        theta = np.linspace(0.0, TWO_PI, 3, endpoint=False)
        w = np.zeros((1, 8, 8), dtype=np.float64).ravel()

        with pytest.raises(ValueError, match="preserve zero"):
            _attnres_mojo.attnres_modulate_mojo(
                knm_flat,
                theta,
                w,
                w.copy(),
                w.copy(),
                w.copy(),
                3,
                1,
                -1,
                1.0,
                0.25,
            )


class TestBackendTypingContracts:
    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (attnres_modulate_go, "go"),
            (attnres_modulate_julia, "julia"),
            (attnres_modulate_mojo, "mojo"),
        ],
    )
    def test_backend_annotations_use_float64_ndarray(self, fn, label: str) -> None:
        hints = get_type_hints(fn)
        for name in ("knm_flat", "theta", "w_q", "w_k", "w_v", "w_o", "return"):
            text = str(hints[name])
            assert_precise_ndarray_hint(hints[name], context=f"{label}:{name}")
            assert "numpy.float64" in text, f"{label}:{name} missing float64"


# ---------------------------------------------------------------------
# Rust parity
# ---------------------------------------------------------------------


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built on this host")

    @given(
        n=st.integers(min_value=4, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=12,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_bit_exact_parity(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        knm = _symmetric_knm(n, seed=seed)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        py = _python_reference(knm, theta, lambda_=0.5)
        rs = _force_backend("rust", knm, theta, lambda_=0.5)
        np.testing.assert_allclose(rs, py, atol=1e-12)

    def test_lambda_zero_passthrough(self) -> None:
        knm = _symmetric_knm(8, seed=99)
        theta = np.arange(8, dtype=np.float64) * 0.1
        py = _python_reference(knm, theta, lambda_=0.0)
        rs = _force_backend("rust", knm, theta, lambda_=0.0)
        np.testing.assert_array_equal(rs, py)

    def test_block_size_honoured(self) -> None:
        """Rust kernel respects ``block_size`` the same way Python does."""
        n = 12
        rng = np.random.default_rng(3)
        knm = _symmetric_knm(n, seed=3)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        py = _python_reference(knm, theta, block_size=2, lambda_=0.5)
        rs = _force_backend("rust", knm, theta, block_size=2, lambda_=0.5)
        np.testing.assert_allclose(rs, py, atol=1e-12)


# ---------------------------------------------------------------------
# Julia parity
# ---------------------------------------------------------------------


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available on this host")

    @pytest.mark.parametrize("n", [6, 10, 14])
    def test_bit_exact_parity(self, n: int) -> None:
        """juliacall's bootstrap is expensive; use parametrised seeds
        rather than Hypothesis."""
        rng = np.random.default_rng(42 + n)
        knm = _symmetric_knm(n, seed=42 + n)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        py = _python_reference(knm, theta, lambda_=0.5)
        jl = _force_backend("julia", knm, theta, lambda_=0.5)
        np.testing.assert_allclose(jl, py, atol=1e-12)

    def test_symmetry_preserved(self) -> None:
        n = 10
        rng = np.random.default_rng(7)
        knm = _symmetric_knm(n, seed=7)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        jl = _force_backend("julia", knm, theta, lambda_=0.5)
        np.testing.assert_allclose(jl, jl.T, atol=1e-12)


# ---------------------------------------------------------------------
# Go parity
# ---------------------------------------------------------------------


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built on this host")

    @given(
        n=st.integers(min_value=4, max_value=14),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_bit_exact_parity(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        knm = _symmetric_knm(n, seed=seed)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        py = _python_reference(knm, theta, lambda_=0.5)
        go = _force_backend("go", knm, theta, lambda_=0.5)
        np.testing.assert_allclose(go, py, atol=1e-12)

    def test_invalid_block_size_surfaces(self) -> None:
        n = 4
        knm = _symmetric_knm(n, seed=0)
        theta = np.zeros(n)
        # block_size=0 is rejected at the Python layer before Go sees it.
        with pytest.raises(ValueError, match=r"(?i)block"):
            _force_backend("go", knm, theta, block_size=0)


# ---------------------------------------------------------------------
# Mojo parity
# ---------------------------------------------------------------------


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built on this host")

    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_numerical_parity(self, n: int) -> None:
        rng = np.random.default_rng(13 + n)
        knm = _symmetric_knm(n, seed=13 + n)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        py = _python_reference(knm, theta, lambda_=0.5)
        mj = _force_backend("mojo", knm, theta, lambda_=0.5)
        # 17-digit repr round-trip budget: float64 has 15–17 decimal
        # digits; allow 1e-13.
        np.testing.assert_allclose(mj, py, atol=1e-13)

    def test_shape_preserved(self) -> None:
        n = 8
        knm = _symmetric_knm(n, seed=3)
        theta = np.linspace(0.0, TWO_PI, n, endpoint=False)
        mj = _force_backend("mojo", knm, theta, lambda_=0.5)
        assert mj.shape == (n, n)


# ---------------------------------------------------------------------
# Cross-backend consistency
# ---------------------------------------------------------------------


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only the Python fallback is available",
    )
    def test_all_backends_agree(self) -> None:
        rng = np.random.default_rng(2026)
        n = 10
        knm = _symmetric_knm(n, seed=2026)
        theta = rng.uniform(0.0, TWO_PI, size=n)

        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-13,
            "python": 0.0,
        }

        ref = _python_reference(knm, theta, lambda_=0.5)
        for backend in AVAILABLE_BACKENDS:
            out = _force_backend(backend, knm, theta, lambda_=0.5)
            atol = tolerances[backend]
            np.testing.assert_allclose(
                out,
                ref,
                atol=atol,
                err_msg=(
                    f"backend {backend!r} differs from python reference "
                    f"by more than atol={atol}"
                ),
            )


class TestBackendLoaderDispatch:
    def test_mojo_loader_invokes_toolchain_bootstrap(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ensure_calls: list[str] = []
        backend_calls: list[tuple[np.ndarray, np.ndarray]] = []

        def _ensure_exe() -> None:
            ensure_calls.append("called")

        def _backend(
            knm_flat: np.ndarray,
            theta: np.ndarray,
            w_q: np.ndarray,
            w_k: np.ndarray,
            w_v: np.ndarray,
            w_o: np.ndarray,
            n: int,
            n_heads: int,
            block_size: int,
            temperature: float,
            lambda_: float,
        ) -> np.ndarray:
            backend_calls.append((knm_flat, np.asarray(theta)))
            return knm_flat

        fake_module = types.ModuleType(
            "scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_mojo"
        )
        fake_module._ensure_exe = _ensure_exe
        fake_module.attnres_modulate_mojo = _backend
        monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

        loader = attnres_mod._load_mojo()
        knm = _symmetric_knm(3, seed=11)
        theta = np.linspace(0.0, TWO_PI, 3, endpoint=False)
        w = np.zeros((1, 8, 8), dtype=np.float64)

        out = loader(
            knm.ravel(),
            theta,
            w.ravel(),
            w.ravel(),
            w.ravel(),
            np.zeros((8, 8), dtype=np.float64),
            3,
            1,
            -1,
            1.0,
            0.25,
        )

        assert ensure_calls == ["called"]
        assert backend_calls and backend_calls[0][0].shape == (9,)
        np.testing.assert_array_equal(out, knm.ravel())

    def test_go_loader_invokes_shared_object_loader(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        load_calls: list[str] = []

        def _load_lib() -> None:
            load_calls.append("loaded")

        def _backend(
            knm_flat: np.ndarray,
            theta: np.ndarray,
            w_q: np.ndarray,
            w_k: np.ndarray,
            w_v: np.ndarray,
            w_o: np.ndarray,
            n: int,
            n_heads: int,
            block_size: int,
            temperature: float,
            lambda_: float,
        ) -> np.ndarray:
            return np.asarray(knm_flat, dtype=np.float64)

        fake_module = types.ModuleType(
            "scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_go"
        )
        fake_module._load_lib = _load_lib
        fake_module.attnres_modulate_go = _backend
        monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

        loader = attnres_mod._load_go()
        knm = _symmetric_knm(2, seed=5)
        theta = np.linspace(0.0, TWO_PI, 2, endpoint=False)
        w = np.zeros((1, 8, 8), dtype=np.float64)

        out = loader(
            knm.ravel(),
            theta,
            w.ravel(),
            w.ravel(),
            w.ravel(),
            np.zeros((8, 8), dtype=np.float64),
            2,
            1,
            -1,
            1.0,
            0.25,
        )

        assert load_calls == ["loaded"]
        np.testing.assert_array_equal(out, knm.ravel())

    def test_julia_loader_imports_juliacall(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_backend = object()

        fake_juliacall = types.ModuleType("juliacall")
        fake_juliacall.Main = object()
        fake_julia = types.ModuleType(
            "scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_julia"
        )
        fake_julia.attnres_modulate_julia = fake_backend
        monkeypatch.setitem(sys.modules, "juliacall", fake_juliacall)
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_julia",
            fake_julia,
        )

        assert attnres_mod._load_julia() is fake_backend

    def test_resolve_backends_chooses_first_healthy_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        def _fail() -> object:
            calls.append("fail")
            raise RuntimeError("backend unavailable")

        def _go() -> object:
            calls.append("go")
            return lambda *args: np.array([0.0], dtype=np.float64)

        monkeypatch.setitem(attnres_mod._LOADERS, "rust", _fail)
        monkeypatch.setitem(attnres_mod._LOADERS, "mojo", _fail)
        monkeypatch.setitem(attnres_mod._LOADERS, "julia", _fail)
        monkeypatch.setitem(attnres_mod._LOADERS, "go", _go)

        active, available = attnres_mod._resolve_backends()
        assert active == "go"
        assert available == ["go", "python"]
        assert calls == ["fail", "fail", "fail", "go"]

    def test_resolve_backends_falls_back_to_python_when_all_backends_fail(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        def _fail() -> object:
            calls.append("fail")
            raise RuntimeError("backend unavailable")

        monkeypatch.setitem(attnres_mod._LOADERS, "rust", _fail)
        monkeypatch.setitem(attnres_mod._LOADERS, "mojo", _fail)
        monkeypatch.setitem(attnres_mod._LOADERS, "julia", _fail)
        monkeypatch.setitem(attnres_mod._LOADERS, "go", _fail)

        active, available = attnres_mod._resolve_backends()

        assert active == "python"
        assert available == ["python"]
        assert calls == ["fail", "fail", "fail", "fail"]
