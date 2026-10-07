# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity tests for multi-head AttnRes

"""Exercise real phase attention runtimes and negative ingress/output contracts.

Explicit owner requests cannot fall back. Available ports compute actual
results; an absent optional owner must raise ImportError. Required profile
assertions in the real-runtime suite separately prevent missing compiled
owners from qualifying a declared native lane. No skip or fake loader supplies
successful numerical evidence. Numerical parity uses stated float64
allclose tolerances and does not assert bitwise equality.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Unpack, cast, get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from benchmarks.attnres_reference import (
    AttnResOptions,
    FloatArray,
    phase_attention_oracle,
)
from scpn_phase_orchestrator.coupling import (
    attention_residuals as attnres_mod,
)
from scpn_phase_orchestrator.coupling.attention_residuals import (
    AVAILABLE_BACKENDS,
    attnres_modulate,
    default_projections,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling import (
    _attnres_julia,
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
    """Direct ingress refuses a malformed graph before optional runtime access."""
    knm, theta, q, key, value, out, n, heads, radius, temp, strength = _direct_payload()
    knm[0] = 0.1
    for backend in (attnres_modulate_go, attnres_modulate_julia, attnres_modulate_mojo):
        with pytest.raises(ValueError, match="diagonal"):
            backend(knm, theta, q, key, value, out, n, heads, radius, temp, strength)


RefusalCall = Callable[
    [
        object,
        object,
        object,
        object,
        object,
        object,
        object,
        object,
        object,
        object,
        object,
    ],
    FloatArray,
]


TWO_PI = 2.0 * np.pi
AttnResDirectBackend = Callable[
    [
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        object,
        object,
        object,
        object,
        object,
    ],
    FloatArray,
]


def _symmetric_knm(n: int, strength: float = 0.3, seed: int = 0) -> FloatArray:
    """Draw a reproducible undirected graph with no self-coupling."""
    rng = np.random.default_rng(seed)
    half = rng.uniform(0.0, 2.0 * strength, size=(n, n))
    knm = 0.5 * (half + half.T)
    np.fill_diagonal(knm, 0.0)
    return knm.astype(np.float64)


def _force_backend(
    backend: str, knm: FloatArray, theta: FloatArray, **kw: Unpack[AttnResOptions]
) -> FloatArray:
    """Require the named real backend through the supported public API."""
    return attnres_modulate(knm, theta, backend=backend, **kw)


def _python_reference(
    knm: FloatArray, theta: FloatArray, **kw: Unpack[AttnResOptions]
) -> FloatArray:
    """Compute the real NumPy reference without changing dispatch state."""
    return attnres_modulate(knm, theta, backend="python", **kw)


def _direct_payload(
    n: int = 3,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    int,
    int,
    int,
    float,
    float,
]:
    """Prepare valid flat buffers for direct ingress and refusal tests."""
    knm = _symmetric_knm(n, seed=11).ravel()
    theta = np.linspace(0.0, TWO_PI, n, endpoint=False)
    w = np.zeros((1, 8, 8), dtype=np.float64).ravel()
    return knm, theta, w, w.copy(), w.copy(), w.copy(), n, 1, -1, 1.0, 0.25


def _mojo_proc(stdout: str) -> object:
    """Represent intentionally malformed subprocess output for refusal controls."""
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
            ("knm", np.array(["0.0"] * 9), ValueError, "numeric-string"),
            ("knm", np.array([0.0, np.nan] + [0.0] * 7), ValueError, "finite"),
            ("knm", np.zeros((3, 3)), ValueError, "one-dimensional"),
            ("knm", np.zeros(8), ValueError, "n\\*n"),
            ("theta", np.array([0.0, np.inf, 1.0]), ValueError, "finite"),
            ("theta", np.array(["0.0", "1.0", "2.0"]), ValueError, "numeric-string"),
            ("theta", np.array([0.0, 1.0 + 0.0j, 2.0]), ValueError, "real-valued"),
            ("theta", np.zeros(2), ValueError, "theta length"),
            ("w_q", np.array([True] * 64), ValueError, "w_q"),
            ("w_q", np.array(["0.0"] * 64), ValueError, "numeric-string"),
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
        """Reject corrupted direct buffers and controls before optional loading."""
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
            cast(
                "RefusalCall",
                backend,
            )(*payload)

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
        """Return an empty vector without requiring an optional runtime."""
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


class TestDirectJuliaBoundaryContracts:
    """Direct Julia AttnRes adapter rejects numeric-string raw returns."""

    def test_julia_raw_return_rejects_numeric_string_aliases(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject injected numeric text from the Julia return boundary."""

        class _JuliaAttnRes:
            """Supply numeric text for the Julia output refusal boundary."""

            @staticmethod
            def attnres_modulate(*_args: object) -> NDArray[np.str_]:
                """Return deliberately invalid numeric text for output validation."""
                return np.array(
                    ["0.0", "0.25", "0.25", "0.0"],
                    dtype=str,
                )

        monkeypatch.setattr(
            _attnres_julia,
            "_ensure_julia_loaded",
            lambda: _JuliaAttnRes(),
        )

        with pytest.raises(ValueError, match="numeric-string"):
            _attnres_julia.attnres_modulate_julia(*_direct_payload(n=2))


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
        """Reject injected stdout with wrong cardinality or non-finite topology."""
        monkeypatch.setattr(_attnres_mojo, "_ensure_exe", lambda: "attnres")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *_args, **_kwargs: _mojo_proc(stdout),
        )

        with pytest.raises(ValueError, match=match):
            _attnres_mojo.attnres_modulate_mojo(*_direct_payload())

    def test_mojo_runner_rejects_output_that_creates_zero_edges(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject injected stdout that creates absent graph edges."""
        monkeypatch.setattr(_attnres_mojo, "_ensure_exe", lambda: "attnres")
        monkeypatch.setattr(
            subprocess,
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
    """Check maintained float64 annotations on the actual bridge APIs."""

    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (attnres_modulate_go, "go"),
            (attnres_modulate_julia, "julia"),
            (attnres_modulate_mojo, "mojo"),
        ],
    )
    def test_backend_annotations_use_float64_ndarray(
        self, fn: object, label: str
    ) -> None:
        """Retain precise float64 buffer annotations on direct bridge APIs."""
        hints = get_type_hints(fn)
        for name in ("knm_flat", "theta", "w_q", "w_k", "w_v", "w_o", "return"):
            text = str(hints[name])
            assert_precise_ndarray_hint(hints[name], context=f"{label}:{name}")
            assert "numpy.float64" in text, f"{label}:{name} missing float64"


# ---------------------------------------------------------------------
# Rust parity
# ---------------------------------------------------------------------


class TestRustParity:
    """Exercise the real Rust numerical owner or its explicit missing-runtime error."""

    @given(
        n=st.integers(min_value=4, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=12,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_numerical_parity(self, n: int, seed: int) -> None:
        """Match the real reference within the stated float64 tolerance."""
        if "rust" not in AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                attnres_modulate(
                    np.array([[0.0, 0.3], [0.3, 0.0]]),
                    np.array([0.1, 0.7]),
                    backend="rust",
                )
            return
        rng = np.random.default_rng(seed)
        knm = _symmetric_knm(n, seed=seed)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        py = _python_reference(knm, theta, lambda_=0.5)
        rs = _force_backend("rust", knm, theta, lambda_=0.5)
        np.testing.assert_allclose(rs, py, atol=1e-12)

    def test_lambda_zero_passthrough(self) -> None:
        """Preserve exact identity semantics independently of native availability."""
        knm = _symmetric_knm(8, seed=99)
        theta = np.arange(8, dtype=np.float64) * 0.1
        py = _python_reference(knm, theta, lambda_=0.0)
        rs = _force_backend("rust", knm, theta, lambda_=0.0)
        np.testing.assert_array_equal(rs, py)

    def test_block_size_honoured(self) -> None:
        """Rust kernel respects ``block_size`` the same way Python does."""
        if "rust" not in AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                attnres_modulate(
                    np.array([[0.0, 0.3], [0.3, 0.0]]),
                    np.array([0.1, 0.7]),
                    backend="rust",
                )
            return
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
    """Exercise the real Julia numerical owner or its explicit missing-runtime error."""

    @pytest.mark.parametrize("n", [6, 10, 14])
    def test_numerical_parity(self, n: int) -> None:
        """Match Julia numerics across deterministic network sizes."""
        if "julia" not in AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                attnres_modulate(
                    np.array([[0.0, 0.3], [0.3, 0.0]]),
                    np.array([0.1, 0.7]),
                    backend="julia",
                )
            return
        rng = np.random.default_rng(42 + n)
        knm = _symmetric_knm(n, seed=42 + n)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        py = _python_reference(knm, theta, lambda_=0.5)
        jl = _force_backend("julia", knm, theta, lambda_=0.5)
        np.testing.assert_allclose(jl, py, atol=1e-12)

    def test_symmetry_preserved(self) -> None:
        """Preserve undirected coupling under arbitrary oscillator phases."""
        if "julia" not in AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                attnres_modulate(
                    np.array([[0.0, 0.3], [0.3, 0.0]]),
                    np.array([0.1, 0.7]),
                    backend="julia",
                )
            return
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
    """Exercise the real Go numerical owner or its explicit missing-runtime error."""

    @given(
        n=st.integers(min_value=4, max_value=14),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_numerical_parity(self, n: int, seed: int) -> None:
        """Match the real reference within the stated float64 tolerance."""
        if "go" not in AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                attnres_modulate(
                    np.array([[0.0, 0.3], [0.3, 0.0]]),
                    np.array([0.1, 0.7]),
                    backend="go",
                )
            return
        rng = np.random.default_rng(seed)
        knm = _symmetric_knm(n, seed=seed)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        py = _python_reference(knm, theta, lambda_=0.5)
        go = _force_backend("go", knm, theta, lambda_=0.5)
        np.testing.assert_allclose(go, py, atol=1e-12)

    def test_invalid_block_size_surfaces(self) -> None:
        """Reject zero-band configuration before a Go computation can start."""
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
    """Exercise the real Mojo numerical owner or its explicit missing-runtime error."""

    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_numerical_parity(self, n: int) -> None:
        """Match the real Mojo output within the documented numeric tolerance."""
        if "mojo" not in AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                attnres_modulate(
                    np.array([[0.0, 0.3], [0.3, 0.0]]),
                    np.array([0.1, 0.7]),
                    backend="mojo",
                )
            return
        rng = np.random.default_rng(13 + n)
        knm = _symmetric_knm(n, seed=13 + n)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        py = _python_reference(knm, theta, lambda_=0.5)
        mj = _force_backend("mojo", knm, theta, lambda_=0.5)
        # 17-digit repr round-trip budget: float64 has 15–17 decimal
        # digits; allow 1e-13.
        np.testing.assert_allclose(mj, py, atol=1e-13)

    def test_shape_preserved(self) -> None:
        """Preserve coupling cardinality through an actual Mojo call."""
        if "mojo" not in AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                attnres_modulate(
                    np.array([[0.0, 0.3], [0.3, 0.0]]),
                    np.array([0.1, 0.7]),
                    backend="mojo",
                )
            return
        n = 8
        knm = _symmetric_knm(n, seed=3)
        theta = np.linspace(0.0, TWO_PI, n, endpoint=False)
        mj = _force_backend("mojo", knm, theta, lambda_=0.5)
        assert mj.shape == (n, n)


# ---------------------------------------------------------------------
# Cross-backend consistency
# ---------------------------------------------------------------------


class TestCrossBackendConsistency:
    """Compare computations from the actual discovered optional owners."""

    def test_all_backends_agree(self) -> None:
        """Compare every discovered original owner on the same graph and phases."""
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


class TestPublicOwnerSelection:
    """Exercise supported public selection rather than substitute private loaders."""

    def test_named_mojo_executes_nondefault_width_or_refuses_absence(self) -> None:
        """Exercise the original mojo runtime on non-default projection width."""
        knm = _symmetric_knm(4, seed=11)
        theta = np.array([0.1, 0.7, 1.8, 2.4])
        if "mojo" not in AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                attnres_modulate(knm, theta, backend="mojo")
            return
        weights = default_projections(n_heads=4, d_model=12, seed=6)
        expected = phase_attention_oracle(knm, theta, weights)
        out = attnres_modulate(
            knm,
            theta,
            w_q=weights[0],
            w_k=weights[1],
            w_v=weights[2],
            w_o=weights[3],
            n_heads=4,
            backend="mojo",
        )
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)
        assert np.max(np.abs(out - knm)) > 1e-4

    def test_named_go_executes_nondefault_width_or_refuses_absence(self) -> None:
        """Exercise the original go runtime on non-default projection width."""
        knm = _symmetric_knm(4, seed=11)
        theta = np.array([0.1, 0.7, 1.8, 2.4])
        if "go" not in AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                attnres_modulate(knm, theta, backend="go")
            return
        weights = default_projections(n_heads=4, d_model=12, seed=6)
        expected = phase_attention_oracle(knm, theta, weights)
        out = attnres_modulate(
            knm,
            theta,
            w_q=weights[0],
            w_k=weights[1],
            w_v=weights[2],
            w_o=weights[3],
            n_heads=4,
            backend="go",
        )
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)
        assert np.max(np.abs(out - knm)) > 1e-4

    def test_named_julia_executes_nondefault_width_or_refuses_absence(self) -> None:
        """Exercise the original julia runtime on non-default projection width."""
        knm = _symmetric_knm(4, seed=11)
        theta = np.array([0.1, 0.7, 1.8, 2.4])
        if "julia" not in AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                attnres_modulate(knm, theta, backend="julia")
            return
        weights = default_projections(n_heads=4, d_model=12, seed=6)
        expected = phase_attention_oracle(knm, theta, weights)
        out = attnres_modulate(
            knm,
            theta,
            w_q=weights[0],
            w_k=weights[1],
            w_v=weights[2],
            w_o=weights[3],
            n_heads=4,
            backend="julia",
        )
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)
        assert np.max(np.abs(out - knm)) > 1e-4

    def test_automatic_and_named_active_owner_agree(self) -> None:
        """Default public dispatch computes the same law as the named active owner."""
        knm = _symmetric_knm(5, seed=42)
        theta = np.linspace(0.0, TWO_PI, 5, endpoint=False)
        actual = attnres_modulate(knm, theta)
        explicit = attnres_modulate(knm, theta, backend=attnres_mod.ACTIVE_BACKEND)
        np.testing.assert_allclose(actual, explicit, rtol=0.0, atol=1e-12)
        assert np.max(np.abs(actual - knm)) > 1e-4

    def test_explicit_numpy_matches_independent_law(self) -> None:
        """The always-available NumPy owner computes the independent coupling oracle."""
        knm = _symmetric_knm(4, seed=8)
        theta = np.linspace(0.0, TWO_PI, 4, endpoint=False)
        weights = default_projections(seed=8)
        expected = phase_attention_oracle(knm, theta, weights)
        actual = attnres_modulate(knm, theta, backend="python", projection_seed=8)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("source_state", ["missing", "invalid"])
def test_real_julia_loader_refuses_missing_or_invalid_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source_state: str
) -> None:
    """Reject actual missing files and Julia syntax failures without substitution."""
    coupling = np.array([[0.0, 0.3], [0.3, 0.0]])
    phases = np.array([0.1, 0.7])
    if "julia" not in AVAILABLE_BACKENDS:
        with pytest.raises(ImportError):
            attnres_modulate(coupling, phases, backend="julia")
        return
    source = tmp_path / "attnres.jl"
    if source_state == "invalid":
        source.write_text("function broken(\n", encoding="utf-8")
    monkeypatch.setattr(_attnres_julia, "_JULIA_FILE", source)
    monkeypatch.setattr(_attnres_julia, "_JULIA_MODULE", None)
    with pytest.raises(ImportError, match="not found|cannot load"):
        attnres_modulate(coupling, phases, backend="julia")


@pytest.mark.parametrize("boundary", ["include", "compute"])
def test_julia_unexpected_python_errors_are_not_relabelled_as_numeric_failures(
    monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """Injected non-Julia failures must propagate; they supply no numerical result."""
    coupling = np.array([[0.0, 0.3], [0.3, 0.0]])
    phases = np.array([0.1, 0.7])
    if "julia" not in AVAILABLE_BACKENDS:
        with pytest.raises(ImportError):
            attnres_modulate(coupling, phases, backend="julia")
        return

    class RefusingBoundary:
        """Raise only unexpected Python exceptions at a negative runtime boundary."""

        @staticmethod
        def include(_path: str) -> None:
            """Refuse inclusion without returning a substitute Julia module."""
            raise RuntimeError("unexpected include failure")

        @staticmethod
        def attnres_modulate(*_arguments: object) -> object:
            """Refuse computation without returning a substitute coupling matrix."""
            raise RuntimeError("unexpected compute failure")

    if boundary == "include":
        monkeypatch.setattr(_attnres_julia, "_JULIA_MODULE", None)
        monkeypatch.setattr(_attnres_julia, "require_julia_main", RefusingBoundary)
    else:
        monkeypatch.setattr(_attnres_julia, "_JULIA_MODULE", RefusingBoundary())
    with pytest.raises(RuntimeError, match=f"unexpected {boundary} failure"):
        attnres_modulate(coupling, phases, backend="julia")
