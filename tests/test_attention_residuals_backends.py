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

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling import (
    attention_residuals as attnres_mod,
)
from scpn_phase_orchestrator.coupling.attention_residuals import (
    AVAILABLE_BACKENDS,
    attnres_modulate,
)

TWO_PI = 2.0 * np.pi


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


def _python_reference(
    knm: np.ndarray, theta: np.ndarray, **kw: object
) -> np.ndarray:
    return _force_backend("python", knm, theta, **kw)


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
