# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for basin stability

"""Cross-backend parity of the ``steady_state_r`` trial kernel.

All five backends (Rust / Mojo / Julia / Go / Python) integrate the
Kuramoto ODE via explicit Euler with full-snapshot step semantics
and must reproduce R within numerical tolerance for identical inputs. This file
names each actual owner explicitly, runs the same problem,
and cross-checks against the Python reference with a tight tolerance.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from benchmarks.kuramoto_trial_reference import scalar_trial
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _basin_stability_go as basin_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _basin_stability_julia as basin_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _basin_stability_mojo as basin_mojo,
)
from scpn_phase_orchestrator.upde import basin_stability as b_mod
from scpn_phase_orchestrator.upde.basin_stability import (
    basin_stability,
    steady_state_r,
)

FloatArray = NDArray[np.float64]
Payload = tuple[
    FloatArray, FloatArray, FloatArray, FloatArray, int, float, float, int, int
]
TOL = 1e-12
DirectBackend = Callable[
    [
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        int,
        float,
        float,
        int,
        int,
    ],
    float,
]
DIRECT_BACKENDS = (
    basin_go.steady_state_r_go,
    basin_julia.steady_state_r_julia,
    basin_mojo.steady_state_r_mojo,
)


def test__basin_stability_validation_linkage() -> None:
    """A malformed actual public trial exercises the shared ingress contract."""
    with pytest.raises(ValueError, match="omegas shape"):
        steady_state_r(np.zeros(2), np.zeros(1), np.zeros((2, 2)), backend="python")


def _all_to_all(n: int, strength: float = 1.0) -> FloatArray:
    k = np.ones((n, n)) * strength / n
    np.fill_diagonal(k, 0.0)
    return k


def _direct_payload(n: int = 5) -> Payload:
    rng = np.random.default_rng(17)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n)
    omegas = rng.normal(0.0, 0.2, size=n)
    knm = _all_to_all(n, strength=2.5).ravel()
    alpha = np.zeros(n * n, dtype=np.float64)
    return phases, omegas, knm, alpha, n, 1.0, 0.01, 20, 10


def _mojo_proc(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["negative-control"], 0, stdout, "")


def _reference_R(n: int, strength: float, seed: int) -> float:
    """Use a separately implemented scalar oracle on the same initial phases."""
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)
    return scalar_trial(
        phases.tolist(),
        np.ones(n).tolist(),
        _all_to_all(n, strength).tolist(),
        np.zeros((n, n)).tolist(),
        dt=0.01,
        transient=200,
        measure=100,
    )


def _backend_R(backend: str, n: int, strength: float, seed: int) -> float:
    """Call the named original public owner without mutation or fallback."""
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)
    return steady_state_r(
        phases,
        np.ones(n),
        _all_to_all(n, strength),
        dt=0.01,
        n_transient=200,
        n_measure=100,
        backend=backend,
    )


def _direct(backend: DirectBackend, payload: list[object]) -> float:
    """Forward hostile original values through a direct boundary negative control."""
    return backend(
        cast(FloatArray, payload[0]),
        cast(FloatArray, payload[1]),
        cast(FloatArray, payload[2]),
        cast(FloatArray, payload[3]),
        cast(int, payload[4]),
        cast(float, payload[5]),
        cast(float, payload[6]),
        cast(int, payload[7]),
        cast(int, payload[8]),
    )


class TestDirectBackendBoundaryContracts:
    """Actual direct adapters reject invalid ingress and malformed stdout."""

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
            (6, lambda payload: 0.0),
            (6, lambda payload: float("inf")),
            (7, lambda payload: True),
            (7, lambda payload: -1),
            (8, lambda payload: True),
            (8, lambda payload: -1),
        ],
    )
    def test_invalid_inputs_fail_before_optional_runtime_loading(
        self,
        backend: DirectBackend,
        index: int,
        replacement: Callable[[Payload], object],
    ) -> None:
        """Direct Go/Julia/Mojo wrappers share the steady-state R contract."""
        payload: list[object] = list(_direct_payload())
        payload[index] = replacement(cast(Payload, tuple(payload)))
        with pytest.raises((TypeError, ValueError)):
            _direct(backend, payload)

    @pytest.mark.parametrize("owner", ("go", "julia", "mojo"))
    def test_zero_measure_validates_named_runtime(self, owner: str) -> None:
        """A named zero-window request still requires that original owner to load."""
        if owner not in b_mod.AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                steady_state_r(
                    np.zeros(2),
                    np.zeros(2),
                    np.zeros((2, 2)),
                    n_measure=0,
                    backend=owner,
                )
        else:
            assert (
                steady_state_r(
                    np.zeros(2),
                    np.zeros(2),
                    np.zeros((2, 2)),
                    n_measure=0,
                    backend=owner,
                )
                == 0.0
            )

    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    @pytest.mark.parametrize(
        ("index", "replacement"),
        [
            (0, lambda payload: np.array(["0.1"] * payload[4], dtype=object)),
            (1, lambda payload: np.array(["0.0"] * payload[4], dtype=object)),
            (2, lambda payload: np.array(["0.0"] * (payload[4] ** 2), dtype=object)),
            (3, lambda payload: np.array(["0.0"] * (payload[4] ** 2), dtype=object)),
            (5, lambda _payload: "1.0"),
            (6, lambda _payload: "0.01"),
            (7, lambda _payload: "1"),
            (8, lambda _payload: "1"),
        ],
    )
    def test_numeric_string_inputs_fail_before_optional_runtime_loading(
        self,
        backend: DirectBackend,
        index: int,
        replacement: Callable[[Payload], object],
    ) -> None:
        """Direct Go/Julia/Mojo inputs must reject numeric-string aliases."""
        payload: list[object] = list(_direct_payload())
        payload[index] = replacement(cast(Payload, tuple(payload)))
        with pytest.raises(ValueError, match="numeric-string"):
            _direct(backend, payload)

    @pytest.mark.parametrize(
        ("stdout", "match"),
        [
            ("", "Mojo STEADY returned 0 lines, expected 1"),
            ("\n", "one finite steady-state R"),
            ("0.5\n0.6\n", "expected 1"),
            ("not-a-number\n", "one finite steady-state R"),
            ("nan\n", "steady-state R must be finite"),
            ("inf\n", "steady-state R must be finite"),
            ("1.2\n", "steady-state R must lie in \\[0, 1\\]"),
        ],
    )
    def test_mojo_steady_state_rejects_malformed_stdout(
        self, monkeypatch: pytest.MonkeyPatch, stdout: str, match: str
    ) -> None:
        """Injected malformed stdout cannot become a finite trial measurement."""
        monkeypatch.setattr(basin_mojo, "_ensure_exe", lambda: "basin_stability")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *_args, **_kwargs: _mojo_proc(stdout),
        )

        with pytest.raises(ValueError, match=match):
            basin_mojo.steady_state_r_mojo(*_direct_payload())


class TestSteadyStateRParity:
    """Named original owners agree with independent scalar Euler trials."""

    def test_rust_matches_python(self) -> None:
        """The original named Rust trial agrees with the scalar Euler oracle."""
        if "rust" not in b_mod.AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                _backend_R("rust", 3, 1.0, 0)
            return
        ref = _reference_R(6, strength=3.0, seed=0)
        got = _backend_R("rust", 6, strength=3.0, seed=0)
        assert abs(got - ref) < TOL

    def test_julia_matches_python(self) -> None:
        """The original named Julia trial agrees with the scalar Euler oracle."""
        if "julia" not in b_mod.AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                _backend_R("julia", 3, 1.0, 0)
            return
        ref = _reference_R(6, strength=3.0, seed=1)
        got = _backend_R("julia", 6, strength=3.0, seed=1)
        assert abs(got - ref) < TOL

    def test_go_matches_python(self) -> None:
        """The original named Go trial agrees with the scalar Euler oracle."""
        if "go" not in b_mod.AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                _backend_R("go", 3, 1.0, 0)
            return
        ref = _reference_R(6, strength=3.0, seed=2)
        got = _backend_R("go", 6, strength=3.0, seed=2)
        assert abs(got - ref) < TOL

    def test_mojo_matches_python(self) -> None:
        """The original named Mojo trial agrees within floating-point tolerance."""
        if "mojo" not in b_mod.AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                _backend_R("mojo", 3, 1.0, 0)
            return
        ref = _reference_R(5, strength=2.5, seed=3)
        got = _backend_R("mojo", 5, strength=2.5, seed=3)
        # Mojo text round-trip introduces ≤ 1e-14 drift over ~300 steps.
        assert abs(got - ref) < 1e-10


class TestBasinStabilityParity:
    """S_B must agree across backends for identical RNG seed."""

    def _compare(self, backend: str) -> None:
        if backend not in b_mod.AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                _backend_R(backend, 5, 2.5, 42)
            return
        n = 5
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=2.5)
        ref = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=100,
            n_measure=50,
            n_samples=6,
            R_threshold=0.5,
            seed=42,
            backend="python",
        )
        got = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=100,
            n_measure=50,
            n_samples=6,
            R_threshold=0.5,
            seed=42,
            backend=backend,
        )
        np.testing.assert_allclose(got.R_final, ref.R_final, atol=1e-10)
        assert got.S_B == ref.S_B
        assert got.n_converged == ref.n_converged

    def test_rust(self) -> None:
        """Actual Rust sampling retains the public NumPy seed and classification."""
        self._compare("rust")

    def test_julia(self) -> None:
        """Actual Julia sampling retains the public NumPy seed and classification."""
        self._compare("julia")

    def test_go(self) -> None:
        """Actual Go sampling retains the public NumPy seed and classification."""
        self._compare("go")

    def test_mojo(self) -> None:
        """Actual Mojo sampling retains the public NumPy seed and classification."""
        self._compare("mojo")


class TestHypothesisParity:
    """Sampled original owner comparisons against scalar Euler values."""

    @given(
        n=st.integers(min_value=2, max_value=6),
        strength=st.floats(min_value=0.5, max_value=4.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rust_hypothesis(self, n: int, strength: float, seed: int) -> None:
        """Sampled Rust trials agree with independent scalar measurements."""
        if "rust" not in b_mod.AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                _backend_R("rust", n, strength, seed)
            return
        ref = _reference_R(n, strength, seed)
        got = _backend_R("rust", n, strength, seed)
        assert abs(got - ref) < TOL

    @given(
        n=st.integers(min_value=2, max_value=6),
        strength=st.floats(min_value=0.5, max_value=4.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(self, n: int, strength: float, seed: int) -> None:
        """Sampled Go trials agree with independent scalar measurements."""
        if "go" not in b_mod.AVAILABLE_BACKENDS:
            with pytest.raises(ImportError):
                _backend_R("go", n, strength, seed)
            return
        ref = _reference_R(n, strength, seed)
        got = _backend_R("go", n, strength, seed)
        assert abs(got - ref) < TOL
