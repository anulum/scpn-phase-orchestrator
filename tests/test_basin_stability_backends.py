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
and must produce bit-exact R values for identical inputs. This file
pins the dispatcher to each backend in turn, runs the same problem,
and cross-checks against the Python reference with a tight tolerance.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _basin_stability_go as basin_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _basin_stability_julia as basin_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _basin_stability_mojo as basin_mojo,
)
from scpn_phase_orchestrator.upde import (
    _basin_stability_validation as basin_validation,
)
from scpn_phase_orchestrator.upde import basin_stability as b_mod
from scpn_phase_orchestrator.upde.basin_stability import (
    basin_stability,
    steady_state_r,
)

TOL = 1e-12
DirectBackend = Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
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
    assert callable(basin_validation.validate_basin_stability_inputs)
    assert callable(basin_validation.validate_basin_stability_output)


@contextlib.contextmanager
def _force_backend(name: str):
    prev = b_mod.ACTIVE_BACKEND
    b_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        b_mod.ACTIVE_BACKEND = prev


def _all_to_all(n: int, strength: float = 1.0) -> np.ndarray:
    k = np.ones((n, n)) * strength / n
    np.fill_diagonal(k, 0.0)
    return k


def _direct_payload(n: int = 5):
    rng = np.random.default_rng(17)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n)
    omegas = rng.normal(0.0, 0.2, size=n)
    knm = _all_to_all(n, strength=2.5).ravel()
    alpha = np.zeros(n * n, dtype=np.float64)
    return phases, omegas, knm, alpha, n, 1.0, 0.01, 20, 10


def _mojo_proc(stdout: str) -> object:
    return type("Proc", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()


def _reference_R(n: int, strength: float, seed: int) -> float:
    omegas = np.ones(n)
    knm = _all_to_all(n, strength=strength)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, n)
    with _force_backend("python"):
        return steady_state_r(
            phases,
            omegas,
            knm,
            dt=0.01,
            n_transient=200,
            n_measure=100,
        )


def _backend_R(backend: str, n: int, strength: float, seed: int) -> float:
    if backend not in b_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    omegas = np.ones(n)
    knm = _all_to_all(n, strength=strength)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, n)
    with _force_backend(backend):
        return steady_state_r(
            phases,
            omegas,
            knm,
            dt=0.01,
            n_transient=200,
            n_measure=100,
        )


class TestDirectBackendBoundaryContracts:
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
        replacement: Callable[[tuple], object],
    ) -> None:
        """Direct Go/Julia/Mojo wrappers share the steady-state R contract."""

        payload = list(_direct_payload())
        payload[index] = replacement(tuple(payload))
        with pytest.raises((TypeError, ValueError)):
            backend(*payload)

    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    def test_zero_measure_returns_zero_without_optional_runtime(
        self,
        backend: DirectBackend,
    ) -> None:
        payload = list(_direct_payload())
        payload[8] = 0
        assert backend(*payload) == 0.0

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
        monkeypatch.setattr(basin_mojo, "_ensure_exe", lambda: "basin_stability")
        monkeypatch.setattr(
            basin_mojo.subprocess,
            "run",
            lambda *_args, **_kwargs: _mojo_proc(stdout),
        )

        with pytest.raises(ValueError, match=match):
            basin_mojo.steady_state_r_mojo(*_direct_payload())


class TestSteadyStateRParity:
    def test_rust_matches_python(self):
        ref = _reference_R(6, strength=3.0, seed=0)
        got = _backend_R("rust", 6, strength=3.0, seed=0)
        assert abs(got - ref) < TOL

    def test_julia_matches_python(self):
        ref = _reference_R(6, strength=3.0, seed=1)
        got = _backend_R("julia", 6, strength=3.0, seed=1)
        assert abs(got - ref) < TOL

    def test_go_matches_python(self):
        ref = _reference_R(6, strength=3.0, seed=2)
        got = _backend_R("go", 6, strength=3.0, seed=2)
        assert abs(got - ref) < TOL

    def test_mojo_matches_python(self):
        ref = _reference_R(5, strength=2.5, seed=3)
        got = _backend_R("mojo", 5, strength=2.5, seed=3)
        # Mojo text round-trip introduces ≤ 1e-14 drift over ~300 steps.
        assert abs(got - ref) < 1e-10


class TestBasinStabilityParity:
    """S_B must agree across backends for identical RNG seed."""

    def _compare(self, backend: str):
        if backend not in b_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"backend {backend!r} unavailable")
        n = 5
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=2.5)
        with _force_backend("python"):
            ref = basin_stability(
                omegas,
                knm,
                dt=0.01,
                n_transient=100,
                n_measure=50,
                n_samples=6,
                R_threshold=0.5,
                seed=42,
            )
        with _force_backend(backend):
            got = basin_stability(
                omegas,
                knm,
                dt=0.01,
                n_transient=100,
                n_measure=50,
                n_samples=6,
                R_threshold=0.5,
                seed=42,
            )
        np.testing.assert_allclose(got.R_final, ref.R_final, atol=1e-10)
        assert got.S_B == ref.S_B
        assert got.n_converged == ref.n_converged

    def test_rust(self):
        self._compare("rust")

    def test_julia(self):
        self._compare("julia")

    def test_go(self):
        self._compare("go")

    def test_mojo(self):
        self._compare("mojo")


class TestHypothesisParity:
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
    def test_rust_hypothesis(self, n, strength, seed):
        if "rust" not in b_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
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
    def test_go_hypothesis(self, n, strength, seed):
        if "go" not in b_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        ref = _reference_R(n, strength, seed)
        got = _backend_R("go", n, strength, seed)
        assert abs(got - ref) < TOL
