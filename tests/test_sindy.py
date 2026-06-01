# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PhaseSINDy tests

"""Behavioural tests for the Phase-SINDy symbolic coupling discoverer.

The pre-existing single happy-path case exercised recovery on a clean
two-oscillator signal. This suite adds the edge and error paths flagged
by the S6 audit.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

import scpn_phase_orchestrator.autotune.sindy as sindy_module
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy


def _simulate(
    phases_init: np.ndarray,
    omega: np.ndarray,
    K: np.ndarray,
    dt: float,
    steps: int,
) -> np.ndarray:
    n = len(omega)
    phases = np.zeros((steps, n))
    phases[0] = phases_init
    for t in range(1, steps):
        d = omega.copy()
        for i in range(n):
            for j in range(n):
                if i != j:
                    d[i] += K[i, j] * np.sin(phases[t - 1, j] - phases[t - 1, i])
        phases[t] = (phases[t - 1] + dt * d) % (2 * np.pi)
    return phases


def test_sindy_recovery():
    """Baseline: two oscillators, bidirectional coupling, clean recovery."""
    n = 2
    dt = 0.05
    t_max = 20.0
    steps = int(t_max / dt)

    omega = np.array([1.5, 2.0])
    K = np.array([[0.0, 0.5], [0.5, 0.0]])

    phases = np.zeros((steps, n))
    phases[0] = [0.0, 1.0]
    for t in range(1, steps):
        d0 = omega[0] + K[0, 1] * np.sin(phases[t - 1, 1] - phases[t - 1, 0])
        d1 = omega[1] + K[1, 0] * np.sin(phases[t - 1, 0] - phases[t - 1, 1])
        phases[t] = (phases[t - 1] + dt * np.array([d0, d1])) % (2 * np.pi)

    sindy = PhaseSINDy(threshold=0.1)
    coeffs = sindy.fit(phases, dt)

    assert abs(coeffs[0][0] - 1.5) < 0.01
    assert abs(coeffs[0][1] - 0.5) < 0.01
    assert abs(coeffs[1][0] - 2.0) < 0.01
    assert abs(coeffs[1][1] - 0.5) < 0.01

    equations = sindy.get_equations()
    assert "1.5000 * 1" in equations[0]
    assert "0.5000 * sin(theta_1 - theta_0)" in equations[0]


def test_sindy_recover_single_oscillator_frequency():
    """One oscillator should recover its constant frequency term."""
    phases = np.array([[0.0], [0.2], [0.4], [0.6], [0.8]], dtype=np.float64)
    sindy = PhaseSINDy(threshold=0.0)
    coeffs = sindy.fit(phases, 0.1)

    assert len(coeffs) == 1
    assert coeffs[0].shape == (1,)
    assert abs(coeffs[0][0] - 2.0) < 1e-2


def test_sindy_zero_coupling():
    """Uncoupled oscillators: SINDy must return K ≈ 0 under the threshold."""
    dt = 0.05
    steps = 400
    omega = np.array([1.0, 1.3])
    K = np.zeros((2, 2))
    phases = _simulate(np.array([0.0, 0.5]), omega, K, dt, steps)

    sindy = PhaseSINDy(threshold=0.05)
    coeffs = sindy.fit(phases, dt)

    # Frequency term close to ground truth, coupling term sparsified to ~0.
    assert abs(coeffs[0][0] - 1.0) < 0.02
    assert abs(coeffs[1][0] - 1.3) < 0.02
    # Coupling slot should either be missing or near-zero.
    if coeffs[0].size > 1:
        assert abs(coeffs[0][1]) < 0.05
    if coeffs[1].size > 1:
        assert abs(coeffs[1][1]) < 0.05


def test_sindy_large_n_stability():
    """N=5 network, moderate coupling, no blow-up."""
    n = 5
    dt = 0.02
    steps = 500
    rng = np.random.default_rng(12)
    omega = rng.uniform(0.8, 1.5, n)
    K = 0.3 * (np.ones((n, n)) - np.eye(n))  # uniform, no self
    phases = _simulate(rng.uniform(0, 2 * np.pi, n), omega, K, dt, steps)

    sindy = PhaseSINDy(threshold=0.05)
    coeffs = sindy.fit(phases, dt)

    assert len(coeffs) == n
    for i in range(n):
        assert np.all(np.isfinite(coeffs[i]))
        # Natural frequency term roughly recovers
        assert abs(coeffs[i][0] - omega[i]) < 0.1


def test_sindy_rejects_zero_max_iter():
    """Constructor guards max_iter — regression for U1b validation."""
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        PhaseSINDy(max_iter=0)


def test_sindy_rejects_negative_threshold():
    """Negative threshold would invert the sparsification logic."""
    with pytest.raises(ValueError, match="threshold.*non-negative"):
        PhaseSINDy(threshold=-0.1)


@pytest.mark.parametrize(
    "threshold",
    [float("nan"), float("inf"), True, np.bool_(True), "0.1"],
)
def test_sindy_rejects_non_finite_or_non_numeric_threshold(threshold: object):
    """Threshold must be a finite numeric sparsity bound."""
    with pytest.raises(ValueError, match="threshold"):
        PhaseSINDy(threshold=cast(Any, threshold))


@pytest.mark.parametrize("max_iter", [1.5, "2", True, np.bool_(True)])
def test_sindy_rejects_non_integer_max_iter(max_iter: object):
    """STLSQ iterations must be a positive integer count."""
    with pytest.raises(ValueError, match="max_iter"):
        PhaseSINDy(max_iter=cast(Any, max_iter))


@pytest.mark.parametrize(
    "dt",
    [float("nan"), float("inf"), 0.0, -0.05, True, np.bool_(True), "0.05"],
)
def test_sindy_rejects_malformed_dt(dt: object):
    """PhaseSINDy requires a finite positive timestep."""
    sindy = PhaseSINDy()
    with pytest.raises(ValueError, match="dt"):
        sindy.fit(np.zeros((4, 2)), cast(Any, dt))


@pytest.mark.parametrize(
    "phases",
    [
        np.array([0.0, 1.0]),
        np.zeros((2, 2, 1)),
        np.array([[0.0, 1.0], [1.0, 2.0, 3.0]], dtype=object),
    ],
)
def test_sindy_rejects_malformed_phase_shapes(phases: Any):
    """Phase trajectories must be a numeric 2D array."""
    sindy = PhaseSINDy()
    with pytest.raises(ValueError, match="2D|finite"):
        sindy.fit(phases, 0.05)


@pytest.mark.parametrize(
    "phases",
    [
        np.array([[0.0, 1.0], [np.nan, 1.5]], dtype=float),
        np.array([[0.0, 1.0], [np.inf, 1.5]], dtype=float),
    ],
)
def test_sindy_rejects_non_finite_phases(phases: np.ndarray):
    """Finite-input precondition must hold for finite-difference regression."""
    sindy = PhaseSINDy()
    with pytest.raises(ValueError, match="finite"):
        sindy.fit(phases, 0.05)


@pytest.mark.parametrize(
    "phases",
    [
        np.array([[True, False], [False, True]], dtype=bool),
        np.array([[0.0, 1.0], [True, False]], dtype=object),
        np.array([[0.0, 1.0], [np.bool_(True), np.bool_(False)]], dtype=object),
        np.array([[0.0 + 0.1j, 1.0], [0.2, 1.5]], dtype=complex),
    ],
)
def test_sindy_rejects_boolean_and_complex_phases(phases: np.ndarray):
    """Logical masks and phasor values must not enter real phase regression."""
    sindy = PhaseSINDy()

    with pytest.raises(ValueError, match="boolean|finite 2D"):
        sindy.fit(phases, 0.05)


def test_sindy_threshold_sparsifies_weak_terms():
    """High threshold wipes small coupling coefficients below it."""
    dt = 0.05
    steps = 400
    omega = np.array([1.0, 1.2])
    K = np.array([[0.0, 0.04], [0.04, 0.0]])  # below threshold
    phases = _simulate(np.array([0.0, 0.1]), omega, K, dt, steps)

    sindy_high = PhaseSINDy(threshold=0.2)
    coeffs_high = sindy_high.fit(phases, dt)
    sindy_low = PhaseSINDy(threshold=0.005)
    coeffs_low = sindy_low.fit(phases, dt)

    # High threshold (0.2) is above the true coupling (0.04), so the
    # STLSQ sparsifier must zero the coupling slot exactly.
    assert coeffs_high[0].size > 1, "fitted coefficients must include coupling term"
    assert coeffs_high[0][1] == 0.0, (
        f"coupling term should be sparsified under threshold 0.2, "
        f"got {coeffs_high[0][1]:.4f}"
    )
    # Low threshold (0.005) leaves the coupling intact; within regression
    # noise it recovers the ground-truth 0.04 to 0.005 precision.
    assert coeffs_low[0].size > 1
    assert abs(coeffs_low[0][1] - 0.04) < 0.005, (
        f"low-threshold fit should recover K_01 ≈ 0.04, got {coeffs_low[0][1]:.4f}"
    )


def test_sindy_threshold_zero_retains_weak_coupling_terms():
    """A zero threshold must not zero weak but present coupling terms."""
    dt = 0.05
    steps = 320
    omega = np.array([1.0, 1.2])
    K = np.array([[0.0, 0.04], [0.04, 0.0]])
    phases = _simulate(np.array([0.0, 0.5]), omega, K, dt, steps)

    sindy = PhaseSINDy(threshold=0.0)
    coeffs = sindy.fit(phases, dt)

    assert abs(coeffs[0][1]) > 1e-6
    assert abs(coeffs[1][1]) > 1e-6


def test_sindy_equations_dump_format():
    """get_equations() returns one string per node with the 'theta_' marker."""
    n = 3
    dt = 0.05
    steps = 200
    rng = np.random.default_rng(3)
    omega = rng.uniform(0.8, 1.2, n)
    K = 0.2 * (np.ones((n, n)) - np.eye(n))
    phases = _simulate(rng.uniform(0, 2 * np.pi, n), omega, K, dt, steps)

    sindy = PhaseSINDy(threshold=0.05)
    sindy.fit(phases, dt)
    eqs = sindy.get_equations()

    assert len(eqs) == n
    for node_i, eq in enumerate(eqs):
        assert isinstance(eq, str)
        assert f"d(theta_{node_i})/dt" in eq


def test_sindy_repeated_fit_is_deterministic():
    """Running fit twice over the same trajectory is deterministic."""
    n = 3
    dt = 0.04
    steps = 180
    rng = np.random.default_rng(3)
    omega = rng.uniform(0.7, 1.4, size=n)
    K = 0.2 * (np.ones((n, n)) - np.eye(n))
    phases = _simulate(rng.uniform(0.0, 2.0, n), omega, K, dt, steps)

    sindy = PhaseSINDy()
    first = sindy.fit(phases, dt)
    first_eq = sindy.get_equations()
    second = sindy.fit(phases, dt)
    second_eq = sindy.get_equations()

    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)
    assert first_eq == second_eq


def test_sindy_import_detects_available_rust_backend(monkeypatch: pytest.MonkeyPatch):
    """Import-time backend detection must enable the Rust path when available."""

    fake_kernel = ModuleType("spo_kernel")
    fake_kernel.sindy_fit_rust = lambda *_args: np.zeros(1, dtype=np.float64)
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_kernel)

    module_path = Path(sindy_module.__file__).resolve()
    spec = importlib.util.spec_from_file_location("_sindy_rust_available", module_path)
    assert spec is not None
    assert spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(loaded)

    assert loaded._HAS_RUST is True


def test_sindy_rust_backend_remaps_matrix_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust backend returns an NxN matrix that PhaseSINDy remaps per oscillator."""

    captured: dict[str, float | int | tuple[float, ...]] = {}

    def fake_rust_sindy_fit(
        p_flat: np.ndarray,
        n_oscillators: int,
        n_steps: int,
        dt: float,
        threshold: float,
        max_iter: int,
    ) -> np.ndarray:
        captured["flat"] = tuple(float(v) for v in p_flat)
        captured["n_oscillators"] = n_oscillators
        captured["n_steps"] = n_steps
        captured["dt"] = dt
        captured["threshold"] = threshold
        captured["max_iter"] = max_iter
        return np.array(
            [
                1.1,
                0.2,
                0.3,
                0.4,
                2.2,
                0.5,
                0.6,
                0.7,
                3.3,
            ],
            dtype=np.float64,
        )

    phases = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1],
        ],
        dtype=np.float64,
    )
    monkeypatch.setattr(sindy_module, "_HAS_RUST", True)
    monkeypatch.setattr(
        sindy_module,
        "_rust_sindy_fit",
        fake_rust_sindy_fit,
        raising=False,
    )

    sindy = PhaseSINDy(threshold=0.125, max_iter=7)
    coeffs = sindy.fit(phases, dt=0.05)

    assert captured == {
        "flat": tuple(float(v) for v in phases.ravel()),
        "n_oscillators": 3,
        "n_steps": 4,
        "dt": 0.05,
        "threshold": 0.125,
        "max_iter": 7,
    }
    assert [c.tolist() for c in coeffs] == [
        [1.1, 0.2, 0.3],
        [2.2, 0.4, 0.5],
        [3.3, 0.6, 0.7],
    ]
    assert sindy.feature_names == [
        ["1", "sin(theta_1 - theta_0)", "sin(theta_2 - theta_0)"],
        ["1", "sin(theta_0 - theta_1)", "sin(theta_2 - theta_1)"],
        ["1", "sin(theta_0 - theta_2)", "sin(theta_1 - theta_2)"],
    ]


def test_sindy_rust_backend_rejects_malformed_output_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed Rust payloads must fail with a clear validation error."""
    phases = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1],
        ],
        dtype=np.float64,
    )

    def fake_rust_sindy_fit(
        _p_flat: np.ndarray,
        _n_oscillators: int,
        _n_steps: int,
        _dt: float,
        _threshold: float,
        _max_iter: int,
    ) -> np.ndarray:
        return np.array([1.1, 0.2, 2.2, 0.4], dtype=np.float64)

    monkeypatch.setattr(sindy_module, "_HAS_RUST", True)
    monkeypatch.setattr(
        sindy_module,
        "_rust_sindy_fit",
        fake_rust_sindy_fit,
        raising=False,
    )
    sindy = PhaseSINDy(threshold=0.125, max_iter=7)

    with pytest.raises(ValueError, match="expected|Rust SINDy"):
        sindy.fit(phases, dt=0.05)


def test_sindy_rust_backend_rejects_non_finite_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend coefficients must stay finite before equation formatting."""
    phases = np.array(
        [
            [0.0, 0.1],
            [0.3, 0.4],
            [0.6, 0.7],
        ],
        dtype=np.float64,
    )

    def fake_rust_sindy_fit(
        _p_flat: np.ndarray,
        _n_oscillators: int,
        _n_steps: int,
        _dt: float,
        _threshold: float,
        _max_iter: int,
    ) -> np.ndarray:
        return np.array([1.0, np.nan, 0.1, 2.0], dtype=np.float64)

    monkeypatch.setattr(sindy_module, "_HAS_RUST", True)
    monkeypatch.setattr(
        sindy_module,
        "_rust_sindy_fit",
        fake_rust_sindy_fit,
        raising=False,
    )

    with pytest.raises(ValueError, match="non-finite coefficients"):
        PhaseSINDy().fit(phases, dt=0.05)


def test_sindy_python_lstsq_rejects_non_finite_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python regression must not accept non-finite fitted coefficients."""

    def fake_lstsq(theta: np.ndarray, _target: np.ndarray) -> tuple[np.ndarray, ...]:
        return (np.full(theta.shape[1], np.nan, dtype=np.float64),)

    monkeypatch.setattr(sindy_module, "_HAS_RUST", False)
    monkeypatch.setattr(sindy_module, "lstsq", fake_lstsq)

    phases = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="least-squares.*non-finite"):
        PhaseSINDy().fit(phases, dt=0.05)


def test_sindy_get_equations_requires_fit_first():
    """Equation export must fail closed until coefficients are fitted."""
    with pytest.raises(RuntimeError, match="called before fit"):
        PhaseSINDy().get_equations()


@pytest.mark.parametrize(
    "phases",
    [
        np.zeros((0, 2), dtype=np.float64),
        np.array([[0.0, 0.3]], dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((4, 0), dtype=np.float64),
    ],
)
def test_sindy_rejects_underdetermined_phase_series(phases: np.ndarray) -> None:
    """SINDy fitting requires derivative samples for at least one oscillator."""
    sindy = PhaseSINDy()

    with pytest.raises(ValueError, match="at least"):
        sindy.fit(phases, 0.01)


# Pipeline wiring: PhaseSINDy feeds the auto-tune pipeline (see
# autotune/pipeline.py). These tests pin the contract every consumer
# relies on: zero-coupling must sparsify, the threshold must suppress
# weak terms, and invalid constructor arguments must fail loudly.


# Salvaged module-specific behavioural contracts from deleted broad tests.
class TestSindyValidation:
    def test_rejects_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            PhaseSINDy(threshold=-0.01)

    def test_rejects_zero_max_iter(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            PhaseSINDy(max_iter=0)

    def test_rejects_negative_max_iter(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            PhaseSINDy(max_iter=-3)
