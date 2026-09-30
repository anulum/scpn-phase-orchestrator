# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public engine shared matrix contracts

"""Exercise public shared-storage integration in real installed environments."""

from __future__ import annotations

import cProfile
import importlib.util
import sys
from functools import partial
from types import CodeType

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import engine as engine_module
from scpn_phase_orchestrator.upde.engine import UPDEEngine

CPU_BACKENDS = tuple(
    backend
    for backend in engine_module.AVAILABLE_BACKENDS
    if backend in {"rust", "mojo", "julia", "go", "python"}
)


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("overlap", ["identical", "partial"])
def test_public_step_shared_matrices_match_independent_inputs(
    method: str, overlap: str
) -> None:
    """Aliased coupling and lag preserve phases, state and scientific inputs."""
    allocation = np.array([0.0, 0.3, 0.4, 0.0, 0.2, 0.1])
    coupling = allocation[:4].reshape(2, 2)
    lag = coupling if overlap == "identical" else allocation[2:].reshape(2, 2)
    original = allocation.copy()
    phases = np.array([0.1, 0.7])
    frequencies = np.array([0.8, 1.2])
    actual_engine = UPDEEngine(2, 0.01, method)
    reference_engine = UPDEEngine(2, 0.01, method)
    expected = reference_engine.step(
        phases.copy(), frequencies.copy(), coupling.copy(), 0.2, 0.4, lag.copy()
    )
    actual = actual_engine.step(phases, frequencies, coupling, 0.2, 0.4, lag)
    np.testing.assert_array_equal(actual, expected)
    assert actual_engine.time == reference_engine.time
    np.testing.assert_array_equal(actual_engine.omega_current, frequencies)
    np.testing.assert_array_equal(allocation, original)
    np.testing.assert_array_equal(phases, [0.1, 0.7])
    np.testing.assert_array_equal(frequencies, [0.8, 1.2])
    np.testing.assert_array_equal(
        actual_engine.compute_order_parameter(actual),
        reference_engine.compute_order_parameter(expected),
    )


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
def test_public_callable_frequency_run_readonly_contract(
    backend: str, method: str
) -> None:
    """Callable-frequency runs inherit the selected schedule buffer contract."""

    def frequency_at(time: float) -> NDArray[np.float64]:
        """Return a changing natural-frequency vector in radians per second."""
        return np.array([0.8 + 0.1 * time, 1.2 - 0.1 * time])

    phases = np.array([0.1, 0.7])
    shared = np.array([[0.0, 0.3], [0.4, 0.0]])
    original = shared.copy()
    shared.setflags(write=False)
    previous_backend = engine_module.ACTIVE_BACKEND
    try:
        engine_module.ACTIVE_BACKEND = backend
        engine = UPDEEngine(2, 0.01, method, omega=frequency_at)
        omega_before = engine.omega_current.copy()
        reference = UPDEEngine(2, 0.01, method, omega=frequency_at)
        expected = reference.run(
            phases, knm=shared.copy(), alpha=shared.copy(), n_steps=2
        )
        if backend == "rust":
            with pytest.raises(ValueError):
                engine.run(phases, knm=shared, alpha=shared, n_steps=2)
            assert engine.time == 0.0
            np.testing.assert_array_equal(engine.omega_current, omega_before)
        else:
            actual = engine.run(phases, knm=shared, alpha=shared, n_steps=2)
            np.testing.assert_array_equal(actual, expected)
            assert engine.time == reference.time
            np.testing.assert_array_equal(engine.omega_current, reference.omega_current)
    finally:
        engine_module.ACTIVE_BACKEND = previous_backend
    np.testing.assert_array_equal(shared, original)
    np.testing.assert_array_equal(phases, [0.1, 0.7])


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
def test_public_zero_shared_matrix_preserves_exact_trajectory(method: str) -> None:
    """Zero shared coupling and lag retain the uncoupled analytical trajectory."""
    shared = np.zeros((2, 2))
    phases = np.array([0.1, 0.7])
    frequencies = np.array([0.8, 1.2])
    engine = UPDEEngine(2, 0.01, method)
    actual = engine.step(phases, frequencies, shared, 0.0, 0.0, shared)
    np.testing.assert_allclose(actual, phases + 0.01 * frequencies, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(shared, np.zeros((2, 2)))
    assert engine.time == 0.01


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("schedule", [False, True])
def test_public_batched_aliases_reach_actual_available_backends(
    backend: str, method: str, schedule: bool
) -> None:
    """Shared matrices retain values through available fixed/scheduled CPU paths."""
    profiler = cProfile.Profile()
    selected_name = {
        "rust": "_rust_run",
        "mojo": "upde_run_mojo",
        "julia": "upde_run_julia",
        "go": "upde_run_go",
        "python": "upde_run_python",
    }[backend]
    if schedule:
        selected_name = {
            "rust": "_rust_run_schedule",
            "mojo": "upde_run_omega_schedule_mojo",
            "julia": "upde_run_omega_schedule_julia",
            "go": "upde_run_omega_schedule_go",
            "python": "upde_run_omega_schedule_python",
        }[backend]
    phases = np.array([0.1, 0.7])
    frequencies = np.array([0.8, 1.2])
    shared = np.array([[0.0, 0.3], [0.4, 0.0]])
    original = shared.copy()
    previous_backend = engine_module.ACTIVE_BACKEND
    previous_profile = sys.getprofile()
    try:
        engine_module.ACTIVE_BACKEND = backend
        if schedule:
            frequency_schedule = np.tile(frequencies, (2, 1))
            expected = engine_module.upde_run_omega_schedule(
                phases.copy(),
                frequency_schedule.copy(),
                shared.copy(),
                shared.copy(),
                0.2,
                0.4,
                0.01,
                method,
            )
        else:
            expected = engine_module.upde_run(
                phases.copy(),
                frequencies.copy(),
                shared.copy(),
                shared.copy(),
                0.2,
                0.4,
                0.01,
                2,
                method,
            )
        profiler.enable()
        if schedule:
            actual = engine_module.upde_run_omega_schedule(
                phases, frequency_schedule, shared, shared, 0.2, 0.4, 0.01, method
            )
        else:
            actual = engine_module.upde_run(
                phases, frequencies, shared, shared, 0.2, 0.4, 0.01, 2, method
            )
    finally:
        profiler.disable()
        sys.setprofile(previous_profile)
        engine_module.ACTIVE_BACKEND = previous_backend
    called_names = {
        entry.code.co_name
        for entry in profiler.getstats()
        if isinstance(entry.code, CodeType)
    }
    assert selected_name in called_names
    if backend != "python":
        assert "upde_run_python" not in called_names
        assert "upde_run_omega_schedule_python" not in called_names
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(shared, original)
    np.testing.assert_array_equal(phases, [0.1, 0.7])
    np.testing.assert_array_equal(frequencies, [0.8, 1.2])


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("operation", ["step", "run"])
def test_public_readonly_coupling_preserves_backend_boundary(
    method: str, operation: str
) -> None:
    """Native readonly refusals preserve state; actual NumPy accepts readonly input."""
    shared = np.array([[0.0, 0.3], [0.4, 0.0]])
    original = shared.copy()
    shared.setflags(write=False)
    phases = np.array([0.1, 0.7])
    frequencies = np.array([0.8, 1.2])
    engine = UPDEEngine(2, 0.01, method)
    omega_before = engine.omega_current.copy()
    runner = engine.step if operation == "step" else engine.run
    if importlib.util.find_spec("spo_kernel") is not None:
        with pytest.raises(ValueError):
            runner(phases, frequencies, shared, 0.2, 0.4, shared)
        assert engine.time == 0.0
        np.testing.assert_array_equal(engine.omega_current, omega_before)
    else:
        actual = runner(phases, frequencies, shared, 0.2, 0.4, shared)
        reference = UPDEEngine(2, 0.01, method)
        reference_runner = reference.step if operation == "step" else reference.run
        expected = reference_runner(
            phases, frequencies, shared.copy(), 0.2, 0.4, shared.copy()
        )
        np.testing.assert_array_equal(actual, expected)
        assert engine.time == reference.time
    retry = runner(phases, frequencies, shared.copy(), 0.2, 0.4, shared.copy())
    assert retry.shape == phases.shape
    np.testing.assert_array_equal(shared, original)
    np.testing.assert_array_equal(phases, [0.1, 0.7])


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("schedule", [False, True])
def test_public_batched_readonly_coupling_contract(
    backend: str, method: str, schedule: bool
) -> None:
    """Rust refuses readonly coupling; the other real CPU backends preserve it."""
    phases = np.array([0.1, 0.7])
    frequencies = np.array([0.8, 1.2])
    shared = np.array([[0.0, 0.3], [0.4, 0.0]])
    original = shared.copy()
    shared.setflags(write=False)
    previous_backend = engine_module.ACTIVE_BACKEND
    try:
        engine_module.ACTIVE_BACKEND = backend
        if schedule:
            runner = partial(
                engine_module.upde_run_omega_schedule,
                phases,
                np.tile(frequencies, (2, 1)),
                zeta=0.2,
                psi=0.4,
                dt=0.01,
                method=method,
            )
        else:
            runner = partial(
                engine_module.upde_run,
                phases,
                frequencies,
                zeta=0.2,
                psi=0.4,
                dt=0.01,
                n_steps=2,
                method=method,
            )
        expected = runner(knm=shared.copy(), alpha=shared.copy())
        if backend == "rust":
            with pytest.raises(ValueError):
                runner(knm=shared, alpha=shared)
        else:
            actual = runner(knm=shared, alpha=shared)
            np.testing.assert_array_equal(actual, expected)
    finally:
        engine_module.ACTIVE_BACKEND = previous_backend
    np.testing.assert_array_equal(shared, original)
    np.testing.assert_array_equal(phases, [0.1, 0.7])
