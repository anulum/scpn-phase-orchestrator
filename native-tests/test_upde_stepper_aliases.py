# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Dense native stepper buffer contracts

"""Exercise aliased NumPy storage through the installed dense Rust stepper."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import spo_kernel
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Operation = Literal["step", "run", "omega", "doppler"]
OPERATIONS: tuple[Operation, ...] = ("step", "run", "omega", "doppler")


def advance(
    stepper: spo_kernel.PyUPDEStepper,
    operation: Operation,
    phases: FloatArray,
    frequencies: FloatArray,
    coupling: FloatArray,
    lag: FloatArray,
    velocity: FloatArray,
) -> FloatArray:
    """Advance the real native owner with a two-step schedule where applicable."""
    if operation == "step":
        result = stepper.step(phases, frequencies, coupling, 0.2, 0.4, lag)
    elif operation == "run":
        result = stepper.run(phases, frequencies, coupling, 0.2, 0.4, lag, 2)
    elif operation == "omega":
        result = stepper.run_omega_schedule(
            phases, frequencies, coupling, 0.2, 0.4, lag, 2
        )
    else:
        result = stepper.run_doppler_schedule(
            phases, frequencies, coupling, 0.2, 0.4, lag, velocity, 0.3, 1e-8, 2
        )
    return np.asarray(result, dtype=np.float64)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("plasticity", [False, True])
def test_native_shared_inputs_match_entry_snapshots(
    operation: Operation, method: str, plasticity: bool
) -> None:
    """Shared lag, phases and schedules equal independent entry-time inputs."""
    coupling = np.array([0.0, 0.3, 0.4, 0.0])
    phases = coupling[:2]
    frequencies = coupling[:2] if operation in {"step", "run"} else coupling
    velocity = coupling
    original = coupling.copy()
    reference_coupling = original.copy()
    actual_stepper = spo_kernel.PyUPDEStepper(2, 0.01, method)
    reference_stepper = spo_kernel.PyUPDEStepper(2, 0.01, method)
    if plasticity:
        actual_stepper.set_plasticity(0.7, 0.2, 0.8)
        reference_stepper.set_plasticity(0.7, 0.2, 0.8)
    expected = advance(
        reference_stepper,
        operation,
        phases.copy(),
        frequencies.copy(),
        reference_coupling,
        original.copy(),
        velocity.copy(),
    )
    actual = advance(
        actual_stepper, operation, phases, frequencies, coupling, coupling, velocity
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(coupling, reference_coupling)
    np.testing.assert_array_equal(
        actual_stepper.order_parameter(), reference_stepper.order_parameter()
    )
    assert actual_stepper.n == 2
    assert actual_stepper.last_dt == reference_stepper.last_dt
    if plasticity:
        assert not np.array_equal(coupling, original)
        assert coupling[0] == coupling[3] == 0.0
    else:
        np.testing.assert_array_equal(coupling, original)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_native_partially_overlapping_views(operation: Operation) -> None:
    """Overlapping views with different offsets retain snapshot lag values."""
    allocation = np.array([0.0, 0.3, 0.4, 0.0, 0.2, 0.1])
    coupling = allocation[:4]
    lag = allocation[2:]
    phases = np.array([0.1, 0.7])
    frequencies = np.ones(2 if operation in {"step", "run"} else 4)
    velocity = np.zeros(4)
    expected = advance(
        spo_kernel.PyUPDEStepper(2),
        operation,
        phases.copy(),
        frequencies.copy(),
        coupling.copy(),
        lag.copy(),
        velocity.copy(),
    )
    actual = advance(
        spo_kernel.PyUPDEStepper(2),
        operation,
        phases,
        frequencies,
        coupling,
        lag,
        velocity,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(allocation, [0.0, 0.3, 0.4, 0.0, 0.2, 0.1])


@pytest.mark.parametrize("operation", OPERATIONS)
def test_native_readonly_coupling_refusal_preserves_stepper(
    operation: Operation,
) -> None:
    """An unwritable coupling raises ValueError before state or inputs change."""
    coupling = np.array([0.0, 0.3, 0.4, 0.0])
    coupling.setflags(write=False)
    phases = np.array([0.1, 0.7])
    frequencies = np.ones(2 if operation in {"step", "run"} else 4)
    velocity = np.zeros(4)
    stepper = spo_kernel.PyUPDEStepper(2)
    state_before = stepper.order_parameter()
    dt_before = stepper.last_dt
    with pytest.raises(ValueError):
        advance(stepper, operation, phases, frequencies, coupling, coupling, velocity)
    assert stepper.order_parameter() == state_before
    assert stepper.last_dt == dt_before
    np.testing.assert_array_equal(phases, [0.1, 0.7])
    np.testing.assert_array_equal(coupling, [0.0, 0.3, 0.4, 0.0])
    writable = coupling.copy()
    actual = advance(
        stepper, operation, phases, frequencies, writable, writable, velocity
    )
    expected = advance(
        spo_kernel.PyUPDEStepper(2),
        operation,
        phases,
        frequencies,
        coupling.copy(),
        coupling.copy(),
        velocity,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("operation", "field"),
    [
        (operation, field)
        for operation in OPERATIONS
        for field in ("phases", "frequencies", "coupling", "lag", "velocity")
        if field != "velocity" or operation == "doppler"
    ],
)
def test_native_noncontiguous_buffer_refusal(operation: Operation, field: str) -> None:
    """Required contiguous buffers refuse strided storage without a native panic."""
    phases = np.array([0.1, 0.7])
    frequencies = np.ones(2 if operation in {"step", "run"} else 4)
    coupling = np.array([0.0, 0.3, 0.4, 0.0])
    lag = np.zeros(4)
    velocity = np.zeros(4)
    buffers = {
        "phases": phases,
        "frequencies": frequencies,
        "coupling": coupling,
        "lag": lag,
        "velocity": velocity,
    }
    selected = buffers[field]
    backing = np.repeat(selected, 2)
    buffers[field] = backing[::2]
    with pytest.raises(ValueError):
        advance(
            spo_kernel.PyUPDEStepper(2),
            operation,
            buffers["phases"],
            buffers["frequencies"],
            buffers["coupling"],
            buffers["lag"],
            buffers["velocity"],
        )
    np.testing.assert_array_equal(coupling, [0.0, 0.3, 0.4, 0.0])


def test_native_plasticity_can_be_disabled() -> None:
    """Disabling plasticity preserves shared coupling on subsequent steps."""
    stepper = spo_kernel.PyUPDEStepper(2)
    stepper.set_plasticity(1.0)
    coupling = np.array([0.0, 0.3, 0.4, 0.0])
    phases = np.array([0.1, 0.7])
    advance(stepper, "step", phases, np.ones(2), coupling, coupling, np.zeros(4))
    updated = coupling.copy()
    stepper.disable_plasticity()
    advance(stepper, "step", phases, np.ones(2), coupling, coupling, np.zeros(4))
    np.testing.assert_array_equal(coupling, updated)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_native_solver_refusal_preserves_shared_storage(operation: Operation) -> None:
    """Malformed phase cardinality retains coupling and previous solver state."""
    coupling = np.array([0.0, 0.3, 0.4, 0.0])
    frequencies = np.ones(2 if operation in {"step", "run"} else 4)
    stepper = spo_kernel.PyUPDEStepper(2)
    state_before = stepper.order_parameter()
    with pytest.raises(ValueError):
        advance(
            stepper,
            operation,
            np.array([0.1]),
            frequencies,
            coupling,
            coupling,
            np.zeros(4),
        )
    assert stepper.order_parameter() == state_before
    np.testing.assert_array_equal(coupling, [0.0, 0.3, 0.4, 0.0])


@pytest.mark.parametrize("method", ["invalid", "euler"])
def test_native_constructor_preserves_configuration_refusals(method: str) -> None:
    """Unknown integration methods and zero dimensions remain ValueError faults."""
    with pytest.raises(ValueError):
        spo_kernel.PyUPDEStepper(0, method=method)


def test_native_plasticity_preserves_rate_refusal() -> None:
    """A refused learning rule leaves coupling unchanged on the next step."""
    stepper = spo_kernel.PyUPDEStepper(2)
    with pytest.raises(ValueError):
        stepper.set_plasticity(-1.0)
    coupling = np.array([0.0, 0.3, 0.4, 0.0])
    advance(
        stepper,
        "step",
        np.array([0.1, 0.7]),
        np.ones(2),
        coupling,
        coupling,
        np.zeros(4),
    )
    np.testing.assert_array_equal(coupling, [0.0, 0.3, 0.4, 0.0])


def test_native_moving_frame_shared_readonly_matrices() -> None:
    """The distinct readonly moving-frame contract preserves ballistic positions."""
    coupling = np.array([0.0, 0.3, 0.4, 0.0])
    coupling.setflags(write=False)
    stepper = spo_kernel.PyUPDEStepper(2)
    result = stepper.run_moving_frame_schedule(
        np.array([0.1, 0.7]),
        np.array([0.0, 1.0]),
        np.ones(4),
        coupling,
        0.0,
        0.0,
        coupling,
        np.array([0.2, -0.3, 0.2, -0.3]),
        0.3,
        0,
        1.0,
        1.0,
        1e-8,
        0.0,
        1e-8,
        2,
    )
    np.testing.assert_allclose(
        np.asarray(result)[2:], [0.004, 0.994], atol=1e-14, rtol=0
    )
    np.testing.assert_array_equal(coupling, [0.0, 0.3, 0.4, 0.0])


@pytest.mark.parametrize(
    "field",
    ["phases", "positions", "omega_schedule", "knm", "alpha", "velocity_schedule"],
)
@pytest.mark.parametrize("fault", ["strided", "cardinality"])
def test_native_moving_frame_buffer_refusals(field: str, fault: str) -> None:
    """The relocated readonly owner retains contiguity and dimension refusals."""
    arrays = {
        "phases": np.array([0.1, 0.7]),
        "positions": np.array([0.0, 1.0]),
        "omega_schedule": np.ones(4),
        "knm": np.array([0.0, 0.3, 0.4, 0.0]),
        "alpha": np.zeros(4),
        "velocity_schedule": np.array([0.2, -0.3, 0.2, -0.3]),
    }
    original_coupling = arrays["knm"].copy()
    selected = arrays[field]
    arrays[field] = np.repeat(selected, 2)[::2] if fault == "strided" else selected[:-1]
    stepper = spo_kernel.PyUPDEStepper(2)
    state_before = stepper.order_parameter()
    with pytest.raises(ValueError):
        stepper.run_moving_frame_schedule(
            **arrays,
            zeta=0.0,
            psi=0.0,
            spatial_k_base=0.3,
            spatial_decay_form=0,
            spatial_decay_exponent=1.0,
            spatial_decay_length_scale=1.0,
            spatial_epsilon=1e-8,
            doppler_strength=0.0,
            doppler_epsilon=1e-8,
            n_steps=2,
        )
    assert stepper.order_parameter() == state_before
    np.testing.assert_array_equal(
        arrays["knm"],
        original_coupling
        if field != "knm" or fault == "strided"
        else original_coupling[:-1],
    )
