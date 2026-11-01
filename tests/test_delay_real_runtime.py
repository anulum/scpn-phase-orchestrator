# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual delayed integration and recovery contracts

"""Qualify explicit runtime owners, scalar history and accepted-state recovery."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Callable
from sys import maxsize
from typing import TypeAlias, TypeVar

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._delay_go import (
    delayed_kuramoto_run_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._delay_julia import (
    delayed_kuramoto_run_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._delay_mojo import (
    _EXE_PATH,
    delayed_kuramoto_run_mojo,
)
from scpn_phase_orchestrator.upde.delay import DelayBuffer, DelayedEngine

_R_CALL = TypeVar("_R_CALL")


def _call(fn: Callable[..., _R_CALL], *args: object, **kwargs: object) -> _R_CALL:
    """Call ``fn`` with arguments mypy cannot check: invalid or dynamic kwargs."""
    return fn(*args, **kwargs)


FloatArray: TypeAlias = NDArray[np.float64]
DirectOwner: TypeAlias = Callable[
    [
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        int,
        float,
        float,
        float,
        int,
        int,
    ],
    FloatArray,
]
OWNERS = ("python", "rust", "go", "julia", "mojo")


def _scalar_history(
    phases: FloatArray,
    frequencies: FloatArray,
    coupling: FloatArray,
    lag: FloatArray,
    delay: int,
    steps: int,
) -> FloatArray:
    """Evaluate simultaneous scalar Euler updates against full saved snapshots."""
    current = [float(value) for value in phases]
    saved: list[list[float]] = []
    for step in range(steps):
        saved.append(current.copy())
        past = saved[step - delay] if step >= delay else current
        candidate = []
        for i, phase in enumerate(current):
            force = sum(
                float(coupling[i, j]) * math.sin(past[j] - phase - float(lag[i, j]))
                for j in range(len(current))
            )
            raw = phase + 0.01 * (
                float(frequencies[i]) + force + 0.3 * math.sin(0.7 - phase)
            )
            wrapped = raw % math.tau
            candidate.append(0.0 if wrapped >= math.tau or wrapped == 0.0 else wrapped)
        current = candidate
    return np.array(current)


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", OWNERS)
@pytest.mark.parametrize("n", [2, 257])
@pytest.mark.parametrize("lag", [0.0, 0.17])
@pytest.mark.parametrize("delay", [2, 7])
def test_explicit_owner_matches_independent_history(
    backend: str, n: int, lag: float, delay: int
) -> None:
    """Required owners reproduce directed history through multiple ring wraps."""
    phases = np.arange(n, dtype=np.float64) * math.tau / n
    frequencies = np.arange(n, dtype=np.float64) / n - 0.7
    coupling = np.asarray(
        ((np.arange(n * n).reshape(n, n) * 3 % 7) - 3) * 0.04, dtype=np.float64
    )
    alpha = np.full((n, n), lag)
    originals = [value.copy() for value in (phases, frequencies, coupling, alpha)]
    expected = _scalar_history(phases, frequencies, coupling, alpha, delay, 17)
    actual = DelayedEngine(n, dt=0.01, delay_steps=delay).run(
        phases, frequencies, coupling, 0.3, 0.7, alpha, n_steps=17, backend=backend
    )
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0.0)
    assert np.all((actual >= 0.0) & (actual < math.tau))
    for value, original in zip(
        (phases, frequencies, coupling, alpha), originals, strict=True
    ):
        np.testing.assert_array_equal(value, original)


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", OWNERS)
def test_explicit_owner_preserves_half_open_endpoints(backend: str) -> None:
    """Rounded endpoints become positive zero; representable interior phases remain."""
    interior = np.nextafter(math.tau, 0.0)
    phases = np.array([-np.nextafter(0.0, 1.0), -0.0, interior])
    actual = DelayedEngine(3, dt=0.01).run(
        phases, np.zeros(3), np.zeros((3, 3)), n_steps=1, backend=backend
    )
    np.testing.assert_array_equal(actual, [0.0, 0.0, interior])
    assert not np.any(np.signbit(actual[:2]))
    assert not np.shares_memory(actual, phases)


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", OWNERS)
def test_explicit_owner_refuses_nonfinite_arithmetic(backend: str) -> None:
    """Finite controls whose derivative overflows never publish nonfinite phases."""
    phases = np.zeros(1)
    with pytest.raises(ValueError):
        DelayedEngine(1, dt=0.01).run(
            phases,
            np.full(1, np.finfo(np.float64).max),
            np.zeros((1, 1)),
            zeta=np.finfo(np.float64).max,
            psi=math.pi / 2,
            n_steps=1,
            backend=backend,
        )
    np.testing.assert_array_equal(phases, [0.0])


@pytest.mark.native_runtime
@pytest.mark.parametrize("backend", OWNERS)
def test_explicit_owner_needs_no_unused_history(backend: str) -> None:
    """Short batches allocate only history they can use."""
    actual = DelayedEngine(1, dt=0.01, delay_steps=maxsize).run(
        np.array([0.4]), np.ones(1), np.zeros((1, 1)), n_steps=2, backend=backend
    )
    np.testing.assert_allclose(actual, [0.42], atol=1e-14, rtol=0.0)


@pytest.mark.parametrize(
    "owner",
    [delayed_kuramoto_run_go, delayed_kuramoto_run_julia, delayed_kuramoto_run_mojo],
    ids=["go", "julia", "mojo"],
)
def test_direct_zero_steps_return_an_unwrapped_copy(owner: DirectOwner) -> None:
    """Zero-step bridge calls validate and copy without requiring a foreign runtime."""
    phases = np.array([-1.0, math.tau + 1.0])
    result = owner(
        phases, np.zeros(2), np.zeros(4), np.zeros(4), 2, 0.0, 0.0, 0.01, 1, 0
    )
    np.testing.assert_array_equal(result, phases)
    assert not np.shares_memory(result, phases)


def test_go_count_conversion_is_refused_before_truncation() -> None:
    """The C ABI never silently truncates a valid Python integration count."""
    with pytest.raises(ValueError, match="signed 32-bit"):
        delayed_kuramoto_run_go(
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            1,
            0.0,
            0.0,
            0.01,
            1,
            2**31,
        )


def test_refused_step_preserves_history_and_recovers() -> None:
    """A refused candidate preserves the source history used by later coupling."""
    actual = DelayedEngine(2, dt=0.01, delay_steps=2)
    reference = DelayedEngine(2, dt=0.01, delay_steps=2)
    phases = np.array([0.4, 1.1])
    frequencies = np.array([1.0, -0.7])
    coupling = np.array([[0.0, 0.5], [-0.2, 0.0]])
    phases = actual.step(phases, frequencies, coupling)
    expected = reference.step(np.array([0.4, 1.1]), frequencies, coupling)
    refused = np.array([0.4, 0.0])
    with pytest.raises(ValueError, match="nonfinite"):
        actual.step(
            refused,
            np.full(2, np.finfo(np.float64).max),
            coupling,
            zeta=np.finfo(np.float64).max,
            psi=math.pi / 2,
        )
    np.testing.assert_array_equal(refused, [0.4, 0.0])
    for _ in range(5):
        phases = actual.step(phases, frequencies, coupling, step_idx=999)
        expected = reference.step(expected, frequencies, coupling)
        np.testing.assert_array_equal(phases, expected)


def test_batch_does_not_modify_stateful_history() -> None:
    """Fresh batches preserve independent caller arrays and step history."""
    actual = DelayedEngine(2, dt=0.01, delay_steps=1)
    reference = DelayedEngine(2, dt=0.01, delay_steps=1)
    phases = np.array([0.4, 1.1])
    omega = np.array([1.0, -0.7])
    coupling = np.array([[0.0, 0.5], [-0.2, 0.0]])
    phases = actual.step(phases, omega, coupling)
    expected = reference.step(np.array([0.4, 1.1]), omega, coupling)
    actual.run(phases, omega, coupling, n_steps=4, backend="python")
    for _ in range(3):
        phases = actual.step(phases, omega, coupling)
        expected = reference.step(expected, omega, coupling)
        np.testing.assert_array_equal(phases, expected)


def test_retrieved_buffer_snapshot_cannot_corrupt_history() -> None:
    """Mutating a retrieved snapshot cannot inject NaN into retained history."""
    history = DelayBuffer(2, max_delay_steps=2)
    history.push(np.array([0.4, 1.1]))
    retrieved = history.get_delayed(1)
    assert retrieved is not None
    retrieved[0] = np.nan
    np.testing.assert_array_equal(history.get_delayed(1), [0.4, 1.1])


@pytest.mark.native_runtime
@pytest.mark.parametrize(
    ("index", "token"),
    [
        (0, "OTHER"),
        (1, "1x"),
        (1, "1e0"),
        (1, str(maxsize + 1)),
        (2, "-1"),
        (3, "1.5"),
        (4, "NaN"),
        (4, "1e9999"),
        (5, "1e+"),
        (5, "."),
        (6, "0"),
        (7, "0.4junk"),
        (8, "1e+-2"),
        (9, "--0"),
        (10, "1.2.3"),
    ],
)
def test_actual_mojo_protocol_refuses_malformed_requests(
    index: int, token: str
) -> None:
    """The actual executable refuses incomplete decimal tokens without phase output."""
    request = ["DELAY", "1", "1", "2", "0", "0", "0.01", "0.4", "1", "0", "0"]
    request[index] = token
    completed = subprocess.run(
        [str(_EXE_PATH)],
        input=" ".join(request) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    assert not completed.stdout.strip()


@pytest.mark.native_runtime
def test_actual_mojo_protocol_accepts_complete_decimal_exponents() -> None:
    """Valid signed decimal and exponent tokens produce the actual rotation."""
    completed = subprocess.run(
        [str(_EXE_PATH)],
        input="DELAY +1 +1 +2 0e0 +.0 1e-2 +4e-1 1.e0 0. 0e1\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = [float(line) for line in completed.stdout.splitlines()]
    np.testing.assert_allclose(result, [0.42], atol=1e-14, rtol=0.0)


@pytest.mark.parametrize("backend", ["", "Rust", "auto ", None, 1])
def test_unknown_public_backend_is_refused(backend: object) -> None:
    """Reject invalid owner selectors before attempting integration or fallback."""
    with pytest.raises(ValueError, match="unknown delay backend"):
        _call(
            DelayedEngine(1, dt=0.01).run,
            np.zeros(1),
            np.zeros(1),
            np.zeros((1, 1)),
            backend=backend,
        )


@pytest.mark.parametrize("control", ["dt", "zeta", "psi"])
def test_public_scalars_refuse_real_integer_overflow(control: str) -> None:
    """Preserve conversion overflow as the cause of a public scalar refusal."""
    huge = 10**400
    with pytest.raises(ValueError, match=control) as error:
        if control == "dt":
            DelayedEngine(1, dt=huge)
        elif control == "zeta":
            DelayedEngine(1, dt=0.01).step(
                np.zeros(1), np.zeros(1), np.zeros((1, 1)), zeta=huge
            )
        else:
            DelayedEngine(1, dt=0.01).step(
                np.zeros(1), np.zeros(1), np.zeros((1, 1)), psi=huge
            )
    assert isinstance(error.value.__cause__, OverflowError)


@pytest.mark.parametrize(
    "owner",
    [delayed_kuramoto_run_go, delayed_kuramoto_run_julia, delayed_kuramoto_run_mojo],
)
@pytest.mark.parametrize("control", ["dt", "zeta", "psi"])
def test_direct_scalars_refuse_real_integer_overflow(
    owner: DirectOwner, control: str
) -> None:
    """Reject unrepresentable real controls before loading a native resource."""
    controls = {"dt": 0.01, "zeta": 0.0, "psi": 0.0}
    controls[control] = 10**400
    with pytest.raises(ValueError, match=control) as error:
        owner(
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            1,
            controls["zeta"],
            controls["psi"],
            controls["dt"],
            1,
            1,
        )
    assert isinstance(error.value.__cause__, OverflowError)
