# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native sparse buffer contracts

"""Exercise sparse public and installed native alias, refusal and recovery paths."""

# LLVM coverage maps an uncalled PyO3-generated function to #[pymethods].
# All handwritten wrapper methods are called through the real extension here;
# macro-generated instantiations are reported separately, without exclusions.
from __future__ import annotations

import numpy as np
import pytest
import spo_kernel
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine


@pytest.mark.parametrize("operation", ["step", "run"])
@pytest.mark.parametrize("alias", ["phases", "omegas", "alpha"])
def test_public_alias_matches_csr_euler_equation(operation: str, alias: str) -> None:
    """Overlapping coupling inputs preserve the actual entry-time Euler equation."""
    coupling = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    phases = coupling if alias == "phases" else np.array([0.3, 0.6, 0.9])
    omegas = coupling if alias == "omegas" else np.ones(3, dtype=np.float64)
    lag = coupling if alias == "alpha" else np.zeros(3, dtype=np.float64)
    row_ptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([1, 2, 0], dtype=np.int64)
    expected = phases.copy()
    for _ in range(1 if operation == "step" else 4):
        derivative = omegas + coupling * np.sin(expected[indices] - expected - lag)
        expected = (expected + 0.01 * derivative) % (2.0 * np.pi)
    engine = SparseUPDEEngine(3, 0.01, method="euler")
    if operation == "step":
        result = engine.step(phases, omegas, row_ptr, indices, coupling, 0.0, 0.0, lag)
    else:
        result = engine.run(
            phases, omegas, row_ptr, indices, coupling, 0.0, 0.0, lag, 4
        )
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(coupling, [0.1, 0.2, 0.3])
    assert not np.shares_memory(result, phases)


@pytest.mark.parametrize("operation", ["step", "run"])
@pytest.mark.parametrize("alias", ["phases", "omegas", "alpha"])
def test_native_plasticity_uses_fixed_readonly_snapshots(
    operation: str, alias: str
) -> None:
    """Plasticity changes coupling while aliased readonly snapshots remain fixed."""
    coupling = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    phases = coupling if alias == "phases" else np.array([0.3, 0.6, 0.9])
    omegas = coupling if alias == "omegas" else np.ones(3, dtype=np.float64)
    lag = coupling if alias == "alpha" else np.zeros(3, dtype=np.float64)
    indices = np.array([1, 2, 0], dtype=np.uintp)
    row_ptr = np.array([0, 1, 2, 3], dtype=np.uintp)
    expected = phases.copy()
    frequencies = omegas.copy()
    lag_snapshot = lag.copy()
    expected_coupling = coupling.copy()
    for _ in range(1 if operation == "step" else 4):
        differences = expected[indices] - expected
        derivative = frequencies + expected_coupling * np.sin(
            differences - lag_snapshot
        )
        expected = (expected + 0.01 * derivative) % (2.0 * np.pi)
        expected_coupling += 0.01 * np.cos(differences)
    engine = spo_kernel.PySparseUPDEStepper(3, 0.01, "euler")
    engine.set_plasticity(1.0)
    if operation == "step":
        result = engine.step(phases, omegas, row_ptr, indices, coupling, 0.0, 0.0, lag)
    else:
        result = engine.run(
            phases, omegas, row_ptr, indices, coupling, 0.0, 0.0, lag, 4
        )
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(coupling, expected_coupling, atol=1e-12, rtol=0.0)
    engine.disable_plasticity()
    assert engine.n == 3
    assert engine.last_dt == 0.01


@pytest.mark.parametrize("operation", ["step", "run"])
def test_public_readonly_coupling_refuses_and_same_engine_recovers(
    operation: str,
) -> None:
    """Readonly native coupling raises ValueError and leaves the solver usable."""
    phases = np.array([0.2, 0.4], dtype=np.float64)
    omegas = np.array([1.0, 2.0], dtype=np.float64)
    row_ptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([1, 0], dtype=np.int64)
    coupling = np.array([0.1, 0.2], dtype=np.float64)
    lag = np.zeros(2, dtype=np.float64)
    engine = SparseUPDEEngine(2, 0.01)

    def advance(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Call the chosen real public operation on the same engine."""
        if operation == "step":
            return engine.step(phases, omegas, row_ptr, indices, values, 0.0, 0.0, lag)
        return engine.run(phases, omegas, row_ptr, indices, values, 0.0, 0.0, lag, 1)

    coupling.setflags(write=False)
    with pytest.raises(ValueError):
        advance(coupling)
    np.testing.assert_array_equal(coupling, [0.1, 0.2])
    np.testing.assert_array_equal(phases, [0.2, 0.4])
    derivative = omegas + coupling * np.sin(phases[indices] - phases)
    np.testing.assert_allclose(
        advance(coupling.copy()), phases + 0.01 * derivative, atol=1e-12
    )


@pytest.mark.parametrize("operation", ["step", "run"])
@pytest.mark.parametrize(
    "buffer", ["phases", "omegas", "rows", "columns", "coupling", "lag"]
)
def test_direct_native_strided_refusal_preserves_inputs_and_recovers(
    operation: str, buffer: str
) -> None:
    """Refuse each non-contiguous native buffer before a successful same-solver call."""
    phases = np.array([0.2, 0.4])
    omegas = np.array([1.0, 2.0])
    rows = np.array([0, 1, 2], dtype=np.uintp)
    columns = np.array([1, 0], dtype=np.uintp)
    coupling = np.array([0.1, 0.2])
    lag = np.zeros(2, dtype=np.float64)
    if buffer == "phases":
        phases = np.repeat(phases, 2)[::2]
    elif buffer == "omegas":
        omegas = np.repeat(omegas, 2)[::2]
    elif buffer == "rows":
        rows = np.repeat(rows, 2)[::2]
    elif buffer == "columns":
        columns = np.repeat(columns, 2)[::2]
    elif buffer == "coupling":
        coupling = np.repeat(coupling, 2)[::2]
    else:
        lag = np.repeat(lag, 2)[::2]
    before_phases = phases.copy()
    before_coupling = coupling.copy()
    engine = spo_kernel.PySparseUPDEStepper(2, 0.01, "euler")
    with pytest.raises(ValueError):
        if operation == "step":
            engine.step(phases, omegas, rows, columns, coupling, 0.0, 0.0, lag)
        else:
            engine.run(phases, omegas, rows, columns, coupling, 0.0, 0.0, lag, 1)
    np.testing.assert_array_equal(phases, before_phases)
    np.testing.assert_array_equal(coupling, before_coupling)
    result = engine.step(
        phases.copy(),
        omegas.copy(),
        rows.copy(),
        columns.copy(),
        coupling.copy(),
        0.0,
        0.0,
        lag.copy(),
    )
    derivative = omegas + coupling * np.sin(phases[columns] - phases - lag)
    np.testing.assert_allclose(result, phases + 0.01 * derivative, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    "n,dt,method",
    [(0, 0.01, "euler"), (2, 0.0, "euler"), (2, 0.01, "unknown")],
)
def test_native_constructor_refuses_invalid_configuration(
    n: int, dt: float, method: str
) -> None:
    """Reject native configuration errors even without the public Python guard."""
    with pytest.raises(ValueError):
        spo_kernel.PySparseUPDEStepper(n, dt, method)


@pytest.mark.parametrize("lr,decay", [(-1.0, 0.0), (1.0, -1.0), (np.nan, 0.0)])
def test_native_plasticity_refusal_preserves_solver(lr: float, decay: float) -> None:
    """Invalid learning rules leave a native solver usable with fixed coupling."""
    engine = spo_kernel.PySparseUPDEStepper(2, 0.01)
    with pytest.raises(ValueError):
        engine.set_plasticity(lr, decay)
    phases = np.array([0.2, 0.4])
    coupling = np.array([0.1, 0.2])
    result = engine.step(
        phases,
        np.ones(2),
        np.array([0, 1, 2], dtype=np.uintp),
        np.array([1, 0], dtype=np.uintp),
        coupling,
        0.0,
        0.0,
        np.zeros(2),
    )
    derivative = 1.0 + coupling * np.sin(phases[::-1] - phases)
    np.testing.assert_allclose(result, phases + 0.01 * derivative, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(coupling, [0.1, 0.2])


def test_native_disable_plasticity_stops_updates_and_exposes_cached_order() -> None:
    """Disabling plasticity fixes coupling and preserves the cached order diagnostic."""
    phases = np.array([0.2, 0.4])
    coupling = np.array([0.1, 0.2])
    frequencies = np.ones(2)
    rows = np.array([0, 1, 2], dtype=np.uintp)
    columns = np.array([1, 0], dtype=np.uintp)
    lag = np.zeros(2)
    engine = spo_kernel.PySparseUPDEStepper(2, 0.01)
    engine.set_plasticity(1.0)
    result = engine.step(phases, frequencies, rows, columns, coupling, 0.0, 0.0, lag)
    np.testing.assert_allclose(
        coupling, [0.1 + 0.01 * np.cos(0.2), 0.2 + 0.01 * np.cos(0.2)]
    )
    # Euler caches the trigonometry used by its derivative at the incoming phases.
    order, mean_phase = engine.order_parameter()
    expected_order = np.mean(np.exp(1j * phases))
    assert order == pytest.approx(abs(expected_order))
    assert mean_phase == pytest.approx(float(np.angle(expected_order)))
    engine.disable_plasticity()
    fixed_values = coupling.copy()
    expected = result + 0.01 * (
        frequencies + coupling * np.sin(result[columns] - result)
    )
    next_result = engine.step(
        result, frequencies, rows, columns, coupling, 0.0, 0.0, lag
    )
    np.testing.assert_allclose(next_result, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(coupling, fixed_values)


@pytest.mark.parametrize("operation", ["step", "run"])
def test_public_readonly_strided_coupling_is_copied_for_native_admission(
    operation: str,
) -> None:
    """Public strided coupling uses writable copies while preserving its source."""
    phases = np.array([0.2, 0.4])
    frequencies = np.ones(2)
    rows = np.array([0, 1, 2], dtype=np.int64)
    columns = np.array([1, 0], dtype=np.int64)
    backing = np.array([0.1, 9.0, 0.2, 9.0])
    coupling = backing[::2]
    coupling.setflags(write=False)
    lag = np.zeros(2)
    engine = SparseUPDEEngine(2, 0.01)
    if operation == "step":
        result = engine.step(
            phases, frequencies, rows, columns, coupling, 0.0, 0.0, lag
        )
    else:
        result = engine.run(
            phases, frequencies, rows, columns, coupling, 0.0, 0.0, lag, 1
        )
    expected = phases + 0.01 * (
        frequencies + coupling * np.sin(phases[columns] - phases)
    )
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(backing, [0.1, 9.0, 0.2, 9.0])
    assert not coupling.flags.writeable


@pytest.mark.parametrize("operation", ["step", "run"])
@pytest.mark.parametrize(
    "case,message",
    [
        ("phases-short", "expected 2, got phases"),
        ("frequencies-short", "expected 2, got phases"),
        ("row-length", "expected row_ptr length"),
        ("edge-count", "sparse edge arrays must match"),
        ("lag-count", "sparse edge arrays must match"),
        ("row-start", "row_ptr must start at 0"),
        ("row-end", "row_ptr must start at 0"),
        ("row-decrease", "row_ptr must be monotonic"),
        ("column-range", "col_indices contains out-of-range"),
        ("phase-nan", "sparse step inputs contain NaN/Inf"),
        ("frequency-nan", "sparse step inputs contain NaN/Inf"),
        ("coupling-nan", "sparse step inputs contain NaN/Inf"),
        ("lag-nan", "sparse step inputs contain NaN/Inf"),
        ("drive-inf", "sparse step inputs contain NaN/Inf"),
        ("target-nan", "sparse step inputs contain NaN/Inf"),
    ],
)
def test_native_csr_refusal_preserves_state_and_recovers(
    operation: str, case: str, message: str
) -> None:
    """Exercise real Rust input refusals and a correct same-solver recovery step."""
    phases = np.array([0.2, 0.4])
    frequencies = np.ones(2)
    rows = np.array([0, 1, 2], dtype=np.uintp)
    columns = np.array([1, 0], dtype=np.uintp)
    coupling = np.array([0.1, 0.2])
    lag = np.zeros(2)
    zeta = psi = 0.0
    if case == "phases-short":
        phases = phases[:1]
    elif case == "frequencies-short":
        frequencies = frequencies[:1]
    elif case == "row-length":
        rows = rows[:2]
    elif case == "edge-count":
        coupling = coupling[:1]
    elif case == "lag-count":
        lag = lag[:1]
    elif case == "row-start":
        rows[0] = 1
    elif case == "row-end":
        rows[-1] = 1
    elif case == "row-decrease":
        rows[1] = 3
    elif case == "column-range":
        columns[0] = 2
    elif case == "phase-nan":
        phases[0] = np.nan
    elif case == "frequency-nan":
        frequencies[0] = np.nan
    elif case == "coupling-nan":
        coupling[0] = np.nan
    elif case == "lag-nan":
        lag[0] = np.nan
    elif case == "drive-inf":
        zeta = np.inf
    else:
        psi = np.nan
    before_phases = phases.copy()
    before_coupling = coupling.copy()
    engine = spo_kernel.PySparseUPDEStepper(2, 0.01)
    with pytest.raises(ValueError, match=message):
        if operation == "step":
            engine.step(phases, frequencies, rows, columns, coupling, zeta, psi, lag)
        else:
            engine.run(phases, frequencies, rows, columns, coupling, zeta, psi, lag, 1)
    np.testing.assert_array_equal(phases, before_phases)
    np.testing.assert_array_equal(coupling, before_coupling)
    recovered = engine.step(
        np.array([0.2, 0.4]),
        np.ones(2),
        np.array([0, 1, 2], dtype=np.uintp),
        np.array([1, 0], dtype=np.uintp),
        np.array([0.1, 0.2]),
        0.0,
        0.0,
        np.zeros(2),
    )
    expected = np.array([0.2, 0.4]) + 0.01 * (
        1.0 + np.array([0.1, 0.2]) * np.sin(np.array([0.2, -0.2]))
    )
    np.testing.assert_allclose(recovered, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("substeps", [1, 2])
def test_native_order_parameter_tracks_final_derivative_stage(
    method: str, substeps: int
) -> None:
    """The cached order describes the final stage for each native integrator."""
    phases = np.array([0.2, 0.4])
    frequencies = np.array([1.0, 0.3])
    coupling = np.array([0.1, 0.2])
    rows = np.array([0, 1, 2], dtype=np.uintp)
    columns = np.array([1, 0], dtype=np.uintp)
    lags = np.zeros(2)
    engine = spo_kernel.PySparseUPDEStepper(2, 0.04, method, n_substeps=substeps)
    result = engine.step(phases, frequencies, rows, columns, coupling, 0.2, 0.5, lags)

    def derivative(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the forced two-node Kuramoto equation independently."""
        return np.array(
            [
                frequencies[0]
                + coupling[0] * np.sin(theta[1] - theta[0])
                + 0.2 * np.sin(0.5 - theta[0]),
                frequencies[1]
                + coupling[1] * np.sin(theta[0] - theta[1])
                + 0.2 * np.sin(0.5 - theta[1]),
            ],
            dtype=np.float64,
        )

    expected = phases.copy()
    cached = phases.copy()
    if method == "rk45":
        cached = result
    else:
        dt = 0.04 / substeps
        for _ in range(substeps):
            k1 = derivative(expected)
            if method == "euler":
                cached = expected.copy()
                expected = expected + dt * k1
            else:
                k2 = derivative(expected + 0.5 * dt * k1)
                k3 = derivative(expected + 0.5 * dt * k2)
                cached = expected + dt * k3
                k4 = derivative(cached)
                expected = expected + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)
    order, mean_phase = engine.order_parameter()
    reference_order = np.mean(np.exp(1j * cached))
    assert order == pytest.approx(abs(reference_order), abs=1e-12)
    assert mean_phase == pytest.approx(float(np.angle(reference_order)), abs=1e-12)
    np.testing.assert_array_equal(phases, [0.2, 0.4])
    np.testing.assert_array_equal(coupling, [0.1, 0.2])
