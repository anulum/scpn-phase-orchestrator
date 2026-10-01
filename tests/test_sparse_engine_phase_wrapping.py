# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sparse public torus crossing contracts

"""Verify CSR phase projection through the actual sparse stateful runtime."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("batch", [False, True])
def test_finite_computation_overflow_refuses_and_recovers(
    method: str, batch: bool
) -> None:
    """Keep the real sparse proposal and caller buffers after output divergence."""
    engine = SparseUPDEEngine(1, 0.01, method=method)
    phases = np.zeros(1)
    omega = np.array([1e308])
    row_ptr = np.zeros(2, dtype=np.int64)
    indices = np.array([], dtype=np.int64)
    values = np.array([], dtype=np.float64)
    before = tuple(a.tobytes() for a in (phases, omega, row_ptr, indices, values))
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="NaN/Inf|nonfinite|finite"),
    ):
        if batch:
            engine.run(
                phases, omega, row_ptr, indices, values, 1e308, np.pi / 2.0, values, 1
            )
        else:
            engine.step(
                phases, omega, row_ptr, indices, values, 1e308, np.pi / 2.0, values
            )
    assert engine.last_dt == 0.01
    assert before == tuple(
        a.tobytes() for a in (phases, omega, row_ptr, indices, values)
    )
    recovered = engine.step(
        phases, np.ones(1), row_ptr, indices, values, 0.0, 0.0, values
    )
    np.testing.assert_allclose(recovered, [0.01], rtol=0.0, atol=2e-17)


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("batch", [False, True])
def test_sparse_step_and_run_phase_publication(method: str, batch: bool) -> None:
    """Keep strict torus guards compatible with real finite negative crossings."""
    tau = 2.0 * np.pi
    interior = np.nextafter(tau, 0.0)
    phases = np.array([0.0, -tau, -2.0 * tau, -0.0, tau, interior, 0.25])
    omega = np.array([-1e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    row = np.zeros(phases.size + 1, dtype=np.int64)
    columns = np.zeros(0, dtype=np.int64)
    edges = np.zeros(0)
    before = tuple(a.tobytes() for a in (phases, omega, row, columns, edges))
    engine = SparseUPDEEngine(phases.size, 0.01, method=method)
    result = (
        engine.run(phases, omega, row, columns, edges, 0.0, 0.0, edges, 1)
        if batch
        else engine.step(phases, omega, row, columns, edges, 0.0, 0.0, edges)
    )
    np.testing.assert_array_equal(result[:5], np.zeros(5))
    assert not np.any(np.signbit(result[:5]))
    assert result[5] == interior
    assert result[6] == pytest.approx(0.27, rel=0.0, abs=2e-16)
    assert np.all((result >= 0.0) & (result < tau))
    assert not np.shares_memory(result, phases)
    assert before == tuple(a.tobytes() for a in (phases, omega, row, columns, edges))


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
def test_csr_coupled_crossing_and_real_refusal(method: str) -> None:
    """Retain CSR admission and same-instance recovery on a genuinely coupled graph."""
    engine = SparseUPDEEngine(2, 0.01, method=method)
    phases = np.zeros(2)
    row = np.array([0, 1, 2], dtype=np.int64)
    columns = np.array([1, 0], dtype=np.int64)
    coupling = np.array([0.8, 0.5])
    alpha = np.zeros(2)
    with pytest.raises(ValueError, match="NaN/Inf"):
        engine.step(
            phases, np.array([np.inf, 0.0]), row, columns, coupling, 0.0, 0.0, alpha
        )
    assert engine.last_dt == 0.01
    result = engine.step(
        phases, np.array([-1e-15, 2e-15]), row, columns, coupling, 0.0, 0.0, alpha
    )
    assert result[0] == 0.0
    assert not np.signbit(result[0])
    assert 0.0 < result[1] < 1e-16
    np.testing.assert_array_equal(phases, np.zeros(2))
