# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mandatory native sparse torus contracts

"""Cross the actual PyO3 CSR boundary with finite phases near the torus cut."""

from __future__ import annotations

import numpy as np
import pytest
from spo_kernel import PySparseUPDEStepper


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("batch", [False, True])
def test_native_sparse_phase_projection(method: str, batch: bool) -> None:
    """Retain signed input buffers while publishing canonical native step/run phases."""
    tau = 2.0 * np.pi
    interior = np.nextafter(tau, 0.0)
    phases = np.array([0.0, -tau, -2.0 * tau, -0.0, tau, interior, 0.25])
    omega = np.array([-1e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    row = np.zeros(phases.size + 1, dtype=np.uint64)
    columns = np.zeros(0, dtype=np.uint64)
    edges = np.zeros(0)
    before = tuple(a.tobytes() for a in (phases, omega, row, columns, edges))
    native = PySparseUPDEStepper(phases.size, dt=0.01, method=method)
    result = (
        native.run(phases, omega, row, columns, edges, 0.0, 0.0, edges, 1)
        if batch
        else native.step(phases, omega, row, columns, edges, 0.0, 0.0, edges)
    )
    np.testing.assert_array_equal(result[:5], np.zeros(5))
    assert not np.any(np.signbit(result[:5]))
    assert result[5] == interior
    assert result[6] == pytest.approx(0.27, rel=0.0, abs=2e-16)
    assert np.all((result >= 0.0) & (result < tau))
    assert not np.shares_memory(result, phases)
    assert before == tuple(a.tobytes() for a in (phases, omega, row, columns, edges))


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("primed", [False, True])
def test_native_sparse_finite_output_refusal_preserves_proposal(
    method: str, primed: bool
) -> None:
    """Preserve cold/previously published CSR diagnostics on real overflow.

    Parameters
    ----------
    method : str
        Actual native integration method.
    primed : bool
        Establish the diagnostic cache through a valid public native step.
    """
    native = PySparseUPDEStepper(1, dt=0.01, method=method)
    phases = np.zeros(1)
    row = np.zeros(2, dtype=np.uint64)
    columns = np.zeros(0, dtype=np.uint64)
    edges = np.zeros(0)
    if primed:
        native.step(np.array([0.3]), np.zeros(1), row, columns, edges, 0.0, 0.0, edges)
    previous_order = native.order_parameter()
    previous_dt = native.last_dt
    with pytest.raises(ValueError, match="output phases contain NaN/Inf"):
        native.step(
            phases, np.array([1e308]), row, columns, edges, 1e308, np.pi / 2.0, edges
        )
    assert native.last_dt == previous_dt
    assert native.order_parameter() == previous_order
    np.testing.assert_array_equal(phases, np.zeros(1))
    recovered = native.step(phases, np.ones(1), row, columns, edges, 0.0, 0.0, edges)
    np.testing.assert_allclose(recovered, [previous_dt], rtol=0.0, atol=2e-17)
