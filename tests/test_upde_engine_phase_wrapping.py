# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Dense public torus crossing contracts

"""Verify dense publication with genuine installed or absent native code.

The shared output-range guard cannot refuse a successful canonicalising real
producer. Boundary/interior assertions exercise that invariant through step
and run; no producer output is replaced to manufacture a guard violation.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.engine import UPDEEngine


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("batch", [False, True])
def test_dense_step_and_run_phase_publication(method: str, batch: bool) -> None:
    """Advance real cut, signed multiple and interior states without mutating inputs."""
    tau = 2.0 * np.pi
    interior = np.nextafter(tau, 0.0)
    phases = np.array([0.0, -tau, -2.0 * tau, -0.0, tau, interior, 0.25])
    omega = np.array([-1e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    matrix = np.zeros((phases.size, phases.size))
    before = tuple(a.tobytes() for a in (phases, omega, matrix))
    engine = UPDEEngine(phases.size, 0.01, method=method)
    result = (
        engine.run(phases, omega, matrix, 0.0, 0.0, matrix, n_steps=1)
        if batch
        else engine.step(phases, omega, matrix, 0.0, 0.0, matrix)
    )
    np.testing.assert_array_equal(result[:5], np.zeros(5))
    assert not np.any(np.signbit(result[:5]))
    assert result[5] == interior
    assert result[6] == pytest.approx(0.27, rel=0.0, abs=2e-16)
    assert np.all((result >= 0.0) & (result < tau))
    assert not np.shares_memory(result, phases)
    assert tuple(a.tobytes() for a in (phases, omega, matrix)) == before


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
def test_coupled_crossing_recovers_after_real_input_refusal(method: str) -> None:
    """Keep the same real engine usable after refusing a nonfinite frequency."""
    engine = UPDEEngine(2, 0.01, method=method)
    phases = np.zeros(2)
    coupling = np.array([[0.0, 0.8], [0.5, 0.0]])
    alpha = np.zeros((2, 2))
    with pytest.raises(ValueError, match="NaN/Inf"):
        engine.step(phases, np.array([np.inf, 0.0]), coupling, 0.0, 0.0, alpha)
    assert engine.time == 0.0
    assert engine.last_dt == 0.01
    result = engine.step(phases, np.array([-1e-15, 2e-15]), coupling, 0.0, 0.0, alpha)
    assert result[0] == 0.0
    assert not np.signbit(result[0])
    assert 0.0 < result[1] < 1e-16
    np.testing.assert_array_equal(phases, np.zeros(2))


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("batch", [False, True])
def test_finite_computation_overflow_refuses_and_recovers(
    method: str, batch: bool
) -> None:
    """Reject actual finite-input divergence before advancing public state."""
    engine = UPDEEngine(1, 0.01, method=method)
    phases = np.zeros(1)
    omega = np.array([1e308])
    matrix = np.zeros((1, 1))
    before = tuple(a.tobytes() for a in (phases, omega, matrix))
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="NaN/Inf|nonfinite|finite"),
    ):
        if batch:
            engine.run(phases, omega, matrix, 1e308, np.pi / 2.0, matrix, n_steps=1)
        else:
            engine.step(phases, omega, matrix, 1e308, np.pi / 2.0, matrix)
    assert engine.time == 0.0
    assert engine.last_dt == 0.01
    assert before == tuple(a.tobytes() for a in (phases, omega, matrix))
    recovered = engine.step(phases, np.ones(1), matrix, 0.0, 0.0, matrix)
    np.testing.assert_allclose(recovered, [0.01], rtol=0.0, atol=2e-17)
    assert engine.time == 0.01
