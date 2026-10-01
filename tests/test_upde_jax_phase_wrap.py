# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual JAX UPDE phase projection contracts

"""Test scpn_phase_orchestrator.upde._jax_phase_wrap via the public JIT engine.

The host's output-range refusal cannot be reached by a successful real producer:
both JIT methods project with the actual dtype period. These tests check the
published bounds instead of replacing the JIT output to reach that guard. The
optional JAX import refusal requires a genuinely JAX-absent interpreter; this
selected module requires the installed runtime and never fabricates absence.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_public_jit_torus_projection(x64: bool, method: str) -> None:
    """Execute actual float32/float64 JIT steps at the cut and interior neighbour."""
    dtype = np.float64 if x64 else np.float32
    period = dtype(2.0 * np.pi)
    interior = np.nextafter(period, dtype(0.0))
    phases = np.array(
        [0.0, -float(period), -0.0, float(period), float(interior), 0.25],
        dtype=np.float64,
    )
    omegas = np.array([-1e-15, 0.0, 0.0, 0.0, 0.0, 2.0])
    matrix = np.zeros((phases.size, phases.size))
    before = tuple(a.tobytes() for a in (phases, omegas, matrix))
    with jax.enable_x64(x64):
        result = JaxUPDEEngine(phases.size, method=method).step(
            phases, omegas, matrix, 0.0, 0.0, matrix
        )
    assert result.dtype == dtype
    np.testing.assert_array_equal(result[:4], np.zeros(4))
    assert not np.any(np.signbit(result[:4]))
    assert result[4] == interior
    assert result[5] == pytest.approx(0.27, rel=0.0, abs=3e-8 if not x64 else 2e-16)
    assert np.all((result >= 0.0) & (result < period))
    assert before == tuple(a.tobytes() for a in (phases, omegas, matrix))
    assert not np.shares_memory(result, phases)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_public_jit_refuses_nonfinite_then_recovers(method: str) -> None:
    """Keep real input admission and same-instance success after refusal."""
    engine = JaxUPDEEngine(1, method=method)
    matrix = np.zeros((1, 1))
    with pytest.raises(ValueError, match="finite"):
        engine.step(np.array([np.inf]), np.zeros(1), matrix, 0.0, 0.0, matrix)
    result = engine.step(np.zeros(1), np.array([-1e-15]), matrix, 0.0, 0.0, matrix)
    assert result[0] == 0.0
    assert not np.signbit(result[0])


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_float32_conversion_overflow_refuses_and_recovers(method: str) -> None:
    """Refuse actual finite-input conversion overflow without hiding it as zero."""
    with jax.enable_x64(False):
        engine = JaxUPDEEngine(1, method=method)
        matrix = np.zeros((1, 1))
        with pytest.raises(ValueError, match="JAX output contains NaN/Inf"):
            engine.step(np.array([1e308]), np.zeros(1), matrix, 0.0, 0.0, matrix)
        result = engine.step(np.zeros(1), np.array([-1e-15]), matrix, 0.0, 0.0, matrix)
        assert result[0] == 0.0
        assert not np.signbit(result[0])
