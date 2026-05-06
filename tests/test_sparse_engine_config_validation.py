# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — sparse engine config validation tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.sparse_engine import SparseUPDEEngine


@pytest.mark.parametrize("n_oscillators", [False, 0, -1, 1.5, "4"])
def test_sparse_engine_rejects_invalid_oscillator_count(
    n_oscillators: Any,
) -> None:
    with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
        SparseUPDEEngine(n_oscillators=n_oscillators, dt=0.01)


@pytest.mark.parametrize("dt", [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"])
def test_sparse_engine_rejects_invalid_timestep(dt: Any) -> None:
    with pytest.raises(ValueError, match="dt must be a finite positive real"):
        SparseUPDEEngine(n_oscillators=4, dt=dt)


@pytest.mark.parametrize("field", ["atol", "rtol"])
@pytest.mark.parametrize(
    "value", [False, 0.0, -1e-6, float("nan"), float("inf"), "1e-6"]
)
def test_sparse_engine_rejects_invalid_tolerances(
    field: str,
    value: Any,
) -> None:
    kwargs: dict[str, Any] = {"n_oscillators": 4, "dt": 0.01, field: value}

    with pytest.raises(ValueError, match=f"{field} must be a finite positive real"):
        SparseUPDEEngine(**kwargs)


@pytest.mark.parametrize("n_steps", [False, -1, 1.5, "10"])
def test_sparse_engine_run_rejects_invalid_step_count(n_steps: Any) -> None:
    engine = SparseUPDEEngine(n_oscillators=4, dt=0.01)
    phases = np.zeros(4, dtype=np.float64)
    omegas = np.ones(4, dtype=np.float64)
    row_ptr = np.zeros(5, dtype=np.uint64)
    col_indices = np.zeros(0, dtype=np.uint64)
    knm_values = np.zeros(0, dtype=np.float64)
    alpha_values = np.zeros(0, dtype=np.float64)

    with pytest.raises(ValueError, match="n_steps must be >= 0"):
        engine.run(
            phases,
            omegas,
            row_ptr,
            col_indices,
            knm_values,
            0.0,
            0.0,
            alpha_values,
            n_steps=n_steps,
        )


def test_sparse_engine_run_zero_steps_returns_copy() -> None:
    engine = SparseUPDEEngine(n_oscillators=4, dt=0.01)
    phases = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    omegas = np.ones(4, dtype=np.float64)
    row_ptr = np.zeros(5, dtype=np.uint64)
    col_indices = np.zeros(0, dtype=np.uint64)
    knm_values = np.zeros(0, dtype=np.float64)
    alpha_values = np.zeros(0, dtype=np.float64)

    result = engine.run(
        phases,
        omegas,
        row_ptr,
        col_indices,
        knm_values,
        0.0,
        0.0,
        alpha_values,
        n_steps=0,
    )

    np.testing.assert_array_equal(result, phases)
    assert result is not phases


def test_sparse_engine_normalises_accepted_numpy_scalars() -> None:
    engine = SparseUPDEEngine(
        n_oscillators=np.int64(4),
        dt=np.float64(0.01),
        atol=np.float64(1e-6),
        rtol=np.float64(1e-3),
    )

    assert engine._n == 4
    assert pytest.approx(0.01) == engine._dt
    assert pytest.approx(1e-6) == engine._atol
    assert pytest.approx(1e-3) == engine._rtol
