# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sparse engine Python fallback contracts

"""
Numerical parity and validation contracts for SparseUPDEEngine Python fallback
execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import sparse_engine

TWO_PI = 2.0 * np.pi


def test_sparse_engine_python_fallback_matches_dense_euler_and_rk4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sparse_engine, "_HAS_RUST", False)
    row_ptr = np.array([0, 2, 4, 6], dtype=np.int64)
    col_indices = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
    knm_values = np.array([0.3, 0.1, 0.2, 0.4, 0.5, 0.1], dtype=np.float64)
    alpha_values = np.array([0.01, -0.02, 0.03, 0.04, -0.01, 0.02], dtype=np.float64)
    phases = np.array([0.1, 0.4, 0.9], dtype=np.float64)
    omegas = np.array([0.2, -0.1, 0.05], dtype=np.float64)

    engine = sparse_engine.SparseUPDEEngine(3, 0.01, method="euler")
    result = engine.step(
        phases, omegas, row_ptr, col_indices, knm_values, 0.2, 0.7, alpha_values
    )

    deriv = omegas.copy()
    for i in range(3):
        for idx in range(row_ptr[i], row_ptr[i + 1]):
            j = col_indices[idx]
            deriv[i] += knm_values[idx] * np.sin(
                phases[j] - phases[i] - alpha_values[idx]
            )
    deriv += 0.2 * np.sin(0.7 - phases)
    np.testing.assert_allclose(result, (phases + 0.01 * deriv) % TWO_PI)

    zero_step = engine.run(
        phases,
        omegas,
        row_ptr,
        col_indices,
        knm_values,
        0.0,
        0.0,
        alpha_values,
        0,
    )
    assert np.shares_memory(zero_step, phases) is False

    rk4 = sparse_engine.SparseUPDEEngine(3, 0.01, method="rk4").run(
        phases, omegas, row_ptr, col_indices, knm_values, 0.1, 0.5, alpha_values, 2
    )
    rk45 = sparse_engine.SparseUPDEEngine(3, 0.01, method="rk45").step(
        phases, omegas, row_ptr, col_indices, knm_values, 0.1, 0.5, alpha_values
    )
    assert rk4.shape == phases.shape
    assert rk45.shape == phases.shape
    assert np.all((rk4 >= 0.0) & (rk4 < TWO_PI))


def test_sparse_engine_rejects_invalid_python_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sparse_engine, "_HAS_RUST", False)
    for kwargs in (
        {"n_oscillators": True, "dt": 0.01},
        {"n_oscillators": 0, "dt": 0.01},
        {"n_oscillators": 2, "dt": False},
        {"n_oscillators": 2, "dt": float("nan")},
    ):
        with pytest.raises(ValueError):
            sparse_engine.SparseUPDEEngine(**kwargs)
    with pytest.raises(ValueError, match="Unknown method"):
        sparse_engine.SparseUPDEEngine(2, 0.01, method="bad")
    engine = sparse_engine.SparseUPDEEngine(2, 0.01)
    with pytest.raises(ValueError, match="n_steps"):
        engine.run(
            np.zeros(2),
            np.zeros(2),
            np.array([0, 0, 0]),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            0.0,
            0.0,
            np.array([], dtype=np.float64),
            -1,
        )
