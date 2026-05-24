# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sheaf engine Python fallback contracts

"""
Numerical parity and validation contracts for SheafUPDEEngine Python fallback
execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import sheaf_engine

TWO_PI = 2.0 * np.pi


def test_sheaf_engine_python_fallback_respects_restriction_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sheaf_engine, "_HAS_RUST", False)
    phases = np.array([[0.1, 0.4], [0.9, 1.2]], dtype=np.float64)
    omegas = np.array([[0.2, -0.1], [0.05, 0.15]], dtype=np.float64)
    restriction = np.zeros((2, 2, 2, 2), dtype=np.float64)
    restriction[0, 1] = np.array([[0.3, 0.1], [0.2, 0.4]])
    restriction[1, 0] = np.array([[0.5, 0.0], [0.1, 0.2]])
    psi = np.array([0.7, 1.0], dtype=np.float64)

    engine = sheaf_engine.SheafUPDEEngine(2, 2, 0.01, method="euler")
    result = engine.step(phases, omegas, restriction, 0.2, psi)

    deriv = omegas.copy()
    for i in range(2):
        for dim in range(2):
            for j in range(2):
                for k in range(2):
                    deriv[i, dim] += restriction[i, j, dim, k] * np.sin(
                        phases[j, k] - phases[i, dim]
                    )
            deriv[i, dim] += 0.2 * np.sin(psi[dim] - phases[i, dim])
    np.testing.assert_allclose(result, (phases + 0.01 * deriv) % TWO_PI)

    rk4 = sheaf_engine.SheafUPDEEngine(2, 2, 0.01, method="rk4").run(
        phases, omegas, restriction, 0.1, psi, 2
    )
    rk45 = sheaf_engine.SheafUPDEEngine(2, 2, 0.01, method="rk45").step(
        phases, omegas, restriction, 0.1, psi
    )
    assert rk4.shape == phases.shape
    assert rk45.shape == phases.shape
    assert np.all((rk4 >= 0.0) & (rk4 < TWO_PI))


def test_sheaf_engine_rejects_invalid_python_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sheaf_engine, "_HAS_RUST", False)
    for kwargs in (
        {"n_oscillators": True, "d_dimensions": 2, "dt": 0.01},
        {"n_oscillators": 2, "d_dimensions": 0, "dt": 0.01},
        {"n_oscillators": 2, "d_dimensions": 2, "dt": False},
        {"n_oscillators": 2, "d_dimensions": 2, "dt": float("inf")},
    ):
        with pytest.raises(ValueError):
            sheaf_engine.SheafUPDEEngine(**kwargs)
    with pytest.raises(ValueError, match="Unknown method"):
        sheaf_engine.SheafUPDEEngine(2, 2, 0.01, method="bad")
    engine = sheaf_engine.SheafUPDEEngine(2, 2, 0.01)
    with pytest.raises(ValueError, match="n_steps"):
        engine.run(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2, 2, 2)),
            0.0,
            np.zeros(2),
            -1,
        )
