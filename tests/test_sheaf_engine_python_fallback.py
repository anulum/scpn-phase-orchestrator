# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sheaf engine Python fallback contracts

"""Verify sheaf numerical contracts with the native runtime and genuine absence."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from scpn_phase_orchestrator.upde import sheaf_engine

TWO_PI = 2.0 * np.pi


def test_sheaf_engine_python_fallback_respects_restriction_maps() -> None:
    """Both real runtimes integrate nonzero anisotropic maps and external drive."""
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

    def derivative(_time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the tensor equation with independent broadcast contraction."""
        theta = state.reshape(2, 2)
        differences = theta[None, :, None, :] - theta[:, None, :, None]
        coupling = np.einsum("ijdk,ijdk->id", restriction, np.sin(differences))
        return np.asarray(omegas + coupling + 0.1 * np.sin(psi - theta)).ravel()

    for method, steps in (("rk4", 2), ("rk45", 7)):
        engine = sheaf_engine.SheafUPDEEngine(
            2, 2, 0.01, method=method, atol=1e-11, rtol=1e-11
        )
        output = engine.run(phases, omegas, restriction, 0.1, psi, steps)
        reference = solve_ivp(
            derivative,
            (0.0, 0.01 * steps),
            phases.ravel(),
            method="DOP853",
            atol=1e-13,
            rtol=1e-13,
        )
        assert reference.success
        np.testing.assert_allclose(
            output, reference.y[:, -1].reshape(2, 2), atol=1e-10, rtol=0
        )
        np.testing.assert_array_equal(phases, [[0.1, 0.4], [0.9, 1.2]])


def test_sheaf_engine_rejects_invalid_python_configuration() -> None:
    """Both real runtimes reject malformed configuration and batch counts."""
    for n, d, dt in (
        (True, 2, 0.01),
        (2, 0, 0.01),
        (2, 2, False),
        (2, 2, float("inf")),
    ):
        with pytest.raises(ValueError):
            sheaf_engine.SheafUPDEEngine(n, d, dt)
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
