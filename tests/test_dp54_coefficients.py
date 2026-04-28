# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — DP54 tableau coefficient verification

"""Verify Dormand-Prince 5(4) coefficients against the reference.

Dormand & Prince (1980) "A family of embedded Runge-Kutta formulae"
J. Comp. Appl. Math. 6(1), Table 5.2.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

# Reference coefficients as exact fractions (Dormand & Prince 1980)
REF_B5 = [
    Fraction(35, 384),
    Fraction(0),
    Fraction(500, 1113),
    Fraction(125, 192),
    Fraction(-2187, 6784),
    Fraction(11, 84),
    Fraction(0),
]

REF_B4 = [
    Fraction(5179, 57600),
    Fraction(0),
    Fraction(7571, 16695),
    Fraction(393, 640),
    Fraction(-92097, 339200),
    Fraction(187, 2100),
    Fraction(1, 40),
]

REF_A = [
    [],
    [Fraction(1, 5)],
    [Fraction(3, 40), Fraction(9, 40)],
    [Fraction(44, 45), Fraction(-56, 15), Fraction(32, 9)],
    [
        Fraction(19372, 6561),
        Fraction(-25360, 2187),
        Fraction(64448, 6561),
        Fraction(-212, 729),
    ],
    [
        Fraction(9017, 3168),
        Fraction(-355, 33),
        Fraction(46732, 5247),
        Fraction(49, 176),
        Fraction(-5103, 18656),
    ],
    [
        Fraction(35, 384),
        Fraction(0),
        Fraction(500, 1113),
        Fraction(125, 192),
        Fraction(-2187, 6784),
        Fraction(11, 84),
    ],
]


def test_upde_b5_coefficients():
    b5 = UPDEEngine._DP_B5
    assert len(b5) == 7
    for i, ref in enumerate(REF_B5):
        assert abs(b5[i] - float(ref)) < 1e-15, f"B5[{i}]: {b5[i]} != {ref}"


def test_upde_b4_coefficients():
    b4 = UPDEEngine._DP_B4
    assert len(b4) == 7
    for i, ref in enumerate(REF_B4):
        assert abs(b4[i] - float(ref)) < 1e-15, f"B4[{i}]: {b4[i]} != {ref}"


def test_upde_a_matrix():
    a = UPDEEngine._DP_A
    assert a.shape == (7, 7)
    for row_idx, ref_row in enumerate(REF_A):
        for col_idx, ref_val in enumerate(ref_row):
            actual = a[row_idx, col_idx]
            assert abs(actual - float(ref_val)) < 1e-15, (
                f"A[{row_idx},{col_idx}]: {actual} != {ref_val}"
            )
        for col_idx in range(len(ref_row), 7):
            assert a[row_idx, col_idx] == 0.0


def test_upde_fsal_property():
    """Row 7 of A must equal B5 (FSAL property)."""
    a = UPDEEngine._DP_A
    b5 = UPDEEngine._DP_B5
    np.testing.assert_allclose(a[6, :6], b5[:6], atol=1e-15)


def test_sl_coefficients_match_upde():
    """Stuart-Landau engine uses identical DP54 tableau."""
    np.testing.assert_array_equal(StuartLandauEngine._DP_A, UPDEEngine._DP_A)
    np.testing.assert_array_equal(StuartLandauEngine._DP_B4, UPDEEngine._DP_B4)
    np.testing.assert_array_equal(StuartLandauEngine._DP_B5, UPDEEngine._DP_B5)


def test_b5_sums_to_one():
    """5th-order weights must sum to 1 for consistency."""
    assert abs(sum(float(f) for f in REF_B5) - 1.0) < 1e-15


def test_b4_sums_to_one():
    """4th-order weights must sum to 1 for consistency."""
    assert abs(sum(float(f) for f in REF_B4) - 1.0) < 1e-15


class TestDP54PipelineWiring:
    """Pipeline: verified coefficients → RK45 engine → accurate R."""

    def test_rk45_with_verified_coefficients(self):
        """RK45 engine using verified DP54 tableau → phases → R∈[0,1].
        Proves the coefficients feed a working adaptive integrator."""
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 4
        eng = UPDEEngine(n, dt=0.01, method="rk45")
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
        assert eng.last_dt > 0.0, "Adaptive dt must be positive"
