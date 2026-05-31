# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Three-factor plasticity tests

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.plasticity import (
    compute_eligibility,
    three_factor_update,
)
from tests.typing_contracts import assert_precise_ndarray_hint


def _physical_matrix(n: int, value: float = 1.0) -> np.ndarray:
    matrix = np.full((n, n), value, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def test_public_array_contracts_are_parameterised() -> None:
    """Public plasticity array contracts stay element-typed."""
    for hint in [
        get_type_hints(compute_eligibility)["phases"],
        get_type_hints(compute_eligibility)["return"],
        get_type_hints(three_factor_update)["knm"],
        get_type_hints(three_factor_update)["eligibility"],
        get_type_hints(three_factor_update)["return"],
    ]:
        assert_precise_ndarray_hint(hint)
        assert "float64" in str(hint)


def test_eligibility_synchronised_phases():
    """Identical phases → cos(0) = 1 off-diagonal, 0 on diagonal."""
    phases = np.zeros(5)
    elig = compute_eligibility(phases)
    expected = np.ones((5, 5))
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(elig, expected, atol=1e-12)


def test_eligibility_antiphase():
    """Two oscillators at 0 and pi → cos(pi) = -1."""
    phases = np.array([0.0, np.pi])
    elig = compute_eligibility(phases)
    assert elig[0, 0] == 0.0
    assert elig[1, 1] == 0.0
    np.testing.assert_allclose(elig[0, 1], -1.0, atol=1e-12)
    np.testing.assert_allclose(elig[1, 0], -1.0, atol=1e-12)


def test_eligibility_diagonal_zero():
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, 20)
    elig = compute_eligibility(phases)
    np.testing.assert_allclose(np.diag(elig), 0.0, atol=1e-15)


def test_three_factor_gate_off_no_update():
    knm = _physical_matrix(3)
    elig = _physical_matrix(3)
    result = three_factor_update(knm, elig, modulator=1.0, phase_gate=False, lr=0.1)
    np.testing.assert_array_equal(result, knm)


def test_three_factor_positive_modulator_increases_coupling():
    knm = np.zeros((4, 4))
    elig = np.ones((4, 4))
    np.fill_diagonal(elig, 0.0)
    result = three_factor_update(knm, elig, modulator=1.0, phase_gate=True, lr=0.05)
    expected = 0.05 * elig
    np.testing.assert_allclose(result, expected, atol=1e-15)


def test_three_factor_negative_modulator_decreases_coupling():
    knm = _physical_matrix(3)
    elig = _physical_matrix(3)
    result = three_factor_update(knm, elig, modulator=-1.0, phase_gate=True, lr=0.1)
    off_diag = ~np.eye(3, dtype=bool)
    assert np.all(result[off_diag] < knm[off_diag])
    np.testing.assert_array_equal(np.diag(result), 0.0)


def test_three_factor_does_not_mutate_input():
    knm = _physical_matrix(3)
    original = knm.copy()
    elig = _physical_matrix(3)
    three_factor_update(knm, elig, modulator=1.0, phase_gate=True, lr=0.1)
    np.testing.assert_array_equal(knm, original)


def test_three_factor_zero_modulator_no_change():
    knm = _physical_matrix(5, value=0.5)
    elig = _physical_matrix(5)
    result = three_factor_update(knm, elig, modulator=0.0, phase_gate=True, lr=0.1)
    np.testing.assert_allclose(result, knm, atol=1e-15)


def test_eligibility_symmetry():
    """cos(θ_j - θ_i) = cos(θ_i - θ_j) → eligibility is symmetric."""
    rng = np.random.default_rng(123)
    phases = rng.uniform(0, 2 * np.pi, 10)
    elig = compute_eligibility(phases)
    np.testing.assert_allclose(elig, elig.T, atol=1e-12)


@pytest.mark.parametrize(
    ("phases", "match"),
    [
        (np.array([[0.0, 1.0]]), "phases"),
        (np.array([0.0, np.nan]), "phases"),
        (np.array([True, False]), "phases"),
    ],
)
def test_eligibility_rejects_invalid_phases(phases, match):
    with pytest.raises(ValueError, match=match):
        compute_eligibility(phases)


def test_eligibility_rejects_mixed_boolean_phase_alias():
    with pytest.raises(ValueError, match="phases must not contain boolean"):
        compute_eligibility([True, 0.5])


def test_eligibility_rejects_complex_phase_alias_without_casting():
    with pytest.raises(ValueError, match="phases must be real-valued"):
        compute_eligibility(np.array([0.0 + 0.5j, 1.0 + 0.0j]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "knm": np.array([[True, False], [False, True]]),
                "eligibility": np.ones((2, 2)),
            },
            "knm",
        ),
        (
            {
                "knm": _physical_matrix(2),
                "eligibility": np.array([[True, False], [False, True]]),
            },
            "eligibility",
        ),
        (
            {"knm": _physical_matrix(2), "eligibility": np.ones((2, 3))},
            "eligibility",
        ),
        (
            {
                "knm": np.array([[1.0, np.inf], [0.0, 1.0]]),
                "eligibility": np.ones((2, 2)),
            },
            "knm",
        ),
    ],
)
def test_three_factor_rejects_invalid_arrays(kwargs, match):
    with pytest.raises(ValueError, match=match):
        three_factor_update(
            kwargs["knm"],
            kwargs["eligibility"],
            modulator=1.0,
            phase_gate=True,
        )


def test_three_factor_rejects_mixed_boolean_matrix_aliases():
    with pytest.raises(ValueError, match="knm must not contain boolean"):
        three_factor_update(
            [[0.0, True], [0.0, 0.0]],
            np.ones((2, 2)),
            modulator=1.0,
            phase_gate=True,
        )

    with pytest.raises(ValueError, match="eligibility must not contain boolean"):
        three_factor_update(
            np.zeros((2, 2)),
            [[0.0, True], [0.0, 0.0]],
            modulator=1.0,
            phase_gate=True,
        )


@pytest.mark.parametrize(
    ("knm", "eligibility", "match"),
    [
        (
            np.array([[0.0, -0.1], [0.2, 0.0]]),
            _physical_matrix(2),
            "knm must be non-negative",
        ),
        (
            np.eye(2),
            _physical_matrix(2),
            "knm diagonal",
        ),
        (
            _physical_matrix(2),
            np.eye(2),
            "eligibility diagonal",
        ),
        (
            _physical_matrix(2),
            np.array([[0.0, 1.25], [0.5, 0.0]]),
            "eligibility values",
        ),
        (
            np.array([[0.0 + 0.0j, 0.1 + 0.2j], [0.1 + 0.0j, 0.0 + 0.0j]]),
            _physical_matrix(2),
            "knm must be real-valued",
        ),
        (
            _physical_matrix(2),
            np.array([[0.0 + 0.0j, 0.1 + 0.2j], [0.1 + 0.0j, 0.0 + 0.0j]]),
            "eligibility must be real-valued",
        ),
    ],
)
def test_three_factor_rejects_non_physical_coupling_contracts(
    knm,
    eligibility,
    match,
):
    with pytest.raises(ValueError, match=match):
        three_factor_update(
            knm,
            eligibility,
            modulator=1.0,
            phase_gate=True,
        )


def test_three_factor_negative_modulator_clamps_to_zero_without_self_coupling():
    knm = np.array([[0.0, 0.02], [0.03, 0.0]])
    eligibility = _physical_matrix(2)

    result = three_factor_update(
        knm,
        eligibility,
        modulator=-1.0,
        phase_gate=True,
        lr=1.0,
    )

    np.testing.assert_array_equal(result, np.zeros((2, 2)))


@pytest.mark.parametrize(
    ("modulator", "phase_gate", "lr", "error"),
    [
        (True, True, 0.01, TypeError),
        (1.0, 1, 0.01, TypeError),
        (1.0, True, True, TypeError),
        (np.nan, True, 0.01, ValueError),
        (1.0, True, -0.01, ValueError),
    ],
)
def test_three_factor_rejects_invalid_scalars(modulator, phase_gate, lr, error):
    with pytest.raises(error):
        three_factor_update(
            _physical_matrix(2),
            _physical_matrix(2),
            modulator=modulator,
            phase_gate=phase_gate,
            lr=lr,
        )


class TestPlasticityPipelineWiring:
    """Pipeline: engine phases → eligibility → three-factor → updated K_nm."""

    def test_plasticity_loop_changes_coupling(self):
        """UPDEEngine → phases → eligibility → three_factor_update →
        engine uses updated K_nm. Proves plasticity isn't decorative."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        # Run 50 steps, then apply plasticity, then run 50 more
        for _ in range(50):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        elig = compute_eligibility(phases)
        knm_updated = three_factor_update(
            knm,
            elig,
            modulator=1.0,
            phase_gate=True,
            lr=0.01,
        )
        assert not np.allclose(knm, knm_updated), "Plasticity must change K_nm"

        for _ in range(50):
            phases = eng.step(phases, omegas, knm_updated, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
