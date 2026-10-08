# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PhaseSINDy tests

"""Verify public phase regression against independent physical and analytic data.

The Euler generator defines a known directed network independently of the
estimator. Tests run on whichever real installation owns this process; separate
profile tests prove actual native use and genuine Python-only execution.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy

FloatArray = NDArray[np.float64]


def _simulate(
    phases_init: FloatArray,
    omega: FloatArray,
    coupling: FloatArray,
    dt: float,
    steps: int,
) -> FloatArray:
    """Generate forward-Euler samples of an independently specified network.

    Parameters
    ----------
    phases_init, omega : FloatArray
        Initial phases and angular velocities, one entry per oscillator.
    coupling : FloatArray
        Target-by-source coupling matrix; diagonal entries are unused.
    dt : float
        Sample period in seconds.
    steps : int
        Number of stored samples, including the initial state.

    Returns
    -------
    FloatArray
        Wrapped phases with shape ``(steps, len(omega))``.
    """
    phases = np.empty((steps, omega.size), dtype=np.float64)
    phases[0] = phases_init
    for time in range(1, steps):
        previous = phases[time - 1]
        derivative = omega.copy()
        for target in range(omega.size):
            for source in range(omega.size):
                if target != source:
                    derivative[target] += coupling[target, source] * math.sin(
                        float(previous[source] - previous[target])
                    )
        phases[time] = np.remainder(previous + dt * derivative, math.tau)
    return phases


@pytest.mark.parametrize("nodes", [2, 3, 5])
def test_sindy_recovery(nodes: int) -> None:
    """Recover every directed coefficient, including asymmetric and negative terms."""
    rng = np.random.default_rng(812 + nodes)
    omega = np.linspace(0.7, 2.4, nodes, dtype=np.float64)
    coupling = rng.uniform(-0.3, 0.3, (nodes, nodes))
    np.fill_diagonal(coupling, 0.0)
    phases = _simulate(rng.uniform(0.0, 2.0, nodes), omega, coupling, 0.02, 600)
    model = PhaseSINDy(threshold=0.0, max_iter=3)
    coefficients = model.fit(phases, 0.02)
    for target, actual in enumerate(coefficients):
        expected = np.concatenate(
            (omega[target : target + 1], np.delete(coupling[target], target))
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-8)
        assert actual.shape == (nodes,)
        assert actual.dtype == np.float64
        assert np.isfinite(actual).all()
    equations = model.get_equations()
    assert len(equations) == nodes
    for target in range(nodes):
        assert model.feature_names[target] == ["1"] + [
            f"sin(theta_{source} - theta_{target})"
            for source in range(nodes)
            if source != target
        ]
        assert f"{omega[target]:.4f} * 1" in equations[target]


@pytest.mark.parametrize("turns", [-9, -3, 0, 3, 9])
def test_sindy_recovers_principal_frequency_for_arbitrary_turn_aliases(
    turns: int,
) -> None:
    """Whole-turn sampling aliases cannot change the principal angular velocity."""
    phases = np.arange(12, dtype=np.float64).reshape(-1, 1) * (0.1 + turns * math.tau)
    coefficients = PhaseSINDy(threshold=0.0).fit(phases, 0.1)
    np.testing.assert_allclose(coefficients, [[1.0]], rtol=0.0, atol=2e-11)


@pytest.mark.parametrize("increment", [math.pi, -math.pi, 3 * math.pi, -3 * math.pi])
def test_sindy_preserves_exact_half_turn_sign(increment: float) -> None:
    """At the ambiguous half-turn boundary retain the original increment sign."""
    coefficients = PhaseSINDy(threshold=0.0).fit(
        np.array([[0.0], [increment]], dtype=np.float64), 1.0
    )
    np.testing.assert_allclose(
        coefficients, [[math.copysign(math.pi, increment)]], rtol=0.0, atol=1e-14
    )


@pytest.mark.parametrize(("dt", "offset"), [(0.01, 0.4), (0.015625, 0.5)])
def test_sindy_rank_deficiency_has_analytic_minimum_norm_solution(
    dt: float, offset: float
) -> None:
    """Dependent sine/constant features yield the closed-form pseudoinverse solution."""
    times = np.arange(40, dtype=np.float64) * dt
    phases = np.column_stack((times, times + offset))
    sine = math.sin(offset)
    inverse_norm = 1.0 / (1.0 + sine * sine)
    expected = [
        [inverse_norm, sine * inverse_norm],
        [inverse_norm, -sine * inverse_norm],
    ]
    coefficients = PhaseSINDy(threshold=0.0, max_iter=3).fit(phases, dt)
    np.testing.assert_allclose(coefficients, expected, rtol=0.0, atol=2e-12)


def test_sindy_nodewise_turn_shifts_and_strides_preserve_directed_fit() -> None:
    """Whole-turn aliases and noncontiguous input preserve directed coefficients."""
    omega = np.array([1.1, 1.8, 2.6], dtype=np.float64)
    coupling = np.array([[0.0, 0.2, 0.35], [-0.1, 0.0, 0.13], [0.07, -0.18, 0.0]])
    phases = _simulate(np.array([0.0, 0.7, 2.1]), omega, coupling, 0.03, 500)
    shifts = np.arange(500)[:, None] * np.array([7, -3, 11])[None, :] * math.tau
    storage = np.empty((500, 6), dtype=np.float64)
    storage[:, ::2] = phases + shifts
    strided = storage[:, ::2]
    assert not strided.flags.c_contiguous
    coefficients = PhaseSINDy(threshold=0.0, max_iter=3).fit(strided, 0.03)
    for target, actual in enumerate(coefficients):
        expected = np.concatenate(
            (omega[target : target + 1], np.delete(coupling[target], target))
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-8)


def test_benchmark_refuses_corrupt_reference_after_actual_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An intentionally wrong oracle cannot admit real public numerical work."""
    from benchmarks import phase_sindy_benchmark as diagnostic

    original_cases = diagnostic._cases

    def wrong_reference() -> list[diagnostic._Case]:
        """Damage only expected data; retain the original trajectory and solver."""
        case = original_cases()[0]
        return [replace(case, reference=case.reference + 1.0)]

    monkeypatch.setattr(diagnostic, "_cases", wrong_reference)
    with pytest.raises(RuntimeError, match="coefficient error"):
        diagnostic.benchmark_phase_sindy(repeats=1)


def test_sindy_zero_coupling() -> None:
    """Independent rotating oscillators have zero couplings and correct frequencies."""
    omega = np.array([1.0, 1.3])
    phases = _simulate(np.array([0.0, 0.5]), omega, np.zeros((2, 2)), 0.05, 400)
    coefficients = PhaseSINDy(threshold=0.05).fit(phases, 0.05)
    np.testing.assert_allclose(
        coefficients, [[1.0, 0.0], [1.3, 0.0]], rtol=0.0, atol=2e-11
    )


def test_sindy_threshold_sparsifies_weak_terms() -> None:
    """Remove weak edges at high thresholds and recover them at smaller thresholds."""
    omega = np.array([1.0, 1.2])
    coupling = np.array([[0.0, 0.04], [0.04, 0.0]])
    phases = _simulate(np.array([0.0, 0.5]), omega, coupling, 0.05, 400)
    high = PhaseSINDy(threshold=0.2).fit(phases, 0.05)
    low = PhaseSINDy(threshold=0.005).fit(phases, 0.05)
    zero = PhaseSINDy(threshold=0.0).fit(phases, 0.05)
    assert high[0][1] == high[1][1] == 0.0
    np.testing.assert_allclose(
        [low[0][1], low[1][1]], [0.04, 0.04], rtol=0.0, atol=1e-10
    )
    np.testing.assert_allclose(low, zero, rtol=0.0, atol=1e-10)


def test_sindy_empty_and_threshold_equal_equations() -> None:
    """Empty support formats as zero, while exact threshold equality retains a term."""
    phases = np.array([[0.0], [0.125]], dtype=np.float64)
    retained = PhaseSINDy(threshold=0.125, max_iter=1)
    retained.fit(phases, 1.0)
    assert retained.get_equations() == ["d(theta_0)/dt = 0.1250 * 1"]
    removed = PhaseSINDy(threshold=0.2)
    np.testing.assert_array_equal(removed.fit(phases, 1.0), [[0.0]])
    assert removed.get_equations() == ["d(theta_0)/dt = 0"]


def test_sindy_repeated_fit_is_deterministic() -> None:
    """Repeated public fits replace complete state with identical numerical results."""
    phases = np.arange(20, dtype=np.float64).reshape(-1, 1) * 0.2
    model = PhaseSINDy(threshold=0.0)
    first = model.fit(phases, 0.1)
    first_equations = model.get_equations()
    second = model.fit(phases, 0.1)
    np.testing.assert_array_equal(first, second)
    assert first_equations == model.get_equations() == ["d(theta_0)/dt = 2.0000 * 1"]


class TestSindyValidation:
    """Preserve the constructor refusal contracts salvaged from broad former tests."""

    def test_rejects_negative_threshold(self) -> None:
        """A negative sparsity threshold cannot define meaningful hard thresholding."""
        with pytest.raises(ValueError, match="threshold.*non-negative"):
            PhaseSINDy(threshold=-0.01)

    def test_rejects_zero_max_iter(self) -> None:
        """Zero iterations are rejected instead of silently skipping sparse fitting."""
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            PhaseSINDy(max_iter=0)

    def test_rejects_negative_max_iter(self) -> None:
        """A negative iteration budget is rejected by the public constructor."""
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            PhaseSINDy(max_iter=-3)
