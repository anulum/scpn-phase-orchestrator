# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public composite ownership and fault controls

"""Original public composite calls; injected outputs are negative controls only."""

from __future__ import annotations

import cProfile
import math

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import bifurcation as bif
from scpn_phase_orchestrator.upde.basin_stability import AVAILABLE_BACKENDS


def test_named_rust_sweep_observes_the_original_batched_export() -> None:
    """Nonzero-lag public trials use the original batch owner and scalar Euler law."""
    omega = np.array([0.2, -0.1])
    graph = np.array([[0.0, -0.7], [0.3, 0.0]])
    lag = np.array([[0.0, 0.4], [-0.2, 0.0]])
    if "rust" not in AVAILABLE_BACKENDS:
        with pytest.raises(ImportError, match="requested basin backend 'rust'"):
            bif.trace_sync_transition(
                omega,
                graph,
                lag,
                n_points=3,
                n_transient=0,
                n_measure=1,
                backend="rust",
            )
        return
    phases = np.random.default_rng(29).uniform(0, 2 * np.pi, 2)
    expected = []
    for scale in (0.0, 0.5, 1.0):
        delta = float(phases[1] - phases[0])
        theta0 = phases[0] + 0.01 * (0.2 - 0.7 * scale * math.sin(delta - 0.4))
        theta1 = phases[1] + 0.01 * (-0.1 + 0.3 * scale * math.sin(-delta + 0.2))
        expected.append(abs(math.cos(float(theta1 - theta0) / 2)))
    with cProfile.Profile() as profile:
        result = bif.trace_sync_transition(
            omega,
            graph,
            lag,
            K_range=(0.0, 1.0),
            n_points=3,
            n_transient=0,
            n_measure=1,
            seed=29,
            backend="rust",
        )
    profile.create_stats()
    assert any(
        "spo_kernel.spo_kernel.trace_sync_transition_rust" in key[2]
        for key in profile.stats
    )
    assert not any(key[2] == "_python_steady_state_r" for key in profile.stats)
    np.testing.assert_allclose(result.R_values, expected, rtol=0, atol=2e-15)


def test_named_rust_zero_window_keeps_unset_sweep_and_nan_search() -> None:
    """The original composites preserve their distinct no-crossing representations."""
    omega = np.zeros(2)
    if "rust" not in AVAILABLE_BACKENDS:
        with pytest.raises(ImportError, match="requested basin backend 'rust'"):
            bif.trace_sync_transition(omega, n_points=3, n_measure=0, backend="rust")
        with pytest.raises(ImportError, match="requested basin backend 'rust'"):
            bif.find_critical_coupling(omega, n_measure=0, backend="rust")
        return
    result = bif.trace_sync_transition(omega, n_points=3, n_measure=0, backend="rust")
    np.testing.assert_array_equal(result.R_values, [0.0, 0.0, 0.0])
    assert result.K_critical is None
    assert math.isnan(bif.find_critical_coupling(omega, n_measure=0, backend="rust"))


def test_named_rust_first_upcrossing_matches_seeded_two_oscillator_algebra() -> None:
    """An actual finite sampled crossing agrees with independent interpolation."""
    omega = np.zeros(2)
    graph = np.array([[0.0, 1.0], [1.0, 0.0]])
    if "rust" not in AVAILABLE_BACKENDS:
        with pytest.raises(ImportError, match="requested basin backend 'rust'"):
            bif.trace_sync_transition(omega, graph, n_points=5, backend="rust")
        return
    phases = np.random.default_rng(233).uniform(0, 2 * np.pi, 2)
    delta = float(phases[1] - phases[0])
    values = [
        abs(math.cos((delta - 0.1 * scale * math.sin(delta)) / 2))
        for scale in (0.0, 5.0, 10.0, 15.0, 20.0)
    ]
    assert values[2] < 0.1 <= values[3]
    expected = 10.0 + 5.0 * (0.1 - values[2]) / (values[3] - values[2])
    result = bif.trace_sync_transition(
        omega,
        graph,
        K_range=(0.0, 20.0),
        n_points=5,
        dt=0.05,
        n_transient=0,
        n_measure=1,
        seed=233,
        backend="rust",
    )
    np.testing.assert_allclose(result.R_values, values, rtol=0, atol=2e-15)
    assert result.K_critical == pytest.approx(expected, abs=2e-13)


def test_named_rust_search_exercises_original_subthreshold_midpoints() -> None:
    """Actual bisection follows analytic lower and upper midpoint classifications."""
    omega = np.zeros(2)
    graph = np.array([[0.0, 1.0], [1.0, 0.0]])
    if "rust" not in AVAILABLE_BACKENDS:
        with pytest.raises(ImportError, match="requested basin backend 'rust'"):
            bif.find_critical_coupling(omega, graph, backend="rust")
        return
    phases = np.random.default_rng(233).uniform(0, 2 * np.pi, 2)
    delta = float(phases[1] - phases[0])
    low = abs(math.cos((delta - math.sin(delta)) / 2))
    high = abs(math.cos((delta - 1.5 * math.sin(delta)) / 2))
    assert low < 0.1 <= high
    with cProfile.Profile() as profile:
        actual = bif.find_critical_coupling(
            omega,
            graph,
            dt=0.05,
            n_transient=0,
            n_measure=1,
            tol=6.0,
            seed=233,
            backend="rust",
        )
    profile.create_stats()
    assert any(
        "spo_kernel.spo_kernel.find_critical_coupling_bif_rust" in key[2]
        for key in profile.stats
    )
    assert actual == 12.5


def test_python_sweep_observes_original_trial_calls() -> None:
    """Explicit Python ownership executes five original deterministic trials."""
    with cProfile.Profile() as profile:
        diagram = bif.trace_sync_transition(
            np.array([1.0, 1.2, 0.9, 1.1]),
            K_range=(0.0, 2.0),
            n_points=5,
            dt=0.01,
            n_transient=50,
            n_measure=30,
            seed=7,
            backend="python",
        )
    profile.create_stats()
    trials = [
        item
        for key, item in profile.stats.items()
        if key[2] == "_python_steady_state_r"
    ]
    assert len(trials) == 1 and trials[0][1] == 5
    assert len(diagram.points) == 5
    assert np.all((diagram.R_values >= 0) & (diagram.R_values <= 1))


def test_python_search_observes_original_trial_calls() -> None:
    """The original Python search runs its upper probe and bisection trials."""
    with cProfile.Profile() as profile:
        critical = bif.find_critical_coupling(
            np.array([1.0, 1.2, 0.9, 1.1]),
            dt=0.01,
            n_transient=50,
            n_measure=30,
            tol=0.1,
            seed=7,
            backend="python",
        )
    profile.create_stats()
    trials = [
        item
        for key, item in profile.stats.items()
        if key[2] == "_python_steady_state_r"
    ]
    assert len(trials) == 1 and trials[0][1] >= 2
    assert math.isfinite(critical)


def test_python_independent_sweep_reaches_a_locked_finite_window() -> None:
    """A specified homogeneous finite network has increasing sampled R here."""
    diagram = bif.trace_sync_transition(
        np.zeros(5),
        K_range=(0.0, 3.0),
        n_points=4,
        dt=0.01,
        n_transient=100,
        n_measure=50,
        seed=1,
        backend="python",
    )
    assert diagram.R_values[-1] > diagram.R_values[0]
    assert diagram.R_values[-1] > 0.9


@pytest.mark.parametrize(
    ("k_values", "r_values", "critical", "match"),
    [
        (["0.0", "1.0"], [0.1, 0.2], 0.5, "numeric"),
        ([0.0, 1.0], ["0.1", "0.2"], 0.5, "numeric"),
        ([1j, 1 - 2j], [0.1, 0.2], 0.5, "real-valued"),
        ([0.0, 1.0], np.array([0.1, 0.2 + 3j], dtype=object), 0.5, "real-valued"),
        ([0.0], [0.1], 0.5, "unexpected shape"),
        ([0.0, np.nan], [0.1, 0.2], 0.5, "non-finite"),
        ([0.0, 1.0], [0.1, 1.2], 0.5, "R outside"),
        ([1.0, 0.0], [0.1, 0.2], 0.5, "non-monotone"),
        ([0.0, 3.0], [0.1, 0.2], 0.5, "outside K_range"),
        ([0.0, 1.0], [0.1, 0.2], False, "K_critical"),
        ([0.0, 1.0], [0.1, 0.2], float("inf"), "K_critical"),
    ],
)
def test_actual_rust_composite_rejects_injected_invalid_outputs(
    monkeypatch: pytest.MonkeyPatch,
    k_values: object,
    r_values: object,
    critical: object,
    match: str,
) -> None:
    """Fault injection exercises the original native branch only when installed."""
    if not bif._HAS_COMPOSITE_RUST:
        with pytest.raises(ImportError, match="requested basin backend"):
            bif.trace_sync_transition(
                np.zeros(2), n_points=2, n_measure=1, backend="rust"
            )
        return

    def corrupt(*_args: object) -> tuple[object, object, object]:
        """Negative control: native results deliberately violate the contract."""
        return k_values, r_values, critical

    monkeypatch.setattr(bif, "_rust_trace", corrupt)
    with pytest.raises(ValueError, match=match):
        bif.trace_sync_transition(
            np.array([-1.0, 1.0]),
            K_range=(0.0, 1.0),
            n_points=2,
            n_transient=3,
            n_measure=2,
            backend="rust",
        )


@pytest.mark.parametrize("critical", [float("inf"), -0.1, object()])
def test_actual_rust_search_rejects_injected_invalid_outputs(
    monkeypatch: pytest.MonkeyPatch,
    critical: object,
) -> None:
    """Invalid native search scalars are refused without switching owners."""
    if not bif._HAS_COMPOSITE_RUST:
        with pytest.raises(ImportError, match="requested basin backend"):
            bif.find_critical_coupling(np.zeros(2), n_measure=1, backend="rust")
        return

    def corrupt(*_args: object) -> object:
        """Negative control: an invalid native scalar cannot become a result."""
        return critical

    monkeypatch.setattr(bif, "_rust_find_kc", corrupt)
    with pytest.raises(ValueError, match="invalid K_c"):
        bif.find_critical_coupling(
            np.array([-0.4, 0.0, 0.4]),
            dt=0.03,
            n_transient=9,
            n_measure=6,
            tol=0.2,
            seed=5,
            backend="rust",
        )


@pytest.mark.parametrize(
    ("grid", "values", "critical", "match"),
    [
        ([0.0, 0.25, 1.0], [0.05, 0.2, 0.8], 0.5, "different coupling grid"),
        ([0.0, 0.5, 1.0], [0.05, 0.2, 0.8], float("nan"), "sampled upcrossing"),
        ([0.0, 0.5, 1.0], [0.2, 0.3, 0.8], 0.5, "sampled upcrossing"),
        ([0.0, 0.5, 1.0], [0.05, 0.2, 0.8], 0.9, "sampled interpolation"),
    ],
)
def test_actual_composite_refuses_inconsistent_grid_and_critical_evidence(
    monkeypatch: pytest.MonkeyPatch,
    grid: list[float],
    values: list[float],
    critical: float,
    match: str,
) -> None:
    """Finite but inconsistent native evidence is a rejected negative control."""
    if not bif._HAS_COMPOSITE_RUST:
        with pytest.raises(ImportError):
            bif.trace_sync_transition(
                np.zeros(2), n_points=3, n_measure=1, backend="rust"
            )
        return

    def corrupt(*_args: object) -> tuple[object, object, object]:
        """Deliberately violate the requested grid or its threshold interpolation."""
        return grid, values, critical

    monkeypatch.setattr(bif, "_rust_trace", corrupt)
    with pytest.raises(ValueError, match=match):
        bif.trace_sync_transition(
            np.zeros(2),
            K_range=(0.0, 1.0),
            n_points=3,
            n_transient=0,
            n_measure=1,
            backend="rust",
        )
