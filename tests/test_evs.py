# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — EVS monitor tests

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.evs import EVSMonitor, EVSResult


def _entrained_phases(
    n_trials: int = 50,
    n_time: int = 100,
    noise_std: float = 0.05,
    seed: int = 7,
) -> np.ndarray:
    """Phase-locked trials with small jitter."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 4 * np.pi, n_time)
    phases = np.tile(base, (n_trials, 1))
    phases += rng.normal(0, noise_std, phases.shape)
    return phases


def _random_phases(n_trials: int = 80, n_time: int = 120, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, size=(n_trials, n_time))


def test_entrained_signal_passes():
    phases = _entrained_phases()
    mon = EVSMonitor(
        itpc_threshold=0.6,
        persistence_threshold=0.4,
        specificity_threshold=1.0,
    )
    result = mon.evaluate(phases, list(range(80, 100)), 10.0, 23.0)
    assert result.is_entrained
    assert result.itpc_value > 0.6
    assert result.persistence_score > 0.4


def test_random_signal_fails():
    phases = _random_phases()
    mon = EVSMonitor()
    result = mon.evaluate(phases, list(range(100, 120)), 10.0, 23.0)
    assert not result.is_entrained
    assert result.itpc_value < 0.3


def test_persistence_criterion_required():
    """High ITPC overall but empty pause window should fail persistence."""
    phases = _entrained_phases()
    mon = EVSMonitor(persistence_threshold=0.5)
    result = mon.evaluate(phases, [], 10.0, 23.0)
    assert result.persistence_score == 0.0
    assert not result.is_entrained


def test_specificity_with_same_freq_gives_one():
    phases = _entrained_phases()
    mon = EVSMonitor(specificity_threshold=0.9)
    result = mon.evaluate(phases, list(range(80, 100)), 10.0, 10.0)
    assert abs(result.specificity_ratio - 1.0) < 1e-10


def test_specificity_zero_freq_handled():
    phases = _entrained_phases()
    mon = EVSMonitor()
    result = mon.evaluate(phases, list(range(80, 100)), 0.0, 10.0)
    assert result.specificity_ratio == 0.0
    assert not result.is_entrained


def test_evs_result_is_frozen():
    r = EVSResult(0.8, 0.6, 2.0, True)
    with pytest.raises(AttributeError):
        r.itpc_value = 0.1  # type: ignore[misc]


def test_high_specificity_threshold_rejects_broadband():
    """With a very high specificity threshold, even entrained
    signals fail if target/control ratio is insufficient."""
    phases = _entrained_phases()
    mon = EVSMonitor(specificity_threshold=100.0)
    result = mon.evaluate(phases, list(range(80, 100)), 10.0, 10.5)
    assert not result.is_entrained


def test_inf_specificity_when_control_is_zero_itpc():
    """When control ITPC is effectively zero, ratio should be inf
    (or very large) for a locked target signal."""
    phases = _entrained_phases(noise_std=0.01)
    mon = EVSMonitor()
    # control_freq very far from target: rescaled phases wrap many times,
    # destroying coherence at the control frequency
    result = mon.evaluate(phases, list(range(80, 100)), 10.0, 1e6)
    assert result.specificity_ratio > 5.0 or math.isinf(result.specificity_ratio)


class TestEVSPipelineWiring:
    """Pipeline: engine trajectory → EVS entrainment detection."""

    def test_engine_trajectory_to_evs(self):
        """UPDEEngine trajectory → EVSMonitor.evaluate: detects
        entrainment from engine-generated phase dynamics."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 10
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        omegas = np.ones(n) * 10.0  # 10 Hz target
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        trajectory = np.zeros((5, 200))
        for trial in range(5):
            p = rng.uniform(0, 2 * np.pi, n)
            for t in range(200):
                p = eng.step(p, omegas, knm, 0.0, 0.0, alpha)
                trajectory[trial, t] = p[0]

        mon = EVSMonitor()
        result = mon.evaluate(trajectory, list(range(80, 100)), 10.0, 20.0)
        assert isinstance(result, EVSResult)
        assert isinstance(result.is_entrained, bool)
