# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — EVS monitor tests

from __future__ import annotations

import importlib
import math
import sys
import types
from typing import Any, cast, get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import evs as evs_mod
from scpn_phase_orchestrator.monitor.evs import EVSMonitor, EVSResult
from tests.typing_contracts import assert_precise_ndarray_hint


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


def test_public_array_contracts_are_parameterised():
    evaluate_hints = get_type_hints(EVSMonitor.evaluate)
    specificity_hints = get_type_hints(EVSMonitor._frequency_specificity)
    hints = (
        evaluate_hints["phases_trials"],
        evaluate_hints["pause_indices"],
        specificity_hints["phases_trials"],
    )
    for hint in hints:
        assert_precise_ndarray_hint(hint)
    assert "float64" in str(evaluate_hints["phases_trials"])
    assert "int64" in str(evaluate_hints["pause_indices"])
    assert "float64" in str(specificity_hints["phases_trials"])


def test_random_signal_fails():
    phases = _random_phases()
    mon = EVSMonitor()
    result = mon.evaluate(phases, list(range(100, 120)), 10.0, 23.0)
    assert not result.is_entrained
    assert result.itpc_value < 0.3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"itpc_threshold": -0.1}, "itpc_threshold"),
        ({"itpc_threshold": 1.1}, "itpc_threshold"),
        ({"persistence_threshold": math.nan}, "persistence_threshold"),
        ({"specificity_threshold": 0.0}, "specificity_threshold"),
        ({"specificity_threshold": True}, "specificity_threshold"),
    ],
)
def test_constructor_rejects_invalid_thresholds(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        EVSMonitor(**kwargs)


@pytest.mark.parametrize(
    ("phases", "match"),
    [
        (np.array([0.0, 1.0, 2.0]), "phases_trials"),
        (np.empty((0, 4)), "phases_trials"),
        (np.array([[0.0, np.nan], [1.0, 2.0]]), "phases_trials"),
        (np.array([[True, False], [False, True]]), "phases_trials"),
        (np.array([[0.0, True], [1.0, 2.0]], dtype=object), "phases_trials"),
        (np.array([[0.0, "not-a-phase"]], dtype=object), "phases_trials"),
    ],
)
def test_evaluate_rejects_invalid_phase_trials(phases, match):
    with pytest.raises(ValueError, match=match):
        EVSMonitor().evaluate(phases, [0], 10.0, 20.0)


@pytest.mark.parametrize(
    "pause_indices",
    [[0.5], [True], [0, True], np.array([[0, 1]]), ["not-an-index"], [math.inf]],
)
def test_evaluate_rejects_invalid_pause_indices(pause_indices):
    phases = _entrained_phases(n_trials=4, n_time=8)
    with pytest.raises((TypeError, ValueError), match="pause_indices"):
        EVSMonitor().evaluate(phases, pause_indices, 10.0, 20.0)


@pytest.mark.parametrize(
    ("target_freq", "control_freq"),
    [(0.0, 20.0), (10.0, 0.0), (math.inf, 20.0), (10.0, math.nan), (True, 20.0)],
)
def test_evaluate_rejects_invalid_frequencies(target_freq, control_freq):
    phases = _entrained_phases(n_trials=4, n_time=8)
    with pytest.raises((TypeError, ValueError), match="freq"):
        EVSMonitor().evaluate(phases, [0], target_freq, control_freq)


def test_persistence_criterion_required():
    """High ITPC overall but empty pause window should fail persistence."""
    phases = _entrained_phases()
    mon = EVSMonitor(persistence_threshold=0.5)
    result = mon.evaluate(phases, [], 10.0, 23.0)
    assert result.persistence_score == 0.0
    assert not result.is_entrained


def test_pause_indices_are_clipped_to_valid_range():
    phases = np.tile(np.array([0.0, np.pi / 2, np.pi]), (4, 1))
    mon = EVSMonitor(
        itpc_threshold=0.95,
        persistence_threshold=0.95,
        specificity_threshold=0.95,
    )

    result = mon.evaluate(
        phases,
        pause_indices=[-3, 0, 1, 999, 4],
        target_freq=5.0,
        control_freq=5.0,
    )

    assert result.itpc_value == pytest.approx(1.0, abs=1e-12)
    assert result.persistence_score == pytest.approx(1.0, abs=1e-12)
    assert result.is_entrained


def test_specificity_with_same_freq_gives_one():
    phases = _entrained_phases()
    mon = EVSMonitor(specificity_threshold=0.9)
    result = mon.evaluate(phases, list(range(80, 100)), 10.0, 10.0)
    assert abs(result.specificity_ratio - 1.0) < 1e-10


def test_specificity_zero_freq_handled():
    phases = _entrained_phases()
    mon = EVSMonitor()
    with pytest.raises(ValueError, match="target_freq"):
        mon.evaluate(phases, list(range(80, 100)), 0.0, 10.0)


def test_module_detects_available_rust_frequency_specificity(monkeypatch):
    fake_spo = types.ModuleType("spo_kernel")
    fake_spo.frequency_specificity_rust = lambda *_args: 1.75
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

    reloaded = importlib.reload(evs_mod)
    try:
        assert reloaded._HAS_RUST is True
        assert reloaded._rust_freq_spec is fake_spo.frequency_specificity_rust
    finally:
        monkeypatch.setitem(sys.modules, "spo_kernel", None)
        importlib.reload(evs_mod)


def test_specificity_zero_control_mean_preserves_target_lock(monkeypatch):
    phases = _entrained_phases()
    calls = iter([np.full(phases.shape[1], 0.8), np.zeros(phases.shape[1])])
    monkeypatch.setattr(evs_mod, "compute_itpc", lambda _phases: next(calls))

    specificity = EVSMonitor._frequency_specificity(phases, 10.0, 20.0)

    assert math.isinf(specificity)


def test_specificity_zero_control_and_zero_target_returns_zero(monkeypatch):
    phases = _random_phases()
    calls = iter([np.zeros(phases.shape[1]), np.zeros(phases.shape[1])])
    monkeypatch.setattr(evs_mod, "compute_itpc", lambda _phases: next(calls))

    specificity = EVSMonitor._frequency_specificity(phases, 10.0, 20.0)

    assert specificity == 0.0


def test_rust_frequency_specificity_receives_flat_phase_trials(monkeypatch):
    phases = _entrained_phases(n_trials=3, n_time=5)
    captured: dict[str, object] = {}

    def rust_freq_spec(
        flat: np.ndarray,
        n_trials: int,
        n_tp: int,
        target_freq: float,
        control_freq: float,
    ) -> float:
        captured["flat_contiguous"] = flat.flags.c_contiguous
        captured["shape"] = (n_trials, n_tp)
        captured["frequencies"] = (target_freq, control_freq)
        return 2.5

    monkeypatch.setattr(evs_mod, "_HAS_RUST", True)
    monkeypatch.setattr(evs_mod, "_rust_freq_spec", rust_freq_spec, raising=False)

    specificity = EVSMonitor._frequency_specificity(phases[:, ::-1], 12.0, 18.0)

    assert specificity == 2.5
    assert captured == {
        "flat_contiguous": True,
        "shape": (3, 5),
        "frequencies": (12.0, 18.0),
    }


def test_evs_result_is_frozen():
    r = EVSResult(0.8, 0.6, 2.0, True)
    mutable = cast(Any, r)
    with pytest.raises(AttributeError):
        mutable.itpc_value = 0.1


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((-0.1, 0.6, 2.0, True), "itpc_value"),
        ((0.8, 1.1, 2.0, True), "persistence_score"),
        ((0.8, 0.6, -0.1, True), "specificity_ratio"),
        ((0.8, 0.6, math.nan, True), "specificity_ratio"),
        ((0.8, 0.6, 2.0, 1), "is_entrained"),
    ],
)
def test_evs_result_rejects_invalid_physics_scores(args, match):
    with pytest.raises((TypeError, ValueError), match=match):
        EVSResult(*args)


def test_evs_result_accepts_infinite_specificity_for_zero_control_itpc():
    result = EVSResult(0.8, 0.6, math.inf, True)

    assert math.isinf(result.specificity_ratio)


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
        assert isinstance(result, evs_mod.EVSResult)
        assert isinstance(result.is_entrained, bool)


# Salvaged module-specific behavioural contracts from deleted broad tests.
class TestEVSBehavioural:
    """Verify EVSMonitor produces structured results with valid fields."""

    def test_single_trial_returns_result(self):
        from scpn_phase_orchestrator.monitor.evs import EVSMonitor

        m = EVSMonitor()
        phases = np.random.default_rng(0).uniform(0, 2 * np.pi, (1, 100))
        result = m.evaluate(
            phases,
            pause_indices=[50],
            target_freq=10.0,
            control_freq=20.0,
        )
        assert hasattr(result, "is_entrained")
        assert isinstance(result.is_entrained, bool)

    def test_multi_trial_aggregation(self):
        from scpn_phase_orchestrator.monitor.evs import EVSMonitor

        m = EVSMonitor()
        phases = np.random.default_rng(1).uniform(0, 2 * np.pi, (5, 200))
        result = m.evaluate(
            phases,
            pause_indices=[100],
            target_freq=10.0,
            control_freq=20.0,
        )
        assert hasattr(result, "is_entrained")
