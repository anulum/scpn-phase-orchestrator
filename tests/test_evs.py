# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — EVS monitor tests

from __future__ import annotations

import cmath
import importlib
import importlib.util
import math
from collections.abc import Callable
from numbers import Real
from typing import Any, cast, get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import evs as evs_mod
from scpn_phase_orchestrator.monitor.evs import EVSMonitor, EVSResult
from tests.typing_contracts import assert_precise_ndarray_hint

# Calls that deliberately pass values outside the declared types.
_monitor_any: Callable[..., EVSMonitor] = EVSMonitor
_result_any: Callable[..., EVSResult] = EVSResult


def _evaluate_any(monitor: EVSMonitor) -> Callable[..., EVSResult]:
    """Return ``monitor.evaluate`` for calls with deliberately invalid inputs."""
    return monitor.evaluate


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


def test_entrained_signal_passes() -> None:
    """Phase-locked trials pass all three criteria."""
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


def test_public_array_contracts_are_parameterised() -> None:
    """Public array parameters carry precise float64 hints."""
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


def test_random_signal_fails() -> None:
    """Random phases fail entrainment."""
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
def test_constructor_rejects_invalid_thresholds(
    kwargs: dict[str, object], match: str
) -> None:
    """Out-of-range or invalid thresholds are refused."""
    with pytest.raises((TypeError, ValueError), match=match):
        _monitor_any(**kwargs)


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
def test_evaluate_rejects_invalid_phase_trials(phases: np.ndarray, match: str) -> None:
    """Malformed phase trials are refused."""
    with pytest.raises(ValueError, match=match):
        EVSMonitor().evaluate(phases, [0], 10.0, 20.0)


def test_evaluate_rejects_object_complex_phase_trials_as_non_real() -> None:
    """Object-dtype complex phases are refused as non-real."""
    phases = np.asarray(
        [[0.0, complex(0.1, 0.0)], [0.2, 0.3]],
        dtype=object,
    )

    with pytest.raises(ValueError, match="phases_trials must contain real-valued"):
        EVSMonitor().evaluate(phases, [0], 10.0, 20.0)


def test_evaluate_rejects_numeric_string_phase_aliases() -> None:
    """Numeric-string phases are refused."""
    phases = np.asarray([["0.0", "1.0"], ["0.2", "1.2"]])

    with pytest.raises(ValueError, match="phases_trials"):
        EVSMonitor().evaluate(phases, [0], 10.0, 20.0)


def test_evaluate_wraps_a_broken_phase_array_protocol() -> None:
    """A broken array protocol becomes a ValueError."""

    class BrokenArray:
        def __array__(self) -> np.ndarray:
            raise TypeError("broken")

    with pytest.raises(ValueError, match="phases_trials"):
        _evaluate_any(EVSMonitor())(BrokenArray(), [0], 10.0, 20.0)


def test_evaluate_accepts_real_numeric_object_phase_samples() -> None:
    """Real numbers in an object array are accepted."""
    phases = np.asarray([[0, 1.0], [0.2, np.float32(1.2)]], dtype=object)

    result = EVSMonitor(specificity_threshold=1.0).evaluate(
        phases,
        [0],
        10.0,
        20.0,
    )

    assert isinstance(result, EVSResult)


def test_evaluate_wraps_a_broken_real_scalar_conversion() -> None:
    """A real scalar that fails conversion becomes a ValueError."""

    class BrokenReal:
        def __float__(self) -> float:
            raise TypeError("broken")

    Real.register(BrokenReal)
    phases = np.asarray([[BrokenReal()]], dtype=object)

    with pytest.raises(ValueError, match="phases_trials"):
        EVSMonitor().evaluate(phases, [0], 10.0, 20.0)


@pytest.mark.parametrize(
    "pause_indices",
    [[0.5], [True], [0, True], np.array([[0, 1]]), ["not-an-index"], [math.inf]],
)
def test_evaluate_rejects_invalid_pause_indices(pause_indices: object) -> None:
    """Invalid pause indices are refused."""
    phases = _entrained_phases(n_trials=4, n_time=8)
    with pytest.raises((TypeError, ValueError), match="pause_indices"):
        _evaluate_any(EVSMonitor())(phases, pause_indices, 10.0, 20.0)


def test_pause_indices_are_bounded_and_deduplicated_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pause indices reach persistence bounded and deduplicated."""
    phases = _entrained_phases(n_trials=4, n_time=3)
    captured: dict[str, np.ndarray] = {}

    monkeypatch.setattr(evs_mod, "compute_itpc", lambda _phases: np.ones(3))

    def persistence(_phases: np.ndarray, pause_indices: np.ndarray) -> float:
        captured["pause_indices"] = pause_indices
        return 1.0

    monkeypatch.setattr(evs_mod, "itpc_persistence", persistence)
    monkeypatch.setattr(
        EVSMonitor,
        "_frequency_specificity",
        staticmethod(lambda *_args, **_kwargs: 1.0),
    )

    EVSMonitor(specificity_threshold=1.0).evaluate(
        phases,
        [-2, 0, 0, 1, 99],
        10.0,
        20.0,
    )

    np.testing.assert_array_equal(captured["pause_indices"], np.array([0, 1]))


def test_pause_indices_reject_int64_overflow_and_broken_array_protocol() -> None:
    """Overflowing or unconvertible pause indices are refused."""
    phases = _entrained_phases(n_trials=4, n_time=3)

    with pytest.raises(ValueError, match="pause_indices"):
        EVSMonitor().evaluate(phases, [2**63], 10.0, 20.0)

    class BrokenArray:
        def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
            raise TypeError("broken")

    with pytest.raises(ValueError, match="pause_indices"):
        _evaluate_any(EVSMonitor())(phases, BrokenArray(), 10.0, 20.0)


@pytest.mark.parametrize(
    ("target_freq", "control_freq"),
    [(0.0, 20.0), (10.0, 0.0), (math.inf, 20.0), (10.0, math.nan), (True, 20.0)],
)
def test_evaluate_rejects_invalid_frequencies(
    target_freq: float, control_freq: float
) -> None:
    """Non-positive or non-finite frequencies are refused."""
    phases = _entrained_phases(n_trials=4, n_time=8)
    with pytest.raises((TypeError, ValueError), match="freq"):
        _evaluate_any(EVSMonitor())(phases, [0], target_freq, control_freq)


@pytest.mark.parametrize(
    ("target_freq", "control_freq", "match"),
    [
        (np.asarray(complex(10.0, 0.0), dtype=object), 20.0, "target_freq.*real"),
        (10.0, np.asarray(complex(20.0, 0.0), dtype=object), "control_freq.*real"),
    ],
)
def test_evaluate_rejects_object_complex_frequency_controls(
    target_freq: object,
    control_freq: object,
    match: str,
) -> None:
    """Object-dtype complex frequencies are refused."""
    phases = _entrained_phases(n_trials=4, n_time=8)

    with pytest.raises((TypeError, ValueError), match=match):
        _evaluate_any(EVSMonitor())(phases, [0], target_freq, control_freq)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"itpc_threshold": np.asarray(complex(0.6, 0.0), dtype=object)},
            "itpc_threshold.*real",
        ),
        (
            {"persistence_threshold": np.asarray(complex(0.4, 0.0), dtype=object)},
            "persistence_threshold.*real",
        ),
        (
            {"specificity_threshold": np.asarray(complex(1.5, 0.0), dtype=object)},
            "specificity_threshold.*real",
        ),
    ],
)
def test_constructor_rejects_object_complex_threshold_controls(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Object-dtype complex thresholds are refused."""
    with pytest.raises((TypeError, ValueError), match=match):
        _monitor_any(**kwargs)


def test_persistence_criterion_required() -> None:
    """High ITPC overall but empty pause window should fail persistence."""
    phases = _entrained_phases()
    mon = EVSMonitor(persistence_threshold=0.5)
    result = mon.evaluate(phases, [], 10.0, 23.0)
    assert result.persistence_score == 0.0
    assert not result.is_entrained


def test_pause_indices_are_clipped_to_valid_range() -> None:
    """Pause indices beyond the trace are dropped."""
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


def test_specificity_with_same_freq_gives_one() -> None:
    """Equal target and control frequencies give a ratio of one."""
    phases = _entrained_phases()
    mon = EVSMonitor(specificity_threshold=0.9)
    result = mon.evaluate(phases, list(range(80, 100)), 10.0, 10.0)
    assert abs(result.specificity_ratio - 1.0) < 1e-10


def test_specificity_zero_freq_handled() -> None:
    """A zero target frequency is refused."""
    phases = _entrained_phases()
    mon = EVSMonitor()
    with pytest.raises(ValueError, match="target_freq"):
        mon.evaluate(phases, list(range(80, 100)), 0.0, 10.0)


def test_named_rust_owner_is_unavailable_without_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the compiled kernel a named Rust owner refuses, at once."""
    monkeypatch.setattr(evs_mod, "_HAS_RUST", False)
    with pytest.raises(RuntimeError, match="EVS backend 'rust' unavailable"):
        evs_mod.frequency_specificity_backend("rust")
    with pytest.raises(RuntimeError, match="EVS backend 'rust' unavailable"):
        EVSMonitor(backend="rust")
    assert evs_mod.frequency_specificity_backend() == "python"


def test_module_detects_available_rust_frequency_specificity() -> None:
    """The module finds the compiled kernel exactly when ``spo_kernel`` installs."""
    installed = importlib.util.find_spec("spo_kernel") is not None
    assert evs_mod._HAS_RUST is installed
    assert evs_mod.frequency_specificity_backend() == (
        "rust" if installed else "python"
    )


def test_specificity_zero_control_mean_preserves_target_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NumPy law gives an infinite ratio for a zero control ITPC."""
    phases = _entrained_phases()
    calls = iter([np.full(phases.shape[1], 0.8), np.zeros(phases.shape[1])])
    monkeypatch.setattr(evs_mod, "compute_itpc", lambda _phases: next(calls))

    specificity = EVSMonitor._frequency_specificity(
        phases, 10.0, 20.0, backend="python"
    )

    assert math.isinf(specificity)


def test_specificity_zero_control_and_zero_target_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NumPy law gives zero when both ITPC means are zero."""
    phases = _random_phases()
    calls = iter([np.zeros(phases.shape[1]), np.zeros(phases.shape[1])])
    monkeypatch.setattr(evs_mod, "compute_itpc", lambda _phases: next(calls))

    specificity = EVSMonitor._frequency_specificity(
        phases, 10.0, 20.0, backend="python"
    )

    assert specificity == 0.0


def test_rust_frequency_specificity_receives_flat_phase_trials() -> None:
    """The compiled kernel gets non-contiguous trials flattened in row order."""
    if not evs_mod._HAS_RUST:
        pytest.skip("Rust backend not built")
    phases = _entrained_phases(n_trials=3, n_time=5)[:, ::-1]
    assert not phases.flags.c_contiguous

    specificity = EVSMonitor._frequency_specificity(phases, 12.0, 18.0, backend="rust")

    assert specificity == pytest.approx(
        _oracle_specificity(phases.tolist(), 12.0, 18.0), rel=1e-12
    )


def _oracle_specificity(
    trials: list[list[float]], target: float, control: float
) -> float:
    """Return mean target ITPC over mean control ITPC by plain loops."""

    def mean_itpc(rows: list[list[float]]) -> float:
        """Average ``|mean_k exp(iθ_kt)|`` over time points."""
        n = len(rows)
        values = [
            abs(sum(cmath.exp(1j * row[t]) for row in rows) / n)
            for t in range(len(rows[0]))
        ]
        return math.fsum(values) / len(values)

    scaled = [[v * control / target for v in row] for row in trials]
    target_mean, control_mean = mean_itpc(trials), mean_itpc(scaled)
    if control_mean < 1e-12:
        return math.inf if target_mean > 0 else 0.0
    return target_mean / control_mean


@pytest.mark.parametrize(
    "backend_value",
    [math.nan, -0.1, True, np.asarray(complex(1.2, 0.0), dtype=object)],
)
def test_rust_frequency_specificity_rejects_invalid_backend_output(
    monkeypatch: pytest.MonkeyPatch,
    backend_value: object,
) -> None:
    """Invalid kernel results are refused."""
    phases = _entrained_phases(n_trials=3, n_time=5)

    monkeypatch.setattr(evs_mod, "_HAS_RUST", True)
    monkeypatch.setattr(
        evs_mod,
        "_rust_freq_spec",
        lambda *_args: backend_value,
        raising=False,
    )

    with pytest.raises((TypeError, ValueError), match="specificity_ratio"):
        EVSMonitor._frequency_specificity(phases, 12.0, 18.0)


def test_rust_frequency_specificity_rejects_plausible_wrong_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plausible but wrong kernel result is refused against the reference.

    A defective kernel cannot be produced from a real build, so this negative
    contract installs one. Without ``spo_kernel`` the module has no
    ``_rust_freq_spec`` attribute, hence ``raising=False``.
    """
    phases = _entrained_phases(n_trials=3, n_time=5)
    monkeypatch.setattr(evs_mod, "_HAS_RUST", True)
    monkeypatch.setattr(evs_mod, "_rust_freq_spec", lambda *_args: 1.25, raising=False)

    with pytest.raises(ValueError, match="diverged from exact reference"):
        EVSMonitor._frequency_specificity(phases, 12.0, 18.0)


def test_evs_result_rejects_object_complex_specificity_ratio() -> None:
    """A complex specificity ratio is refused."""
    with pytest.raises((TypeError, ValueError), match="specificity_ratio.*real"):
        _result_any(0.8, 0.6, np.asarray(complex(1.2, 0.0), dtype=object), True)


def test_evs_result_is_frozen() -> None:
    """EVSResult fields cannot be reassigned."""
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
def test_evs_result_rejects_invalid_physics_scores(
    args: tuple[object, ...], match: str
) -> None:
    """Out-of-range scores are refused."""
    with pytest.raises((TypeError, ValueError), match=match):
        _result_any(*args)


def test_evs_result_accepts_infinite_specificity_for_zero_control_itpc() -> None:
    """An infinite specificity ratio is accepted."""
    result = EVSResult(0.8, 0.6, math.inf, True)

    assert math.isinf(result.specificity_ratio)


def test_high_specificity_threshold_rejects_broadband() -> None:
    """With a very high specificity threshold, even entrained
    signals fail if target/control ratio is insufficient."""
    phases = _entrained_phases()
    mon = EVSMonitor(specificity_threshold=100.0)
    result = mon.evaluate(phases, list(range(80, 100)), 10.0, 10.5)
    assert not result.is_entrained


def test_inf_specificity_when_control_is_zero_itpc() -> None:
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

    def test_engine_trajectory_to_evs(self) -> None:
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

    def test_single_trial_returns_result(self) -> None:
        """A single trial still yields a result."""
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

    def test_multi_trial_aggregation(self) -> None:
        """Several trials aggregate into one result."""
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
