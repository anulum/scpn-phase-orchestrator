# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Explosive-sync early-warning monitor tests

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.explosive_sync import (
    ExplosiveSyncWarning,
    _first_sustained_breach,
    explosive_sync_warning,
)


def _transition_signals(
    n_nodes: int = 8, length: int = 4000, switch: int | None = None, seed: int = 0
) -> tuple[np.ndarray, int]:
    """Noise (high entropy) → locked oscillation (low entropy) per node."""
    rng = np.random.default_rng(seed)
    switch = length // 2 if switch is None else switch
    t = np.arange(length)
    signals = np.empty((n_nodes, length))
    for node in range(n_nodes):
        noise = rng.standard_normal(length)
        locked = np.sin(2.0 * np.pi * 0.05 * t + node * 0.1)
        locked += 0.02 * rng.standard_normal(length)
        series = noise.copy()
        series[switch:] = locked[switch:]
        signals[node] = series
    return signals, switch


class TestWarningDetection:
    def test_fires_on_regularisation_transition(self) -> None:
        signals, switch = _transition_signals()
        result = explosive_sync_warning(
            signals,
            window=200,
            step=25,
            baseline_fraction=0.2,
            z_threshold=3.0,
            drop_threshold=0.15,
            persistence=2,
        )
        assert result.warning_triggered
        assert result.warning_sample is not None
        # The alarm must land in the neighbourhood of the regularisation,
        # not in the disordered baseline.
        assert result.warning_sample >= switch - result.window
        assert result.robust_z.min() < -result.z_threshold

    def test_silent_on_stationary_noise(self) -> None:
        rng = np.random.default_rng(1)
        signals = rng.standard_normal((6, 4000))
        result = explosive_sync_warning(
            signals, window=200, step=25, z_threshold=3.0, drop_threshold=0.15
        )
        assert not result.warning_triggered
        assert result.warning_window is None
        assert result.warning_sample is None

    def test_one_dimensional_treated_as_single_node(self) -> None:
        signals, _ = _transition_signals(n_nodes=1)
        result = explosive_sync_warning(signals[0], window=200, step=25)
        assert result.per_node_entropy.shape[1] == 1
        assert result.warning_triggered

    def test_persistence_requires_sustained_breach(self) -> None:
        signals, _ = _transition_signals()
        lenient = explosive_sync_warning(
            signals, window=200, step=25, drop_threshold=0.15, persistence=1
        )
        strict = explosive_sync_warning(
            signals, window=200, step=25, drop_threshold=0.15, persistence=8
        )
        assert lenient.warning_triggered
        if strict.warning_window is not None and lenient.warning_window is not None:
            assert strict.warning_window >= lenient.warning_window

    def test_high_drop_threshold_suppresses_warning(self) -> None:
        signals, _ = _transition_signals()
        result = explosive_sync_warning(
            signals, window=200, step=25, drop_threshold=0.99
        )
        assert not result.warning_triggered


class TestResultStructure:
    def test_shapes_and_window_starts(self) -> None:
        signals, _ = _transition_signals(n_nodes=5, length=2000)
        result = explosive_sync_warning(signals, window=200, step=50)
        n_windows = result.entropy_index.shape[0]
        assert result.per_node_entropy.shape == (n_windows, 5)
        assert result.window_starts.shape == (n_windows,)
        assert result.robust_z.shape == (n_windows,)
        assert result.relative_drop.shape == (n_windows,)
        np.testing.assert_array_equal(result.window_starts, np.arange(n_windows) * 50)

    def test_entropy_index_is_node_mean(self) -> None:
        signals, _ = _transition_signals(n_nodes=4, length=1500)
        result = explosive_sync_warning(signals, window=200, step=50)
        np.testing.assert_allclose(
            result.entropy_index, result.per_node_entropy.mean(axis=1), atol=1e-12
        )

    def test_entropy_index_bounded(self) -> None:
        signals, _ = _transition_signals()
        result = explosive_sync_warning(signals, window=200, step=25)
        assert result.entropy_index.min() >= 0.0
        assert result.entropy_index.max() <= 1.0

    def test_summary_keys(self) -> None:
        signals, _ = _transition_signals(length=2000)
        result = explosive_sync_warning(signals, window=200, step=50)
        summary = result.summary()
        for key in (
            "n_windows",
            "n_baseline_windows",
            "baseline_median",
            "baseline_scale",
            "min_entropy_index",
            "min_robust_z",
            "max_relative_drop",
            "warning_triggered",
            "warning_window",
            "warning_sample",
        ):
            assert key in summary

    def test_result_is_frozen(self) -> None:
        signals, _ = _transition_signals(length=2000)
        result = explosive_sync_warning(signals, window=200, step=50)
        assert isinstance(result, ExplosiveSyncWarning)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.baseline_median = 0.0  # type: ignore[misc]

    def test_baseline_window_count(self) -> None:
        signals, _ = _transition_signals(n_nodes=3, length=3000)
        result = explosive_sync_warning(
            signals, window=200, step=50, baseline_fraction=0.3, min_baseline_windows=2
        )
        n_windows = result.entropy_index.shape[0]
        assert result.n_baseline_windows == max(2, int(np.ceil(0.3 * n_windows)))


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"window": 4}, "window"),
            ({"window": "x"}, "window"),
            ({"window": 100000}, "exceeds"),
            ({"dimension": 1}, "dimension"),
            ({"delay": 0}, "delay"),
            ({"step": 0}, "step"),
            ({"step": 2.5}, "step"),
            ({"baseline_fraction": 0.0}, "baseline_fraction"),
            ({"baseline_fraction": 1.0}, "baseline_fraction"),
            ({"baseline_fraction": "x"}, "baseline_fraction"),
            ({"z_threshold": -1.0}, "z_threshold"),
            ({"z_threshold": "x"}, "z_threshold"),
            ({"drop_threshold": -0.1}, "drop_threshold"),
            ({"drop_threshold": "x"}, "drop_threshold"),
            ({"persistence": 0}, "persistence"),
            ({"min_baseline_windows": 0}, "min_baseline_windows"),
        ],
    )
    def test_rejects_invalid_parameters(
        self, kwargs: dict[str, Any], match: str
    ) -> None:
        signals = np.random.default_rng(0).standard_normal((4, 1000))
        base = {"window": 200, "step": 50}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            explosive_sync_warning(signals, **base)

    @pytest.mark.parametrize(
        ("signals", "match"),
        [
            (np.zeros((2, 2, 2)), "two-dimensional"),
            (np.array([[0.0, np.nan, 1.0, 2.0]]), "finite"),
            (np.array([[True, False, True, False]]), "boolean"),
            (np.array([[0.0 + 1.0j, 1.0, 2.0, 3.0]]), "real-valued"),
            (np.array(["a", "b", "c", "d"], dtype=object), "real float array"),
        ],
    )
    def test_rejects_invalid_signals(self, signals: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            explosive_sync_warning(signals, window=4, step=1)


class TestFirstSustainedBreach:
    @pytest.mark.parametrize(
        ("breaches", "persistence", "expected"),
        [
            ([False, False, False], 1, None),
            ([False, True, False], 1, 1),
            ([False, True, True, False], 2, 1),
            ([True, False, True, True], 2, 2),
            ([True, True, True], 3, 0),
            ([True, False, True], 2, None),
        ],
    )
    def test_run_detection(
        self, breaches: list[bool], persistence: int, expected: int | None
    ) -> None:
        result = _first_sustained_breach(
            np.array(breaches, dtype=np.bool_), persistence
        )
        assert result == expected


class TestZeroBaselineMedian:
    def test_constant_signal_yields_zero_relative_drop(self) -> None:
        # A constant (monotone) signal has zero transition entropy in every
        # window, so the baseline median is zero and the relative-drop
        # denominator collapses to the guarded zero branch.
        signals = np.tile(np.arange(2000, dtype=np.float64), (3, 1))
        result = explosive_sync_warning(signals, window=200, step=50)
        assert result.baseline_median == 0.0
        np.testing.assert_array_equal(
            result.relative_drop, np.zeros_like(result.relative_drop)
        )
        assert not result.warning_triggered


class TestConstantBaselineGuard:
    def test_zero_scale_does_not_divide_by_zero(self) -> None:
        # A perfectly regular baseline has zero MAD; the guarded scale floor
        # must keep robust_z finite.
        t = np.linspace(0.0, 400.0 * np.pi, 4000)
        regular = np.tile(np.sin(t), (3, 1))
        result = explosive_sync_warning(regular, window=200, step=50)
        assert np.all(np.isfinite(result.robust_z))
        assert result.baseline_scale >= 0.0
