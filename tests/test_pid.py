# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for time-series Partial Information Decomposition

from __future__ import annotations

import functools

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import pid as pid_module
from scpn_phase_orchestrator.monitor.pid import redundancy, synergy

TWO_PI = 2.0 * np.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = pid_module.ACTIVE_BACKEND
        pid_module.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            pid_module.ACTIVE_BACKEND = prev

    return wrapper


def _coherent_groups(t: int, seed: int = 0) -> np.ndarray:
    """8-oscillator history where group A sweeps one phase and group B
    another, so the global target depends on both sources."""
    idx = np.arange(t, dtype=np.float64)
    a_phase = TWO_PI * idx / t
    b_phase = TWO_PI * 2.0 * idx / t
    history = np.zeros((t, 8), dtype=np.float64)
    history[:, 0:4] = a_phase[:, None]
    history[:, 4:8] = b_phase[:, None]
    return history


class TestPidSemantics:
    @_python
    def test_components_non_negative(self):
        rng = np.random.default_rng(7)
        history = rng.uniform(0, TWO_PI, (800, 8))
        assert redundancy(history, [0, 1, 2, 3], [4, 5, 6, 7], 16) >= 0.0
        assert synergy(history, [0, 1, 2, 3], [4, 5, 6, 7], 16) >= 0.0

    @_python
    def test_co_varying_sources_have_positive_synergy(self):
        history = _coherent_groups(2000)
        s = synergy(history, [0, 1, 2, 3], [4, 5, 6, 7], 8)
        assert s > 1e-2

    @_python
    def test_fully_redundant_has_zero_synergy(self):
        """A single sweeping phase shared by all oscillators → A = B = Y, so
        the shared information is redundant and synergy vanishes."""
        idx = np.arange(2000, dtype=np.float64)
        history = np.tile((TWO_PI * idx / 2000)[:, None], (1, 8))
        red = redundancy(history, [0, 1, 2, 3], [4, 5, 6, 7], 8)
        syn = synergy(history, [0, 1, 2, 3], [4, 5, 6, 7], 8)
        assert red > 0.0
        assert syn == pytest.approx(0.0, abs=1e-9)

    @_python
    def test_single_snapshot_is_zero(self):
        history = np.arange(8, dtype=np.float64).reshape(1, 8)
        assert redundancy(history, [0, 1, 2, 3], [4, 5, 6, 7]) == 0.0
        assert synergy(history, [0, 1, 2, 3], [4, 5, 6, 7]) == 0.0

    @_python
    def test_one_dimensional_history_treated_as_single_snapshot(self):
        assert redundancy(np.zeros(8), [0, 1], [2, 3]) == 0.0
        assert synergy(np.zeros(8), [0, 1], [2, 3]) == 0.0

    @_python
    def test_empty_history(self):
        assert redundancy(np.zeros((0, 0)), [0], [1]) == 0.0
        assert synergy(np.zeros((0, 0)), [0], [1]) == 0.0

    @_python
    def test_empty_groups(self):
        history = _coherent_groups(64)
        assert redundancy(history, [], [4, 5]) == 0.0
        assert synergy(history, [0, 1], []) == 0.0

    @_python
    def test_deterministic(self):
        history = _coherent_groups(512, seed=3)
        ga, gb = [0, 1, 2, 3], [4, 5, 6, 7]
        assert redundancy(history, ga, gb, 8) == redundancy(history, ga, gb, 8)
        assert synergy(history, ga, gb, 8) == synergy(history, ga, gb, 8)

    @_python
    def test_accepts_numpy_integer_bin_count(self):
        history = _coherent_groups(256)
        assert redundancy(history, [0, 1], [2, 3], n_bins=np.int64(8)) >= 0.0


class TestPidValidation:
    @_python
    @pytest.mark.parametrize(
        "history",
        [
            np.zeros((2, 2, 2)),
            np.array([[0.0, np.nan], [1.0, 2.0]]),
            np.array([[True, False], [False, True]]),
            np.array([[0.0, 1.0 + 0.0j], [1.0, 2.0]]),
        ],
    )
    def test_rejects_invalid_history(self, history):
        with pytest.raises(ValueError, match="phases"):
            redundancy(history, [0], [1])

    @_python
    def test_object_complex_history_rejected_as_real_valued(self):
        history = np.array([[0.0, complex(1.0, 0.0)]], dtype=object)
        with pytest.raises(ValueError, match="real-valued"):
            synergy(history, [0], [1])

    @_python
    @pytest.mark.parametrize(
        ("group", "error"),
        [
            ([0.5], TypeError),
            ([True], TypeError),
            ([-1], IndexError),
            ([8], IndexError),
            (np.array([[0]]), ValueError),
        ],
    )
    def test_rejects_invalid_group_indices(self, group, error):
        history = _coherent_groups(32)
        with pytest.raises(error, match="group_a"):
            redundancy(history, group, [4, 5])

    @_python
    @pytest.mark.parametrize("n_bins", [0, 1, False, np.bool_(True), 4.5])
    def test_rejects_invalid_bin_count(self, n_bins):
        history = _coherent_groups(32)
        with pytest.raises((TypeError, ValueError), match="n_bins"):
            redundancy(history, [0, 1], [4, 5], n_bins=n_bins)


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert pid_module.AVAILABLE_BACKENDS
        assert "python" in pid_module.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert pid_module.AVAILABLE_BACKENDS[0] == pid_module.ACTIVE_BACKEND

    def test_dispatch_returns_none_for_python(self, monkeypatch):
        monkeypatch.setattr(pid_module, "ACTIVE_BACKEND", "python")
        assert pid_module._dispatch() is None


class TestPidPipelineWiring:
    """Pipeline: engine phase history → PID redundancy/synergy between groups."""

    @_python
    def test_engine_history_to_pid(self):
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(0.5, 1.5, n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        history = np.empty((300, n), dtype=np.float64)
        for step in range(300):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            history[step] = phases

        red = redundancy(history, [0, 1, 2, 3], [4, 5, 6, 7], 8)
        syn = synergy(history, [0, 1, 2, 3], [4, 5, 6, 7], 8)
        assert red >= 0.0
        assert np.isfinite(syn)
        assert syn >= 0.0
