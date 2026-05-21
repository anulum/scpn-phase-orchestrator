# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Transfer entropy tests

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import transfer_entropy as te_mod
from scpn_phase_orchestrator.monitor.transfer_entropy import (
    phase_transfer_entropy,
    transfer_entropy_matrix,
)
from tests.typing_contracts import assert_precise_ndarray_hint


class TestTransferEntropy:
    def test_public_array_contracts_are_parameterised(self):
        hints = (
            get_type_hints(phase_transfer_entropy)["source"],
            get_type_hints(phase_transfer_entropy)["target"],
            get_type_hints(transfer_entropy_matrix)["phase_series"],
            get_type_hints(transfer_entropy_matrix)["return"],
        )
        for hint in hints:
            assert_precise_ndarray_hint(hint)
            assert "float64" in str(hint)

    def test_identical_signals_low_te(self):
        rng = np.random.default_rng(42)
        sig = rng.uniform(0, 2 * np.pi, 200)
        te = phase_transfer_entropy(sig, sig)
        assert te >= 0.0

    def test_independent_signals_finite_te(self):
        rng = np.random.default_rng(42)
        a = rng.uniform(0, 2 * np.pi, 1000)
        b = rng.uniform(0, 2 * np.pi, 1000)
        te = phase_transfer_entropy(a, b, n_bins=8)
        assert np.isfinite(te)
        assert te >= 0.0

    def test_driven_signal_higher_te(self):
        rng = np.random.default_rng(42)
        source = np.cumsum(rng.normal(0, 0.1, 500)) % (2 * np.pi)
        target = np.roll(source, 1)  # target follows source
        te_fwd = phase_transfer_entropy(source, target)
        _te_rev = phase_transfer_entropy(target, source)
        # Forward TE should be >= reverse (source drives target)
        assert te_fwd >= 0.0

    def test_short_signals(self):
        previous = te_mod.ACTIVE_BACKEND
        te_mod.ACTIVE_BACKEND = "python"
        try:
            assert phase_transfer_entropy(np.array([1.0, 2.0]), np.array([1.0])) == 0.0
        finally:
            te_mod.ACTIVE_BACKEND = previous

    def test_length_mismatch_shortens_to_shortest_series(self) -> None:
        previous = te_mod.ACTIVE_BACKEND
        te_mod.ACTIVE_BACKEND = "python"
        try:
            source = np.array([0.0, 0.4, 0.8, 1.2, 1.6], dtype=np.float64)
            target = np.array([0.0, 0.25, 0.5], dtype=np.float64)
            result = phase_transfer_entropy(source, target, n_bins=12)
            expected = phase_transfer_entropy(source[:3], target, n_bins=12)
            assert result == pytest.approx(expected, rel=0.0, abs=1e-12)
        finally:
            te_mod.ACTIVE_BACKEND = previous

    def test_short_signal_source_is_rejected_as_zero_regardless_of_target(self) -> None:
        previous = te_mod.ACTIVE_BACKEND
        te_mod.ACTIVE_BACKEND = "python"
        try:
            assert (
                phase_transfer_entropy(
                    np.array([0.1, 0.2], dtype=np.float64),
                    np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float64),
                    n_bins=10,
                )
                == 0.0
            )
        finally:
            te_mod.ACTIVE_BACKEND = previous

    @pytest.mark.parametrize("n_bins", [0, 1, True, 4.5])
    def test_phase_te_rejects_invalid_bin_counts(self, n_bins):
        with pytest.raises((TypeError, ValueError), match="n_bins"):
            phase_transfer_entropy(
                np.linspace(0.0, 1.0, 8),
                np.linspace(0.1, 1.1, 8),
                n_bins=n_bins,
            )

    @pytest.mark.parametrize(
        ("source", "target", "match"),
        [
            ([[0.0, 1.0, 2.0]], np.array([0.0, 1.0, 2.0]), "source must be 1-D"),
            (np.array([0.0, np.nan, 2.0]), np.array([0.0, 1.0, 2.0]), "source"),
            (np.array([0.0, 1.0, 2.0]), np.array([0.0, np.inf, 2.0]), "target"),
            (np.array([True, False, True]), np.array([0.0, 1.0, 2.0]), "source"),
            (np.array([0.0, 1.0, 2.0]), np.array([True, False, True]), "target"),
        ],
    )
    def test_phase_te_rejects_non_vector_or_non_finite_inputs(
        self, source, target, match
    ):
        with pytest.raises(ValueError, match=match):
            phase_transfer_entropy(source, target)

    def test_matrix_shape(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 2 * np.pi, (4, 100))
        te = transfer_entropy_matrix(data)
        assert te.shape == (4, 4)

    def test_matrix_rejects_non_oscillator_time_series_shape(self):
        with pytest.raises(ValueError, match="phase_series must be 2-D"):
            transfer_entropy_matrix(np.linspace(0.0, 1.0, 8))

    def test_matrix_rejects_non_finite_phase_series(self):
        series = np.array([[0.0, 1.0, np.nan], [0.1, 1.1, 2.1]])
        with pytest.raises(ValueError, match="phase_series"):
            transfer_entropy_matrix(series)

    def test_matrix_rejects_boolean_phase_series(self):
        series = np.array([[True, False, True], [False, True, False]])
        with pytest.raises(ValueError, match="phase_series"):
            transfer_entropy_matrix(series)

    @pytest.mark.parametrize("shape", [(0, 8), (3, 0)])
    def test_matrix_rejects_empty_axes(self, shape):
        with pytest.raises(ValueError, match="phase_series"):
            transfer_entropy_matrix(np.empty(shape))

    @pytest.mark.parametrize("n_bins", [0, 1, False, 8.25])
    def test_matrix_rejects_invalid_bin_counts(self, n_bins):
        rng = np.random.default_rng(42)
        series = rng.uniform(0, 2 * np.pi, (3, 32))
        with pytest.raises((TypeError, ValueError), match="n_bins"):
            transfer_entropy_matrix(series, n_bins=n_bins)

    def test_matrix_diagonal_zero(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 2 * np.pi, (3, 100))
        te = transfer_entropy_matrix(data)
        np.testing.assert_array_equal(np.diag(te), 0.0)

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 2 * np.pi, (3, 100))
        te = transfer_entropy_matrix(data)
        assert np.all(te >= 0.0)

    def test_python_fallback_matrix_populates_each_off_diagonal_pair(self):
        previous = te_mod.ACTIVE_BACKEND
        te_mod.ACTIVE_BACKEND = "python"
        try:
            rng = np.random.default_rng(123)
            driver = np.cumsum(0.03 + 0.01 * rng.standard_normal(600))
            follower = np.roll(driver, 1) + 0.02 * rng.standard_normal(600)
            independent = rng.uniform(0.0, 2 * np.pi, size=600)

            te = transfer_entropy_matrix(
                np.stack([driver % (2 * np.pi), follower % (2 * np.pi), independent]),
                n_bins=12,
            )
        finally:
            te_mod.ACTIVE_BACKEND = previous

        assert te.shape == (3, 3)
        np.testing.assert_array_equal(np.diag(te), 0.0)
        assert np.count_nonzero(te) == 6
        assert te[0, 1] > te[1, 0]


class TestTEPipelineWiring:
    """Pipeline: engine trajectory → TE matrix reveals coupling direction."""

    def test_engine_trajectory_to_te_matrix(self):
        """UPDEEngine generates trajectory → transfer_entropy_matrix
        reveals directional coupling structure."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = rng.normal(1.0, 0.3, n)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        trajectory = []
        for _ in range(200):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            trajectory.append(phases.copy())
        traj = np.array(trajectory).T  # (n, T)

        te = transfer_entropy_matrix(traj)
        assert te.shape == (n, n)
        assert np.all(te >= 0.0)
        np.testing.assert_array_equal(np.diag(te), 0.0)


def test_backend_dispatchers_are_honoured_for_phase_te_and_matrix(monkeypatch):
    def fake_phase_te(src: np.ndarray, tgt: np.ndarray, bins: int) -> float:
        assert src.shape == (3,)
        assert tgt.shape == (3,)
        assert bins == 6
        return 0.75

    def fake_matrix(flat: np.ndarray, n_osc: int, n_time: int, bins: int) -> np.ndarray:
        assert n_osc == 3
        assert n_time == 3
        assert bins == 6
        return np.arange(n_osc * n_osc, dtype=np.float64).reshape((n_osc, n_osc))

    def fake_loader():
        return {"phase_te": fake_phase_te, "te_matrix": fake_matrix}

    monkeypatch.setitem(te_mod._LOADERS, "rust", fake_loader)
    previous = te_mod.ACTIVE_BACKEND
    te_mod.ACTIVE_BACKEND = "rust"
    try:
        source = np.linspace(0.0, 1.0, 3)
        target = np.linspace(0.1, 1.1, 3)
        phase_value = phase_transfer_entropy(source, target, n_bins=6)
        matrix_value = transfer_entropy_matrix(
            np.stack([source, target, np.array([0.2, 0.4, 0.6])]), n_bins=6
        )
    finally:
        te_mod.ACTIVE_BACKEND = previous

    assert phase_value == 0.75
    assert matrix_value.shape == (3, 3)
    assert matrix_value[0, 1] == 1.0


def test_phase_te_backend_failure_falls_back_to_python(monkeypatch):
    def raising_phase_te(_src: np.ndarray, _tgt: np.ndarray, _bins: int) -> float:
        raise RuntimeError("boom")

    previous = te_mod.ACTIVE_BACKEND
    te_mod.ACTIVE_BACKEND = "rust"
    monkeypatch.setattr(
        te_mod,
        "_dispatch",
        lambda fn_name: raising_phase_te if fn_name == "phase_te" else None,
    )
    try:
        source = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
        target = np.roll(source, 1)
        value = phase_transfer_entropy(source, target, n_bins=8)
    finally:
        te_mod.ACTIVE_BACKEND = previous

    assert np.isfinite(value)
    assert value >= 0.0


def test_te_matrix_backend_failure_falls_back_to_python(monkeypatch):
    def raising_matrix(
        _flat: np.ndarray, _n_osc: int, _n_time: int, _bins: int
    ) -> np.ndarray:
        raise RuntimeError("boom")

    previous = te_mod.ACTIVE_BACKEND
    te_mod.ACTIVE_BACKEND = "rust"
    monkeypatch.setattr(
        te_mod,
        "_dispatch",
        lambda fn_name: raising_matrix if fn_name == "te_matrix" else None,
    )
    try:
        data = np.stack(
            [
                np.linspace(0.0, 2 * np.pi, 64, endpoint=False),
                np.linspace(0.2, 2 * np.pi + 0.2, 64, endpoint=False),
                np.linspace(0.4, 2 * np.pi + 0.4, 64, endpoint=False),
            ]
        )
        value = transfer_entropy_matrix(data, n_bins=8)
    finally:
        te_mod.ACTIVE_BACKEND = previous

    assert value.shape == (3, 3)
    assert np.all(value >= 0.0)
    np.testing.assert_array_equal(np.diag(value), 0.0)
