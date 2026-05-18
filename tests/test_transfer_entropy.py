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
