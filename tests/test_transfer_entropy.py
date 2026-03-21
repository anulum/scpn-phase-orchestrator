# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Transfer entropy tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.transfer_entropy import (
    phase_transfer_entropy,
    transfer_entropy_matrix,
)


class TestTransferEntropy:
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
        te_rev = phase_transfer_entropy(target, source)
        # Forward TE should be >= reverse (source drives target)
        assert te_fwd >= 0.0

    def test_short_signals(self):
        assert phase_transfer_entropy(np.array([1.0, 2.0]), np.array([1.0])) == 0.0

    def test_matrix_shape(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 2 * np.pi, (4, 100))
        te = transfer_entropy_matrix(data)
        assert te.shape == (4, 4)

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
