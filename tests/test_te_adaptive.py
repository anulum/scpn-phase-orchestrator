# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — TE adaptive coupling tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling import te_adaptive as te_mod
from scpn_phase_orchestrator.coupling.te_adaptive import te_adapt_coupling


class TestTEAdaptive:
    def test_output_shape(self):
        knm = np.full((4, 4), 0.5)
        np.fill_diagonal(knm, 0.0)
        rng = np.random.default_rng(42)
        history = rng.uniform(0, 2 * np.pi, (4, 100))
        result = te_adapt_coupling(knm, history)
        assert result.shape == (4, 4)

    def test_zero_diagonal(self):
        knm = np.full((3, 3), 0.5)
        np.fill_diagonal(knm, 0.0)
        rng = np.random.default_rng(42)
        history = rng.uniform(0, 2 * np.pi, (3, 100))
        result = te_adapt_coupling(knm, history)
        np.testing.assert_array_equal(np.diag(result), 0.0)

    def test_non_negative(self):
        knm = np.full((3, 3), 0.1)
        np.fill_diagonal(knm, 0.0)
        rng = np.random.default_rng(42)
        history = rng.uniform(0, 2 * np.pi, (3, 200))
        result = te_adapt_coupling(knm, history, lr=0.01)
        assert np.all(result >= 0)

    def test_coupling_increases(self):
        knm = np.full((4, 4), 0.1)
        np.fill_diagonal(knm, 0.0)
        rng = np.random.default_rng(42)
        history = rng.uniform(0, 2 * np.pi, (4, 200))
        result = te_adapt_coupling(knm, history, lr=1.0)
        # With lr=1.0, TE contribution should increase off-diagonal
        assert float(result.sum()) >= float(knm.sum())

    def test_decay_reduces(self):
        knm = np.full((3, 3), 1.0)
        np.fill_diagonal(knm, 0.0)
        rng = np.random.default_rng(42)
        history = rng.uniform(0, 2 * np.pi, (3, 100))
        result = te_adapt_coupling(knm, history, lr=0.0, decay=0.5)
        assert float(result.sum()) < float(knm.sum())

    def test_rust_dispatch_uses_flattened_transfer_entropy_matrix(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: dict[str, object] = {}

        def fake_transfer_entropy_matrix(history, *, n_bins):
            calls["history_shape"] = tuple(history.shape)
            calls["n_bins"] = n_bins
            return np.array([[0.0, 0.4], [0.2, 0.0]], dtype=np.float64)

        def fake_rust(k_flat, te_flat, n, lr, decay):
            calls["k_flat"] = tuple(float(value) for value in k_flat)
            calls["te_flat"] = tuple(float(value) for value in te_flat)
            calls["n"] = n
            calls["lr"] = lr
            calls["decay"] = decay
            return np.array([0.0, 0.8, 0.3, 0.0], dtype=np.float64)

        monkeypatch.setattr(te_mod, "_HAS_RUST", True)
        monkeypatch.setattr(te_mod, "_rust_te_adapt", fake_rust, raising=False)
        monkeypatch.setattr(
            te_mod,
            "transfer_entropy_matrix",
            fake_transfer_entropy_matrix,
        )

        knm = np.array([[0.0, 0.1], [0.2, 0.0]], dtype=np.float64)
        history = np.ones((2, 5), dtype=np.float64)

        result = te_adapt_coupling(knm, history, lr=0.25, decay=0.1, n_bins=5)

        np.testing.assert_allclose(result, [[0.0, 0.8], [0.3, 0.0]])
        assert calls == {
            "history_shape": (2, 5),
            "n_bins": 5,
            "k_flat": (0.0, 0.1, 0.2, 0.0),
            "te_flat": (0.0, 0.4, 0.2, 0.0),
            "n": 2,
            "lr": 0.25,
            "decay": 0.1,
        }


class TestTEAdaptivePipelineWiring:
    """Pipeline: engine trajectory → TE → adapt coupling → engine."""

    def test_te_adapted_knm_drives_engine(self):
        """Engine trajectory → te_adapt_coupling → updated K_nm → engine."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.array([1.0, 1.5, 2.0, 0.5])
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        trajectory = []
        for _ in range(200):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            trajectory.append(phases.copy())
        traj = np.array(trajectory).T  # (n, T)

        knm_adapted = te_adapt_coupling(knm, traj)
        assert knm_adapted.shape == (n, n)

        # Use adapted coupling in engine
        for _ in range(100):
            phases = eng.step(
                phases,
                omegas,
                knm_adapted,
                0.0,
                0.0,
                alpha,
            )
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
