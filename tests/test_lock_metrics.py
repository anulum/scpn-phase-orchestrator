# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lock metrics tests

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.metrics import (
    LayerState,
    LockSignature,
    UPDEState,
)


# ---------------------------------------------------------------------------
# LockSignature: cross-layer phase locking
# ---------------------------------------------------------------------------


class TestLockSignature:
    """Verify LockSignature stores cross-layer coupling metrics
    with correct invariants."""

    def test_plv_range(self):
        """PLV must be in [0, 1]."""
        sig = LockSignature(source_layer=0, target_layer=1, plv=0.95, mean_lag=0.02)
        assert 0.0 <= sig.plv <= 1.0

    def test_field_preservation(self):
        sig = LockSignature(source_layer=2, target_layer=5, plv=0.88, mean_lag=0.15)
        assert sig.source_layer == 2
        assert sig.target_layer == 5
        assert sig.mean_lag == 0.15

    def test_frozen(self):
        sig = LockSignature(source_layer=0, target_layer=1, plv=0.5, mean_lag=0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            sig.plv = 0.9

    def test_self_locking_identity(self):
        """Layer locked to itself: PLV=1, lag=0."""
        sig = LockSignature(source_layer=0, target_layer=0, plv=1.0, mean_lag=0.0)
        assert sig.plv == 1.0
        assert sig.mean_lag == 0.0


# ---------------------------------------------------------------------------
# LayerState: per-layer oscillator statistics
# ---------------------------------------------------------------------------


class TestLayerState:
    """Verify LayerState enforces R∈[0,1] and integrates with
    LockSignature for cross-layer analysis."""

    def test_r_in_unit_interval(self):
        ls = LayerState(R=0.75, psi=1.2)
        assert 0.0 <= ls.R <= 1.0

    def test_with_lock_signatures(self):
        sig = LockSignature(
            source_layer=0, target_layer=1, plv=0.88, mean_lag=0.01
        )
        ls = LayerState(R=0.5, psi=0.0, lock_signatures={"0_1": sig})
        assert "0_1" in ls.lock_signatures
        assert ls.lock_signatures["0_1"].plv == 0.88

    def test_default_lock_signatures_empty(self):
        ls = LayerState(R=0.5, psi=0.0)
        assert ls.lock_signatures == {}

    def test_multiple_lock_signatures(self):
        """Layer can have locks to multiple other layers."""
        sigs = {
            "0_1": LockSignature(0, 1, plv=0.9, mean_lag=0.01),
            "0_2": LockSignature(0, 2, plv=0.7, mean_lag=0.05),
        }
        ls = LayerState(R=0.6, psi=0.3, lock_signatures=sigs)
        assert len(ls.lock_signatures) == 2
        assert ls.lock_signatures["0_1"].plv > ls.lock_signatures["0_2"].plv


# ---------------------------------------------------------------------------
# UPDEState: full system state
# ---------------------------------------------------------------------------


class TestUPDEState:
    """Verify UPDEState aggregates layer states and cross-layer alignment
    into a coherent system snapshot."""

    def test_construction_with_layers(self):
        layers = [
            LayerState(R=0.9, psi=0.1),
            LayerState(R=0.7, psi=0.5),
            LayerState(R=0.3, psi=1.0),
        ]
        state = UPDEState(
            layers=layers,
            cross_layer_alignment=np.eye(3),
            stability_proxy=0.8,
            regime_id="nominal",
        )
        assert len(state.layers) == 3
        assert state.stability_proxy == 0.8
        assert state.regime_id == "nominal"
        np.testing.assert_array_equal(state.cross_layer_alignment, np.eye(3))

    def test_empty_layers_critical(self):
        state = UPDEState(
            layers=[],
            cross_layer_alignment=np.array([]),
            stability_proxy=0.0,
            regime_id="critical",
        )
        assert len(state.layers) == 0
        assert state.regime_id == "critical"

    def test_r_values_accessible(self):
        """R values from all layers must be extractable for analysis."""
        layers = [LayerState(R=r, psi=0.0) for r in [0.9, 0.5, 0.2]]
        state = UPDEState(
            layers=layers,
            cross_layer_alignment=np.eye(3),
            stability_proxy=0.5,
            regime_id="degraded",
        )
        r_values = [ls.R for ls in state.layers]
        assert r_values == [0.9, 0.5, 0.2]
        assert np.mean(r_values) == pytest.approx(0.533, abs=0.01)


# ---------------------------------------------------------------------------
# Pipeline wiring + performance
# ---------------------------------------------------------------------------


class TestMetricsPipelineWiring:
    """Verify metrics types wire into the regime evaluation pipeline."""

    def test_upde_state_feeds_regime_manager(self):
        """UPDEState → RegimeManager.evaluate → Regime."""
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager

        state = UPDEState(
            layers=[LayerState(R=0.1, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.1,
            regime_id="nominal",
        )
        mgr = RegimeManager()
        regime = mgr.evaluate(state, BoundaryState())
        assert regime == Regime.CRITICAL

    def test_layer_state_construction_performance(self):
        """Creating 1000 LayerState objects must take <10ms."""
        t0 = time.perf_counter()
        for _ in range(1000):
            LayerState(R=0.5, psi=1.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.01, (
            f"1000 LayerState constructions: {elapsed * 1000:.1f}ms > 10ms"
        )

    def test_upde_state_construction_performance(self):
        """Creating UPDEState with 16 layers must take <1ms."""
        layers = [LayerState(R=0.5, psi=0.0) for _ in range(16)]
        t0 = time.perf_counter()
        for _ in range(100):
            UPDEState(
                layers=layers,
                cross_layer_alignment=np.eye(16),
                stability_proxy=0.5,
                regime_id="nominal",
            )
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.001, (
            f"UPDEState(16 layers): {elapsed * 1000:.2f}ms > 1ms"
        )
