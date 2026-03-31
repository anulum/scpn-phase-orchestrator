# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL monitor tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.monitor.stl import HAS_RTAMT, STLMonitor

needs_rtamt = pytest.mark.skipif(not HAS_RTAMT, reason="rtamt not installed")


class TestSTLMonitorConstants:
    def test_sync_threshold_contains_r(self):
        assert "R" in STLMonitor.SYNC_THRESHOLD

    def test_coupling_bound_contains_k(self):
        assert "K" in STLMonitor.COUPLING_BOUND


class TestSTLImportGuard:
    @pytest.mark.skipif(HAS_RTAMT, reason="rtamt is installed")
    def test_import_error_without_rtamt(self):
        with pytest.raises(ImportError, match="rtamt"):
            STLMonitor("always (x <= 1.0)")


@needs_rtamt
class TestSTLMonitor:
    def test_constant_satisfied(self):
        mon = STLMonitor("always (x <= 1.0)")
        rob = mon.evaluate({"x": [0.5, 0.6, 0.7, 0.8]})
        assert rob > 0

    def test_constant_violated(self):
        mon = STLMonitor("always (x <= 1.0)")
        rob = mon.evaluate({"x": [0.5, 0.6, 1.5, 0.8]})
        assert rob < 0

    def test_sync_threshold_spec(self):
        mon = STLMonitor(STLMonitor.SYNC_THRESHOLD)
        rob_ok = mon.evaluate({"R": [0.9, 0.8, 0.7, 0.5]})
        assert rob_ok > 0
        mon2 = STLMonitor(STLMonitor.SYNC_THRESHOLD)
        rob_fail = mon2.evaluate({"R": [0.9, 0.2, 0.7, 0.5]})
        assert rob_fail < 0

    def test_coupling_bound_spec(self):
        mon = STLMonitor(STLMonitor.COUPLING_BOUND)
        rob = mon.evaluate({"K": [1.0, 2.0, 5.0, 9.0]})
        assert rob > 0

    def test_empty_trace_raises(self):
        mon = STLMonitor("always (x <= 1.0)")
        with pytest.raises(ValueError, match="at least one signal"):
            mon.evaluate({})

    def test_empty_signal_raises(self):
        mon = STLMonitor("always (x <= 1.0)")
        with pytest.raises(ValueError, match="non-empty"):
            mon.evaluate({"x": []})

    def test_unequal_lengths_raises(self):
        mon = STLMonitor("always (x <= y)")
        with pytest.raises(ValueError, match="equal length"):
            mon.evaluate({"x": [1.0, 2.0], "y": [1.0]})

    def test_robustness_magnitude(self):
        mon = STLMonitor("always (x <= 10.0)")
        rob = mon.evaluate({"x": [5.0, 5.0, 5.0]})
        assert rob >= 5.0 - 1e-6


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
