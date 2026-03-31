# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sleep staging tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.sleep_staging import (
    classify_sleep_stage,
    ultradian_phase,
)


def test_n3_high_synchrony():
    assert classify_sleep_stage(0.85) == "N3"
    assert classify_sleep_stage(0.70) == "N3"


def test_n2_moderate_synchrony():
    assert classify_sleep_stage(0.55) == "N2"
    assert classify_sleep_stage(0.40) == "N2"


def test_n1_light_sleep():
    assert classify_sleep_stage(0.35) == "N1"
    assert classify_sleep_stage(0.30) == "N1"


def test_rem_with_functional_desync():
    assert classify_sleep_stage(0.25, functional_desync=True) == "REM"
    assert classify_sleep_stage(0.35, functional_desync=True) == "REM"


def test_wake_low_r_no_desync():
    assert classify_sleep_stage(0.15) == "Wake"
    assert classify_sleep_stage(0.0) == "Wake"


def test_wake_very_low_r_even_with_desync():
    assert classify_sleep_stage(0.10, functional_desync=True) == "Wake"


def test_n1_without_desync_not_rem():
    assert classify_sleep_stage(0.32) == "N1"
    assert classify_sleep_stage(0.32, functional_desync=False) == "N1"


def test_boundary_at_n3_threshold():
    assert classify_sleep_stage(0.699) == "N2"
    assert classify_sleep_stage(0.700) == "N3"


def test_ultradian_phase_at_n3_onset():
    ts = np.array([0.0, 30.0, 60.0])
    stages = ["Wake", "N1", "N3"]
    phase = ultradian_phase(ts, stages)
    assert phase == 0.0


def test_ultradian_phase_halfway():
    # 45 minutes = half of 90-minute cycle
    ts = np.array([0.0, 45.0 * 60.0])
    stages = ["N3", "REM"]
    phase = ultradian_phase(ts, stages)
    np.testing.assert_allclose(phase, 0.5, atol=1e-6)


def test_ultradian_phase_wraps():
    # 90 minutes exactly → wraps to 0
    ts = np.array([0.0, 90.0 * 60.0])
    stages = ["N3", "N2"]
    phase = ultradian_phase(ts, stages)
    np.testing.assert_allclose(phase, 0.0, atol=1e-6)


def test_ultradian_no_n3_returns_zero():
    ts = np.array([0.0, 100.0, 200.0])
    stages = ["Wake", "N1", "N2"]
    assert ultradian_phase(ts, stages) == 0.0


def test_ultradian_empty_input():
    assert ultradian_phase(np.array([]), []) == 0.0


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
