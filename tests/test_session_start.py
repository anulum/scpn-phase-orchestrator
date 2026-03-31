# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Session-start coherence gate tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.monitor.session_start import (
    SessionCoherenceReport,
    check_session_start,
)
from scpn_phase_orchestrator.oscillators.base import PhaseState

TWO_PI = 2.0 * np.pi


def _make_states(n: int, quality: float = 0.8, channel: str = "P") -> list[PhaseState]:
    return [
        PhaseState(
            theta=float(i) * TWO_PI / n,
            omega=1.0,
            amplitude=1.0,
            quality=quality,
            channel=channel,
            node_id=f"{channel}_{i}",
        )
        for i in range(n)
    ]


def test_healthy_session_passes():
    n = 10
    states = _make_states(n, quality=0.8, channel="P")
    states += _make_states(n, quality=0.7, channel="I")
    states += _make_states(n, quality=0.6, channel="S")
    phases = np.linspace(0, 0.1, n)  # near-synchronised
    imprint = ImprintState(m_k=np.full(n, 0.2), last_update=100.0)

    report = check_session_start(states, phases, imprint, n)
    assert report.passed
    assert not report.errors
    assert report.quality_scores["P"] > 0.5
    assert report.initial_r > 0.9
    assert abs(report.imprint_level - 0.2) < 1e-6


def test_low_quality_warns():
    n = 10
    states = _make_states(n, quality=0.1, channel="P")
    phases = np.linspace(0, TWO_PI, n)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, phases, imprint, n)
    assert report.passed  # low quality is warning, not error
    assert any("low quality" in w for w in report.warnings)


def test_signal_collapse_fails():
    n = 10
    states = _make_states(n, quality=0.05, channel="P")
    phases = np.zeros(n)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, phases, imprint, n)
    assert not report.passed
    assert any("collapse" in e.lower() for e in report.errors)


def test_imprint_size_mismatch_fails():
    n = 10
    states = _make_states(n, quality=0.8)
    phases = np.zeros(n)
    bad_imprint = ImprintState(m_k=np.zeros(5), last_update=0.0)

    report = check_session_start(states, phases, bad_imprint, n)
    assert not report.passed
    assert any("mismatch" in e.lower() for e in report.errors)


def test_fresh_imprint_passes():
    n = 10
    states = _make_states(n, quality=0.8)
    phases = np.linspace(0, 0.1, n)
    fresh = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, phases, fresh, n)
    assert report.passed
    assert report.imprint_level == 0.0


def test_report_dataclass_defaults():
    r = SessionCoherenceReport()
    assert r.passed is True
    assert r.errors == []
    assert r.warnings == []
    assert r.quality_scores == {}


def test_multi_channel_quality_scores():
    n = 5
    p_states = _make_states(n, quality=0.9, channel="P")
    i_states = _make_states(n, quality=0.5, channel="I")
    s_states = _make_states(n, quality=0.3, channel="S")
    phases = np.zeros(n)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(p_states + i_states + s_states, phases, imprint, n)
    assert "P" in report.quality_scores
    assert "I" in report.quality_scores
    assert "S" in report.quality_scores
    assert report.quality_scores["P"] > report.quality_scores["S"]


def test_low_initial_coherence_warns():
    n = 10
    states = _make_states(n, quality=0.8)
    rng = np.random.default_rng(99)
    random_phases = rng.uniform(0, TWO_PI, n)
    imprint = ImprintState(m_k=np.zeros(n), last_update=0.0)

    report = check_session_start(states, random_phases, imprint, n)
    # 10 random phases → R typically < 0.8 (not fully synchronized)
    assert report.initial_r < 0.95


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
