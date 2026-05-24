# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Session start coherence analysis contracts

"""
Coherence-analysis contracts for session-start diagnostics and imprint shape
validation.
"""
from __future__ import annotations

import numpy as np


class TestSessionStartCoherenceAnalysis:
    """Verify that session_start checks correctly identify low vs high
    initial coherence and produce appropriate warnings."""

    def _run_check(self, phases):
        from scpn_phase_orchestrator.imprint.state import ImprintState
        from scpn_phase_orchestrator.monitor.session_start import check_session_start

        n = len(phases)
        imprint = ImprintState(m_k=np.ones(n), last_update=0.0)
        return check_session_start([], np.array(phases), imprint, n)

    def test_near_chaos_phases_produce_low_coherence_warning(self):
        """Uniformly spread phases (R ≈ 0) must trigger the warning."""
        phases = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        report = self._run_check(phases)
        assert any("Low initial coherence" in w for w in report.warnings)
        assert report.initial_r < 0.05, (
            f"R={report.initial_r:.4f} should be near 0 for uniformly spread phases"
        )

    def test_synchronised_phases_no_warning(self):
        """Nearly identical phases (R ≈ 1) must NOT trigger the warning."""
        phases = [0.01, 0.02, 0.015, 0.005]
        report = self._run_check(phases)
        low_coh = [w for w in report.warnings if "Low initial" in w]
        assert low_coh == [], f"Sync phases → no warning: {low_coh}"
        assert report.initial_r > 0.99, (
            f"R={report.initial_r:.4f} should be near 1 for nearly identical phases"
        )

    def test_moderate_coherence_no_warning(self):
        """R just above 0.05 threshold must not trigger the warning."""
        # Two clusters: [0, 0.1] and [0.3, 0.4] — R well above 0.05
        phases = [0.0, 0.1, 0.3, 0.4]
        report = self._run_check(phases)
        assert report.initial_r > 0.05
        assert not any("Low initial coherence" in w for w in report.warnings)

    def test_imprint_size_mismatch_error(self):
        """If imprint dimension doesn't match oscillator count, report error."""
        from scpn_phase_orchestrator.imprint.state import ImprintState
        from scpn_phase_orchestrator.monitor.session_start import check_session_start

        phases = np.array([0.1, 0.2, 0.3])
        imprint = ImprintState(m_k=np.ones(5), last_update=0.0)  # 5 != 3
        report = check_session_start([], phases, imprint, 3)
        assert not report.passed
        assert any("mismatch" in e.lower() for e in report.errors)
