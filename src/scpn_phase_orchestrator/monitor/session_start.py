# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Session-start coherence gate

"""Session-start validation gate for extractor, imprint, and coherence inputs.

The validator checks startup preconditions across extractor quality signals,
imprint availability, and initial coherence metrics before a session is allowed
to proceed. It returns explicit warnings and errors without mutating source
state or triggering actuation, keeping the gate suitable for dry-run previews,
operator review, and fail-closed orchestration handoffs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

__all__ = ["SessionCoherenceReport", "check_session_start"]

TWO_PI = 2.0 * np.pi
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class SessionCoherenceReport:
    """Results of the session-start coherence gate check."""

    quality_scores: dict[str, float] = field(default_factory=dict)
    initial_r: float = 0.0
    imprint_level: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    passed: bool = True


def check_session_start(
    phase_states: list[PhaseState],
    initial_phases: FloatArray,
    imprint_state: ImprintState,
    n_osc: int,
) -> SessionCoherenceReport:
    """Validate extraction quality, imprint consistency, and initial coherence.

    Parameters
    ----------
    phase_states : list[PhaseState]
        extracted states from all configured channels.
    initial_phases : FloatArray
        phase array that will seed the UPDE engine.
    imprint_state : ImprintState
        loaded (or fresh) imprint state.
    n_osc : int
        expected oscillator count.

    Returns
    -------
    SessionCoherenceReport
        SessionCoherenceReport with pass/fail, quality scores, and diagnostics.
    """
    report = SessionCoherenceReport()
    scorer = PhaseQualityScorer()

    # Quality per channel
    by_channel: dict[str, list[PhaseState]] = {}
    for ps in phase_states:
        by_channel.setdefault(ps.channel, []).append(ps)

    for ch, states in by_channel.items():
        if not states:
            continue
        q = scorer.score(states)
        report.quality_scores[ch] = q
        if q < 0.3:
            report.warnings.append(
                f"Channel {ch}: low quality ({q:.2f}); extraction may be unreliable"
            )

    if scorer.detect_collapse(phase_states):
        report.errors.append("Signal collapse: majority of extractors below threshold")
        report.passed = False

    # Imprint consistency
    if imprint_state.m_k.shape[0] != n_osc:
        report.errors.append(
            f"Imprint size mismatch: {imprint_state.m_k.shape[0]} != {n_osc}"
        )
        report.passed = False
    else:
        report.imprint_level = float(np.mean(imprint_state.m_k))

    # Initial coherence from extracted phases
    if initial_phases.shape[0] == n_osc:
        r, _ = compute_order_parameter(initial_phases)
        report.initial_r = float(r)
        if r < 0.05:
            report.warnings.append(
                f"Low initial coherence (R={r:.3f}); starting from near-chaos"
            )

    return report
