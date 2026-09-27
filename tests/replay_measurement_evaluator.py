# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Numerical replay evaluator for input contracts

"""Evaluate replay rewards from actual UPDE evolution and order parameters."""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.autotune.reward import (
    KnobPolicyCandidate,
    RewardObservation,
)
from scpn_phase_orchestrator.upde.engine import upde_run
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


def evaluate_replay(candidate: KnobPolicyCandidate) -> RewardObservation:
    """Measure coherence after evolving a candidate in a three-oscillator replay.

    Parameters
    ----------
    candidate : KnobPolicyCandidate
        Validated coupling and control proposal.

    Returns
    -------
    RewardObservation
        Coherence measured from the evolved state.
    """
    phases = np.array([0.1, 0.3, 0.8], dtype=np.float64)
    coupling = np.full((3, 3), float(np.mean(candidate.K)), dtype=np.float64)
    alpha = np.full((3, 3), float(np.mean(candidate.alpha)), dtype=np.float64)
    output = upde_run(
        phases,
        np.ones(3),
        coupling,
        alpha,
        float(np.mean(candidate.zeta)),
        float(np.mean(candidate.Psi)),
        0.01,
        4,
    )
    coherence, _ = compute_order_parameter(output)
    previous, _ = compute_order_parameter(phases)
    return RewardObservation(
        coherence=coherence,
        previous_coherence=previous,
        unsafe=False,
        regime_changed=False,
    )
