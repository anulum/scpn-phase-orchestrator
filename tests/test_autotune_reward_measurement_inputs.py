# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — autotune_reward measurement ingress

"""Refuse aliases in replay proposals using an actual numerical evaluator."""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.autotune.reward import (
    KnobPolicyCandidate,
    evaluate_knob_policy,
)
from tests.measurement_samples import source_alias
from tests.replay_measurement_evaluator import evaluate_replay


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """The public proposal operation refuses the original malformed coupling."""
    seed = KnobPolicyCandidate(
        K=source_alias(kind, (3, 3)), alpha=0.0, zeta=0.0, Psi=0.0
    )
    with pytest.raises(ValueError):
        evaluate_knob_policy(
            seed,
            evaluate_replay(KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.0, Psi=0.0)),
        )
