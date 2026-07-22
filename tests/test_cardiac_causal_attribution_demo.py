# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cardiac causal attribution demo tests

from __future__ import annotations

import pytest

from domainpacks.cardiac_rhythm.causal_attribution_demo import run_demo
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor import CausalInterventionEngine
from tests.test_causal_supervisor import _system


def test_cardiac_causal_demo_emits_attribution_record() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "cardiac_rhythm"
    assert payload["scenario"] == "ventricular_disturbance_pacing_counterfactual"
    attribution = payload["attribution"]
    assert isinstance(attribution, dict)
    assert attribution["effect"] in {"stabilising", "neutral", "destabilising"}
    # Honest weak-signal case: the verdict is driven by a small net delta, yet only
    # 13 of 31 steps hold that sign — a low trajectory-consistency the old
    # magnitude-over-threshold "confidence" (clamped to 1.0) masked entirely.
    assert attribution["trajectory_consistency"] == pytest.approx(13 / 31)
    assert attribution["threshold"] == pytest.approx(1e-4)
    counterfactual = payload["counterfactual"]
    assert counterfactual["actions"][0]["knob"] == "zeta"
    assert len(counterfactual["baseline_R"]) == len(counterfactual["intervention_R"])


def test_rollout_attribute_rejects_invalid_threshold() -> None:
    phases, omegas, knm, alpha = _system()
    rollout = CausalInterventionEngine(6, dt=0.01, horizon=2).evaluate_actions(
        phases,
        omegas,
        knm,
        alpha,
        0.0,
        0.0,
        [
            ControlAction(
                knob="K",
                scope="global",
                value=0.1,
                ttl_s=1.0,
                justification="threshold validation",
            )
        ],
    )

    with pytest.raises(ValueError, match="threshold"):
        rollout.attribute(threshold=-1.0)
