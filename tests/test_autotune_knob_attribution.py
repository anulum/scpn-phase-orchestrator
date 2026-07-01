# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — per-knob attribution tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.autotune.knob_attribution import (
    KnobAttributionConfig,
    KnobAttributionReport,
    attribute_knob_policy,
)
from scpn_phase_orchestrator.autotune.reward import (
    AutotuneRewardReport,
    KnobPolicyCandidate,
    RewardConfig,
    RewardObservation,
)


def _score(candidate: KnobPolicyCandidate) -> AutotuneRewardReport:
    k_total = float(np.sum(np.asarray(candidate.K, dtype=np.float64)))
    alpha_total = float(np.sum(np.asarray(candidate.alpha, dtype=np.float64)))
    zeta_total = float(np.sum(np.asarray(candidate.zeta, dtype=np.float64)))
    psi_total = float(np.sum(np.asarray(candidate.Psi, dtype=np.float64)))
    channel_total = float(sum(candidate.channel_weights))
    cross_total = float(sum(candidate.cross_channel_gains))
    reward = (
        2.0 * k_total
        + alpha_total
        + 0.5 * zeta_total
        - psi_total
        + channel_total
        + 0.25 * cross_total
    )
    return AutotuneRewardReport(
        reward=reward,
        components={
            "coupling": 2.0 * k_total,
            "adaptive": alpha_total + 0.5 * zeta_total - psi_total,
            "channels": channel_total + 0.25 * cross_total,
        },
        candidate=candidate,
        observation=RewardObservation(coherence=0.8),
        config=RewardConfig(component_order=("coherence_gain",)),
    )


def test_attribute_knob_policy_exact_shapley_serialises_ranked_components() -> None:
    baseline = KnobPolicyCandidate(K=0.0, alpha=0.0)
    candidate = KnobPolicyCandidate(K=2.0, alpha=3.0)

    report = attribute_knob_policy(candidate, baseline, _score)

    assert isinstance(report, KnobAttributionReport)
    assert report.method == "exact"
    assert report.sample_error is None
    assert report.candidate_reward == pytest.approx(7.0)
    assert report.baseline_reward == pytest.approx(0.0)
    assert report.attributed_total == pytest.approx(7.0)
    assert [item.knob for item in report.attributions[:2]] == ["K", "alpha"]
    assert report.attributions[0].shapley_components["coupling"] == pytest.approx(4.0)
    assert report.attributions[1].shapley_components["adaptive"] == pytest.approx(3.0)

    record = report.to_audit_record()
    assert record["method"] == "exact"
    assert record["attributions"][0]["rank"] == 0


def test_attribute_knob_policy_reports_empty_exact_attribution() -> None:
    baseline = KnobPolicyCandidate(K=np.asarray([0.0, 1.0], dtype=np.float64))

    report = attribute_knob_policy(baseline, baseline, _score)

    assert report.method == "exact"
    assert report.attributions == ()
    assert report.attributed_total == 0.0
    assert report.candidate_reward == report.baseline_reward


def test_attribute_knob_policy_uses_deterministic_sampling_above_exact_limit() -> None:
    baseline = KnobPolicyCandidate(
        K=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        alpha=0.0,
        zeta=0.0,
        Psi=0.0,
        channel_weights=(0.0, 0.0),
        cross_channel_gains=(0.0,),
    )
    candidate = KnobPolicyCandidate(
        K=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        alpha=1.0,
        zeta=2.0,
        Psi=0.5,
        channel_weights=(0.5, 0.25),
        cross_channel_gains=(0.4,),
    )

    report = attribute_knob_policy(
        candidate,
        baseline,
        _score,
        config=KnobAttributionConfig(exact_max_knobs=2, sample_permutations=8, seed=7),
    )

    assert report.method == "sampled"
    assert report.sample_error is not None
    assert report.sample_error >= 0.0
    assert len(report.attributions) == 9
    assert report.attributions[0].rank == 0
    assert report.attributed_total == pytest.approx(
        report.candidate_reward - report.baseline_reward
    )


def test_attribute_knob_policy_rejects_mismatched_candidate_shapes() -> None:
    with pytest.raises(ValueError, match="same knob shape"):
        attribute_knob_policy(
            KnobPolicyCandidate(K=np.asarray([1.0, 2.0], dtype=np.float64)),
            KnobPolicyCandidate(K=0.0),
            _score,
        )


def test_knob_attribution_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="exact_max_knobs"):
        KnobAttributionConfig(exact_max_knobs=0)

    with pytest.raises(ValueError, match="sample_permutations"):
        KnobAttributionConfig(sample_permutations=0)
