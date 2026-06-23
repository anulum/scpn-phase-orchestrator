# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — per-knob attribution tests

"""Tests for per-knob Shapley attribution of autotune candidates.

The Shapley implementation is checked against the four defining axioms
(efficiency, symmetry, dummy, additivity over components), against an
independent permutation-based reference, and on the sampled path, array-valued
knobs, inactive knobs, and the input-validation paths.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping

import numpy as np
import pytest

from scpn_phase_orchestrator.autotune.knob_attribution import (
    KnobAttributionConfig,
    attribute_knob_policy,
)
from scpn_phase_orchestrator.autotune.reward import (
    AutotuneRewardReport,
    KnobPolicyCandidate,
    RewardConfig,
    RewardObservation,
)

_OBSERVATION = RewardObservation(coherence=0.0)
_CONFIG = RewardConfig()


def _report(
    candidate: KnobPolicyCandidate, components: dict[str, float]
) -> AutotuneRewardReport:
    return AutotuneRewardReport(
        reward=float(sum(components.values())),
        components=components,
        candidate=candidate,
        observation=_OBSERVATION,
        config=_CONFIG,
    )


def _linear_interaction_evaluator() -> Callable[
    [KnobPolicyCandidate], AutotuneRewardReport
]:
    """Reward = linear knob terms plus one alpha*zeta interaction component."""

    def evaluate(candidate: KnobPolicyCandidate) -> AutotuneRewardReport:
        alpha = float(candidate.alpha)
        zeta = float(candidate.zeta)
        weight = candidate.channel_weights[0] if candidate.channel_weights else 0.0
        linear = 2.0 * alpha + 3.0 * zeta + 1.5 * weight
        interaction = 4.0 * alpha * zeta
        return _report(candidate, {"linear": linear, "interaction": interaction})

    return evaluate


def _permutation_reference(
    active: list[str],
    value: Callable[[frozenset[str]], float],
) -> dict[str, float]:
    """Independent permutation-definition Shapley estimate (exact for small sets)."""
    phi = dict.fromkeys(active, 0.0)
    permutations = list(itertools.permutations(active))
    for order in permutations:
        coalition: set[str] = set()
        previous = value(frozenset())
        for knob in order:
            coalition.add(knob)
            current = value(frozenset(coalition))
            phi[knob] += current - previous
            previous = current
    return {knob: phi[knob] / len(permutations) for knob in active}


def _coalition_value_factory(
    candidate: KnobPolicyCandidate,
    baseline: KnobPolicyCandidate,
    evaluate: Callable[[KnobPolicyCandidate], AutotuneRewardReport],
) -> Callable[[frozenset[str]], float]:
    candidate_values = {
        "alpha": float(candidate.alpha),
        "zeta": float(candidate.zeta),
        "channel_weights[0]": candidate.channel_weights[0],
    }
    baseline_values = {
        "alpha": float(baseline.alpha),
        "zeta": float(baseline.zeta),
        "channel_weights[0]": baseline.channel_weights[0],
    }

    def value(coalition: frozenset[str]) -> float:
        merged = dict(baseline_values)
        for knob in coalition:
            merged[knob] = candidate_values[knob]
        rebuilt = KnobPolicyCandidate(
            alpha=merged["alpha"],
            zeta=merged["zeta"],
            channel_weights=(merged["channel_weights[0]"],),
        )
        return evaluate(rebuilt).reward

    return value


def test_efficiency_axiom_exact() -> None:
    candidate = KnobPolicyCandidate(alpha=1.0, zeta=2.0, channel_weights=(0.5,))
    baseline = KnobPolicyCandidate(alpha=0.0, zeta=0.0, channel_weights=(0.0,))
    report = attribute_knob_policy(candidate, baseline, _linear_interaction_evaluator())
    assert report.method == "exact"
    assert report.sample_error is None
    spread = report.candidate_reward - report.baseline_reward
    assert report.attributed_total == pytest.approx(spread, abs=1e-9)


def test_exact_matches_permutation_reference() -> None:
    candidate = KnobPolicyCandidate(alpha=1.0, zeta=2.0, channel_weights=(0.5,))
    baseline = KnobPolicyCandidate(alpha=0.0, zeta=0.0, channel_weights=(0.0,))
    evaluate = _linear_interaction_evaluator()
    report = attribute_knob_policy(candidate, baseline, evaluate)
    reference = _permutation_reference(
        ["alpha", "zeta", "channel_weights[0]"],
        _coalition_value_factory(candidate, baseline, evaluate),
    )
    for item in report.attributions:
        assert item.shapley_total == pytest.approx(reference[item.knob], abs=1e-9)


def test_per_component_shapley_sums_to_total() -> None:
    candidate = KnobPolicyCandidate(alpha=1.0, zeta=2.0, channel_weights=(0.5,))
    baseline = KnobPolicyCandidate(alpha=0.0, zeta=0.0, channel_weights=(0.0,))
    report = attribute_knob_policy(candidate, baseline, _linear_interaction_evaluator())
    for item in report.attributions:
        component_sum = sum(item.shapley_components.values())
        assert component_sum == pytest.approx(item.shapley_total, abs=1e-9)


def test_dummy_knob_gets_zero_attribution() -> None:
    # channel_weights[0] differs but the evaluator ignores it -> dummy axiom.
    def evaluate(candidate: KnobPolicyCandidate) -> AutotuneRewardReport:
        value = 2.0 * float(candidate.alpha) + 3.0 * float(candidate.zeta)
        return _report(candidate, {"main": value})

    candidate = KnobPolicyCandidate(alpha=1.0, zeta=1.0, channel_weights=(9.0,))
    baseline = KnobPolicyCandidate(alpha=0.0, zeta=0.0, channel_weights=(0.0,))
    report = attribute_knob_policy(candidate, baseline, evaluate)
    dummy = next(i for i in report.attributions if i.knob == "channel_weights[0]")
    assert dummy.shapley_total == pytest.approx(0.0, abs=1e-12)
    assert dummy.marginal_total == pytest.approx(0.0, abs=1e-12)


def test_symmetry_axiom_equal_contributors() -> None:
    def evaluate(candidate: KnobPolicyCandidate) -> AutotuneRewardReport:
        value = 5.0 * float(candidate.alpha) + 5.0 * float(candidate.zeta)
        return _report(candidate, {"main": value})

    candidate = KnobPolicyCandidate(alpha=1.0, zeta=1.0)
    baseline = KnobPolicyCandidate(alpha=0.0, zeta=0.0)
    report = attribute_knob_policy(candidate, baseline, evaluate)
    contributions = {item.knob: item.shapley_total for item in report.attributions}
    assert contributions["alpha"] == pytest.approx(contributions["zeta"], abs=1e-12)


def test_marginal_is_leave_one_out() -> None:
    candidate = KnobPolicyCandidate(alpha=1.0, zeta=2.0, channel_weights=(0.5,))
    baseline = KnobPolicyCandidate(alpha=0.0, zeta=0.0, channel_weights=(0.0,))
    evaluate = _linear_interaction_evaluator()
    report = attribute_knob_policy(candidate, baseline, evaluate)
    value = _coalition_value_factory(candidate, baseline, evaluate)
    full = value(frozenset({"alpha", "zeta", "channel_weights[0]"}))
    for item in report.attributions:
        without = value(
            frozenset({"alpha", "zeta", "channel_weights[0]"}) - {item.knob}
        )
        assert item.marginal_total == pytest.approx(full - without, abs=1e-9)


def test_ranking_is_by_descending_absolute_shapley() -> None:
    candidate = KnobPolicyCandidate(alpha=1.0, zeta=2.0, channel_weights=(0.5,))
    baseline = KnobPolicyCandidate(alpha=0.0, zeta=0.0, channel_weights=(0.0,))
    report = attribute_knob_policy(candidate, baseline, _linear_interaction_evaluator())
    magnitudes = [abs(item.shapley_total) for item in report.attributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert [item.rank for item in report.attributions] == list(range(len(magnitudes)))


def test_inactive_knobs_are_omitted() -> None:
    candidate = KnobPolicyCandidate(alpha=1.0, zeta=0.0, channel_weights=(0.0,))
    baseline = KnobPolicyCandidate(alpha=0.0, zeta=0.0, channel_weights=(0.0,))
    report = attribute_knob_policy(candidate, baseline, _linear_interaction_evaluator())
    assert {item.knob for item in report.attributions} == {"alpha"}


def test_no_active_knobs_returns_empty_report() -> None:
    candidate = KnobPolicyCandidate(alpha=1.0, zeta=2.0)
    report = attribute_knob_policy(
        candidate, candidate, _linear_interaction_evaluator()
    )
    assert report.attributions == ()
    assert report.method == "exact"
    assert report.attributed_total == pytest.approx(0.0, abs=1e-12)


def test_attributes_uppercase_coupling_and_psi_knobs() -> None:
    # K and Psi are uppercase-led fields; they must be flattened and rebuilt too.
    def evaluate(candidate: KnobPolicyCandidate) -> AutotuneRewardReport:
        value = 3.0 * float(candidate.K) + 5.0 * float(candidate.Psi)
        return _report(candidate, {"main": value})

    candidate = KnobPolicyCandidate(K=1.0, Psi=1.0)
    baseline = KnobPolicyCandidate(K=0.0, Psi=0.0)
    report = attribute_knob_policy(candidate, baseline, evaluate)
    by_knob = {item.knob: item.shapley_total for item in report.attributions}
    assert by_knob["K"] == pytest.approx(3.0, abs=1e-9)
    assert by_knob["Psi"] == pytest.approx(5.0, abs=1e-9)


def test_array_valued_knobs_are_flattened_and_rebuilt() -> None:
    def evaluate(candidate: KnobPolicyCandidate) -> AutotuneRewardReport:
        alpha = np.asarray(candidate.alpha, dtype=float).ravel()
        value = float(2.0 * alpha[0] + 7.0 * alpha[1])
        return _report(candidate, {"main": value})

    candidate = KnobPolicyCandidate(alpha=np.array([1.0, 1.0]), zeta=0.0)
    baseline = KnobPolicyCandidate(alpha=np.array([0.0, 0.0]), zeta=0.0)
    report = attribute_knob_policy(candidate, baseline, evaluate)
    by_knob = {item.knob: item.shapley_total for item in report.attributions}
    assert by_knob["alpha[0]"] == pytest.approx(2.0, abs=1e-9)
    assert by_knob["alpha[1]"] == pytest.approx(7.0, abs=1e-9)


def test_sampled_path_approximates_exact() -> None:
    rng = np.random.default_rng(0)
    weights = rng.normal(size=14)

    def evaluate(candidate: KnobPolicyCandidate) -> AutotuneRewardReport:
        gains = np.asarray(candidate.cross_channel_gains, dtype=float)
        value = float(weights @ gains)
        return _report(candidate, {"main": value})

    candidate = KnobPolicyCandidate(cross_channel_gains=tuple([1.0] * 14))
    baseline = KnobPolicyCandidate(cross_channel_gains=tuple([0.0] * 14))
    config = KnobAttributionConfig(exact_max_knobs=12, sample_permutations=400, seed=1)
    report = attribute_knob_policy(candidate, baseline, evaluate, config=config)
    assert report.method == "sampled"
    assert report.sample_error is not None and report.sample_error >= 0.0
    # A purely additive game gives each knob its own weight as the exact Shapley
    # value; the sampled estimate must land close.
    for item in report.attributions:
        index = int(item.knob.split("[")[1].rstrip("]"))
        assert item.shapley_total == pytest.approx(weights[index], abs=1e-9)


def test_sampled_single_permutation_reports_zero_error() -> None:
    def evaluate(candidate: KnobPolicyCandidate) -> AutotuneRewardReport:
        value = float(sum(candidate.cross_channel_gains))
        return _report(candidate, {"main": value})

    candidate = KnobPolicyCandidate(cross_channel_gains=tuple([1.0] * 13))
    baseline = KnobPolicyCandidate(cross_channel_gains=tuple([0.0] * 13))
    config = KnobAttributionConfig(exact_max_knobs=12, sample_permutations=1)
    report = attribute_knob_policy(candidate, baseline, evaluate, config=config)
    assert report.method == "sampled"
    assert report.sample_error == pytest.approx(0.0, abs=1e-12)


def test_shape_mismatch_raises() -> None:
    candidate = KnobPolicyCandidate(alpha=1.0, channel_weights=(0.5,))
    baseline = KnobPolicyCandidate(alpha=0.0, channel_weights=(0.0, 0.0))
    with pytest.raises(ValueError, match="same knob shape"):
        attribute_knob_policy(candidate, baseline, _linear_interaction_evaluator())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"exact_max_knobs": 0}, "exact_max_knobs"),
        ({"sample_permutations": 0}, "sample_permutations"),
    ],
)
def test_config_validation(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        KnobAttributionConfig(**kwargs)


def test_audit_records_are_json_ready_and_sorted() -> None:
    candidate = KnobPolicyCandidate(alpha=1.0, zeta=2.0, channel_weights=(0.5,))
    baseline = KnobPolicyCandidate(alpha=0.0, zeta=0.0, channel_weights=(0.0,))
    report = attribute_knob_policy(candidate, baseline, _linear_interaction_evaluator())
    record = report.to_audit_record()
    assert record["method"] == "exact"
    first: Mapping[str, object] = record["attributions"][0]  # type: ignore[index]
    components = first["shapley_components"]
    assert isinstance(components, dict)
    assert list(components) == sorted(components)
