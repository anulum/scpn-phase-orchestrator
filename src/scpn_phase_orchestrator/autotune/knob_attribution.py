# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — per-knob attribution for autotune candidates

"""Per-knob attribution — *why this knob* — for autotune policy candidates.

A reward report scores a candidate (a ``KnobPolicyCandidate``) as a whole and
breaks the score into reward *components*. It does not say which *knob* earned
the score: a reviewer asked to trust a candidate wants to know how
much of the gain came from ``alpha`` versus ``zeta`` versus each channel weight,
and whether any knob is doing nothing or actively hurting.

This module answers that. Given a candidate, a baseline to attribute against, and
a value function that scores any candidate, it computes each knob's contribution
to the total reward and to every component by two complementary measures:

* **Shapley value** — the knob's average marginal contribution over every order
  in which the knobs could be switched from the baseline to the candidate. It is
  the unique attribution that is efficient (the contributions sum exactly to the
  candidate-minus-baseline reward), symmetric, and assigns zero to a knob that
  never changes any value function. It is computed exactly over all coalitions
  for a small number of active knobs and by deterministic Monte-Carlo sampling,
  with a reported standard error, above a configurable threshold.
* **Marginal (leave-one-out)** — the reward lost when that one knob alone is
  reset to its baseline. It is cheap and intuitive but, unlike the Shapley value,
  does not credit interactions consistently.

The value function is supplied by the caller and is treated as a black box, so
attribution works for a fixed-observation reward, a replay evaluator that re-runs
the candidate over recorded data, or any other scorer. The module performs no
control actuation and has no side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import factorial, sqrt

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.autotune.reward import (
    AutotuneRewardReport,
    KnobPolicyCandidate,
)

__all__ = [
    "KnobAttribution",
    "KnobAttributionConfig",
    "KnobAttributionReport",
    "attribute_knob_policy",
]

CandidateEvaluator = Callable[[KnobPolicyCandidate], AutotuneRewardReport]
"""A scorer mapping a candidate to its reward report (reward plus components)."""

_TOTAL_KEY = "__total__"


@dataclass(frozen=True)
class KnobAttribution:
    """Attribution of one scalar knob's effect on the reward.

    Parameters
    ----------
    knob : str
        Stable name of the knob, e.g. ``"alpha"``, ``"zeta[2]"`` or
        ``"channel_weights[0]"``.
    baseline_value : float
        The knob's value in the baseline candidate.
    candidate_value : float
        The knob's value in the attributed candidate.
    shapley_total : float
        The knob's Shapley contribution to the total reward.
    marginal_total : float
        The reward delta from resetting this knob alone to its baseline value
        (the leave-one-out contribution).
    shapley_components : Mapping[str, float]
        The knob's Shapley contribution to each reward component, keyed by the
        component name.
    rank : int
        The knob's rank by descending absolute Shapley contribution, starting
        at ``0`` for the most influential knob.
    """

    knob: str
    baseline_value: float
    candidate_value: float
    shapley_total: float
    marginal_total: float
    shapley_components: Mapping[str, float]
    rank: int

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-ready, deterministic record of this attribution.

        Returns
        -------
        dict[str, object]
            A mapping with the knob name, its baseline and candidate values, the
            Shapley and marginal totals, the per-component Shapley contributions
            sorted by component name, and the rank.
        """
        return {
            "knob": self.knob,
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
            "shapley_total": self.shapley_total,
            "marginal_total": self.marginal_total,
            "shapley_components": {
                name: self.shapley_components[name]
                for name in sorted(self.shapley_components)
            },
            "rank": self.rank,
        }


@dataclass(frozen=True)
class KnobAttributionReport:
    """The full per-knob attribution of a candidate against a baseline.

    Parameters
    ----------
    candidate_reward : float
        The total reward of the attributed candidate.
    baseline_reward : float
        The total reward of the baseline candidate.
    attributions : tuple[KnobAttribution, ...]
        One entry per active knob (a knob whose candidate value differs from its
        baseline value), ordered by ascending rank.
    method : str
        ``"exact"`` if Shapley values were computed over all coalitions, or
        ``"sampled"`` if they were estimated by Monte-Carlo permutation sampling.
    sample_error : float | None
        The largest per-knob standard error of the sampled Shapley estimate, or
        ``None`` when ``method`` is ``"exact"``.
    """

    candidate_reward: float
    baseline_reward: float
    attributions: tuple[KnobAttribution, ...]
    method: str
    sample_error: float | None

    @property
    def attributed_total(self) -> float:
        """Return the sum of the Shapley totals over all active knobs.

        Returns
        -------
        float
            For the exact method this equals ``candidate_reward -
            baseline_reward`` to within floating-point tolerance (the Shapley
            efficiency axiom).
        """
        return float(sum(item.shapley_total for item in self.attributions))

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-ready, deterministic record of the report.

        Returns
        -------
        dict[str, object]
            A mapping with the candidate and baseline rewards, the ordered list
            of per-knob audit records, the method, and the sampling error.
        """
        return {
            "candidate_reward": self.candidate_reward,
            "baseline_reward": self.baseline_reward,
            "method": self.method,
            "sample_error": self.sample_error,
            "attributions": [item.to_audit_record() for item in self.attributions],
        }


@dataclass(frozen=True)
class KnobAttributionConfig:
    """Settings controlling exact-versus-sampled Shapley attribution.

    Parameters
    ----------
    exact_max_knobs : int
        The largest number of active knobs for which Shapley values are computed
        exactly over all coalitions. Above this, sampling is used. Must be at
        least ``1``.
    sample_permutations : int
        The number of random knob orderings drawn when sampling. Must be at
        least ``1``.
    seed : int
        Seed for the deterministic permutation sampler.
    """

    exact_max_knobs: int = 12
    sample_permutations: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration bounds.

        Raises
        ------
        ValueError
            If ``exact_max_knobs`` or ``sample_permutations`` is below ``1``.
        """
        if self.exact_max_knobs < 1:
            raise ValueError("exact_max_knobs must be at least 1")
        if self.sample_permutations < 1:
            raise ValueError("sample_permutations must be at least 1")


def _flatten_field(name: str, value: object) -> list[tuple[str, float]]:
    """Flatten one candidate field into ordered named scalar knobs."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [(name, float(value))]
    array = np.asarray(value, dtype=float).ravel()
    return [(f"{name}[{index}]", float(entry)) for index, entry in enumerate(array)]


def _flatten_candidate(candidate: KnobPolicyCandidate) -> list[tuple[str, float]]:
    """Flatten a candidate into an ordered list of ``(name, value)`` knobs."""
    knobs: list[tuple[str, float]] = []
    knobs.extend(_flatten_field("alpha", candidate.alpha))
    knobs.extend(_flatten_field("zeta", candidate.zeta))
    for index, weight in enumerate(candidate.channel_weights):
        knobs.append((f"channel_weights[{index}]", float(weight)))
    for index, gain in enumerate(candidate.cross_channel_gains):
        knobs.append((f"cross_channel_gains[{index}]", float(gain)))
    return knobs


def _rebuild_field(
    name: str,
    template: object,
    values: Mapping[str, float],
) -> float | NDArray[np.float64]:
    """Reconstruct one candidate field from per-knob values."""
    if isinstance(template, (int, float)) and not isinstance(template, bool):
        return values[name]
    length = int(np.asarray(template, dtype=float).ravel().shape[0])
    return np.array(
        [values[f"{name}[{index}]"] for index in range(length)], dtype=float
    )


def _rebuild_candidate(
    template: KnobPolicyCandidate,
    values: Mapping[str, float],
) -> KnobPolicyCandidate:
    """Rebuild a candidate of the template's shape from per-knob values."""
    alpha = _rebuild_field("alpha", template.alpha, values)
    zeta = _rebuild_field("zeta", template.zeta, values)
    channel_weights = tuple(
        values[f"channel_weights[{index}]"]
        for index in range(len(template.channel_weights))
    )
    cross_channel_gains = tuple(
        values[f"cross_channel_gains[{index}]"]
        for index in range(len(template.cross_channel_gains))
    )
    return KnobPolicyCandidate(
        alpha=alpha,
        zeta=zeta,
        channel_weights=channel_weights,
        cross_channel_gains=cross_channel_gains,
    )


class _CoalitionValuer:
    """Memoised value function over coalitions of active knobs."""

    def __init__(
        self,
        *,
        template: KnobPolicyCandidate,
        candidate_values: Mapping[str, float],
        baseline_values: Mapping[str, float],
        active: Sequence[str],
        evaluate: CandidateEvaluator,
    ) -> None:
        self._template = template
        self._candidate_values = candidate_values
        self._baseline_values = baseline_values
        self._active = tuple(active)
        self._evaluate = evaluate
        self._cache: dict[frozenset[str], dict[str, float]] = {}

    def value(self, coalition: frozenset[str]) -> dict[str, float]:
        """Return total and per-component value for a coalition of active knobs."""
        cached = self._cache.get(coalition)
        if cached is not None:
            return cached
        values = dict(self._baseline_values)
        for knob in coalition:
            values[knob] = self._candidate_values[knob]
        report = self._evaluate(_rebuild_candidate(self._template, values))
        scored = dict(report.components)
        scored[_TOTAL_KEY] = report.reward
        self._cache[coalition] = scored
        return scored


def _exact_shapley(
    active: Sequence[str],
    valuer: _CoalitionValuer,
    keys: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Exact Shapley values per knob and per value key over all coalitions."""
    others_of = {knob: [peer for peer in active if peer != knob] for knob in active}
    total = len(active)
    weights = {
        size: factorial(size) * factorial(total - size - 1) / factorial(total)
        for size in range(total)
    }
    result: dict[str, dict[str, float]] = {
        knob: dict.fromkeys(keys, 0.0) for knob in active
    }
    for knob in active:
        peers = others_of[knob]
        for mask in range(1 << len(peers)):
            subset = frozenset(
                peers[bit] for bit in range(len(peers)) if mask & (1 << bit)
            )
            with_knob = valuer.value(subset | {knob})
            without_knob = valuer.value(subset)
            weight = weights[len(subset)]
            knob_result = result[knob]
            for key in keys:
                knob_result[key] += weight * (with_knob[key] - without_knob[key])
    return result


def _sampled_shapley(
    active: Sequence[str],
    valuer: _CoalitionValuer,
    keys: Sequence[str],
    *,
    permutations: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], float]:
    """Monte-Carlo Shapley estimate with the largest per-knob standard error."""
    rng = np.random.default_rng(seed)
    order = list(active)
    sums = {knob: dict.fromkeys(keys, 0.0) for knob in active}
    squares = {knob: dict.fromkeys(keys, 0.0) for knob in active}
    for _ in range(permutations):
        rng.shuffle(order)
        coalition: set[str] = set()
        previous = valuer.value(frozenset())
        for knob in order:
            coalition.add(knob)
            current = valuer.value(frozenset(coalition))
            for key in keys:
                contribution = current[key] - previous[key]
                sums[knob][key] += contribution
                squares[knob][key] += contribution * contribution
            previous = current
    result: dict[str, dict[str, float]] = {}
    max_error = 0.0
    for knob in active:
        result[knob] = {key: sums[knob][key] / permutations for key in keys}
        if permutations > 1:
            for key in keys:
                mean = result[knob][key]
                variance = max(0.0, squares[knob][key] / permutations - mean * mean)
                error = sqrt(variance / permutations)
                max_error = max(max_error, error)
    return result, max_error


def attribute_knob_policy(
    candidate: KnobPolicyCandidate,
    baseline: KnobPolicyCandidate,
    evaluate: CandidateEvaluator,
    *,
    config: KnobAttributionConfig | None = None,
) -> KnobAttributionReport:
    """Attribute a candidate's reward to its individual knobs against a baseline.

    Each knob that differs between ``candidate`` and ``baseline`` is credited
    with its Shapley contribution to the total reward and to every reward
    component, plus its leave-one-out marginal contribution. Knobs that are
    identical in both candidates are inactive: they contribute nothing by
    construction and are omitted from the report.

    Parameters
    ----------
    candidate : KnobPolicyCandidate
        The candidate whose reward is being explained.
    baseline : KnobPolicyCandidate
        The reference candidate to attribute against. It must have the same
        shape as ``candidate`` (matching scalar-versus-array fields and tuple
        lengths).
    evaluate : CandidateEvaluator
        A side-effect-free scorer returning a reward report for any candidate of
        the shared shape. It is called once per distinct coalition and the
        results are memoised.
    config : KnobAttributionConfig | None
        Exact-versus-sampled settings. Defaults to
        :class:`KnobAttributionConfig`.

    Returns
    -------
    KnobAttributionReport
        The candidate and baseline rewards and one attribution per active knob,
        ordered by descending absolute Shapley contribution.

    Raises
    ------
    ValueError
        If ``candidate`` and ``baseline`` do not share the same knob shape.
    """
    active_config = config or KnobAttributionConfig()
    candidate_values = dict(_flatten_candidate(candidate))
    baseline_values = dict(_flatten_candidate(baseline))
    if candidate_values.keys() != baseline_values.keys():
        raise ValueError("candidate and baseline must share the same knob shape")

    active = [
        knob
        for knob in candidate_values
        if candidate_values[knob] != baseline_values[knob]
    ]

    valuer = _CoalitionValuer(
        template=candidate,
        candidate_values=candidate_values,
        baseline_values=baseline_values,
        active=active,
        evaluate=evaluate,
    )
    component_keys = sorted(
        key for key in valuer.value(frozenset()) if key != _TOTAL_KEY
    )
    keys = [_TOTAL_KEY, *component_keys]

    full = valuer.value(frozenset(active))
    empty = valuer.value(frozenset())
    candidate_reward = full[_TOTAL_KEY]
    baseline_reward = empty[_TOTAL_KEY]

    if not active:
        return KnobAttributionReport(
            candidate_reward=candidate_reward,
            baseline_reward=baseline_reward,
            attributions=(),
            method="exact",
            sample_error=None,
        )

    if len(active) <= active_config.exact_max_knobs:
        shapley = _exact_shapley(active, valuer, keys)
        method = "exact"
        sample_error: float | None = None
    else:
        shapley, sample_error = _sampled_shapley(
            active,
            valuer,
            keys,
            permutations=active_config.sample_permutations,
            seed=active_config.seed,
        )
        method = "sampled"

    marginals = {
        knob: full[_TOTAL_KEY] - valuer.value(frozenset(active) - {knob})[_TOTAL_KEY]
        for knob in active
    }

    ordered = sorted(active, key=lambda knob: -abs(shapley[knob][_TOTAL_KEY]))
    attributions = tuple(
        KnobAttribution(
            knob=knob,
            baseline_value=baseline_values[knob],
            candidate_value=candidate_values[knob],
            shapley_total=shapley[knob][_TOTAL_KEY],
            marginal_total=marginals[knob],
            shapley_components={key: shapley[knob][key] for key in component_keys},
            rank=rank,
        )
        for rank, knob in enumerate(ordered)
    )
    return KnobAttributionReport(
        candidate_reward=candidate_reward,
        baseline_reward=baseline_reward,
        attributions=attributions,
        method=method,
        sample_error=sample_error,
    )
