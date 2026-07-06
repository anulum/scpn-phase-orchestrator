# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid modal-growth matched-false-alarm significance

"""The matched-false-alarm significance layer over the grid modal-growth detector.

The detector itself — the passive growth-rate primitives — is the product-side
``scpn_phase_orchestrator.monitor.grid_modal_growth``; they are re-exported here so the
offline benchmarks keep one import. This module adds the *offline evaluation* around
them: it calibrates the per-segment growth rate ``σ`` to a matched false alarm on damped
null segments and tests the instability alarm count with the shared label-permutation
core (:func:`~bench.early_warning_domain.permutation_significance_from_alarms`), so the
modal detector's p-value is directly comparable to the generic SCPN suite on the *same*
segments.

On the PSML 23-bus corpus (Zheng et al. 2021), labelling transitions by *disturbance
type* — generator trips (instability-prone) against damped bus faults and branch trips,
a label independent of the growth statistic, so the comparison is not circular — the
growth-rate detector leads far more transitions than any generic member at the same
operating point (see :mod:`bench.grid_modal_head_to_head`).

References
----------
* Zheng et al. 2021 — the PSML power-system dataset (23-bus millisecond-level PMU
  measurements) with disturbance-type annotations.
* Kundur 1994, *Power System Stability and Control* — small-signal (modal) stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bench.early_warning_domain import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
    PermutationSignificance,
    calibrate_score_threshold,
    permutation_significance_from_alarms,
)
from scpn_phase_orchestrator.monitor.grid_modal_growth import (
    DEFAULT_AGGREGATION,
    DEFAULT_RECENCY_TOP,
    FloatArray,
    cross_bus_deviation,
    envelope_growth_rate,
    modal_growth_score,
    per_bus_deviation,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_AGGREGATION",
    "DEFAULT_RECENCY_TOP",
    "ModalGrowthSignificance",
    "cross_bus_deviation",
    "envelope_growth_rate",
    "modal_growth_score",
    "modal_growth_significance",
    "per_bus_deviation",
]


@dataclass(frozen=True)
class ModalGrowthSignificance:
    """The grid modal-growth detector's matched-false-alarm result on a corpus.

    Attributes
    ----------
    aggregation : str
        The channel aggregation used (``"mean"`` or ``"focal"``).
    recency_top : float
        The recency weighting used in the growth-rate fit.
    score_threshold : float
        The matched-false-alarm growth-rate threshold set on the damped null segments.
    achieved_false_alarm : float
        The fraction of null segments that alarmed at that threshold.
    significance : PermutationSignificance
        The label-permutation significance of the instability alarm count — the same
        test the generic SCPN suite is scored by.
    """

    aggregation: str
    recency_top: float
    score_threshold: float
    achieved_false_alarm: float
    significance: PermutationSignificance

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the grid modal-growth result."""
        return {
            "detector": f"modal_envelope_growth_rate_{self.aggregation}",
            "aggregation": self.aggregation,
            "recency_top": self.recency_top,
            "score_threshold": self.score_threshold,
            "achieved_false_alarm": self.achieved_false_alarm,
            "significance": self.significance.to_audit_record(),
        }


def modal_growth_significance(
    transition_segments: Sequence[FloatArray],
    null_segments: Sequence[FloatArray],
    *,
    rate: float,
    aggregation: str = DEFAULT_AGGREGATION,
    recency_top: float = DEFAULT_RECENCY_TOP,
    target_fa: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> ModalGrowthSignificance:
    """Score the modal-growth detector through the matched-false-alarm protocol.

    Each pre-onset transition segment and damped null segment is reduced to its modal
    growth rate ``σ`` (:func:`modal_growth_score`) under the chosen aggregation and
    recency weighting; the threshold is calibrated on the null rates to a matched false
    alarm; a segment alarms when its ``σ`` meets the threshold; and the instability
    alarm count is tested for significance by the shared label-permutation core, so the
    p-value is directly comparable to the generic suite.

    Parameters
    ----------
    transition_segments : sequence of FloatArray
        Each instability transition's pre-onset segment, shape ``(buses, samples)``.
    null_segments : sequence of FloatArray
        Each damped-disturbance null segment, same shape convention.
    rate : float
        Sampling rate in Hz.
    aggregation : str
        ``"mean"`` (whole network) or ``"focal"`` (most unstable bus).
    recency_top : float
        Recency weighting passed to :func:`envelope_growth_rate`.
    target_fa : float
        Target false-alarm rate the growth-rate threshold is held at or below.
    n_permutations : int
        Number of random relabellings for the significance test.
    seed : int
        Seed of the resampling, so the p-value is reproducible.

    Returns
    -------
    ModalGrowthSignificance
        The aggregation, recency weighting, calibrated threshold, achieved false-alarm
        rate, and permutation significance.

    Raises
    ------
    ValueError
        If either segment set is empty.
    """
    if not transition_segments:
        raise ValueError("transition_segments must not be empty")
    if not null_segments:
        raise ValueError("null_segments must not be empty")
    transition_scores = [
        modal_growth_score(
            s, rate=rate, aggregation=aggregation, recency_top=recency_top
        )
        for s in transition_segments
    ]
    null_scores = [
        modal_growth_score(
            s, rate=rate, aggregation=aggregation, recency_top=recency_top
        )
        for s in null_segments
    ]
    threshold = calibrate_score_threshold(null_scores, target_fa=target_fa)
    transition_alarms = [score >= threshold for score in transition_scores]
    null_alarms = [score >= threshold for score in null_scores]
    achieved = float(np.mean(null_alarms))
    significance = permutation_significance_from_alarms(
        transition_alarms, null_alarms, n_permutations=n_permutations, seed=seed
    )
    return ModalGrowthSignificance(
        aggregation=aggregation,
        recency_top=recency_top,
        score_threshold=threshold,
        achieved_false_alarm=achieved,
        significance=significance,
    )
