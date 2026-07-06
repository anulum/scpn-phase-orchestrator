# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — dynamical network biomarker (DNB) early-warning detector

"""Dynamical network biomarkers, run through the same matched-false-alarm protocol.

The four physical domains (brain, heart, grid, climate) are *many time-samples × few
oscillator nodes* scored by a sliding-window critical-slowing-down monitor. Molecular
early warning of a cell-fate or disease transition is the opposite regime — *few
timepoints × many genes* — and its canonical detector is the **dynamical network
biomarker** (DNB) of Chen, Liu, Liu & Aihara 2012 (*Sci Rep* 2:342): as a biological
system nears a tipping point a group of genes (the DNB module) shows, over the sample
population at that timepoint, a sharp rise in variability and in-module correlation and
a fall in its correlation with the rest of the network. Two published index forms are
implemented here:

* the **bulk composite index** :func:`dnb_index` — ``SD_in · PCC_in / PCC_out`` over a
  chosen module against the rest of the network (Chen/Aihara), with the module found by
  :func:`select_dnb_module`; used for bulk expression corpora such as GSE2565;
* the **single-cell transition index** :func:`single_cell_transition_index` — the ratio
  of mean gene–gene to mean cell–cell correlation over a curated panel (Mojtahedi et al.
  2016, *PLoS Biol* 14:e2000640), where the panel *is* the critical module so no
  selection is needed; validatable against that paper's published transition index.

Either index gives one value per timepoint; :func:`dnb_trend_score` reduces the
pre-transition rising limb to a Kendall-τ trend — the same trend statistic the
AR(1)-Kendall-τ competitor uses — and :func:`dnb_significance` calibrates a matched
false alarm on a temporally-shuffled surrogate null and runs the *identical*
label-permutation test the SCPN suite and the competitor are scored by
(:func:`~bench.early_warning_domain.permutation_significance_from_alarms`). So the DNB
p-value is directly comparable to the four physical domains: same operating point, same
significance test, only the detector and the modality differ.

References
----------
* Chen, Liu, Liu & Aihara 2012, *Sci Rep* 2:342 — the dynamical network biomarker and
  its composite index ``SD_in · PCC_in / PCC_out``.
* Mojtahedi, Skupin, Zhou, Castaño, … Huang 2016, *PLoS Biol* 14:e2000640 — the
  single-cell transition index (gene–gene over cell–cell correlation) at a leukaemic
  fate bifurcation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau

from bench.early_warning_domain import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
    PermutationSignificance,
    calibrate_score_threshold,
    permutation_significance_from_alarms,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Callable, Sequence

    #: A one-dimensional real series the detectors coerce with ``np.asarray``: a plain
    #: float sequence or a float array.
    RealSeries = Sequence[float] | NDArray[np.float64]
    #: A corpus of such series.
    RealSeriesCorpus = Sequence[RealSeries]

FloatArray = NDArray[np.float64]

_CORRELATION_FLOOR = 1.0e-9
_DEFAULT_CANDIDATE_FRACTION = 0.1
_DEFAULT_MIN_MODULE = 3

__all__ = [
    "DnbSignificance",
    "dnb_index",
    "dnb_regression_slope",
    "dnb_significance",
    "dnb_trend_score",
    "select_dnb_module",
    "single_cell_transition_index",
]


def dnb_index(expression: FloatArray, module: Sequence[int]) -> float:
    """Return the Chen/Aihara composite DNB index of a gene module.

    Over the samples at one condition/timepoint, the module's members show — near a
    tipping point — high per-gene variability, tight in-module correlation, and loose
    correlation with the rest of the network. The composite index quantifies this as
    ``SD_in · PCC_in / PCC_out``, where ``SD_in`` is the mean per-gene standard
    deviation of the module, ``PCC_in`` the mean absolute Pearson correlation among
    module genes, and ``PCC_out`` the mean absolute correlation between module genes and
    the rest.

    Parameters
    ----------
    expression : FloatArray
        Expression matrix at one timepoint, shape ``(samples, genes)``.
    module : sequence of int
        Column indices of the module genes; at least two, all distinct and in range,
        and not naming every gene (the rest of the network must be non-empty).

    Returns
    -------
    float
        The composite DNB index; larger as the module approaches the transition.

    Raises
    ------
    ValueError
        If ``expression`` is not a two-dimensional matrix of at least two samples, or
        the module is malformed (fewer than two genes, out of range, or leaving no
        outside genes).
    """
    values = np.asarray(expression, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("expression must be two-dimensional (samples × genes)")
    n_samples, n_genes = values.shape
    if n_samples < 2:
        raise ValueError(
            "expression needs at least two samples for a standard deviation"
        )
    member = _module_indices(module, n_genes)
    outside = [gene for gene in range(n_genes) if gene not in set(member)]
    if not outside:
        raise ValueError("module must leave at least one gene outside for PCC_out")
    sd_in = float(np.mean(np.std(values[:, member], axis=0, ddof=1)))
    correlation = _gene_correlation(values)
    pcc_in = _mean_abs_offdiagonal(correlation[np.ix_(member, member)])
    pcc_out = float(np.mean(np.abs(correlation[np.ix_(member, outside)])))
    return sd_in * pcc_in / max(pcc_out, _CORRELATION_FLOOR)


def select_dnb_module(
    expression_by_timepoint: Sequence[FloatArray],
    *,
    transition_index: int,
    candidate_fraction: float = _DEFAULT_CANDIDATE_FRACTION,
    min_module: int = _DEFAULT_MIN_MODULE,
) -> tuple[int, ...]:
    """Select the DNB module by greedy growth of the composite index at the transition.

    The Chen/Aihara signature is that the module's variability rises toward the
    transition, so the candidates are the genes whose standard deviation rises most from
    the baseline (first) timepoint to the transition timepoint. Among those, the module
    is grown greedily: seed with the two highest-rising candidates, then repeatedly add
    the candidate that most increases :func:`dnb_index` at the transition, stopping when
    no candidate improves it. This recovers the co-varying, tightly-correlated,
    loosely-outward-coupled group the composite index rewards, without the curated panel
    a single-cell study supplies.

    Parameters
    ----------
    expression_by_timepoint : sequence of FloatArray
        One ``(samples, genes)`` matrix per timepoint, sharing the same gene axis;
        at least two timepoints (a baseline and the transition).
    transition_index : int
        Index into ``expression_by_timepoint`` of the (candidate) tipping point.
    candidate_fraction : float
        Fraction of genes taken as rising-variability candidates (at least
        ``min_module`` genes are always considered).
    min_module : int
        Minimum candidate-pool size, so a small network still admits a module.

    Returns
    -------
    tuple of int
        The selected module's gene indices, sorted ascending.

    Raises
    ------
    ValueError
        If fewer than two timepoints are given, the timepoints disagree on the gene
        axis, the transition index is out of range, or fewer than two genes show a
        rising standard deviation at the transition.
    """
    frames = [np.asarray(frame, dtype=np.float64) for frame in expression_by_timepoint]
    if len(frames) < 2:
        raise ValueError("need at least two timepoints (a baseline and the transition)")
    if frames[0].ndim != 2:
        raise ValueError("each timepoint must be two-dimensional (samples × genes)")
    n_genes = int(frames[0].shape[1])
    for frame in frames:
        if frame.ndim != 2 or int(frame.shape[1]) != n_genes:
            raise ValueError("every timepoint must share the same gene axis")
    transition = frames[_index_in_range(transition_index, len(frames))]
    sd_rise = np.std(transition, axis=0, ddof=1) - np.std(frames[0], axis=0, ddof=1)
    pool_size = max(min_module, int(np.ceil(candidate_fraction * n_genes)))
    ranked = np.argsort(sd_rise)[::-1]
    candidates = [int(gene) for gene in ranked[:pool_size] if sd_rise[int(gene)] > 0.0]
    if len(candidates) < 2:
        raise ValueError("fewer than two genes show a rising standard deviation")
    module = [candidates[0], candidates[1]]
    remaining = candidates[2:]
    best = dnb_index(transition, module)
    while remaining:
        gain, gene = max(
            (dnb_index(transition, [*module, candidate]), candidate)
            for candidate in remaining
        )
        if gain <= best:
            break
        module.append(gene)
        remaining.remove(gene)
        best = gain
    return tuple(sorted(module))


def single_cell_transition_index(expression: FloatArray) -> float:
    """Return the Mojtahedi single-cell transition index of a cell population.

    The single-cell DNB analogue: over the cells at one timepoint, on the curated gene
    panel, the index is the mean *signed* gene–gene Pearson correlation (across cells)
    divided by the mean *signed* cell–cell Pearson correlation (across genes) — the
    signed average of Mojtahedi et al., not the absolute average of the bulk composite
    index. Toward the fate bifurcation the genes co-vary more (numerator rises) while
    the cells diverge (denominator falls), so the index peaks at the tipping point. The
    panel is itself the critical module, so no selection step is needed.

    Parameters
    ----------
    expression : FloatArray
        Single-cell expression on the panel at one timepoint, shape ``(cells, genes)``;
        at least two cells and two genes.

    Returns
    -------
    float
        The transition index — mean gene–gene over mean cell–cell correlation.

    Raises
    ------
    ValueError
        If ``expression`` is not a two-dimensional matrix of at least two cells and two
        genes.
    """
    values = np.asarray(expression, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("expression must be two-dimensional (cells × genes)")
    n_cells, n_genes = values.shape
    if n_cells < 2:
        raise ValueError("expression needs at least two cells")
    if n_genes < 2:
        raise ValueError("expression needs at least two genes")
    gene_gene = _mean_offdiagonal(_gene_correlation(values))
    cell_cell = _mean_offdiagonal(_gene_correlation(values.T))
    return gene_gene / max(cell_cell, _CORRELATION_FLOOR)


def dnb_trend_score(trajectory: RealSeries) -> float:
    """Return the Kendall-τ rising trend of a DNB-index trajectory.

    An operational early-warning detector sees the index only up to the present, so it
    scores the *rising limb* toward the transition. Reducing that limb to the Kendall
    rank correlation τ against timepoint index mirrors the AR(1)-Kendall-τ competitor:
    a positive τ is the DNB index climbing toward the tipping point; τ near zero or
    negative is no rise. A temporally-shuffled surrogate destroys this order, which is
    what the matched-false-alarm null in :func:`dnb_significance` exploits.

    Parameters
    ----------
    trajectory : sequence of float
        The DNB index at each pre-transition timepoint, in real time order.

    Returns
    -------
    float
        The Kendall τ of the trajectory against time, in ``[-1, 1]``; ``0.0`` when
        fewer than two timepoints are given or τ is undefined (a constant trajectory).
    """
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("trajectory must be one-dimensional")
    if values.shape[0] < 2:
        return 0.0
    tau = kendalltau(np.arange(values.shape[0]), values)[0]
    return float(tau) if np.isfinite(tau) else 0.0


def dnb_regression_slope(trajectory: RealSeries) -> float:
    """Return the ordinary-least-squares slope of a DNB-index trajectory over time.

    A continuous rising-trend statistic, the alternative to the rank-based
    :func:`dnb_trend_score` for the few timepoints a molecular corpus supplies. On four
    published timepoints a Kendall τ takes only a handful of discrete values, so a
    matched-false-alarm threshold cannot land on a clean operating point; the slope of
    the index against its (uniform) timepoint index is continuous and order-sensitive —
    a temporally-shuffled surrogate regresses to a slope near zero — so a matched false
    alarm is placed exactly. The positions are uniform ranks, as in
    :func:`dnb_trend_score`, so a non-uniform real-time spacing does not weight the fit.

    Parameters
    ----------
    trajectory : sequence of float
        The DNB index at each pre-transition timepoint, in real time order.

    Returns
    -------
    float
        The least-squares slope against timepoint index; ``0.0`` when fewer than two
        timepoints are given or the slope is undefined.
    """
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("trajectory must be one-dimensional")
    if values.shape[0] < 2:
        return 0.0
    positions = np.arange(values.shape[0], dtype=np.float64)
    slope = np.polyfit(positions, values, 1)[0]
    return float(slope) if np.isfinite(slope) else 0.0


@dataclass(frozen=True)
class DnbSignificance:
    """The DNB detector's matched-false-alarm result on a corpus of trajectories.

    Attributes
    ----------
    score_threshold : float
        The matched-false-alarm trend-score threshold calibrated on the surrogate null.
    achieved_false_alarm : float
        The fraction of null trajectories that alarmed at that threshold.
    significance : PermutationSignificance
        The label-permutation significance of the transition alarm count — the same
        test the SCPN suite and the AR(1)-Kendall-τ competitor are scored by.
    """

    score_threshold: float
    achieved_false_alarm: float
    significance: PermutationSignificance

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the DNB result.

        Returns
        -------
        dict[str, object]
            The trend-score threshold, the achieved false-alarm rate, and the sealed
            significance record.
        """
        return {
            "detector": "dnb_trend",
            "score_threshold": self.score_threshold,
            "achieved_false_alarm": self.achieved_false_alarm,
            "significance": self.significance.to_audit_record(),
        }


def dnb_significance(
    transition_trajectories: RealSeriesCorpus,
    null_trajectories: RealSeriesCorpus,
    *,
    score: Callable[[RealSeries], float] = dnb_trend_score,
    target_fa: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> DnbSignificance:
    """Score DNB-index trajectories through the matched-false-alarm protocol.

    Each transition and surrogate-null trajectory is reduced to its rising-trend score
    (:func:`dnb_trend_score` by default, or :func:`dnb_regression_slope` for the few
    timepoints where a rank statistic is too coarse); the threshold is calibrated on the
    null scores to a matched false alarm
    (:func:`~bench.early_warning_domain.calibrate_score_threshold`); a trajectory alarms
    when its score meets the threshold; and the transition alarm count is tested for
    significance by the shared label-permutation core, so the p-value is directly
    comparable to the SCPN suite and the AR(1)-Kendall-τ competitor.

    Parameters
    ----------
    transition_trajectories : sequence of sequence of float
        The DNB-index rising limb of each real, time-ordered transition trajectory.
    null_trajectories : sequence of sequence of float
        The DNB-index limb of each temporally-shuffled surrogate (or control-arm) null.
    score : callable
        The per-trajectory rising-trend statistic; :func:`dnb_trend_score` (Kendall τ)
        by default, or :func:`dnb_regression_slope` for coarse-timepoint corpora.
    target_fa : float
        Target false-alarm rate the trend-score threshold is held at or below.
    n_permutations : int
        Number of random relabellings for the significance test.
    seed : int
        Seed of the resampling, so the p-value is reproducible.

    Returns
    -------
    DnbSignificance
        The calibrated threshold, achieved false-alarm rate, and permutation
        significance.

    Raises
    ------
    ValueError
        If either trajectory set is empty.
    """
    if not transition_trajectories:
        raise ValueError("transition_trajectories must not be empty")
    if not null_trajectories:
        raise ValueError("null_trajectories must not be empty")
    transition_scores = [score(t) for t in transition_trajectories]
    null_scores = [score(t) for t in null_trajectories]
    threshold = calibrate_score_threshold(null_scores, target_fa=target_fa)
    transition_alarms = [score >= threshold for score in transition_scores]
    null_alarms = [score >= threshold for score in null_scores]
    achieved = float(np.mean(null_alarms))
    significance = permutation_significance_from_alarms(
        transition_alarms, null_alarms, n_permutations=n_permutations, seed=seed
    )
    return DnbSignificance(
        score_threshold=threshold,
        achieved_false_alarm=achieved,
        significance=significance,
    )


def _gene_correlation(values: FloatArray) -> FloatArray:
    """Return the Pearson correlation matrix over columns, with non-finite set to zero.

    A constant column has zero variance and yields ``nan`` correlations; those are
    replaced by ``0.0`` so a flat gene contributes no spurious in- or out-module
    correlation rather than poisoning the mean.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(values, rowvar=False)
    return np.where(np.isfinite(correlation), correlation, 0.0)


def _mean_abs_offdiagonal(block: FloatArray) -> float:
    """Return the mean absolute off-diagonal entry of a square correlation block."""
    size = int(block.shape[0])
    if size < 2:
        return 0.0
    off_diagonal = ~np.eye(size, dtype=bool)
    return float(np.mean(np.abs(block[off_diagonal])))


def _mean_offdiagonal(block: FloatArray) -> float:
    """Return the mean signed off-diagonal entry of a square correlation block."""
    size = int(block.shape[0])
    if size < 2:
        return 0.0
    off_diagonal = ~np.eye(size, dtype=bool)
    return float(np.mean(block[off_diagonal]))


def _module_indices(module: Sequence[int], n_genes: int) -> list[int]:
    """Return the sorted distinct in-range module indices, or raise ``ValueError``."""
    members = sorted({int(gene) for gene in module})
    if len(members) < 2:
        raise ValueError("module must name at least two distinct genes")
    if members[0] < 0 or members[-1] >= n_genes:
        raise ValueError(f"module genes must lie in [0, {n_genes})")
    return members


def _index_in_range(index: int, length: int) -> int:
    """Return ``index`` if it addresses ``[0, length)``, else raise ``ValueError``."""
    value = int(index)
    if value < 0 or value >= length:
        raise ValueError(f"transition_index {value} out of range [0, {length})")
    return value
