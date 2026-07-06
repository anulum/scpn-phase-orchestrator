# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — dynamical network biomarker detector tests

"""Tests for the DNB early-warning detector.

The bulk composite index, the greedy module selector, the single-cell transition index,
the rising-trend score, and the matched-false-alarm significance are each exercised on
synthetic gene-expression with a planted DNB module, alongside every guard and boundary.
The single-cell index is validated against an independently computed signed
correlation ratio — the exact Mojtahedi form — so the shipped index is the published
one.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.dnb_detector import (
    DnbSignificance,
    _mean_abs_offdiagonal,
    _mean_offdiagonal,
    dnb_index,
    dnb_significance,
    dnb_trend_score,
    select_dnb_module,
    single_cell_transition_index,
)


def _correlated_block(
    rng: np.random.Generator, n_samples: int, spread: float
) -> np.ndarray:
    """Return a length-``n_samples`` shared factor scaled to ``spread``."""
    return rng.normal(0.0, spread, size=n_samples)


def _planted_frames(
    rng: np.random.Generator, *, module: list[int], n_genes: int, n_samples: int
) -> list[np.ndarray]:
    """Return two timepoints where ``module`` co-varies with rising variance."""
    frames = []
    for step in range(2):
        frame = rng.normal(0.0, 0.2, size=(n_samples, n_genes))
        factor = _correlated_block(rng, n_samples, 0.3 + 1.5 * step)
        for gene in module:
            frame[:, gene] = factor + rng.normal(0.0, 0.05, size=n_samples)
        frames.append(frame)
    return frames


# --------------------------------------------------------------------------- #
# dnb_index                                                                    #
# --------------------------------------------------------------------------- #


def test_dnb_index_rewards_a_tight_module_against_the_network() -> None:
    rng = np.random.default_rng(0)
    frames = _planted_frames(rng, module=[1, 3, 5], n_genes=10, n_samples=12)
    index = dnb_index(frames[1], [1, 3, 5])
    assert index > dnb_index(frames[0], [1, 3, 5])  # rises toward the transition
    assert np.isfinite(index)


def test_dnb_index_tolerates_a_constant_gene() -> None:
    # A constant column yields nan correlations; the detector must set them to zero and
    # still return a finite index rather than propagate the nan.
    rng = np.random.default_rng(1)
    frame = rng.normal(0.0, 1.0, size=(8, 6))
    frame[:, 4] = 3.0  # a flat outside gene
    assert np.isfinite(dnb_index(frame, [0, 1, 2]))


def test_dnb_index_rejects_a_non_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        dnb_index(np.zeros(6), [0, 1])


def test_dnb_index_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        dnb_index(np.zeros((1, 6)), [0, 1])


def test_dnb_index_rejects_a_degenerate_module() -> None:
    with pytest.raises(ValueError, match="at least two distinct genes"):
        dnb_index(np.zeros((4, 6)), [2])


def test_dnb_index_rejects_an_out_of_range_module() -> None:
    with pytest.raises(ValueError, match=r"lie in \[0, 6\)"):
        dnb_index(np.zeros((4, 6)), [0, 6])


def test_dnb_index_rejects_a_module_leaving_no_outside_genes() -> None:
    with pytest.raises(ValueError, match="at least one gene outside"):
        dnb_index(np.zeros((4, 3)), [0, 1, 2])


# --------------------------------------------------------------------------- #
# select_dnb_module                                                           #
# --------------------------------------------------------------------------- #


def test_select_module_grows_through_every_candidate() -> None:
    # Three co-varying rising genes and nothing else rising: the pool is exactly the
    # module, so growth consumes every candidate and exits when none remain.
    rng = np.random.default_rng(2)
    frames = _planted_frames(rng, module=[2, 4, 7], n_genes=12, n_samples=14)
    module = select_dnb_module(frames, transition_index=1, min_module=3)
    assert module == (2, 4, 7)


def test_select_module_stops_at_a_decoy_that_lowers_the_index() -> None:
    # A high-variance but uncorrelated decoy sits in the candidate pool; adding it drops
    # the composite index, so greedy growth stops and excludes it.
    rng = np.random.default_rng(3)
    n_samples = 16
    baseline = rng.normal(0.0, 0.1, size=(n_samples, 8))
    transition = rng.normal(0.0, 0.1, size=(n_samples, 8))
    factor = _correlated_block(rng, n_samples, 2.0)
    for gene in (0, 1, 2):
        transition[:, gene] = factor + rng.normal(0.0, 0.05, size=n_samples)
    transition[:, 3] = rng.normal(
        0.0, 1.0, size=n_samples
    )  # high variance, independent
    module = select_dnb_module([baseline, transition], transition_index=1, min_module=4)
    assert set(module) == {0, 1, 2}  # the decoy gene 3 is left out


def test_select_module_rejects_too_few_timepoints() -> None:
    with pytest.raises(ValueError, match="at least two timepoints"):
        select_dnb_module([np.zeros((4, 6))], transition_index=0)


def test_select_module_rejects_a_non_matrix_baseline() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        select_dnb_module([np.zeros(6), np.zeros((4, 6))], transition_index=1)


def test_select_module_rejects_a_ragged_gene_axis() -> None:
    with pytest.raises(ValueError, match="same gene axis"):
        select_dnb_module([np.zeros((4, 6)), np.zeros((4, 5))], transition_index=1)


def test_select_module_rejects_an_out_of_range_transition() -> None:
    with pytest.raises(ValueError, match=r"out of range \[0, 2\)"):
        select_dnb_module([np.zeros((4, 6)), np.zeros((4, 6))], transition_index=2)


def test_select_module_rejects_a_corpus_with_no_rising_variance() -> None:
    # Variance falls from baseline to transition, so no gene is a candidate.
    rng = np.random.default_rng(4)
    baseline = rng.normal(0.0, 2.0, size=(10, 6))
    transition = rng.normal(0.0, 0.01, size=(10, 6))
    with pytest.raises(ValueError, match="rising standard deviation"):
        select_dnb_module([baseline, transition], transition_index=1)


# --------------------------------------------------------------------------- #
# single_cell_transition_index                                                #
# --------------------------------------------------------------------------- #


def test_single_cell_index_matches_the_signed_correlation_ratio() -> None:
    # The Mojtahedi form: mean signed gene-gene over mean signed cell-cell correlation.
    # A per-cell scale makes cells positively correlated, the real single-cell regime.
    rng = np.random.default_rng(5)
    n_cells, n_genes = 40, 8
    scale = rng.normal(20.0, 3.0, size=(n_cells, 1))
    gene_effect = rng.normal(0.0, 1.0, size=(1, n_genes))
    expression = scale + gene_effect + rng.normal(0.0, 0.5, size=(n_cells, n_genes))
    index = single_cell_transition_index(expression)
    gene_gene = np.corrcoef(expression, rowvar=False)
    cell_cell = np.corrcoef(expression.T, rowvar=False)
    gene_off = ~np.eye(n_genes, dtype=bool)
    cell_off = ~np.eye(n_cells, dtype=bool)
    expected = float(gene_gene[gene_off].mean() / cell_cell[cell_off].mean())
    assert index == pytest.approx(expected)


def test_single_cell_index_rejects_a_non_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        single_cell_transition_index(np.zeros(6))


def test_single_cell_index_rejects_too_few_cells() -> None:
    with pytest.raises(ValueError, match="at least two cells"):
        single_cell_transition_index(np.zeros((1, 6)))


def test_single_cell_index_rejects_too_few_genes() -> None:
    with pytest.raises(ValueError, match="at least two genes"):
        single_cell_transition_index(np.zeros((6, 1)))


# --------------------------------------------------------------------------- #
# dnb_trend_score                                                             #
# --------------------------------------------------------------------------- #


def test_trend_score_is_one_for_a_monotone_rise() -> None:
    assert dnb_trend_score([0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)


def test_trend_score_is_zero_below_two_points() -> None:
    assert dnb_trend_score([0.5]) == 0.0


def test_trend_score_is_zero_for_a_constant_trajectory() -> None:
    # A constant trajectory gives an undefined Kendall τ (nan), reported as no trend.
    assert dnb_trend_score([2.0, 2.0, 2.0]) == 0.0


def test_trend_score_rejects_a_non_vector() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        dnb_trend_score([[1.0, 2.0], [3.0, 4.0]])


# --------------------------------------------------------------------------- #
# dnb_significance                                                            #
# --------------------------------------------------------------------------- #


def test_dnb_significance_scores_rising_transitions_against_shuffled_nulls() -> None:
    transitions = [[0.1, 0.2, 0.4], [0.1, 0.25, 0.5], [0.12, 0.18, 0.35]]
    nulls = [
        [0.4, 0.1, 0.2],
        [0.2, 0.5, 0.1],
        [0.3, 0.1, 0.4],
        [0.1, 0.4, 0.2],
        [0.5, 0.2, 0.1],
    ]
    result = dnb_significance(transitions, nulls, n_permutations=2000)
    assert isinstance(result, DnbSignificance)
    assert result.significance.observed_led == 3
    assert result.significance.n_transitions == 3
    assert 0.0 < result.significance.p_value <= 1.0
    # Reproducible under a fixed seed.
    again = dnb_significance(transitions, nulls, n_permutations=2000)
    assert again.significance.p_value == result.significance.p_value


def test_dnb_significance_audit_record_is_json_safe() -> None:
    transitions = [[0.1, 0.2, 0.4]]
    nulls = [[0.4, 0.1, 0.2], [0.2, 0.5, 0.1]]
    record = dnb_significance(transitions, nulls, n_permutations=500).to_audit_record()
    assert record["detector"] == "dnb_trend"
    assert set(record) == {
        "detector",
        "score_threshold",
        "achieved_false_alarm",
        "significance",
    }
    assert isinstance(record["significance"], dict)


def test_dnb_significance_rejects_empty_transitions() -> None:
    with pytest.raises(ValueError, match="transition_trajectories must not be empty"):
        dnb_significance([], [[0.1, 0.2]])


def test_dnb_significance_rejects_empty_nulls() -> None:
    with pytest.raises(ValueError, match="null_trajectories must not be empty"):
        dnb_significance([[0.1, 0.2]], [])


# --------------------------------------------------------------------------- #
# private helpers                                                             #
# --------------------------------------------------------------------------- #


def test_mean_offdiagonal_helpers_are_zero_for_a_singleton_block() -> None:
    singleton = np.array([[1.0]])
    assert _mean_abs_offdiagonal(singleton) == 0.0
    assert _mean_offdiagonal(singleton) == 0.0
