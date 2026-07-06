# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid modal-growth significance-layer tests

"""Tests for the grid modal-growth matched-false-alarm significance layer.

The detector primitives are tested in ``test_grid_modal_growth.py`` (they live in
``scpn_phase_orchestrator.monitor.grid_modal_growth``); here the offline evaluation
around them — the matched-false-alarm calibration, the label-permutation significance,
and the audit record under both aggregations — is exercised on synthetic multi-bus
oscillations with planted growth, separating growing (unstable) segments from damped
ones at a matched false alarm, the behaviour the real PSML head-to-head then
demonstrates against the generic suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.grid_oscillation_detector import (
    DEFAULT_RECENCY_TOP,
    ModalGrowthSignificance,
    modal_growth_significance,
)

_RATE = 238.0
_SAMPLES = 476  # a 2 s segment at the PSML rate


def _oscillation(
    rng: np.random.Generator, *, sigma: float, buses: int = 4
) -> np.ndarray:
    """Return a multi-bus oscillation whose amplitude envelope grows at rate sigma."""
    time = np.arange(_SAMPLES) / _RATE
    envelope = np.exp(sigma * time)
    wave = np.sin(2.0 * np.pi * 1.0 * time)
    return np.stack(
        [
            1.0
            + envelope * wave * rng.uniform(0.8, 1.2)
            + 1e-3 * rng.standard_normal(_SAMPLES)
            for _ in range(buses)
        ]
    )


def test_modal_growth_significance_separates_growing_from_damped() -> None:
    rng = np.random.default_rng(2)
    transitions = [_oscillation(rng, sigma=rng.uniform(0.2, 0.8)) for _ in range(20)]
    nulls = [_oscillation(rng, sigma=rng.uniform(-0.8, -0.1)) for _ in range(20)]
    result = modal_growth_significance(
        transitions, nulls, rate=_RATE, n_permutations=2000
    )
    assert isinstance(result, ModalGrowthSignificance)
    assert result.aggregation == "focal"
    assert result.recency_top == DEFAULT_RECENCY_TOP
    assert result.significance.observed_led >= 18  # the planted growth is detected
    assert result.significance.p_value < 0.05  # beats chance
    record = result.to_audit_record()
    assert record["detector"] == "modal_envelope_growth_rate_focal"
    assert record["aggregation"] == "focal"
    assert record["recency_top"] == DEFAULT_RECENCY_TOP
    assert set(record) == {
        "detector",
        "aggregation",
        "recency_top",
        "score_threshold",
        "achieved_false_alarm",
        "significance",
    }


def test_modal_growth_significance_mean_aggregation_also_separates() -> None:
    rng = np.random.default_rng(8)
    transitions = [_oscillation(rng, sigma=rng.uniform(0.3, 0.8)) for _ in range(20)]
    nulls = [_oscillation(rng, sigma=rng.uniform(-0.8, -0.2)) for _ in range(20)]
    result = modal_growth_significance(
        transitions, nulls, rate=_RATE, aggregation="mean", n_permutations=2000
    )
    assert result.aggregation == "mean"
    assert result.to_audit_record()["detector"] == "modal_envelope_growth_rate_mean"
    assert result.significance.p_value < 0.05


def test_modal_growth_significance_rejects_empty_transitions() -> None:
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="transition_segments must not be empty"):
        modal_growth_significance([], [_oscillation(rng, sigma=-0.3)], rate=_RATE)


def test_modal_growth_significance_rejects_empty_nulls() -> None:
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="null_segments must not be empty"):
        modal_growth_significance([_oscillation(rng, sigma=0.3)], [], rate=_RATE)
