# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Finite quality aggregation contracts
"""Compare public quality scores with exact rational amplitude weighting."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer


@pytest.mark.parametrize(
    ("qualities", "amplitudes"),
    [
        ([0.8, 0.2], [1e308, 1e308]),
        ([0.8, 0.2], [float(np.finfo(np.float64).max)] * 2),
        ([1.0, 1.0, 1.0], [1e308] * 3),
        ([0.0, 0.0, 0.0], [1e308] * 3),
        ([0.9, 0.1, 0.5], [1e308, 5e307, 2.5e307]),
        ([0.0, 1.0], [1e308, 1.0]),
        ([0.8, 0.2], [0.0, -1e308]),
        ([0.2, 0.8], [1e-15, 1e-13]),
        ([0.2, 0.8], [1e-12, 2e-12]),
        ([0.8, 0.2, float("nan")], [1e308, 1e308, 1e308]),
        ([0.8, 0.2, 0.9], [1e308, 1e308, float("inf")]),
        ([float("inf"), 0.25], [1e308, 1e308]),
        ([-1e308, 1e308], [1e308, 1e308]),
        ([0.8], [1e308]),
    ],
)
def test_public_finite_mean_matches_exact_rational_oracle(
    qualities: list[float], amplitudes: list[float]
) -> None:
    """Retain the mathematical mean across overflow, floor and nonfinite cases.

    Parameters
    ----------
    qualities : list[float]
        Real extraction qualities; nonfinite pairs are omitted by contract.
    amplitudes : list[float]
        Corresponding finite or nonfinite extraction amplitudes.
    """
    pairs = [
        (
            Fraction.from_float(min(max(q, 0.0), 1.0)),
            Fraction.from_float(max(a, 1e-12)),
        )
        for q, a in zip(qualities, amplitudes, strict=True)
        if math.isfinite(q) and math.isfinite(a)
    ]
    expected = float(
        sum((q * a for q, a in pairs), Fraction())
        / sum((a for _, a in pairs), Fraction())
    )
    states = [
        PhaseState(0.0, 1.0, a, q, "P", f"n{i}")
        for i, (q, a) in enumerate(zip(qualities, amplitudes, strict=True))
    ]
    before = [(s.quality, s.amplitude) for s in states]
    with np.errstate(over="raise", invalid="raise"):
        result = PhaseQualityScorer().score(states)
        reverse = PhaseQualityScorer().score(list(reversed(states)))
    assert math.isfinite(result) and 0.0 <= result <= 1.0
    assert result == pytest.approx(expected, rel=2e-15, abs=0.0)
    assert reverse == pytest.approx(expected, rel=2e-15, abs=0.0)
    assert [(s.quality, s.amplitude) for s in states] == before


def test_score_preserves_valid_calls_after_bad_source_refusal() -> None:
    """A real alias refusal leaves the same scorer ready for large finite input."""
    from typing import cast

    scorer = PhaseQualityScorer()
    invalid = PhaseState(0.0, 1.0, cast("float", "1e308"), 0.8, "P", "invalid")
    with pytest.raises(ValueError, match="amplitude"):
        scorer.score([invalid])
    states = [
        PhaseState(0.0, 1.0, 1e308, 0.8, "P", "a"),
        PhaseState(0.0, 1.0, 1e308, 0.2, "P", "b"),
    ]
    assert scorer.score(states) == pytest.approx(0.5)
    assert not scorer.detect_collapse(states)
    np.testing.assert_array_equal(scorer.downweight_mask(states), [0.8, 0.0])


def test_real_session_start_receives_finite_channel_confidence() -> None:
    """Large finite amplitudes do not manufacture a low-quality session warning."""
    from scpn_phase_orchestrator.imprint.state import ImprintState
    from scpn_phase_orchestrator.monitor.session_start import check_session_start

    states = [
        PhaseState(0.0, 1.0, 1e308, 0.8, "P", "a"),
        PhaseState(0.1, 1.0, 1e308, 0.2, "P", "b"),
    ]
    phases = np.array([state.theta for state in states], dtype=np.float64)
    imprint = ImprintState(m_k=np.zeros(2, dtype=np.float64), last_update=0.0)
    report = check_session_start(states, phases, imprint, 2)
    assert report.passed
    assert report.quality_scores == {"P": pytest.approx(0.5)}
    assert report.errors == []
    assert report.warnings == []


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "uint64"])
def test_public_numpy_measurements_retain_original_type_validation(dtype: str) -> None:
    """Accepted NumPy scalars retain the same weighting, majority and mask.

    Parameters
    ----------
    dtype : str
        Real scalar dtype supplied directly through public PhaseState records.
    """
    qualities = np.array([0, 1], dtype=dtype)
    amplitudes = np.array([1, 2], dtype=dtype)
    states = [
        PhaseState(0.0, 1.0, cast("float", a), cast("float", q), "P", f"n{i}")
        for i, (q, a) in enumerate(zip(qualities, amplitudes, strict=True))
    ]
    scorer = PhaseQualityScorer()
    assert scorer.score(states) == pytest.approx(2.0 / 3.0)
    assert not scorer.detect_collapse(states)
    np.testing.assert_array_equal(scorer.downweight_mask(states), [0.0, 1.0])
