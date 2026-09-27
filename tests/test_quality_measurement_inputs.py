# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase quality measurement inputs

"""Verify source types and numerical quality policies through public methods."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer


@pytest.mark.parametrize("method", ["score", "collapse", "mask"])
@pytest.mark.parametrize(
    "value",
    [
        "0.8",
        b"0.8",
        True,
        np.bool_(True),
        np.timedelta64(1, "ms"),
        np.datetime64("2026-01-01"),
        complex(0.8, 0.1),
    ],
)
def test_quality_aliases_fail_before_scoring(method: str, value: object) -> None:
    """No public quality operation can interpret aliases as real quality."""
    state = PhaseState(0.0, 1.0, 1.0, cast("float", value), "probe", "node")
    scorer = PhaseQualityScorer()
    with pytest.raises(ValueError, match="quality"):
        if method == "score":
            scorer.score([state])
        elif method == "collapse":
            scorer.detect_collapse([state])
        else:
            scorer.downweight_mask([state])


@pytest.mark.parametrize("value", ["1.0", True, np.timedelta64(1, "ms")])
def test_amplitude_aliases_fail_before_weighting(value: object) -> None:
    """Amplitude weights cannot contain strings, booleans or temporal units."""
    state = PhaseState(0.0, 1.0, cast("float", value), 0.8, "probe", "node")
    with pytest.raises(ValueError, match="amplitude"):
        PhaseQualityScorer().score([state])


@pytest.mark.parametrize(
    "argument", ["collapse_threshold", "min_quality", "threshold", "mask_min_quality"]
)
@pytest.mark.parametrize("value", [True, "0.3", np.timedelta64(1, "ms")])
def test_quality_threshold_aliases_fail(argument: str, value: object) -> None:
    """Constructor and per-call thresholds share the plain real contract."""
    invalid = cast("float", value)
    state = PhaseState(0.0, 1.0, 1.0, 0.8, "probe", "node")
    with pytest.raises(ValueError):
        if argument == "collapse_threshold":
            PhaseQualityScorer(collapse_threshold=invalid)
        elif argument == "min_quality":
            PhaseQualityScorer(min_quality=invalid)
        elif argument == "threshold":
            PhaseQualityScorer().detect_collapse([state], threshold=invalid)
        else:
            PhaseQualityScorer().downweight_mask([state], min_quality=invalid)


def test_quality_nan_and_real_numeric_policies_survive_source_validation() -> None:
    """Real source validation preserves documented nonfinite quality policy."""
    states = [
        PhaseState(0.0, 1.0, 2.0, 0.8, "probe", "a"),
        PhaseState(0.0, 1.0, 1.0, np.nan, "probe", "b"),
        PhaseState(0.0, 1.0, 1.0, 0.1, "probe", "c"),
    ]
    scorer = PhaseQualityScorer()
    assert scorer.score(states) == pytest.approx((2 * 0.8 + 0.1) / 3)
    assert scorer.detect_collapse(states, threshold=0.3)
    np.testing.assert_array_equal(scorer.downweight_mask(states), [0.8, 0.0, 0.0])
