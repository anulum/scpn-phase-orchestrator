# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — quality scorer Rust/Python semantic parity

"""Prove the quality scorer gives the Rust kernel's answer on every path.

The Rust kernel skips non-finite qualities and amplitudes, clamps quality to
``[0, 1]`` and counts a non-finite quality as collapsed. The Python path
compared NaN directly, so five NaN-quality states were reported as *not*
collapsed and ``score`` returned NaN or values outside ``[0, 1]``.

Default thresholds run on the Rust kernel when it is installed; the override
thresholds always run the Python path, and CI jobs without the kernel run
Python for both. No backend is substituted.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer

NAN = float("nan")


def _state(quality: float, amplitude: float = 1.0) -> PhaseState:
    return PhaseState(
        theta=0.0,
        omega=1.0,
        amplitude=amplitude,
        quality=quality,
        channel="P",
        node_id="n",
    )


SCORER = PhaseQualityScorer(collapse_threshold=0.1, min_quality=0.3)


@pytest.mark.parametrize("threshold", [0.1, 0.2])
def test_non_finite_quality_counts_as_collapsed(threshold: float) -> None:
    states = [_state(NAN)] * 5
    assert SCORER.detect_collapse(states, threshold=threshold) is True


@pytest.mark.parametrize("threshold", [0.1, 0.2])
def test_infinite_quality_counts_as_collapsed(threshold: float) -> None:
    states = [_state(float("inf"))] * 3 + [_state(0.9)] * 2
    assert SCORER.detect_collapse(states, threshold=threshold) is True


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        ([_state(NAN)] * 3, 0.0),
        ([_state(5.0)], 1.0),
        ([_state(-1.0)], 0.0),
        ([_state(0.9, NAN), _state(0.1, 1.0)], 0.1),
        ([_state(NAN, 2.0), _state(0.4, 1.0)], 0.4),
        ([_state(0.2, 1.0), _state(0.8, 3.0)], 0.65),
    ],
)
def test_score_skips_non_finite_and_clamps(
    states: list[PhaseState], expected: float
) -> None:
    assert SCORER.score(states) == pytest.approx(expected)


@pytest.mark.parametrize("min_quality", [0.3, 0.35])
def test_mask_zeroes_non_finite_and_clamps(min_quality: float) -> None:
    states = [_state(NAN), _state(float("inf")), _state(1.5), _state(0.5), _state(0.2)]
    mask = SCORER.downweight_mask(states, min_quality=min_quality)
    np.testing.assert_allclose(mask, [0.0, 0.0, 1.0, 0.5, 0.0])
