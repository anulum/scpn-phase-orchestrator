# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — resilience scoring refuses non-finite trajectories

"""A NaN order parameter must not score as "no drop" or "recovered"."""

from __future__ import annotations

import math

import pytest

from scpn_phase_orchestrator.runtime.chaos import compute_resilience

NOMINAL = (0.9,) * 10


@pytest.mark.parametrize(
    ("nominal", "perturbed"),
    [
        pytest.param(
            NOMINAL,
            (0.9, 0.1, 0.1, math.nan, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
            id="nan-mid-perturbed",
        ),
        pytest.param(NOMINAL, (0.9,) * 9 + (math.nan,), id="nan-final-perturbed"),
        pytest.param(NOMINAL, (0.9,) * 9 + (math.inf,), id="inf-final-perturbed"),
        pytest.param((math.nan,) + (0.9,) * 9, NOMINAL, id="nan-nominal"),
    ],
)
def test_non_finite_history_is_refused(
    nominal: tuple[float, ...], perturbed: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        compute_resilience(
            nominal,
            perturbed,
            fault_onset_step=1,
            last_fault_end=4,
            recovery_tolerance=0.05,
        )


def test_finite_drop_is_reported() -> None:
    metrics = compute_resilience(
        NOMINAL,
        (0.9, 0.1, 0.1, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
        fault_onset_step=1,
        last_fault_end=4,
        recovery_tolerance=0.05,
    )
    assert metrics.max_coherence_drop == pytest.approx(0.8)
    assert metrics.recovered is True
    assert metrics.recovery_steps == 0
