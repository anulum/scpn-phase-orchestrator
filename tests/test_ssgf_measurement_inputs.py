# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SSGF measurement ingress

"""Exercise cost and observer ingress through real production computations."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytest

from scpn_phase_orchestrator.ssgf.costs import compute_ssgf_costs
from scpn_phase_orchestrator.ssgf.pgbo import PGBO


@pytest.mark.parametrize("surface", ["costs", "observer"])
@pytest.mark.parametrize("argument", ["phases", "W"])
@pytest.mark.parametrize(
    "kind",
    [
        "U8",
        "S8",
        "timedelta64[ms]",
        "timedelta64[ns]",
        "datetime64[ms]",
        "object_time",
        "mixed_bool",
    ],
)
def test_ssgf_rejects_measurement_aliases_before_observation(
    surface: str, argument: str, kind: str
) -> None:
    """Invalid arrays cannot publish costs or mutate the observer history."""
    phases = np.array([0.0, 1.0, 2.0])
    matrix = np.ones((3, 3)) - np.eye(3)
    raw = phases if argument == "phases" else matrix
    if kind == "object_time":
        invalid = raw.astype(object)
        invalid.flat[0] = np.timedelta64(1, "ms")
    elif kind == "mixed_bool":
        invalid = raw.tolist()
        if argument == "phases":
            invalid[0] = True
        else:
            invalid[0][0] = True
    else:
        invalid = raw.astype(kind)
    observer = PGBO()
    with pytest.raises(ValueError, match=argument):
        if surface == "costs":
            compute_ssgf_costs(
                invalid if argument == "W" else matrix,
                invalid if argument == "phases" else phases,
            )
        else:
            observer.observe(
                invalid if argument == "phases" else phases,
                invalid if argument == "W" else matrix,
            )
    assert observer.history == []


def test_ssgf_numeric_objects_preserve_cost_and_observer_results() -> None:
    """Compatible object storage preserves the real numerical pipeline."""
    phases = np.array([0.0, 1.0, 2.0])
    matrix = np.ones((3, 3)) - np.eye(3)
    expected = compute_ssgf_costs(matrix, phases)
    actual = compute_ssgf_costs(matrix.astype(object), phases.astype(object))
    assert actual == expected
    snapshot = PGBO().observe(phases.astype(object), matrix.astype(object))
    assert snapshot.costs == expected
    assert snapshot.step == 1
    assert np.isfinite(snapshot.gauge_curvature)


@pytest.mark.parametrize("surface", ["costs", "observer"])
@pytest.mark.parametrize("unit", ["ms", "ns"])
def test_ssgf_rejects_temporal_cost_weights(
    surface: str, unit: Literal["ms", "ns"]
) -> None:
    """Duration weights fail as source types even though NumPy calls them Real."""
    weight = cast("float", np.timedelta64(1, unit))
    weights = (weight, 0.5, 0.1, 0.1)
    with pytest.raises(ValueError, match="weights"):
        if surface == "costs":
            compute_ssgf_costs(np.eye(3), np.array([0.0, 1.0, 2.0]), weights)
        else:
            PGBO(weights)
