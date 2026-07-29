# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — coherence monitor config validation tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.coherence import CoherenceMonitor


@pytest.mark.parametrize("field", ["good_layers", "bad_layers"])
@pytest.mark.parametrize("value", [False, -1, 1.5, "1"])
def test_coherence_monitor_rejects_invalid_layer_indices(
    field: str,
    value: Any,
) -> None:
    kwargs: dict[str, Any] = {"good_layers": [0], "bad_layers": [1]}
    kwargs[field] = [value]

    with pytest.raises(
        ValueError,
        match=f"{field} must contain non-negative integer indices",
    ):
        CoherenceMonitor(**kwargs)


def test_coherence_monitor_allows_empty_layer_groups() -> None:
    monitor = CoherenceMonitor(good_layers=[], bad_layers=[])

    assert monitor._good == []
    assert monitor._bad == []


def test_coherence_monitor_copies_layer_groups() -> None:
    good_layers = [0]
    bad_layers = [1]

    monitor = CoherenceMonitor(good_layers=good_layers, bad_layers=bad_layers)
    good_layers.append(2)
    bad_layers.append(3)

    assert monitor._good == [0]
    assert monitor._bad == [1]


@pytest.mark.parametrize("field", ["good_layers", "bad_layers"])
def test_coherence_monitor_requires_list_layer_groups(field: str) -> None:
    kwargs: dict[str, Any] = {"good_layers": [0], "bad_layers": [1]}
    kwargs[field] = (0,)

    with pytest.raises(TypeError, match=f"{field} must be a list"):
        CoherenceMonitor(**kwargs)


@pytest.mark.parametrize("field", ["good_layers", "bad_layers"])
def test_coherence_monitor_rejects_duplicate_layer_indices(field: str) -> None:
    kwargs: dict[str, Any] = {"good_layers": [0], "bad_layers": [1]}
    kwargs[field] = [0, 0]

    with pytest.raises(ValueError, match=f"{field} must not contain duplicates"):
        CoherenceMonitor(**kwargs)


def test_coherence_monitor_rejects_overlapping_partitions() -> None:
    with pytest.raises(ValueError, match="good_layers and bad_layers must be disjoint"):
        CoherenceMonitor(good_layers=[0, 1], bad_layers=[1, 2])


def test_coherence_monitor_accepts_numpy_integer_indices() -> None:
    monitor = CoherenceMonitor(
        good_layers=[np.int64(0)],  # type: ignore[list-item]
        bad_layers=[np.int64(1)],  # type: ignore[list-item]
    )

    assert monitor._good == [0]
    assert monitor._bad == [1]
