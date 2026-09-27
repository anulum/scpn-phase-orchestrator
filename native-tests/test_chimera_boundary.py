# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native chimera boundary tests

"""Exercise the installed Rust chimera ingress without backend substitution."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
import spo_kernel
from numpy.typing import NDArray

Detect = Callable[
    [NDArray[np.float64], NDArray[np.float64], object],
    tuple[object, object, float, NDArray[np.float64]],
]
detect = cast(Detect, spo_kernel.detect_chimera_rust)


@pytest.mark.parametrize(
    "count",
    [
        True,
        np.bool_(True),
        "1",
        1.0,
        np.timedelta64(1, "ms"),
        np.datetime64("2026-01-01"),
    ],
)
def test_original_count_refusal(count: object) -> None:
    """Non-integer aliases fail before kernel dispatch."""
    with pytest.raises((ValueError, TypeError)):
        detect(np.zeros(1), np.zeros(1), count)


@pytest.mark.parametrize("field", ["phases", "knm"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_refusal(field: str, bad: float) -> None:
    """Nonfinite source samples fail before classification."""
    p = np.zeros(3)
    k = np.zeros(9)
    (p if field == "phases" else k)[0] = bad
    with pytest.raises(ValueError):
        detect(p, k, 3)


@pytest.mark.parametrize(
    "n,p_size,k_size", [(3, 2, 9), (3, 3, 8), (0, 1, 0), (0, 0, 1), (2**63, 0, 0)]
)
def test_cardinality_refusal(n: int, p_size: int, k_size: int) -> None:
    """Mismatched buffers and overflowing matrix sizes fail safely."""
    with pytest.raises(ValueError):
        detect(np.zeros(p_size), np.zeros(k_size), n)


def test_diagonal_refusal() -> None:
    """Self coupling cannot participate in neighbourhood classification."""
    with pytest.raises(ValueError):
        detect(np.zeros(3), np.eye(3).ravel(), 3)


def test_empty_network() -> None:
    """The valid empty network keeps its zero boundary fraction."""
    coherent, incoherent, index, local = detect(np.zeros(0), np.zeros(0), 0)
    assert coherent == [] and incoherent == [] and index == 0
    assert local.size == 0
