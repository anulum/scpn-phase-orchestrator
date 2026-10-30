# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Independent transfer-entropy oracle for tests

"""Transfer entropy computed with ``bisect`` and ``collections.Counter``.

Phases are wrapped into ``[0, 2π)`` and binned on ``n_bins`` equal arcs whose
left edges are found with :func:`bisect.bisect_right`; conditional entropies
are sums over counted states. Conditions seen fewer than twice contribute
nothing, as in the package law.
"""

from __future__ import annotations

import bisect
import collections
import math
from collections.abc import Sequence


def _bins(values: Sequence[float], n_bins: int) -> list[int]:
    """Return the equal-arc bin of each wrapped phase."""
    edges = [2 * math.pi * k / n_bins for k in range(n_bins + 1)]
    return [
        min(max(bisect.bisect_right(edges, v % (2 * math.pi)) - 1, 0), n_bins - 1)
        for v in values
    ]


def _conditional_entropy(target: list[int], condition: list[object]) -> float:
    """Return ``H(target | condition)`` from counted states."""
    total = len(target)
    by_condition: dict[object, list[int]] = collections.defaultdict(list)
    for t, c in zip(target, condition, strict=True):
        by_condition[c].append(t)
    entropy = 0.0
    for members in by_condition.values():
        if len(members) < 2:
            continue
        counts = collections.Counter(members)
        size = len(members)
        entropy -= (size / total) * math.fsum(
            n / size * math.log(n / size) for n in counts.values()
        )
    return entropy


def oracle_phase_te(
    source: Sequence[float], target: Sequence[float], n_bins: int
) -> float:
    """Return ``TE(source → target) = H(Y⁺ | Y) − H(Y⁺ | Y, X)``."""
    n = min(len(source), len(target)) - 1
    if n < 2:
        return 0.0
    src = _bins(source[:n], n_bins)
    tgt = _bins(target[:n], n_bins)
    nxt = _bins(target[1 : n + 1], n_bins)
    value = _conditional_entropy(nxt, list(tgt)) - _conditional_entropy(
        nxt, list(zip(tgt, src, strict=True))
    )
    return max(value, 0.0)


def oracle_te_matrix(
    series: Sequence[Sequence[float]], n_bins: int
) -> list[list[float]]:
    """Return ``[i][j] = TE(i → j)`` with a zero diagonal."""
    n = len(series)
    return [
        [
            0.0 if i == j else oracle_phase_te(series[i], series[j], n_bins)
            for j in range(n)
        ]
        for i in range(n)
    ]
