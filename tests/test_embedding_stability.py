# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability tests for delay embedding

"""Long-run invariants for the embedding primitives.

* ``delay_embed`` on a 10 000-sample signal — allocation-free result
  of the right shape.
* ``mutual_information`` stable under translation (MI(x+c, lag) =
  MI(x, lag)) — bin edges are computed from min/max so the shift
  should preserve MI exactly up to float ULP.
* ``nearest_neighbor_distances`` symmetric reciprocity on a small
  cloud: if j is i's NN, then i is in j's top-2 neighbours.

Marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import embedding as em_mod
from scpn_phase_orchestrator.monitor.embedding import (
    delay_embed,
    mutual_information,
    nearest_neighbor_distances,
)

pytestmark = pytest.mark.slow


def _python(func):
    def wrapper(*args, **kwargs):
        prev = em_mod.ACTIVE_BACKEND
        em_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            em_mod.ACTIVE_BACKEND = prev

    wrapper.__name__ = func.__name__
    return wrapper


@_python
def test_delay_embed_stress():
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 1, 10_000)
    emb = delay_embed(sig, delay=10, dimension=5)
    assert emb.shape == (10_000 - 40, 5)
    assert np.all(np.isfinite(emb))


@_python
def test_mi_translation_invariance():
    rng = np.random.default_rng(3)
    sig = rng.normal(0, 1, 800)
    shifted = sig + 42.0  # uniform offset
    mi1 = mutual_information(sig, 5, 32)
    mi2 = mutual_information(shifted, 5, 32)
    # The min/max-shifted binning preserves MI exactly modulo float noise.
    assert abs(mi1 - mi2) < 1e-9


@_python
def test_nn_reciprocity_top2():
    """For a small point cloud, if ``j = NN(i)`` then ``i`` is
    typically among ``j``'s top-2 neighbours. This is a sanity check
    for the kNN primitive — it does not have to hold for every point
    but should for the vast majority on a smooth trajectory."""
    t = np.linspace(0, 4 * math.pi, 60)
    emb = np.column_stack([np.sin(t), np.cos(t)])
    _, idx = nearest_neighbor_distances(emb)
    # Build j → set of points whose NN is j.
    reciprocal = 0
    for i, j in enumerate(idx):
        # Is i among j's two nearest neighbours? Compute locally.
        diffs = emb - emb[int(j)]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists[int(j)] = np.inf
        order = np.argsort(dists)
        if i in order[:2]:
            reciprocal += 1
    # Expect the vast majority of the 60 points to be reciprocal.
    assert reciprocal >= 50
