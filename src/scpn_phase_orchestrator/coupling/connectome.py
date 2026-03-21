# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Synthetic HCP-inspired connectome loader
#
# NOT real Human Connectome Project data. Generates a structured coupling
# matrix that mimics known macroscale brain connectivity patterns:
#   - Strong intra-hemispheric coupling
#   - Weaker inter-hemispheric coupling (corpus callosum pattern)
#   - Default mode network (DMN) hub structure
#
# For real HCP data, use neurolib (Cakan & Obermayer 2021, Neuroimage 227:117474)
# or the HCP1200 parcellation directly.

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["load_hcp_connectome"]

# Hagmann et al. 2008, PLoS Biol. 6:e159 — structural connectivity statistics
_INTRA_HEMI_STRENGTH = 0.5
_INTER_HEMI_STRENGTH = 0.15
_DMN_HUB_BOOST = 0.3


def load_hcp_connectome(n_regions: int) -> NDArray:
    """Generate a synthetic HCP-inspired coupling matrix.

    Args:
        n_regions: number of cortical regions (must be >= 2, even recommended).

    Returns:
        Symmetric coupling matrix, shape (n_regions, n_regions), zero diagonal.
    """
    if n_regions < 2:
        msg = f"n_regions must be >= 2, got {n_regions}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed=42)
    knm = np.zeros((n_regions, n_regions), dtype=np.float64)
    half = n_regions // 2

    # Intra-hemispheric: exponential distance decay within each hemisphere
    for hemi_start in (0, half):
        hemi_end = half if hemi_start == 0 else n_regions
        size = hemi_end - hemi_start
        idx = np.arange(size)
        dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
        block = _INTRA_HEMI_STRENGTH * np.exp(-0.3 * dist)
        # Small noise for biological realism
        block += rng.normal(0, 0.02, block.shape)
        block = np.clip(block, 0, None)
        np.fill_diagonal(block, 0.0)
        knm[hemi_start:hemi_end, hemi_start:hemi_end] = block

    # Inter-hemispheric: corpus callosum — homotopic connections strongest
    for i in range(min(half, n_regions - half)):
        j = i + half
        weight = _INTER_HEMI_STRENGTH * np.exp(-0.1 * abs(i - i))  # homotopic = strongest
        # Add distance decay for non-homotopic callosal fibers
        spread = min(3, half)
        for offset in range(-spread, spread + 1):
            ji = i + offset
            jj = j + offset - (offset if offset else 0)
            if 0 <= ji < half and half <= jj < n_regions:
                w = weight * np.exp(-0.5 * abs(offset))
                knm[ji, jj] = w
                knm[jj, ji] = w

    # DMN hubs: mPFC, PCC/precuneus, lateral parietal, MTL
    # Placed at roughly anatomical proportions of the parcellation
    dmn_fractions = [0.15, 0.45, 0.65, 0.85]
    dmn_left = [int(f * half) for f in dmn_fractions]
    dmn_right = [h + half for h in dmn_left]
    dmn_nodes = [n for n in dmn_left + dmn_right if n < n_regions]

    for hub in dmn_nodes:
        for other in dmn_nodes:
            if hub != other:
                knm[hub, other] += _DMN_HUB_BOOST

    # Symmetrise and clean diagonal
    knm = (knm + knm.T) / 2.0
    np.fill_diagonal(knm, 0.0)
    knm = np.clip(knm, 0, None)

    return knm
