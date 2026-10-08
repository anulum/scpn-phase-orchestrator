# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Synthetic HCP-inspired connectome loader
#
# NOT real Human Connectome Project data. Generates a structured coupling
# matrix that mimics known macroscale brain connectivity patterns:
#   - Higher intra-hemispheric coupling
#   - Weaker inter-hemispheric coupling (corpus callosum pattern)
#   - Default mode network (DMN) hub structure
#
# For real HCP data, use neurolib (Cakan, Jajcay & Obermayer 2021,
# doi:10.1007/s12559-021-09931-9) or the HCP1200 parcellation directly.

"""Synthetic and optional neurolib HCP coupling loaders.

`load_hcp_connectome` generates a deterministic HCP-inspired synthetic matrix
with explicit non-real-data provenance. `load_neurolib_hcp` is the optional real
HCP path and fails with an import error when `neurolib` is unavailable. Both
paths return non-negative zero-diagonal structural coupling matrices suitable
for examples, validation, and explicit downstream review.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._array_types import require_real_values
from scpn_phase_orchestrator.coupling._connectome_validation import (
    _coerce_connectome_matrix,
    _validate_connectome_matrix,
    _validate_n_regions,
    _validate_seed,
)

try:
    _rust_load_hcp = cast(
        "ConnectomeBackend | None",
        getattr(import_module("spo_kernel"), "load_hcp_connectome_rust", None),
    )
except ImportError:
    _rust_load_hcp = None

_HAS_RUST = callable(_rust_load_hcp)

__all__ = ["load_hcp_connectome", "load_neurolib_hcp"]

FloatArray: TypeAlias = NDArray[np.float64]
ConnectomeBackend: TypeAlias = Callable[[int, int], object]


class _HCPDataset(Protocol):
    """Structural weights supplied by the optional neurolib dataset loader."""

    Cmat: object


_NEUROLIB_HCP_SIZE = 80  # neurolib 0.6.2 cortical AAL2/LRLR dataset


def load_neurolib_hcp(n_regions: int = 80) -> FloatArray:
    """Load real HCP structural connectivity from neurolib.

    Parameters
    ----------
    n_regions : int
        Genuine non-boolean integer from two through 80. Smaller counts return
        the top-left square slice of the original subject-average matrix.

    Returns
    -------
    FloatArray
        Independent writable C-contiguous float64 matrix of shape
        ``(n_regions, n_regions)``, symmetric, non-negative and zero diagonal.

    Raises
    ------
    ImportError
        If neurolib is not installed.
    TypeError
        If the region count is not a genuine non-boolean integer.
    ValueError
        If the region count or original provider's structural weights are invalid.

    Notes
    -----
    The qualified neurolib 0.6.2 provider averages separately normalised subject
    weights in the 80-region cortical AAL2/LRLR ordering. The slice preserves
    that ordering; it does not select a new anatomical parcellation. The original
    provider diagonal is canonicalised to zero after source admission.
    """
    n_regions = _validate_n_regions(n_regions, max_regions=_NEUROLIB_HCP_SIZE)
    try:
        dataset_factory = cast(
            "Callable[[str], _HCPDataset]",
            import_module("neurolib.utils.loadData").Dataset,
        )
    except ModuleNotFoundError:
        raise ImportError(
            "neurolib is required for real HCP data: pip install neurolib"
        ) from None

    ds = dataset_factory("hcp")
    sc = _coerce_connectome_matrix(
        ds.Cmat,
        n_regions=_NEUROLIB_HCP_SIZE,
        source="neurolib HCP",
    )[:n_regions, :n_regions].copy()
    np.fill_diagonal(sc, 0.0)
    return _validate_connectome_matrix(
        sc,
        n_regions=n_regions,
        source="neurolib HCP",
    )


# Hagmann et al. 2008, PLoS Biol. 6:e159 — structural connectivity statistics
_INTRA_HEMI_STRENGTH = 0.5
_INTER_HEMI_STRENGTH = 0.15
_DMN_HUB_BOOST = 0.3


def load_hcp_connectome(n_regions: int, seed: int = 42) -> FloatArray:
    """Generate a synthetic HCP-inspired coupling matrix.

    Parameters
    ----------
    n_regions : int
        Number of synthetic regions, at least two. The dense float64 matrix
        must fit the platform's addressable storage; even counts are optional.
    seed : int
        Non-boolean integer in the unsigned 64-bit range, default 42.

    Returns
    -------
    FloatArray
        Independent writable C-contiguous symmetric non-negative matrix with
        shape ``(n_regions, n_regions)`` and zero diagonal.

    Raises
    ------
    TypeError
        If a region count or seed is not a genuine non-boolean integer.
    ValueError
        If a count, seed, allocation or returned structural matrix is invalid.

    Notes
    -----
    The default uses the original Rust builtin when available, otherwise NumPy.
    Python uses PCG64 Gaussian noise; Rust uses LCG uniform noise. Seeds are
    repeatable within an owner, without elementwise cross-owner equivalence.
    Up to 128 matrices are cached; every call returns a copy. These are synthetic
    weights, not HCP observations or a calibrated brain model.
    """
    n_regions = _validate_n_regions(n_regions)
    seed = _validate_seed(seed)
    backend: ConnectomeBackend | None = _rust_load_hcp if _HAS_RUST else None
    try:
        return _load_hcp_connectome_cached(n_regions, seed, backend).copy()
    except MemoryError as exc:
        raise ValueError("cannot allocate connectome matrix") from exc


@lru_cache(maxsize=128)
def _load_hcp_connectome_cached(
    n_regions: int, seed: int, backend: ConnectomeBackend | None
) -> FloatArray:
    """Cache the selected original generator and its validated structural weights.

    Parameters
    ----------
    n_regions : int
        Publicly admitted dense matrix dimension.
    seed : int
        Publicly admitted unsigned 64-bit seed.
    backend : Callable or None
        Actual selected native callable, or genuine NumPy fallback selection.

    Returns
    -------
    FloatArray
        Cached structural matrix; the public caller publishes a separate copy.

    Raises
    ------
    ValueError
        If original native weights or allocation are invalid.
    MemoryError
        If NumPy allocation fails, translated by the public caller.
    """
    if backend is not None:
        raw_array = np.asarray(backend(n_regions, seed))
        if np.issubdtype(raw_array.dtype, np.bool_):
            raise ValueError(
                "Rust HCP connectome output must not contain boolean values"
            )
        if np.issubdtype(raw_array.dtype, np.complexfloating):
            raise ValueError(
                "Rust HCP connectome output must contain real-valued weights"
            )
        if raw_array.dtype == np.dtype(object):
            raw_flat = raw_array.ravel()
            if any(isinstance(item, bool | np.bool_) for item in raw_flat):
                raise ValueError(
                    "Rust HCP connectome output must not contain boolean values"
                )
            if any(isinstance(item, complex | np.complexfloating) for item in raw_flat):
                raise ValueError(
                    "Rust HCP connectome output must contain real-valued weights"
                )
        for item in raw_array.ravel():
            if not isinstance(item, str | bytes | np.str_ | np.bytes_):
                continue
            try:
                float(item)
            except (TypeError, ValueError):
                continue
            raise ValueError(
                "Rust HCP connectome output must not contain numeric-string aliases"
            )
        try:
            require_real_values(
                raw_array, name="Rust HCP connectome output", allow_object=True
            )
            flat: FloatArray = np.asarray(raw_array, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Rust HCP connectome output must contain real-valued weights"
            ) from exc
        return _validate_connectome_matrix(
            flat.reshape(n_regions, n_regions),
            n_regions=n_regions,
            source="Rust HCP",
        )

    rng = np.random.default_rng(seed=seed)
    knm: FloatArray = np.zeros((n_regions, n_regions), dtype=np.float64)
    half = n_regions // 2

    # Intra-hemispheric: exponential distance decay within each hemisphere
    for hemi_start in (0, half):
        hemi_end = half if hemi_start == 0 else n_regions
        size = hemi_end - hemi_start
        idx = np.arange(size)
        dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
        block = _INTRA_HEMI_STRENGTH * np.exp(-0.3 * dist)
        # Seeded weight perturbation; no biological calibration is implied.
        block += rng.normal(0, 0.02, block.shape)
        block = np.clip(block, 0, None)
        np.fill_diagonal(block, 0.0)
        knm[hemi_start:hemi_end, hemi_start:hemi_end] = block

    # Inter-hemispheric: corpus callosum — homotopic connections largest
    for i in range(min(half, n_regions - half)):
        j = i + half
        weight = _INTER_HEMI_STRENGTH * np.exp(-0.1 * abs(i - i))  # homotopic = largest
        # Add distance decay for non-homotopic callosal fibres
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
    knm = np.asarray((knm + knm.T) / 2.0, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    result: FloatArray = np.clip(knm, 0, None)
    return _validate_connectome_matrix(
        result,
        n_regions=n_regions,
        source="synthetic HCP",
    )
