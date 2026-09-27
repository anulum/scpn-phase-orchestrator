# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Embedding measurement source types

"""Exercise original signal and parameter types through real embedding APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _embedding_go,
    _embedding_julia,
    _embedding_mojo,
)
from scpn_phase_orchestrator.monitor import embedding

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
EmbedFn: TypeAlias = Callable[[FloatArray, int, int], FloatArray]
InformationFn: TypeAlias = Callable[[FloatArray, int, int], float]
NeighborFn: TypeAlias = Callable[[FloatArray, int, int], tuple[FloatArray, IntArray]]

BRIDGES: dict[str, tuple[EmbedFn, InformationFn, NeighborFn]] = {
    "go": (
        _embedding_go.delay_embed_go,
        _embedding_go.mutual_information_go,
        _embedding_go.nearest_neighbor_distances_go,
    ),
    "julia": (
        _embedding_julia.delay_embed_julia,
        _embedding_julia.mutual_information_julia,
        _embedding_julia.nearest_neighbor_distances_julia,
    ),
    "mojo": (
        _embedding_mojo.delay_embed_mojo,
        _embedding_mojo.mutual_information_mojo,
        _embedding_mojo.nearest_neighbor_distances_mojo,
    ),
}


def _run(backend: str, surface: str, data: FloatArray) -> object:
    """Exercise one public primitive or its real native bridge."""
    if backend == "public":
        if surface == "embed":
            return embedding.delay_embed(data, 2, 3)
        if surface == "mi":
            return embedding.mutual_information(data, 2, 4)
        return embedding.nearest_neighbor_distances(data.reshape(8, 3))
    embed, information, neighbor = BRIDGES[backend]
    if surface == "embed":
        return embed(data, 2, 3)
    if surface == "mi":
        return information(data, 2, 4)
    return neighbor(data, 8, 3)


@pytest.mark.parametrize("backend", ["public", *BRIDGES])
@pytest.mark.parametrize("surface", ["embed", "mi", "nn"])
@pytest.mark.parametrize(
    "dtype", ["timedelta64[ms]", "datetime64[ns]", "object-time", "bool", "U16"]
)
def test_embedding_rejects_measurement_aliases(
    backend: str, surface: str, dtype: str
) -> None:
    """A duration remains a duration until the original-source refusal."""
    value = (
        np.array([np.timedelta64(i, "ns") for i in range(24)], dtype=object)
        if dtype == "object-time"
        else np.arange(24, dtype=np.float64).astype(dtype)
    )
    with pytest.raises(ValueError):
        _run(backend, surface, cast(FloatArray, value))


@pytest.mark.parametrize("backend", ["public", *BRIDGES])
@pytest.mark.parametrize("surface", ["embed", "mi", "nn"])
def test_embedding_preserves_real_numeric_objects(backend: str, surface: str) -> None:
    """Real object storage preserves samples, information and neighbor geometry."""
    value = np.sin(np.arange(24, dtype=np.float64) / 3)
    np.testing.assert_array_equal(
        _run(backend, surface, value),
        _run(backend, surface, cast(FloatArray, value.astype(object))),
    )


@pytest.mark.parametrize(
    "field", ["delay", "dimension", "max_lag", "n_bins", "rtol", "atol"]
)
def test_public_embedding_rejects_temporal_metadata(field: str) -> None:
    """Temporal metadata cannot become dimensions, delays or tolerances."""
    value = np.timedelta64(2, "ns")
    series = np.arange(24, dtype=np.float64)
    with pytest.raises(ValueError):
        if field in {"delay", "dimension"}:
            embedding.delay_embed(
                series,
                value if field == "delay" else 2,
                value if field == "dimension" else 3,
            )
        elif field in {"max_lag", "n_bins"}:
            embedding.optimal_delay(
                series,
                value if field == "max_lag" else 4,
                value if field == "n_bins" else 4,
            )
        else:
            embedding.optimal_dimension(
                series,
                1,
                3,
                value if field == "rtol" else 15.0,
                value if field == "atol" else 2.0,
            )
