# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling measurement ingress

"""Exercise measurement source types through public coupling and native bridges."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.hodge import hodge_decomposition
from scpn_phase_orchestrator.coupling.infer import infer_coupling_from_timeseries
from scpn_phase_orchestrator.coupling.spatial_modulator import spatial_modulate
from scpn_phase_orchestrator.coupling.spectral import fiedler_value
from scpn_phase_orchestrator.experimental.accelerators.coupling import (
    _spatial_modulator_go as spatial_go,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling import (
    _spatial_modulator_julia as spatial_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling import (
    _spatial_modulator_mojo as spatial_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_go import (
    hodge_decomposition_go,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_julia import (
    hodge_decomposition_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_mojo import (
    hodge_decomposition_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._spectral_go import (
    spectral_eig_go,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._spectral_julia import (
    spectral_eig_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._spectral_mojo import (
    spectral_eig_mojo,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
HodgeFn: TypeAlias = Callable[
    [FloatArray, FloatArray, int, IntArray, int, IntArray, int],
    tuple[FloatArray, FloatArray, FloatArray],
]
SpatialFn: TypeAlias = Callable[
    [FloatArray, FloatArray, int, int, float, int, float, float, float], FloatArray
]
SpectralFn: TypeAlias = Callable[[FloatArray, int], tuple[FloatArray, FloatArray]]


@pytest.mark.parametrize(
    "surface",
    ["hodge_phases", "hodge_knm", "spectral", "spatial_positions", "spatial_knm"],
)
@pytest.mark.parametrize(
    "kind", ["timedelta64[ms]", "timedelta64[ns]", "datetime64[ms]", "object_time"]
)
def test_public_coupling_rejects_temporal_measurements(surface: str, kind: str) -> None:
    """Time-bearing source types cannot become phases, weights or coordinates."""
    matrix = np.ones((3, 3)) - np.eye(3)
    phases = np.array([0.0, 1.0, 2.0])
    raw = phases if surface.endswith(("phases", "positions")) else matrix
    invalid = raw.astype(object if kind == "object_time" else kind)
    if kind == "object_time":
        invalid.flat[0] = np.timedelta64(1, "ms")
    with pytest.raises(ValueError):
        if surface == "hodge_phases":
            hodge_decomposition(matrix, invalid)
        elif surface == "hodge_knm":
            hodge_decomposition(invalid, phases)
        elif surface == "spectral":
            fiedler_value(invalid)
        elif surface == "spatial_positions":
            spatial_modulate(matrix, invalid)
        else:
            spatial_modulate(invalid, phases)


@pytest.mark.parametrize(
    "backend", [spectral_eig_go, spectral_eig_julia, spectral_eig_mojo]
)
@pytest.mark.parametrize("kind", ["timedelta64[ms]", "datetime64[ms]", "object_time"])
def test_direct_spectral_bridges_reject_temporal_weights(
    backend: SpectralFn, kind: str
) -> None:
    """All direct spectral bridges check source units before loader dispatch."""
    raw = (np.ones((3, 3)) - np.eye(3)).ravel()
    invalid = raw.astype(object if kind == "object_time" else kind)
    if kind == "object_time":
        invalid[1] = np.timedelta64(1, "ms")
    with pytest.raises(ValueError, match="knm_flat"):
        backend(invalid, 3)


@pytest.mark.parametrize(
    "kind", ["U8", "S8", "timedelta64[ms]", "datetime64[ms]", "object_time"]
)
def test_inference_rejects_text_and_temporal_phase_series(kind: str) -> None:
    """Invalid time-series ingress cannot publish an inferred coupling graph."""
    raw = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
    invalid = raw.astype(object if kind == "object_time" else kind)
    if kind == "object_time":
        invalid[0, 1] = np.timedelta64(1, "ms")
    with pytest.raises(ValueError, match="phase_series"):
        infer_coupling_from_timeseries(invalid)


def test_coupling_numeric_objects_preserve_real_computations() -> None:
    """Compatible object storage preserves spectral, Hodge and spatial results."""
    matrix = np.ones((3, 3)) - np.eye(3)
    phases = np.array([0.0, 1.0, 2.0])
    assert fiedler_value(matrix.astype(object)) == pytest.approx(fiedler_value(matrix))
    expected = hodge_decomposition(matrix, phases)
    actual = hodge_decomposition(matrix.astype(object), phases.astype(object))
    np.testing.assert_allclose(actual.gradient, expected.gradient, atol=1e-12)
    np.testing.assert_allclose(actual.curl, expected.curl, atol=1e-12)
    np.testing.assert_allclose(actual.harmonic, expected.harmonic, atol=1e-12)
    np.testing.assert_allclose(
        spatial_modulate(matrix.astype(object), phases.astype(object)),
        spatial_modulate(matrix, phases),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "backend",
    [hodge_decomposition_go, hodge_decomposition_julia, hodge_decomposition_mojo],
)
@pytest.mark.parametrize("argument", ["knm_flat", "phases", "edges", "triangles"])
@pytest.mark.parametrize("kind", ["timedelta64[ms]", "datetime64[ms]", "object_time"])
def test_direct_hodge_bridges_reject_temporal_measurements(
    backend: HodgeFn, argument: str, kind: str
) -> None:
    """Hodge phases, weights and topology indices retain their source units."""
    matrix = (np.ones((3, 3)) - np.eye(3)).ravel()
    phases = np.array([0.0, 1.0, 2.0])
    edges = np.array([0, 1, 0, 2, 1, 2], dtype=np.int64)
    triangles = np.array([0, 1, 2], dtype=np.int64)
    raw = {
        "knm_flat": matrix,
        "phases": phases,
        "edges": edges,
        "triangles": triangles,
    }[argument]
    invalid = raw.astype(object if kind == "object_time" else kind)
    if kind == "object_time":
        invalid.flat[0] = np.timedelta64(1, "ms")
    with pytest.raises(ValueError):
        backend(
            invalid if argument == "knm_flat" else matrix,
            invalid if argument == "phases" else phases,
            3,
            invalid if argument == "edges" else edges,
            3,
            invalid if argument == "triangles" else triangles,
            1,
        )


@pytest.mark.parametrize(
    "backend",
    [
        spatial_go.spatial_modulate_go,
        spatial_julia.spatial_modulate_julia,
        spatial_mojo.spatial_modulate_mojo,
    ],
)
@pytest.mark.parametrize("argument", ["coupling", "positions"])
@pytest.mark.parametrize("kind", ["timedelta64[ms]", "datetime64[ms]", "object_time"])
def test_direct_spatial_bridges_reject_temporal_measurements(
    backend: SpatialFn, argument: str, kind: str
) -> None:
    """All spatial bridges check source coordinates/weights before dispatch."""
    matrix = (np.ones((3, 3)) - np.eye(3)).ravel()
    positions = np.array([0.0, 1.0, 2.0])
    raw = matrix if argument == "coupling" else positions
    invalid = raw.astype(object if kind == "object_time" else kind)
    if kind == "object_time":
        invalid.flat[0] = np.timedelta64(1, "ms")
    with pytest.raises(ValueError):
        backend(
            invalid if argument == "coupling" else matrix,
            invalid if argument == "positions" else positions,
            3,
            1,
            1.0,
            0,
            1.0,
            1.0,
            1e-12,
        )


@pytest.mark.parametrize(
    "backend",
    [hodge_decomposition_go, hodge_decomposition_julia, hodge_decomposition_mojo],
)
@pytest.mark.parametrize("argument", ["edges", "triangles"])
@pytest.mark.parametrize("value", [0.5, np.nan, np.inf, 2**64])
def test_hodge_bridges_reject_lossy_index_conversion(
    backend: HodgeFn, argument: str, value: object
) -> None:
    """Invalid indices cannot silently truncate, wrap or become graph vertices."""
    matrix = (np.ones((3, 3)) - np.eye(3)).ravel()
    phases = np.array([0.0, 1.0, 2.0])
    edges = np.array([0, 1, 0, 2, 1, 2], dtype=np.int64)
    triangles = np.array([0, 1, 2], dtype=np.int64)
    invalid = (edges if argument == "edges" else triangles).astype(object)
    invalid[0] = value
    with pytest.raises(ValueError, match="integer"):
        backend(
            matrix,
            phases,
            3,
            invalid if argument == "edges" else edges,
            3,
            invalid if argument == "triangles" else triangles,
            1,
        )
