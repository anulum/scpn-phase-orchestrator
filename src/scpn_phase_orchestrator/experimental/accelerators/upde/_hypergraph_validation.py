# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - hypergraph backend validation

"""Shared validation for direct hypergraph accelerator bridges."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
TWO_PI = 2.0 * np.pi

__all__ = ["TWO_PI", "validate_hypergraph_inputs", "validate_hypergraph_output"]


def _as_real_vector(value: Any, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite real vector, else raise."""
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional float64 vector")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be a finite real-valued vector")
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real-valued")
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _as_index_vector(value: Any, *, name: str) -> IntArray:
    """Return ``value`` as a validated integer index vector, else raise."""
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int64 vector")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must contain non-boolean integer indices")
    return np.ascontiguousarray(arr, dtype=np.int64)


def _validate_positive_int(value: Any, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be >= 1 as a non-boolean integer")
    return int(value)


def _validate_non_negative_int(value: Any, *, name: str) -> int:
    """Return ``value`` as a non-negative integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be >= 0 as a non-boolean integer")
    return int(value)


def _validate_real_scalar(value: Any, *, name: str) -> float:
    """Return ``value`` as a finite real scalar, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite real")
    return out


def _validate_positive_real(value: Any, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    out = _validate_real_scalar(value, name=name)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _validate_square_flat(value: Any, *, name: str, n: int) -> FloatArray:
    """Return ``value`` as a validated flattened square matrix, else raise."""
    arr = _as_real_vector(value, name=name)
    if arr.size not in (0, n * n):
        raise ValueError(f"{name} must be empty or have exactly n*n entries")
    if arr.size == n * n:
        mat = arr.reshape(n, n)
        if np.any(np.diag(mat) != 0.0):
            raise ValueError(f"{name} diagonal must be zero")
    return arr


def _validate_edge_encoding(
    edge_nodes: Any,
    edge_offsets: Any,
    edge_strengths: Any,
    *,
    n: int,
) -> tuple[IntArray, IntArray, FloatArray]:
    """Return the validated flat hyperedge encoding, else raise."""
    nodes = _as_index_vector(edge_nodes, name="edge_nodes")
    offsets = _as_index_vector(edge_offsets, name="edge_offsets")
    strengths = _as_real_vector(edge_strengths, name="edge_strengths")
    if offsets.size != strengths.size:
        raise ValueError("edge_offsets and edge_strengths must have equal length")
    if offsets.size == 0:
        if nodes.size != 0:
            raise ValueError("edge_nodes must be empty when edge_offsets is empty")
        return nodes, offsets, strengths
    if nodes.size == 0:
        raise ValueError("edge_nodes must not be empty when edges are present")
    if offsets[0] != 0:
        raise ValueError("edge_offsets must start at zero")
    if np.any(offsets < 0) or np.any(offsets >= nodes.size):
        raise ValueError("edge_offsets must reference edge_nodes positions")
    if offsets.size > 1 and np.any(np.diff(offsets) <= 0):
        raise ValueError("edge_offsets must be strictly increasing")
    if np.any(nodes < 0) or np.any(nodes >= n):
        raise ValueError("edge_nodes entries must be valid oscillator indices")
    stops = np.concatenate((offsets[1:], np.array([nodes.size], dtype=np.int64)))
    for edge_index, (start, stop) in enumerate(zip(offsets, stops, strict=True)):
        if int(stop - start) < 2:
            raise ValueError(f"hyperedge {edge_index} must contain at least two nodes")
        edge = nodes[int(start) : int(stop)]
        if np.unique(edge).size != edge.size:
            raise ValueError(f"hyperedge {edge_index} must not repeat nodes")
    return nodes, offsets, strengths


def validate_hypergraph_inputs(
    phases: Any,
    omegas: Any,
    n: Any,
    edge_nodes: Any,
    edge_offsets: Any,
    edge_strengths: Any,
    knm_flat: Any,
    alpha_flat: Any,
    zeta: Any,
    psi: Any,
    dt: Any,
    n_steps: Any,
) -> tuple[
    FloatArray,
    FloatArray,
    int,
    IntArray,
    IntArray,
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    float,
    float,
    int,
]:
    """Validate and normalise direct hypergraph backend inputs."""
    n_i = _validate_positive_int(n, name="n")
    p = _as_real_vector(phases, name="phases")
    o = _as_real_vector(omegas, name="omegas")
    if p.size != n_i or o.size != n_i:
        raise ValueError("phases and omegas must both have length n")
    nodes, offsets, strengths = _validate_edge_encoding(
        edge_nodes,
        edge_offsets,
        edge_strengths,
        n=n_i,
    )
    knm = _validate_square_flat(knm_flat, name="knm_flat", n=n_i)
    alpha = _validate_square_flat(alpha_flat, name="alpha_flat", n=n_i)
    zeta_f = _validate_real_scalar(zeta, name="zeta")
    psi_f = _validate_real_scalar(psi, name="psi")
    dt_f = _validate_positive_real(dt, name="dt")
    steps_i = _validate_non_negative_int(n_steps, name="n_steps")
    return (
        p,
        o,
        n_i,
        nodes,
        offsets,
        strengths,
        knm,
        alpha,
        zeta_f,
        psi_f,
        dt_f,
        steps_i,
    )


def validate_hypergraph_output(value: Any, *, n: int) -> FloatArray:
    """Validate direct backend phase output."""
    out = _as_real_vector(value, name="hypergraph backend output")
    if out.size != n:
        raise ValueError(f"hypergraph backend output must have length {n}")
    if np.any(out < -1e-12) or np.any(out >= TWO_PI + 1e-12):
        raise ValueError("hypergraph backend output phases must be in [0, 2*pi)")
    return np.mod(out, TWO_PI)
