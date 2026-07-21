# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Neuromorphic IR (NIR-structural) export

"""Deterministic NIR-structural graph export for SNN schedule manifests.

This module serialises the LIF populations and inter-population projections of a
reviewable neuromorphic schedule into a portable, deterministic, SHA-256-hashed
graph in the *shape* of the Neuromorphic Intermediate Representation
(`neuromorphs/NIR <https://github.com/neuromorphs/NIR>`_): a directed graph whose
nodes are neuron populations and whose edges are ``(source, target)`` connection
tuples with weights.

Honesty boundary
----------------
The output is a **structural subset**, not a spec-validated NIR export, and it
says so in its metadata (``conformance = "structural_subset"``). The reference
NIR ``LIF`` primitive is parametrised by ``tau`` [ms], ``R`` [Ω], ``v_leak`` [mV],
``v_reset`` [mV], and ``v_threshold`` [mV] (verified at source, 2026-07-21). The
SCPN SNN bridge models an Abbott-1999 analytic-rate LIF that genuinely defines
only the membrane and refractory time constants and a *normalised* firing
threshold; it does not define the NIR physical parameters ``R``, ``v_leak``, or
``v_reset``. Rather than fabricate those values, the export emits only the
parameters the model actually holds, marks the firing threshold as normalised,
and lists the unmodelled NIR parameters explicitly in
``unmodelled_nir_lif_parameters``. It adds no dependency and touches no hardware.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

__all__ = [
    "NIR_STRUCTURAL_FORMAT",
    "NIR_STRUCTURAL_FORMAT_VERSION",
    "UNMODELLED_NIR_LIF_PARAMETERS",
    "NeuromorphicIRGraph",
    "to_nir_graph",
]

#: Identifier of the emitted format. Deliberately not ``"nir"`` — the payload is a
#: structural subset of NIR, not a validated NIR export.
NIR_STRUCTURAL_FORMAT = "scpn-neuromorphic-ir-structural"

#: Semantic version of the structural-subset schema emitted by this module.
NIR_STRUCTURAL_FORMAT_VERSION = "0.1.0"

#: NIR ``LIF`` primitive parameters the SCPN Abbott-rate LIF does not model, so
#: they are omitted from node attributes rather than fabricated.
UNMODELLED_NIR_LIF_PARAMETERS: tuple[str, ...] = (
    "R_ohm",
    "v_leak_mV",
    "v_reset_mV",
)


def _require_finite_non_negative(value: object, *, field: str) -> float:
    """Return *value* as a finite, non-negative float, else raise ``ValueError``.

    Parameters
    ----------
    value : object
        The candidate value.
    field : str
        The field name used in the error message.

    Returns
    -------
    float
        The validated non-negative finite float.

    Raises
    ------
    ValueError
        If *value* is boolean, non-real, non-finite, or negative.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a real number")
    parsed = float(value)
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite")
    if parsed < 0.0:
        raise ValueError(f"{field} must be >= 0")
    return parsed


def _require_text(value: object, *, field: str) -> str:
    """Return *value* as a non-empty string, else raise ``ValueError``."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class NeuromorphicIRGraph:
    """A deterministic NIR-structural graph of LIF nodes and weighted edges."""

    nodes: tuple[dict[str, object], ...]
    edges: tuple[dict[str, object], ...]
    metadata: dict[str, object]

    def to_record(self) -> dict[str, object]:
        """Return a deterministic, JSON-safe mapping of the graph.

        Returns
        -------
        dict[str, object]
            The ``metadata`` / ``nodes`` / ``edges`` mapping, with the node and
            edge tuples materialised as lists in insertion order.
        """
        return {
            "metadata": dict(self.metadata),
            "nodes": [dict(node) for node in self.nodes],
            "edges": [dict(edge) for edge in self.edges],
        }

    def canonical_json(self) -> str:
        """Return the canonical (sorted, compact) JSON encoding of the graph.

        Returns
        -------
        str
            A deterministic JSON string with sorted keys and no whitespace.
        """
        return json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        """Return the SHA-256 hex digest of the canonical JSON encoding.

        Returns
        -------
        str
            The 64-character lowercase hexadecimal SHA-256 digest.
        """
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _node_from_population(
    population: dict[str, object],
    *,
    tau_membrane_ms: float,
    tau_refractory_ms: float,
    v_threshold_normalised: float,
) -> dict[str, object]:
    """Return a NIR-structural LIF node from one schedule population record."""
    if not isinstance(population, dict):
        raise ValueError("each population must be a mapping")
    name = _require_text(population.get("name"), field="population.name")
    estimated_rate_hz = _require_finite_non_negative(
        population.get("estimated_rate_hz", 0.0), field="population.estimated_rate_hz"
    )
    return {
        "id": name,
        "type": "LIF",
        "tau_membrane_ms": tau_membrane_ms,
        "tau_refractory_ms": tau_refractory_ms,
        "v_threshold_normalised": v_threshold_normalised,
        "estimated_rate_hz": estimated_rate_hz,
    }


def _edge_from_projection(projection: dict[str, object]) -> dict[str, object]:
    """Return a NIR-structural weighted edge from one projection record."""
    if not isinstance(projection, dict):
        raise ValueError("each projection must be a mapping")
    source = _require_text(projection.get("source"), field="projection.source")
    target = _require_text(projection.get("target"), field="projection.target")
    weight = _require_finite_non_negative(
        projection.get("weight"), field="projection.weight"
    )
    delay_ms = _require_finite_non_negative(
        projection.get("delay_ms", 0.0), field="projection.delay_ms"
    )
    return {
        "type": "Linear",
        "source": source,
        "target": target,
        "weight": weight,
        "delay_ms": delay_ms,
    }


def to_nir_graph(
    populations: list[dict[str, object]],
    projections: list[dict[str, object]],
    *,
    tau_membrane_ms: float,
    tau_refractory_ms: float,
    v_threshold_normalised: float = 1.0,
) -> NeuromorphicIRGraph:
    """Compile schedule populations and projections into a NIR-structural graph.

    Parameters
    ----------
    populations : list[dict[str, object]]
        Per-population records from a neuromorphic schedule manifest; each must
        carry a non-empty ``name`` and a non-negative ``estimated_rate_hz``.
    projections : list[dict[str, object]]
        Inter-population projection records; each must carry non-empty
        ``source`` / ``target`` node names and a non-negative ``weight``.
    tau_membrane_ms : float
        LIF membrane time constant in milliseconds.
    tau_refractory_ms : float
        LIF refractory period in milliseconds.
    v_threshold_normalised : float, optional
        The normalised firing threshold of the Abbott-rate LIF (dimensionless;
        not the NIR ``v_threshold`` in mV, which the model does not define).

    Returns
    -------
    NeuromorphicIRGraph
        The deterministic NIR-structural graph. Every edge's ``source`` and
        ``target`` is guaranteed to reference a declared node id.

    Raises
    ------
    ValueError
        If a record is malformed, or an edge references an undeclared node.
    """
    tau_membrane_ms = _require_finite_non_negative(
        tau_membrane_ms, field="tau_membrane_ms"
    )
    tau_refractory_ms = _require_finite_non_negative(
        tau_refractory_ms, field="tau_refractory_ms"
    )
    v_threshold_normalised = _require_finite_non_negative(
        v_threshold_normalised, field="v_threshold_normalised"
    )
    nodes = tuple(
        _node_from_population(
            population,
            tau_membrane_ms=tau_membrane_ms,
            tau_refractory_ms=tau_refractory_ms,
            v_threshold_normalised=v_threshold_normalised,
        )
        for population in populations
    )
    node_ids = {node["id"] for node in nodes}
    edges = tuple(_edge_from_projection(projection) for projection in projections)
    for edge in edges:
        for endpoint in ("source", "target"):
            if edge[endpoint] not in node_ids:
                raise ValueError(
                    f"projection {endpoint} {edge[endpoint]!r} references an "
                    "undeclared node"
                )
    metadata: dict[str, object] = {
        "format": NIR_STRUCTURAL_FORMAT,
        "format_version": NIR_STRUCTURAL_FORMAT_VERSION,
        "reference_spec": "neuromorphs/NIR",
        "conformance": "structural_subset",
        "node_type": "LIF",
        "edge_type": "Linear",
        "v_threshold_unit": "normalised",
        "unmodelled_nir_lif_parameters": list(UNMODELLED_NIR_LIF_PARAMETERS),
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
    return NeuromorphicIRGraph(nodes=nodes, edges=edges, metadata=metadata)
