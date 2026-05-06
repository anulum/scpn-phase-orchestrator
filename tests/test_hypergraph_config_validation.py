# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — hypergraph engine config validation tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.hypergraph import Hyperedge, HypergraphEngine


@pytest.mark.parametrize("n_oscillators", [False, 0, -1, 1.5, "4"])
def test_hypergraph_engine_rejects_invalid_oscillator_count(
    n_oscillators: Any,
) -> None:
    with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
        HypergraphEngine(n_oscillators=n_oscillators, dt=0.01)


@pytest.mark.parametrize("dt", [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"])
def test_hypergraph_engine_rejects_invalid_timestep(dt: Any) -> None:
    with pytest.raises(ValueError, match="dt must be positive finite real"):
        HypergraphEngine(n_oscillators=4, dt=dt)


@pytest.mark.parametrize(
    "edge",
    [
        Hyperedge(nodes=(0,), strength=1.0),
        Hyperedge(nodes=(0, 0), strength=1.0),
        Hyperedge(nodes=(0, 4), strength=1.0),
        Hyperedge(nodes=(0, False), strength=1.0),
        Hyperedge(nodes=(0, 1), strength=float("nan")),
        Hyperedge(nodes=(0, 1), strength=True),
    ],
)
def test_hypergraph_engine_rejects_invalid_constructor_edges(edge: Hyperedge) -> None:
    with pytest.raises(ValueError):
        HypergraphEngine(n_oscillators=4, dt=0.01, hyperedges=[edge])


@pytest.mark.parametrize(
    ("nodes", "strength"),
    [
        ((0,), 1.0),
        ((0, 0), 1.0),
        ((0, 4), 1.0),
        ((0, "1"), 1.0),
        ((0, 1), float("inf")),
        ((0, 1), False),
    ],
)
def test_hypergraph_engine_add_edge_rejects_invalid_edges(
    nodes: tuple[Any, ...],
    strength: Any,
) -> None:
    engine = HypergraphEngine(n_oscillators=4, dt=0.01)

    with pytest.raises(ValueError):
        engine.add_edge(nodes=nodes, strength=strength)


@pytest.mark.parametrize("order", [False, 0, 1, -1, 1.5, "3", 5])
def test_hypergraph_engine_add_all_to_all_rejects_invalid_order(order: Any) -> None:
    engine = HypergraphEngine(n_oscillators=4, dt=0.01)

    with pytest.raises(ValueError):
        engine.add_all_to_all(order=order, strength=1.0)


@pytest.mark.parametrize("strength", [False, float("nan"), float("inf"), "1.0"])
def test_hypergraph_engine_add_all_to_all_rejects_invalid_strength(
    strength: Any,
) -> None:
    engine = HypergraphEngine(n_oscillators=4, dt=0.01)

    with pytest.raises(ValueError):
        engine.add_all_to_all(order=3, strength=strength)


@pytest.mark.parametrize("n_steps", [False, 0, -1, 1.5, "10"])
def test_hypergraph_engine_run_rejects_invalid_step_count(n_steps: Any) -> None:
    engine = HypergraphEngine(n_oscillators=4, dt=0.01)
    phases = np.zeros(4, dtype=np.float64)
    omegas = np.ones(4, dtype=np.float64)

    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        engine.run(phases, omegas, n_steps=n_steps)


def test_hypergraph_engine_normalises_accepted_numpy_scalars_and_edges() -> None:
    engine = HypergraphEngine(
        n_oscillators=np.int64(4),
        dt=np.float64(0.01),
        hyperedges=[
            Hyperedge(nodes=(np.int64(0), np.int64(1)), strength=np.float64(0.5))
        ],
    )

    assert engine._n == 4
    assert pytest.approx(0.01) == engine._dt
    assert engine._hyperedges == [Hyperedge(nodes=(0, 1), strength=0.5)]
