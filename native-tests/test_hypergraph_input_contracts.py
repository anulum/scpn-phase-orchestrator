# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native hypergraph input contracts

"""Exercise malformed native hypergraph buffers without crossing into Rust panics."""

from __future__ import annotations

import numpy as np
import pytest
import spo_kernel


@pytest.mark.parametrize(
    "fault",
    [
        "offset_count",
        "negative_offset",
        "terminal_offset",
        "unordered_offsets",
        "negative_node",
        "large_node",
        "single_node",
        "duplicate_node",
        "phase_size",
        "omega_size",
        "knm_size",
        "alpha_size",
        "nonfinite",
        "zero_n",
        "bad_dt",
    ],
)
def test_flat_encoding_is_refused(fault: str) -> None:
    """The installed entry point reports invalid input as a Python ValueError."""
    phases = np.zeros(3)
    omegas = np.ones(3)
    nodes = np.array([0, 1, 2], dtype=np.int64)
    offsets = np.array([0], dtype=np.int64)
    strengths = np.array([0.1])
    knm = np.zeros(9)
    alpha = np.zeros(9)
    n = 3
    dt = 0.01
    if fault == "offset_count":
        offsets = np.array([0, 3], dtype=np.int64)
    elif fault == "negative_offset":
        offsets[0] = -1
    elif fault == "terminal_offset":
        offsets[0] = 3
    elif fault == "unordered_offsets":
        offsets = np.array([0, 0], dtype=np.int64)
        strengths = np.ones(2)
    elif fault == "negative_node":
        nodes[0] = -1
    elif fault == "large_node":
        nodes[0] = 3
    elif fault == "single_node":
        nodes = np.array([0], dtype=np.int64)
    elif fault == "duplicate_node":
        nodes[1] = 0
    elif fault == "phase_size":
        phases = np.zeros(2)
    elif fault == "omega_size":
        omegas = np.ones(2)
    elif fault == "knm_size":
        knm = np.zeros(8)
    elif fault == "alpha_size":
        alpha = np.zeros(8)
    elif fault == "nonfinite":
        strengths[0] = np.nan
    elif fault == "zero_n":
        n = 0
    elif fault == "bad_dt":
        dt = 0.0
    with pytest.raises(ValueError):
        spo_kernel.hypergraph_run_rust(
            phases, omegas, n, nodes, offsets, strengths, knm, alpha, 0.0, 0.0, dt, 1
        )


@pytest.mark.parametrize("steps", [0, 1, 4])
@pytest.mark.parametrize("fault", ["phase_size", "alpha_size", "node", "nonfinite"])
def test_stateful_run_validates_before_evolution(steps: int, fault: str) -> None:
    """Zero-step runs share the safety contract with advancing runs."""
    phases = np.zeros(2 if fault == "phase_size" else 3)
    alpha = np.zeros(8 if fault == "alpha_size" else 9)
    edges = [([0, 1, 3] if fault == "node" else [0, 1, 2], 0.1)]
    if fault == "nonfinite":
        phases[0] = np.nan
    with pytest.raises(ValueError):
        spo_kernel.PyHypergraphStepper(3).run(
            phases, np.ones(3), edges, np.zeros(9), alpha, 0.0, 0.0, steps
        )


@pytest.mark.parametrize("control", ["n", "n_steps", "zeta", "psi", "dt"])
@pytest.mark.parametrize("alias", [True, np.timedelta64(1, "ns"), "1"])
def test_original_scalar_aliases_are_refused(control: str, alias: object) -> None:
    """Native scalar extraction preserves source type before coercion."""
    kwargs: dict[str, object] = {
        "phases": np.zeros(3),
        "omegas": np.ones(3),
        "n": 3,
        "edge_nodes": np.array([0, 1, 2], dtype=np.int64),
        "edge_offsets": np.array([0], dtype=np.int64),
        "edge_strengths": np.array([0.1]),
        "pairwise_knm": np.zeros(9),
        "alpha": np.zeros(9),
        "zeta": 0.0,
        "psi": 0.0,
        "dt": 0.01,
        "n_steps": 1,
    }
    kwargs[control] = alias
    with pytest.raises(ValueError):
        spo_kernel.hypergraph_run_rust(**kwargs)


def test_flat_and_stateful_runs_match_and_keep_zero_step_identity() -> None:
    """Two real installed interfaces evolve the same valid weighted hyperedge."""
    phases = np.array([0.1, 0.3, 0.8])
    args = (
        phases,
        np.ones(3),
        3,
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([0.1]),
        np.zeros(9),
        np.zeros(9),
        0.0,
        0.0,
        0.01,
    )
    zero = spo_kernel.hypergraph_run_rust(*args, 0)
    np.testing.assert_array_equal(zero, phases)
    flat = spo_kernel.hypergraph_run_rust(*args, 4)
    stateful = spo_kernel.PyHypergraphStepper(3, 0.01).run(
        phases, np.ones(3), [([0, 1, 2], 0.1)], np.zeros(9), np.zeros(9), 0.0, 0.0, 4
    )
    np.testing.assert_allclose(flat, stateful, rtol=1e-13, atol=1e-13)
    assert not np.array_equal(flat, phases)


@pytest.mark.parametrize("pairwise", [False, True])
def test_empty_native_buffers_preserve_uncoupled_and_zero_shift_modes(
    pairwise: bool,
) -> None:
    """Legacy empty buffers retain the real flat and stateful numerical contract."""
    phases = np.array([0.1, 0.3, 0.8])
    omegas = np.ones(3)
    knm = np.zeros(9) if pairwise else np.empty(0)
    expected = np.mod(phases + 0.04, 2.0 * np.pi)
    flat = spo_kernel.hypergraph_run_rust(
        phases,
        omegas,
        3,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0),
        knm,
        np.empty(0),
        0.0,
        0.0,
        0.01,
        4,
    )
    stateful = spo_kernel.PyHypergraphStepper(3, 0.01).run(
        phases, omegas, [], knm, np.empty(0), 0.0, 0.0, 4
    )
    np.testing.assert_allclose(flat, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(stateful, expected, rtol=1e-13, atol=1e-13)
