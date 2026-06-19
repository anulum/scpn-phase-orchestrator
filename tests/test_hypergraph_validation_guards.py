# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend hypergraph validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.upde._hypergraph_validation import (  # noqa: E501
    validate_hypergraph_inputs,
    validate_hypergraph_output,
)


def _inputs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phases": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "omegas": np.zeros(3, dtype=np.float64),
        "n": 3,
        "edge_nodes": np.array([0, 1, 2], dtype=np.int64),
        "edge_offsets": np.array([0], dtype=np.int64),
        "edge_strengths": np.array([1.0], dtype=np.float64),
        "knm_flat": np.zeros(9, dtype=np.float64),
        "alpha_flat": np.zeros(9, dtype=np.float64),
        "zeta": 0.0,
        "psi": 0.0,
        "dt": 0.01,
        "n_steps": 10,
    }
    payload.update(overrides)
    return payload


class TestHypergraphInputs:
    def test_valid_round_trips(self) -> None:
        result = validate_hypergraph_inputs(**_inputs())
        assert result[2] == 3  # n

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n": 0}, "n must be >= 1"),
            ({"phases": np.array([0.1, 0.2])}, "must both have length n"),
            ({"omegas": np.zeros(2)}, "must both have length n"),
            (
                {"edge_offsets": np.array([0, 1]), "edge_strengths": np.array([1.0])},
                "equal length",
            ),
            ({"edge_nodes": np.array([0])}, "hyperedge 0 must contain at least two"),
            (
                {"edge_nodes": np.zeros((1, 3), dtype=np.int64)},
                "edge_nodes must be a one-dimensional",
            ),
            (
                {"edge_nodes": np.array([0.0, 1.0, 2.0])},
                "edge_nodes must contain non-boolean integer",
            ),
            ({"psi": True}, "psi must be finite real"),
            ({"edge_nodes": np.array([0, 1, 9])}, "edge_nodes entries must be valid"),
            ({"edge_nodes": np.array([0, 1, 1])}, "must not repeat nodes"),
            ({"edge_offsets": np.array([1])}, "must start at zero"),
            ({"knm_flat": np.zeros(4)}, "empty or have exactly n.n"),
            ({"knm_flat": np.eye(3).ravel()}, "knm_flat diagonal must be zero"),
            ({"dt": 0.0}, "dt must be positive"),
            ({"n_steps": -1}, "n_steps must be >= 0"),
            ({"zeta": float("inf")}, "zeta must be finite"),
        ],
    )
    def test_rejects_corrupt_input(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_hypergraph_inputs(**_inputs(**overrides))


class TestHypergraphOutput:
    def test_valid_round_trips(self) -> None:
        out = validate_hypergraph_output(np.array([0.1, 0.2, 0.3]), n=3)
        assert out.shape == (3,)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.array([0.1, 0.2]), "must have length 3"),
            (np.array([0.1, 0.2, 10.0]), r"phases must be in \[0, 2\*pi\)"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_hypergraph_output(value, n=3)
