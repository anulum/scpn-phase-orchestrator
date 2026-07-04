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

from scpn_phase_orchestrator.upde._hypergraph_validation import (  # noqa: E501
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

    def test_accepts_empty_edge_encoding(self) -> None:
        result = validate_hypergraph_inputs(
            **_inputs(
                edge_nodes=np.array([], dtype=np.int64),
                edge_offsets=np.array([], dtype=np.int64),
                edge_strengths=np.array([], dtype=np.float64),
            )
        )
        assert result[3].size == 0

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n": 0}, "n must be >= 1"),
            ({"phases": np.zeros((1, 3), dtype=np.float64)}, "phases must be"),
            ({"phases": np.array(["bad", "phase", "value"])}, "phases must"),
            ({"phases": np.array([0.1, 0.2, 0.3j])}, "phases must be real"),
            ({"phases": np.array([0.1, np.inf, 0.3])}, "phases must contain"),
            ({"phases": np.array([0.1, 0.2])}, "must both have length n"),
            ({"omegas": np.zeros(2)}, "must both have length n"),
            (
                {"phases": np.array([True, False, True], dtype=object)},
                "phases must be real-valued, not boolean",
            ),
            (
                {"phases": np.array(["0.1", "0.2", "0.3"])},
                "phases must not contain numeric-string aliases",
            ),
            (
                {"omegas": np.array(["0.0", "0.1", "0.2"])},
                "omegas must not contain numeric-string aliases",
            ),
            (
                {"edge_offsets": np.array([0, 1]), "edge_strengths": np.array([1.0])},
                "equal length",
            ),
            (
                {
                    "edge_offsets": np.array([], dtype=np.int64),
                    "edge_strengths": np.array([], dtype=np.float64),
                },
                "edge_nodes must be empty",
            ),
            (
                {
                    "edge_nodes": np.array([], dtype=np.int64),
                    "edge_offsets": np.array([0], dtype=np.int64),
                },
                "edge_nodes must not be empty",
            ),
            (
                {"edge_offsets": np.array([0, 3]), "edge_strengths": np.ones(2)},
                "edge_offsets must reference",
            ),
            (
                {"edge_offsets": np.array([0, 0]), "edge_strengths": np.ones(2)},
                "edge_offsets must be strictly increasing",
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
            (
                {"edge_nodes": np.array(["0", "1", "2"])},
                "edge_nodes must not contain numeric-string aliases",
            ),
            (
                {"edge_nodes": np.array([0, True, 2], dtype=object)},
                "edge_nodes must contain non-boolean integer",
            ),
            (
                {"edge_strengths": np.array(["0.4"])},
                "edge_strengths must not contain numeric-string aliases",
            ),
            (
                {"knm_flat": np.array(["0.0"] * 9)},
                "knm_flat must not contain numeric-string aliases",
            ),
            (
                {"alpha_flat": np.array(["0.0"] * 9)},
                "alpha_flat must not contain numeric-string aliases",
            ),
            ({"n": "3"}, "n must not be a numeric-string alias"),
            ({"psi": True}, "psi must be finite real"),
            ({"psi": "0.0"}, "psi must not be a numeric-string alias"),
            ({"zeta": "0.0"}, "zeta must not be a numeric-string alias"),
            ({"zeta": "not numeric"}, "zeta must be finite real"),
            ({"zeta": " "}, "zeta must be finite real"),
            ({"dt": "0.01"}, "dt must not be a numeric-string alias"),
            ({"n_steps": True}, "n_steps must be a non-boolean integer"),
            ({"n_steps": "10"}, "n_steps must not be a numeric-string alias"),
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
            (
                np.array(["0.1", "0.2", "0.3"]),
                "hypergraph backend output must not contain numeric-string aliases",
            ),
            (
                "0.1",
                "hypergraph backend output must not contain numeric-string aliases",
            ),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_hypergraph_output(value, n=3)
