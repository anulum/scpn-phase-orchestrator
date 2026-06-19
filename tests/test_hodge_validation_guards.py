# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend Hodge input validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_validation import (  # noqa: E501
    validate_hodge_backend_inputs,
)


def _inputs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "knm_flat": np.zeros(9, dtype=np.float64),
        "phases": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "n": 3,
        "edges_flat": np.array([0, 1, 1, 2, 0, 2], dtype=np.int64),
        "n_edges": 3,
        "tris_flat": np.array([0, 1, 2], dtype=np.int64),
        "n_tris": 1,
    }
    payload.update(overrides)
    return payload


class TestHodgeInputs:
    def test_valid_round_trips(self) -> None:
        k, p, n, edges, n_edges, tris, n_tris = validate_hodge_backend_inputs(
            **_inputs()
        )
        assert (n, n_edges, n_tris) == (3, 3, 1)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n": True}, "n must be a non-negative integer"),
            ({"n": -1}, "n must be non-negative"),
            ({"knm_flat": np.zeros(4)}, "knm_flat length .* does not match"),
            ({"phases": np.array([0.1, 0.2])}, "phases length .* does not match"),
            (
                {"phases": np.array([True, False, False])},
                "phases must not contain boolean",
            ),
            ({"phases": np.array([0.1 + 1j, 0.2, 0.3])}, "phases must be real-valued"),
            (
                {"phases": np.array([0.1, np.inf, 0.3])},
                "phases must contain only finite",
            ),
            ({"n_edges": -1}, "n_edges must be non-negative"),
            (
                {"edges_flat": np.array([True, False])},
                "edges_flat must not contain boolean",
            ),
            ({"edges_flat": np.array([0, 1])}, "edges_flat length .* does not match"),
            (
                {"edges_flat": np.array([0, 9, 1, 2, 0, 2])},
                r"edges_flat indices must lie in \[0, 3\)",
            ),
            ({"tris_flat": np.array([0, 1])}, "tris_flat length .* does not match"),
        ],
    )
    def test_rejects_corrupt_input(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_hodge_backend_inputs(**_inputs(**overrides))
