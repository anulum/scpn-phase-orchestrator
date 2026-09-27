# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Network measurement JSON contracts

from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.visualization.network import (
    coupling_heatmap_json,
    network_graph_json,
)


@pytest.mark.parametrize(
    "encoded_values", ["0.5", "[[0.25, 0.75]]", "[[0.25], [0.75]]"]
)
def test_network_graph_rejects_non_vector_json_node_metrics(
    encoded_values: str,
) -> None:
    """JSON node metrics must remain a vector before graph serialisation."""
    knm = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="R_values must be 1-D"):
        network_graph_json(knm, R_values=json.loads(encoded_values))
    payload = json.loads(network_graph_json(knm, R_values=[0.25, 0.75]))
    assert [node["R"] for node in payload["nodes"]] == [0.25, 0.75]
    assert payload["links"] == [{"source": 0, "target": 1, "weight": 1.0}]


@pytest.mark.parametrize(
    "encoded_names", ['"Alpha"', '{"0": "Alpha", "1": "Beta"}', "1", "true"]
)
def test_network_views_reject_non_list_json_layer_names(encoded_names: str) -> None:
    """Both D3 payload formats refuse JSON names with the wrong container type."""
    knm = np.array([[0.0, -0.5], [-0.5, 0.0]])
    names = json.loads(encoded_names)
    with pytest.raises(ValueError, match="layer_names must be a list"):
        network_graph_json(knm, layer_names=names)
    with pytest.raises(ValueError, match="layer_names must be a list"):
        coupling_heatmap_json(knm, layer_names=names)

    valid_names = ["Alpha", "Beta"]
    graph = json.loads(network_graph_json(knm, layer_names=valid_names))
    heatmap = json.loads(coupling_heatmap_json(knm, layer_names=valid_names))
    assert [node["name"] for node in graph["nodes"]] == valid_names
    assert graph["links"] == [{"source": 0, "target": 1, "weight": -0.5}]
    assert heatmap["labels"] == valid_names
    assert heatmap["matrix"] == [[0.0, -0.5], [-0.5, 0.0]]


@pytest.mark.parametrize(
    "dtype", ["timedelta64[ms]", "timedelta64[ns]", "datetime64[ms]", "object"]
)
def test_network_views_reject_temporal_and_object_coupling(dtype: str) -> None:
    knm = np.array([[0.0, 1.0], [1.0, 0.0]]).astype(dtype)
    with pytest.raises(ValueError, match="knm"):
        network_graph_json(knm)
    with pytest.raises(ValueError, match="knm"):
        coupling_heatmap_json(knm)


@pytest.mark.parametrize(
    "dtype", ["timedelta64[ms]", "timedelta64[ns]", "datetime64[ms]"]
)
def test_network_view_rejects_temporal_order_parameters(dtype: str) -> None:
    knm = np.array([[0.0, 1.0], [1.0, 0.0]])
    values = np.array([0, 1]).astype(dtype)
    with pytest.raises(ValueError, match="R_values"):
        network_graph_json(knm, R_values=cast("list[float]", values))


def test_network_views_preserve_integer_edges_and_real_node_values() -> None:
    payload = json.loads(
        network_graph_json(np.array([[0, 2], [2, 0]]), R_values=[0.25, 0.75])
    )
    assert payload["links"] == [{"source": 0, "target": 1, "weight": 2.0}]
    assert [node["R"] for node in payload["nodes"]] == [0.25, 0.75]


def test_network_view_rejects_boolean_aliases_in_numeric_sequences() -> None:
    with pytest.raises(ValueError, match="R_values"):
        network_graph_json(np.array([[0.0, 1.0], [1.0, 0.0]]), R_values=[True, 0.75])
