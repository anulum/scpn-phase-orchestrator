# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio shared-helper tests

"""Contract tests for shared Studio validation primitives."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from scpn_phase_orchestrator.studio.ui_helpers import _shared as shared
from scpn_phase_orchestrator.studio.ui_helpers._state import StudioReplayResult
from scpn_phase_orchestrator.studio.workflow import ExportManifest


def test_shared_deployment_and_optional_primitives() -> None:
    """Deployment helpers deduplicate blockers and normalise optional values."""
    first = ExportManifest.review_artifact(
        target_kind="docker",
        file_name="docker.json",
        payload="{}",
        command="write docker",
        warnings=("missing review", "missing review"),
    )
    second = ExportManifest.review_artifact(
        target_kind="wasm",
        file_name="wasm.json",
        payload="{}",
        command="write wasm",
        warnings=("operator hold",),
    )

    assert shared._deployment_blocked_reasons((first, second)) == (
        "missing review",
        "operator hold",
    )
    assert shared._blocked_target("docker", ("missing review",)) == {
        "target": "docker",
        "status": "blocked",
        "required_artifacts": (),
        "commands": (),
        "operator_action": "resolve blocked reasons before packaging",
        "blocked_reasons": ["missing review"],
    }
    digest = "a" * 64
    assert shared._optional_sha256_hex(None, "digest") is None
    assert shared._optional_sha256_hex(digest, "digest") == digest
    assert shared._optional_non_negative_int(None, "count") is None
    assert shared._optional_non_negative_int(3, "count") == 3
    assert shared._normalise_optional_text_sequence(None, "labels") == ()
    assert shared._normalise_optional_text_sequence(["a", "b"], "labels") == (
        "a",
        "b",
    )


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("yes", "must be a boolean"),
        (2.0, "finite non-negative"),
    ],
)
def test_shared_boolean_and_integer_type_guards(value: object, match: str) -> None:
    """Boolean and integer validators reject incompatible scalar types."""
    if isinstance(value, str):
        with pytest.raises(ValueError, match=match):
            shared._required_bool(value, "flag")
    else:
        with pytest.raises(ValueError, match="must be an integer"):
            shared._positive_int(value, "count", minimum=1)
        with pytest.raises(ValueError, match="must be an integer"):
            shared._non_negative_int(value, "count")


def test_shared_numeric_validators_enforce_bounds() -> None:
    """Numeric validators accept finite values and reject invalid bounds."""
    assert shared._required_bool(True, "flag") is True
    assert shared._non_negative_float(0.0, "score") == 0.0
    assert shared._unit_interval_number(1.0, "ratio") == 1.0
    assert shared._positive_int(2, "count", minimum=1) == 2
    assert shared._non_negative_int(0, "count") == 0
    assert shared._finite_range(0.5, "range", low=0.0, high=1.0) == 0.5
    assert shared._positive_float(0.1, "step") == 0.1

    with pytest.raises(ValueError, match="finite non-negative"):
        shared._non_negative_float(-0.1, "score")
    with pytest.raises(ValueError, match="finite non-negative"):
        shared._non_negative_float(True, "score")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        shared._unit_interval_number(1.1, "ratio")
    with pytest.raises(ValueError, match="at least 3"):
        shared._positive_int(2, "count", minimum=3)
    with pytest.raises(ValueError, match="non-negative"):
        shared._non_negative_int(-1, "count")
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        shared._finite_range(2.0, "range", low=0.0, high=1.0)
    with pytest.raises(ValueError, match="finite"):
        shared._finite_number(True, "value")
    with pytest.raises(ValueError, match="finite"):
        shared._finite_number(float("inf"), "value")
    with pytest.raises(ValueError, match="positive"):
        shared._positive_float(0.0, "step")


def test_shared_sequence_normalisers_validate_text_float_and_gradients() -> None:
    """Sequence normalisers reject malformed labels, numbers, and gradients."""
    assert shared._normalise_text_sequence([" a ", "b"], "labels") == ("a", "b")
    assert shared._normalise_float_sequence([1, 2.5], "values") == (1.0, 2.5)
    assert shared._normalise_information_geometry_gradient(
        [("K", 0.1), ("alpha", -0.2)],
        "gradient",
    ) == (("K", 0.1), ("alpha", -0.2))
    assert shared._require_sequence([1, 2], "items") == [1, 2]

    with pytest.raises(ValueError, match="sequence of strings"):
        shared._normalise_text_sequence("bad", "labels")
    with pytest.raises(ValueError, match="sequence of finite numbers"):
        shared._normalise_float_sequence("bad", "values")
    with pytest.raises(ValueError, match="must not be empty"):
        shared._normalise_float_sequence([], "values")
    with pytest.raises(ValueError, match="knob/value pairs"):
        shared._normalise_information_geometry_gradient("bad", "gradient")
    with pytest.raises(ValueError, match="must not be empty"):
        shared._normalise_information_geometry_gradient([], "gradient")
    with pytest.raises(ValueError, match="knob/value pairs"):
        shared._normalise_information_geometry_gradient(["K"], "gradient")
    with pytest.raises(ValueError, match="knob/value pairs"):
        shared._normalise_information_geometry_gradient([("K", 0.1, 0.2)], "gradient")
    with pytest.raises(ValueError, match="must be a sequence"):
        shared._require_sequence("bad", "items")


def test_shared_digest_connector_and_count_helpers() -> None:
    """Digest, connector, and count helpers validate Studio mappings."""
    digest = "b" * 64
    replay_result = cast(
        "StudioReplayResult",
        SimpleNamespace(canvas_graph={"node_count": 4}),
    )

    assert shared._require_sha256_hex(digest, "digest") == digest
    assert shared._is_sha256_digest(digest.upper()) is True
    assert shared._is_sha256_digest("bad") is False
    assert shared._connector_by_transport(
        {"connectors": [{"transport": "grpc", "endpoint": "localhost"}]},
        "grpc",
    ) == {"transport": "grpc", "endpoint": "localhost"}
    assert shared._mapping_count({"count": 2}, "count") == 2
    assert shared._canvas_graph_count(replay_result, "node_count") == 4

    with pytest.raises(ValueError, match="SHA-256"):
        shared._require_sha256_hex("bad", "digest")
    with pytest.raises(ValueError, match="connectors must be a sequence"):
        shared._connector_by_transport({"connectors": "bad"}, "grpc")
    with pytest.raises(ValueError, match="connector entries"):
        shared._connector_by_transport({"connectors": ["bad"]}, "grpc")
    with pytest.raises(ValueError, match="not found"):
        shared._connector_by_transport({"connectors": []}, "grpc")


def test_shared_layer_and_text_helpers_are_stable() -> None:
    """Layer metrics, channel IDs, and text guards stay deterministic."""
    assert shared._layer_metrics("bad") == ()
    assert shared._layer_metrics(
        [
            "skip",
            {"name": " layer-a ", "R": 0.4},
            {"R": 0.2},
        ]
    ) == (("layer-a", 0.4), ("layer_2", 0.2))
    assert shared._canvas_channel_id("alpha/beta:gamma") == "channel_alpha_beta_gamma"
    assert shared._require_non_empty_text("  value  ", "text") == "value"

    with pytest.raises(ValueError, match="non-empty string"):
        shared._require_non_empty_text("", "text")
