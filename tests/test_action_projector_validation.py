# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — ActionProjector validation contracts

"""Validation contracts for ActionProjector constructor and projection inputs."""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.actuation.mapper import ControlAction


def test_u1_action_projector_rejects_non_finite_rate_limit() -> None:
    with pytest.raises(ValueError, match="finite >= 0"):
        ActionProjector(rate_limits={"K": float("nan")}, value_bounds={"K": (0.0, 1.0)})


def test_u1_action_projector_rejects_non_dict_value_bounds() -> None:
    with pytest.raises(TypeError, match="value_bounds must be a dict"):
        ActionProjector(  # type: ignore[arg-type]
            rate_limits={"K": 0.1},
            value_bounds=[("K", (0.0, 1.0))],
        )


def test_u1_action_projector_rejects_blank_rate_limit_knob() -> None:
    with pytest.raises(ValueError, match="knob name must be non-empty str"):
        ActionProjector(rate_limits={" ": 0.1}, value_bounds={"K": (0.0, 1.0)})


def test_u1_action_projector_rejects_boolean_rate_limit_value() -> None:
    with pytest.raises(TypeError, match="must be finite real"):
        ActionProjector(
            rate_limits={"K": True},  # type: ignore[dict-item]
            value_bounds={"K": (0.0, 1.0)},
        )


def test_u1_action_projector_rejects_negative_rate_limit_value() -> None:
    with pytest.raises(ValueError, match="finite >= 0"):
        ActionProjector(rate_limits={"K": -0.1}, value_bounds={"K": (0.0, 1.0)})


def test_u1_action_projector_rejects_blank_value_bound_knob() -> None:
    with pytest.raises(ValueError, match="value-bound knob name must be non-empty str"):
        ActionProjector(rate_limits={"K": 0.1}, value_bounds={" ": (0.0, 1.0)})


def test_u1_action_projector_rejects_non_tuple_value_bounds() -> None:
    with pytest.raises(TypeError, match="must be a 2-tuple"):
        ActionProjector(
            rate_limits={"K": 0.1},
            value_bounds={"K": [0.0, 1.0]},  # type: ignore[dict-item]
        )


def test_u1_action_projector_rejects_wrong_arity_value_bounds() -> None:
    with pytest.raises(TypeError, match="must be a 2-tuple"):
        ActionProjector(
            rate_limits={"K": 0.1},
            value_bounds={"K": (0.0, 1.0, 2.0)},  # type: ignore[dict-item]
        )


def test_u1_action_projector_rejects_boolean_value_bounds() -> None:
    with pytest.raises(TypeError, match="must be finite reals"):
        ActionProjector(
            rate_limits={"K": 0.1},
            value_bounds={"K": (True, 1.0)},  # type: ignore[dict-item]
        )


def test_u1_action_projector_rejects_non_finite_value_bounds() -> None:
    with pytest.raises(ValueError, match="must be finite reals"):
        ActionProjector(
            rate_limits={"K": 0.1},
            value_bounds={"K": (0.0, float("inf"))},
        )


def test_u1_action_projector_rejects_inverted_value_bounds() -> None:
    with pytest.raises(ValueError, match="require lo <= hi"):
        ActionProjector(
            rate_limits={"K": 0.1},
            value_bounds={"K": (1.0, 0.0)},
        )


def test_u1_action_projector_rejects_non_dict_rate_limits() -> None:
    with pytest.raises(TypeError, match="rate_limits must be a dict"):
        ActionProjector(  # type: ignore[arg-type]
            rate_limits=[("K", 0.1)],
            value_bounds={"K": (0.0, 1.0)},
        )


def test_u1_action_projector_rejects_non_finite_previous_value() -> None:
    projector = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 1.0)})
    action = ControlAction(
        knob="K",
        value=0.5,
        scope="global",
        ttl_s=1.0,
        justification="u1-test",
    )
    with pytest.raises(ValueError, match="finite real scalar"):
        projector.project(action, float("inf"))


def test_u1_action_projector_rejects_nan_previous_value() -> None:
    projector = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 1.0)})
    action = ControlAction(
        knob="K",
        value=0.5,
        scope="global",
        ttl_s=1.0,
        justification="u1-test",
    )
    with pytest.raises(ValueError, match="finite real scalar"):
        projector.project(action, float("nan"))


def test_u1_action_projector_rejects_boolean_previous_value() -> None:
    projector = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 1.0)})
    action = ControlAction(
        knob="K",
        value=0.5,
        scope="global",
        ttl_s=1.0,
        justification="u1-test",
    )
    with pytest.raises(TypeError, match="finite real scalar"):
        projector.project(action, True)  # type: ignore[arg-type]


def test_u1_action_projector_rejects_non_action_payload() -> None:
    projector = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 1.0)})
    with pytest.raises(TypeError, match="ControlAction"):
        projector.project(object(), 0.0)  # type: ignore[arg-type]


def test_u1_action_projector_rejects_non_finite_action_value() -> None:
    projector = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 1.0)})
    action = ControlAction(
        knob="K",
        value=float("nan"),
        scope="global",
        ttl_s=1.0,
        justification="u1-test",
    )
    with pytest.raises(ValueError, match="action.value must be finite real"):
        projector.project(action, 0.0)
