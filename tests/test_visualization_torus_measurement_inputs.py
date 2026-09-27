# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — visualization torus measurement ingress

"""Verify original source-type refusal through visualization.torus public calls."""

from __future__ import annotations

import json

import numpy as np
import pytest

import scpn_phase_orchestrator.visualization.torus as module
from tests.measurement_samples import source_alias


@pytest.mark.parametrize("encoded_names", ['"Phase"', '{"0": "Phase"}', "0", "false"])
def test_phase_wheel_refuses_non_list_json_names(encoded_names: str) -> None:
    """Malformed JSON names cannot replace the per-oscillator name vector."""
    phases = np.array([0.0, np.pi / 2.0])
    with pytest.raises(ValueError, match="layer_names must be a list"):
        module.phase_wheel_json(phases, layer_names=json.loads(encoded_names))

    payload = json.loads(module.phase_wheel_json(phases, layer_names=["A", "B"]))
    assert payload["oscillators"] == [
        {"name": "A", "phase": 0.0, "x": 1.0, "y": 0.0},
        {"name": "B", "phase": 1.5708, "x": 0.0, "y": 1.0},
    ]
    np.testing.assert_array_equal(phases, [0.0, np.pi / 2.0])


@pytest.mark.parametrize("kind", ["text", "duration", "object_duration"])
def test_original_measurement_types_are_refused(kind: str) -> None:
    """An otherwise valid numerical shape cannot erase text or duration units."""
    with pytest.raises(ValueError):
        module.phase_wheel_json(source_alias(kind, (3,)))
