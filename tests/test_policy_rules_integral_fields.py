# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — policy rule integer-field tests

"""Policy rules refuse non-integral layer indices and fire limits.

The loader read ``condition.layer`` and ``max_fires`` with ``int()``, so a
rule written with ``layer: 1.9`` watched layer 1 and ``max_fires: 2.5`` fired
twice, with no error. A text value such as ``"2"`` was also accepted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules


def _write_policy(tmp_path: Path, *, layer: object, max_fires: object) -> Path:
    """Write a one-rule policy with the given layer index and fire limit."""
    policy = {
        "rules": [
            {
                "name": "guard",
                "regime": ["DEGRADED"],
                "condition": {
                    "metric": "R",
                    "layer": layer,
                    "op": "<",
                    "threshold": 0.5,
                },
                "action": {"knob": "K", "scope": "global", "value": 0.1, "ttl_s": 1.0},
                "max_fires": max_fires,
            }
        ]
    }
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    return path


@pytest.mark.parametrize("layer", [1.9, 1.0, "1", True])
def test_non_integral_layer_is_refused(tmp_path: Path, layer: object) -> None:
    """A layer index that is not an integer does not load."""
    with pytest.raises(ValueError, match="condition.layer must be a non-negative"):
        load_policy_rules(_write_policy(tmp_path, layer=layer, max_fires=1))


@pytest.mark.parametrize("max_fires", [2.5, "2", False])
def test_non_integral_max_fires_is_refused(tmp_path: Path, max_fires: object) -> None:
    """A fire limit that is not an integer does not load."""
    with pytest.raises(ValueError, match="rule.max_fires must be a non-negative"):
        load_policy_rules(_write_policy(tmp_path, layer=1, max_fires=max_fires))


def test_integral_fields_load_unchanged(tmp_path: Path) -> None:
    """Integer layer indices and fire limits load as written."""
    (rule,) = load_policy_rules(_write_policy(tmp_path, layer=1, max_fires=3))

    assert rule.condition.layer == 1  # type: ignore[union-attr]
    assert rule.max_fires == 3
