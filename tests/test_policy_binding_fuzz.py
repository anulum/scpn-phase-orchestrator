# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding and policy fuzz tests

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.binding import validate_binding_spec
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyEngine,
    load_policy_rules,
)
from scpn_phase_orchestrator.supervisor.regimes import Regime
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

_SAFE_TEXT = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters=("_", "-"),
    ),
    min_size=1,
    max_size=24,
)


def _valid_binding_data(
    name: str,
    layer_count: int,
    sample_period_s: float,
    control_multiplier: int,
    base_strength: float,
    decay_alpha: float,
) -> dict[str, Any]:
    layers = [
        {
            "name": f"L{idx}",
            "index": idx,
            "oscillator_ids": [f"osc_{idx}"],
            "omegas": [1.0 + 0.1 * idx],
        }
        for idx in range(layer_count)
    ]
    return {
        "name": name,
        "version": "0.2.0",
        "safety_tier": "research",
        "sample_period_s": sample_period_s,
        "control_period_s": sample_period_s * control_multiplier,
        "layers": layers,
        "oscillator_families": {
            "physical": {"channel": "P", "extractor_type": "hilbert", "config": {}}
        },
        "coupling": {"base_strength": base_strength, "decay_alpha": decay_alpha},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [
            {
                "name": "coherence_floor",
                "variable": "R",
                "lower": 0.0,
                "upper": 1.0,
                "severity": "soft",
            }
        ],
        "actuators": [
            {"name": "K_global", "knob": "K", "scope": "global", "limits": [0.0, 1.0]}
        ],
    }


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    name=_SAFE_TEXT,
    layer_count=st.integers(min_value=1, max_value=5),
    sample_period_s=st.floats(
        min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    control_multiplier=st.integers(min_value=1, max_value=20),
    base_strength=st.floats(
        min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
    decay_alpha=st.floats(
        min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
)
def test_generated_binding_specs_load_and_validate(
    tmp_path: Path,
    name: str,
    layer_count: int,
    sample_period_s: float,
    control_multiplier: int,
    base_strength: float,
    decay_alpha: float,
) -> None:
    """Generated binding specs must load and satisfy semantic validation."""
    data = _valid_binding_data(
        name,
        layer_count,
        sample_period_s,
        control_multiplier,
        base_strength,
        decay_alpha,
    )
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    spec = load_binding_spec(path)

    assert validate_binding_spec(spec) == []
    assert spec.control_period_s >= spec.sample_period_s
    assert all(layer.index >= 0 for layer in spec.layers)


def _valid_policy_data(
    name: str,
    metric: str,
    op: str,
    threshold: float,
    value: float,
    ttl_s: float,
) -> dict[str, Any]:
    return {
        "rules": [
            {
                "name": name,
                "regime": ["NOMINAL", "DEGRADED"],
                "condition": {
                    "metric": metric,
                    "layer": 0 if metric == "R" else None,
                    "op": op,
                    "threshold": threshold,
                },
                "action": {
                    "knob": "K",
                    "scope": "global",
                    "value": value,
                    "ttl_s": ttl_s,
                },
                "cooldown_s": 0.0,
                "max_fires": 3,
            }
        ]
    }


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    name=_SAFE_TEXT,
    metric=st.sampled_from(["R", "stability_proxy", "pac_max"]),
    op=st.sampled_from([">", ">=", "<", "<=", "=="]),
    threshold=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    value=st.floats(
        min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    ttl_s=st.floats(
        min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False
    ),
)
def test_generated_policy_rules_load_and_evaluate(
    tmp_path: Path,
    name: str,
    metric: str,
    op: str,
    threshold: float,
    value: float,
    ttl_s: float,
) -> None:
    """Generated policy rules must parse into finite bounded actions."""
    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump(_valid_policy_data(name, metric, op, threshold, value, ttl_s)),
        encoding="utf-8",
    )

    rules = load_policy_rules(path)
    engine = PolicyEngine(rules)
    state = UPDEState(
        layers=[LayerState(R=0.5, psi=0.0)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=0.5,
        regime_id="test",
        pac_max=0.5,
    )
    actions = engine.evaluate(Regime.NOMINAL, state, [0], [])

    assert len(rules) == 1
    assert all(isfinite(action.value) and action.ttl_s >= 0.0 for action in actions)


_JSON_SCALAR = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-10, max_value=10)
    | st.floats(allow_nan=True, allow_infinity=True, width=32)
    | st.text(max_size=20)
)
_JSONISH = st.recursive(
    _JSON_SCALAR,
    lambda children: (
        st.lists(children, max_size=5)
        | st.dictionaries(st.text(max_size=12), children, max_size=5)
    ),
    max_leaves=20,
)


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(raw=_JSONISH)
def test_malformed_policy_yaml_is_bounded_and_scrubbed(
    tmp_path: Path, raw: Any
) -> None:
    """Malformed policy data must either load safely or fail generically."""
    path = tmp_path / "private" / "policy.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    try:
        rules = load_policy_rules(path)
    except ValueError as exc:
        msg = str(exc)
        assert "private" not in msg
        assert str(tmp_path) not in msg
    else:
        assert isinstance(rules, list)
