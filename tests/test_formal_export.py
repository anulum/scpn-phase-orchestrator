# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Formal exporter tests

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from scpn_phase_orchestrator.cli import main
from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyRule,
    export_petri_net_prism,
    export_policy_rules_prism,
)
from scpn_phase_orchestrator.supervisor.formal_export import PrismExport
from scpn_phase_orchestrator.supervisor.petri_net import (
    Arc,
    Guard,
    Marking,
    PetriNet,
    Place,
    Transition,
)


def _net() -> PetriNet:
    return PetriNet(
        [
            Place("warmup"),
            Place("nominal"),
            Place("cool-down"),
            Place("done"),
        ],
        [
            Transition(
                name="start",
                inputs=[Arc("warmup")],
                outputs=[Arc("nominal")],
                guard=Guard("stability.proxy", ">", 0.6),
            ),
            Transition(
                name="wind down",
                inputs=[Arc("nominal", weight=2)],
                outputs=[Arc("cool-down")],
                guard=Guard("R_bad.0", "<=", 0.3),
            ),
            Transition(
                name="finish",
                inputs=[Arc("cool-down")],
                outputs=[Arc("done")],
            ),
        ],
    )


def test_petri_net_prism_export_serialises_guards_and_arcs() -> None:
    export = export_petri_net_prism(
        _net(),
        Marking(tokens={"warmup": 1, "nominal": 2}),
        module_name="supervisor net",
    )

    assert isinstance(export, PrismExport)
    assert export.metric_names == {
        "R_bad.0": "R_bad_0",
        "stability.proxy": "stability_proxy",
    }
    assert "mdp\n" in export.model
    assert "module supervisor_net" in export.model
    assert "const double R_bad_0;" in export.model
    assert "const double stability_proxy;" in export.model
    assert "warmup : [0..2] init 1;" in export.model
    assert "nominal : [0..2] init 2;" in export.model
    assert "[start] stability_proxy > 0.59999999999999998 & warmup >= 1" in (
        export.model
    )
    assert "[wind_down] R_bad_0 <= 0.29999999999999999 & nominal >= 2" in (export.model)
    assert "(nominal'=nominal-2)" in export.model
    assert "(cool_down'=cool_down+1)" in export.model
    assert 'label "active_done" = done > 0;' in export.model


def test_petri_net_prism_export_is_deterministic() -> None:
    first = export_petri_net_prism(_net(), Marking(tokens={"warmup": 1})).model
    second = export_petri_net_prism(_net(), Marking(tokens={"warmup": 1})).model

    assert first == second


def test_petri_net_prism_export_rejects_bad_token_bound() -> None:
    with pytest.raises(PolicyError, match="max_tokens"):
        export_petri_net_prism(_net(), Marking(tokens={"warmup": 1}), max_tokens=0)


def test_petri_net_prism_export_rejects_initial_tokens_above_bound() -> None:
    with pytest.raises(PolicyError, match="exceeds max_tokens"):
        export_petri_net_prism(_net(), Marking(tokens={"warmup": 3}), max_tokens=2)


def _rules() -> list[PolicyRule]:
    return [
        PolicyRule(
            name="boost K",
            regimes=["DEGRADED", "CRITICAL"],
            condition=PolicyCondition(
                metric="R_good",
                layer=0,
                op="<",
                threshold=0.6,
            ),
            actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=5.0)],
            max_fires=2,
        ),
        PolicyRule(
            name="damp_bad",
            regimes=["CRITICAL"],
            condition=CompoundCondition(
                conditions=[
                    PolicyCondition(
                        metric="R_bad",
                        layer=0,
                        op=">",
                        threshold=0.4,
                    ),
                    PolicyCondition(
                        metric="stability_proxy",
                        layer=None,
                        op="<=",
                        threshold=0.5,
                    ),
                ],
                logic="AND",
            ),
            actions=[
                PolicyAction(knob="alpha", scope="layer_0", value=-0.05, ttl_s=3.0)
            ],
        ),
    ]


def test_policy_rules_prism_export_serialises_rules_and_actions() -> None:
    export = export_policy_rules_prism(_rules(), module_name="policy model")

    assert export.rule_names == {"boost K": "boost_K", "damp_bad": "damp_bad"}
    assert export.metric_names == {
        "R_bad.0": "R_bad_0",
        "R_good.0": "R_good_0",
        "stability_proxy": "stability_proxy",
    }
    assert "module policy_model" in export.model
    assert "//   CRITICAL -> 0" in export.model
    assert "//   DEGRADED -> 1" in export.model
    assert "boost_K_fires : [0..2] init 0;" in export.model
    assert "damp_bad_fires : [0..1] init 0;" in export.model
    assert "[boost_K] (regime = 1 | regime = 0) & R_good_0 < 0.59999999999999998" in (
        export.model
    )
    assert (
        "[damp_bad] (regime = 0) & "
        "(R_bad_0 > 0.40000000000000002 & stability_proxy <= 0.5)"
    ) in export.model
    assert 'label "fires_boost_K" = boost_K_fires > 0;' in export.model
    assert 'label "emits_boost_K_K_global_0" = boost_K_fires > 0;' in export.model


def test_policy_rules_prism_export_rejects_bad_rules() -> None:
    with pytest.raises(PolicyError, match="without rules"):
        export_policy_rules_prism([])

    bad = PolicyRule(
        name="bad",
        regimes=["DEGRADED"],
        condition=PolicyCondition(metric="R", layer=0, op="!=", threshold=0.1),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
    )
    with pytest.raises(PolicyError, match="unsupported operator"):
        export_policy_rules_prism([bad])


def test_formal_export_cli_writes_prism_model(tmp_path: Path) -> None:
    spec = {
        "name": "formal-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
        ],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
        "protocol_net": {
            "places": ["warmup", "nominal"],
            "initial": {"warmup": 1},
            "place_regime": {"warmup": "NOMINAL", "nominal": "NOMINAL"},
            "transitions": [
                {
                    "name": "start",
                    "inputs": [{"place": "warmup"}],
                    "outputs": [{"place": "nominal"}],
                    "guard": "stability_proxy > 0.0",
                },
            ],
        },
    }
    spec_path = tmp_path / "binding_spec.yaml"
    out_path = tmp_path / "protocol.prism"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--output",
            str(out_path),
            "--module-name",
            "formal_test",
        ],
    )

    assert result.exit_code == 0
    assert "PRISM model written:" in result.output
    model = out_path.read_text(encoding="utf-8")
    assert "module formal_test" in model
    assert "[start] stability_proxy > 0 & warmup >= 1" in model


def test_formal_export_cli_writes_policy_prism_model(tmp_path: Path) -> None:
    spec = {
        "name": "formal-policy-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
        ],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    policy = {
        "rules": [
            {
                "name": "boost",
                "regime": ["DEGRADED"],
                "condition": {
                    "metric": "R_good",
                    "layer": 0,
                    "op": "<",
                    "threshold": 0.7,
                },
                "action": {"knob": "K", "scope": "global", "value": 0.1, "ttl_s": 5.0},
            }
        ]
    }
    spec_path = tmp_path / "binding_spec.yaml"
    policy_path = tmp_path / "policy.yaml"
    out_path = tmp_path / "policy.prism"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "policy",
            "--output",
            str(out_path),
            "--module-name",
            "policy_test",
        ],
    )

    assert result.exit_code == 0
    assert "PRISM model written:" in result.output
    model = out_path.read_text(encoding="utf-8")
    assert "module policy_test" in model
    assert "[boost] (regime = 0) & R_good_0 < 0.69999999999999996" in model


def test_formal_export_cli_requires_protocol_net(tmp_path: Path) -> None:
    spec = {
        "name": "formal-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
        ],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    spec_path = tmp_path / "binding_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    result = CliRunner().invoke(main, ["formal-export", str(spec_path)])

    assert result.exit_code == 1
    assert "ERROR: binding spec has no protocol_net" in result.output
