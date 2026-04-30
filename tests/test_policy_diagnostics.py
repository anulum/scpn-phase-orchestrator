# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy dry-run diagnostics tests

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from click.testing import CliRunner

from scpn_phase_orchestrator.cli import main
from scpn_phase_orchestrator.supervisor.policy_diagnostics import (
    dry_run_policy_rules,
)
from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules


def _entries() -> list[dict[str, Any]]:
    return [
        {
            "step": 0,
            "regime": "DEGRADED",
            "stability": 0.4,
            "layers": [{"R": 0.8, "psi": 0.0}, {"R": 0.2, "psi": 0.0}],
        },
        {
            "step": 1,
            "regime": "CRITICAL",
            "stability": 0.2,
            "layers": [{"R": 0.7, "psi": 0.0}, {"R": 0.1, "psi": 0.0}],
        },
    ]


def _policy_data() -> dict[str, Any]:
    return {
        "rules": [
            {
                "name": "boost_degraded",
                "regime": ["DEGRADED"],
                "condition": {
                    "metric": "stability_proxy",
                    "op": "<",
                    "threshold": 0.5,
                },
                "action": {
                    "knob": "K",
                    "scope": "global",
                    "value": 0.1,
                    "ttl_s": 5.0,
                },
            },
            {
                "name": "suppress_bad",
                "regime": ["CRITICAL"],
                "condition": {
                    "metric": "R_bad",
                    "layer": 0,
                    "op": "<",
                    "threshold": 0.2,
                },
                "action": {
                    "knob": "K",
                    "scope": "layer_1",
                    "value": -0.1,
                    "ttl_s": 5.0,
                },
            },
            {
                "name": "never_fires",
                "regime": ["NOMINAL"],
                "condition": {
                    "metric": "R",
                    "layer": 0,
                    "op": ">",
                    "threshold": 0.9,
                },
                "action": {
                    "knob": "zeta",
                    "scope": "global",
                    "value": 0.1,
                    "ttl_s": 5.0,
                },
            },
        ]
    }


def _binding_data() -> dict[str, Any]:
    return {
        "name": "policy-dry-run-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "good", "index": 0, "oscillator_ids": ["g0"]},
            {"name": "bad", "index": 1, "oscillator_ids": ["b0"]},
        ],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": [1]},
        "boundaries": [],
        "actuators": [],
    }


def test_policy_dry_run_reports_unreachable_rules(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(_policy_data()), encoding="utf-8")
    rules = load_policy_rules(policy_path)

    report = dry_run_policy_rules(
        rules,
        _entries(),
        good_layers=[0],
        bad_layers=[1],
    )

    assert report.steps == 2
    assert report.fire_counts["boost_degraded"] == 1
    assert report.fire_counts["suppress_bad"] == 1
    assert report.unreachable_rules == ("never_fires",)
    assert report.step_reports[0].fired_rules == ("boost_degraded",)


def test_policy_dry_run_cli_outputs_json(tmp_path: Path) -> None:
    binding_path = tmp_path / "binding_spec.yaml"
    policy_path = tmp_path / "policy.yaml"
    audit_path = tmp_path / "audit.jsonl"
    binding_path.write_text(yaml.safe_dump(_binding_data()), encoding="utf-8")
    policy_path.write_text(yaml.safe_dump(_policy_data()), encoding="utf-8")
    audit_path.write_text(
        "\n".join(json.dumps(entry) for entry in _entries()) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["policy-dry-run", str(binding_path), str(audit_path), "--json-out"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["steps"] == 2
    assert payload["fire_counts"]["boost_degraded"] == 1
    assert payload["unreachable_rules"] == ["never_fires"]


def test_policy_dry_run_cli_rejects_missing_policy(tmp_path: Path) -> None:
    binding_path = tmp_path / "binding_spec.yaml"
    audit_path = tmp_path / "audit.jsonl"
    binding_path.write_text(yaml.safe_dump(_binding_data()), encoding="utf-8")
    audit_path.write_text(
        "\n".join(json.dumps(entry) for entry in _entries()) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main, ["policy-dry-run", str(binding_path), str(audit_path)]
    )

    assert result.exit_code == 1
    assert "ERROR: policy file not found:" in result.output
