# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI verification contract guards

"""Public CLI contract tests for formal export and supervisor verification."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner, Result

from scpn_phase_orchestrator.runtime.cli import main

Payload = dict[str, object]


def _write_json_payload(path: Path, payload: Mapping[str, object]) -> Path:
    """Write JSON syntax that remains valid YAML for CLI fixture files."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _policy_rule() -> Payload:
    """Return a minimal policy rule accepted by formal-export commands."""
    return {
        "name": "boost",
        "regime": ["DEGRADED"],
        "condition": {
            "metric": "R_good",
            "layer": 0,
            "op": "<",
            "threshold": 0.7,
        },
        "action": {
            "knob": "K",
            "scope": "global",
            "value": 0.1,
            "ttl_s": 5.0,
        },
    }


def _write_formal_export_spec(tmp_path: Path, *, protocol_net: bool = True) -> Path:
    """Write a binding spec fixture for public formal-export invocations."""
    spec: Payload = {
        "name": "formal-cli-verification-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [{"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]}],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    if protocol_net:
        spec["protocol_net"] = {
            "places": ["warmup", "nominal"],
            "initial": {"warmup": 1, "nominal": 0},
            "place_regime": {"warmup": "NOMINAL", "nominal": "NOMINAL"},
            "transitions": [
                {
                    "name": "start",
                    "inputs": [{"place": "warmup"}],
                    "outputs": [{"place": "nominal"}],
                    "guard": "stability_proxy > 0.0",
                }
            ],
        }
    return _write_json_payload(tmp_path / "binding_spec.yaml", spec)


def _write_policy_rules(tmp_path: Path) -> Path:
    """Write the policy file discovered by formal-export package commands."""
    return _write_json_payload(tmp_path / "policy.yaml", {"rules": [_policy_rule()]})


def _scenario_payload() -> Payload:
    """Return a valid supervisor scenario fixture."""
    return {
        "phases": [0.0, 0.2, 2.5, 3.0],
        "omegas": [0.05, 0.02, -0.02, -0.05],
        "base_coupling_off_diagonal": 0.02,
        "good_mask": [1.0, 1.0, 0.0, 0.0],
        "bad_mask": [0.0, 0.0, 1.0, 1.0],
        "dt": 0.04,
        "inner_steps": 3,
        "horizon": 5,
    }


def _baseline_args(
    tmp_path: Path,
    *,
    scenario_json: Path | None = None,
    manifest_json: Path | None = None,
    seed: str = "91",
    dependency_lock: str = "requirements-dev.txt:sha256:test",
    json_out: bool = False,
) -> list[str]:
    """Return a supervisor-baseline command line using temp artifacts."""
    args = [
        "supervisor-baseline-experiment",
        "--config-json",
        str(tmp_path / "supervisor_config.json"),
        "--metrics-jsonl",
        str(tmp_path / "supervisor_metrics.jsonl"),
        "--summary-json",
        str(tmp_path / "supervisor_summary.json"),
        "--git-sha",
        "abc1234",
        "--seed",
        seed,
        "--dependency-lock",
        dependency_lock,
    ]
    if scenario_json is not None:
        args.extend(["--scenario-json", str(scenario_json)])
    if manifest_json is not None:
        args.extend(["--manifest-json", str(manifest_json)])
    if json_out:
        args.append("--json-out")
    return args


def _invoke(args: Sequence[str]) -> Result:
    """Invoke the public SPO CLI with isolated Click state."""
    return CliRunner().invoke(main, list(args))


def _invoke_without_nn_stack(args: Sequence[str]) -> Result:
    """Invoke the CLI with the optional NN/JAX stack forced unavailable.

    Setting the ``jax`` entries in :data:`sys.modules` to ``None`` makes the
    command's in-body ``import jax`` raise :class:`ImportError`, reproducing an
    installation without the optional NN extra.
    """
    with patch.dict(sys.modules, {"jax": None, "jax.numpy": None}):
        return _invoke(args)


def _assert_cli_error(result: Result, expected: str) -> None:
    """Assert that a public CLI invocation failed with ``expected`` text."""
    assert result.exit_code == 1, result.output
    assert expected in result.output


def test_formal_export_package_rejects_specs_without_protocol_net(
    tmp_path: Path,
) -> None:
    """Package export requires a protocol net after policy rules are valid."""
    spec_path = _write_formal_export_spec(tmp_path, protocol_net=False)
    _write_policy_rules(tmp_path)

    result = _invoke(["formal-export", str(spec_path), "--export", "package"])

    _assert_cli_error(result, "ERROR: binding spec has no protocol_net")


def test_formal_export_checker_path_rejects_empty_executable(
    tmp_path: Path,
) -> None:
    """Checker readiness rejects empty executable names through the CLI."""
    spec_path = _write_formal_export_spec(tmp_path)
    _write_policy_rules(tmp_path)

    result = _invoke(
        [
            "formal-export",
            str(spec_path),
            "--export",
            "package",
            "--include-checker-readiness",
            "--checker-path",
            "=/opt/prism/bin/prism",
        ]
    )

    _assert_cli_error(result, "--checker-path executable must not be empty")


def test_supervisor_baseline_rejects_malformed_dependency_lock(
    tmp_path: Path,
) -> None:
    """Supervisor baseline provenance locks must include label and digest."""
    result = _invoke(_baseline_args(tmp_path, dependency_lock="requirements-dev.txt"))

    _assert_cli_error(
        result,
        "--dependency-lock values must use '<label>:<digest>' format",
    )


def test_supervisor_baseline_rejects_negative_seed(tmp_path: Path) -> None:
    """Supervisor baseline seeds are required to be non-negative."""
    result = _invoke(_baseline_args(tmp_path, seed="-1"))

    _assert_cli_error(result, "--seed values must be non-negative")


def test_supervisor_baseline_rejects_invalid_scenario_json(
    tmp_path: Path,
) -> None:
    """Supervisor baseline rejects malformed JSON and non-object payloads."""
    invalid_json = tmp_path / "invalid_scenario.json"
    invalid_json.write_text("{", encoding="utf-8")
    non_object_json = _write_json_payload(tmp_path / "list_scenario.json", {})
    non_object_json.write_text("[1, 2, 3]", encoding="utf-8")

    invalid_result = _invoke(_baseline_args(tmp_path, scenario_json=invalid_json))
    non_object_result = _invoke(_baseline_args(tmp_path, scenario_json=non_object_json))

    _assert_cli_error(invalid_result, "invalid scenario JSON:")
    _assert_cli_error(non_object_result, "scenario JSON must be an object")


def test_supervisor_baseline_rejects_invalid_scenario_fields(
    tmp_path: Path,
) -> None:
    """Supervisor baseline validates scenario lists and positive scalars."""
    empty_phases = _scenario_payload()
    empty_phases["phases"] = []
    non_numeric_omega = _scenario_payload()
    non_numeric_omega["omegas"] = [0.05, "bad", -0.02, -0.05]
    non_positive_dt = _scenario_payload()
    non_positive_dt["dt"] = 0.0
    non_positive_horizon = _scenario_payload()
    non_positive_horizon["horizon"] = 0

    cases = (
        (
            _write_json_payload(tmp_path / "empty_phases.json", empty_phases),
            "scenario phases must be a non-empty list",
        ),
        (
            _write_json_payload(tmp_path / "non_numeric_omega.json", non_numeric_omega),
            "scenario omegas[1] must be numeric",
        ),
        (
            _write_json_payload(tmp_path / "non_positive_dt.json", non_positive_dt),
            "scenario dt must be a positive number",
        ),
        (
            _write_json_payload(
                tmp_path / "non_positive_horizon.json",
                non_positive_horizon,
            ),
            "scenario horizon must be a positive integer",
        ),
    )

    for scenario_path, expected in cases:
        result = _invoke(_baseline_args(tmp_path, scenario_json=scenario_path))
        _assert_cli_error(result, expected)


def test_supervisor_baseline_validates_inputs_before_nn_stack_gate(
    tmp_path: Path,
) -> None:
    """Input-contract errors surface even when the NN/JAX stack is unavailable.

    Guards the ordering fix: input validation must run before the optional
    ``import jax`` gate so a malformed invocation reports its precise contract
    error rather than the "install the NN/JAX stack" cascade.
    """
    malformed_lock = _baseline_args(tmp_path, dependency_lock="requirements-dev.txt")
    negative_seed = _baseline_args(tmp_path, seed="-1")
    invalid_json = tmp_path / "invalid_scenario.json"
    invalid_json.write_text("{", encoding="utf-8")
    invalid_scenario = _baseline_args(tmp_path, scenario_json=invalid_json)

    _assert_cli_error(
        _invoke_without_nn_stack(malformed_lock),
        "--dependency-lock values must use '<label>:<digest>' format",
    )
    _assert_cli_error(
        _invoke_without_nn_stack(negative_seed),
        "--seed values must be non-negative",
    )
    _assert_cli_error(
        _invoke_without_nn_stack(invalid_scenario),
        "invalid scenario JSON:",
    )


def test_supervisor_baseline_requires_nn_stack_for_valid_inputs(
    tmp_path: Path,
) -> None:
    """Valid inputs with the NN/JAX stack absent report the optional-extra error."""
    result = _invoke_without_nn_stack(_baseline_args(tmp_path))

    _assert_cli_error(
        result,
        "supervisor baseline experiments require the optional NN/JAX stack",
    )


def test_supervisor_baseline_reports_text_outputs_and_manifest(
    tmp_path: Path,
) -> None:
    """Supervisor baseline prints artifact paths and writes manifest JSON."""
    manifest_json = tmp_path / "supervisor_manifest.json"

    result = _invoke(
        _baseline_args(
            tmp_path,
            manifest_json=manifest_json,
        )
    )

    assert result.exit_code == 0, result.output
    assert f"Wrote supervisor config: {tmp_path / 'supervisor_config.json'}" in (
        result.output
    )
    assert f"Wrote supervisor metrics: {tmp_path / 'supervisor_metrics.jsonl'}" in (
        result.output
    )
    assert f"Wrote supervisor summary: {tmp_path / 'supervisor_summary.json'}" in (
        result.output
    )
    assert f"Wrote supervisor manifest: {manifest_json}" in result.output
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["actuation_permitted"] is False
    assert manifest["dependency_lock"] == {"requirements-dev.txt": "sha256:test"}
