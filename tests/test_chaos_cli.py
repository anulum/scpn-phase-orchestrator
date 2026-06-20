# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — chaos CLI command tests

"""Tests for the ``spo chaos`` resilience-injection command."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main

SPEC = "domainpacks/minimal_domain/binding_spec.yaml"


def test_chaos_human_output() -> None:
    result = CliRunner().invoke(
        main,
        ["chaos", SPEC, "--fault", "coupling_drop:40:30:0.85", "--steps", "160"],
    )
    assert result.exit_code == 0, result.output
    assert "recovered:" in result.output
    assert "max coherence drop:" in result.output
    assert "nominal/perturbed R:" in result.output


def test_chaos_json_output() -> None:
    result = CliRunner().invoke(
        main,
        [
            "chaos",
            SPEC,
            "--fault",
            "frequency_drift:30:10:0.05",
            "--steps",
            "200",
            "--json-out",
        ],
    )
    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    assert record["non_actuating"] is True
    assert record["steps"] == 200
    assert "metrics" in record


def test_chaos_fail_unrecovered_exits_nonzero() -> None:
    result = CliRunner().invoke(
        main,
        [
            "chaos",
            SPEC,
            "--fault",
            "coupling_drop:40:40:0.95",
            "--steps",
            "120",
            "--recovery-tolerance",
            "0.001",
            "--fail-unrecovered",
        ],
    )
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "fault",
    ["coupling_drop:40:30", "coupling_drop:40:30:0.5:extra"],
)
def test_chaos_rejects_malformed_fault(fault: str) -> None:
    result = CliRunner().invoke(main, ["chaos", SPEC, "--fault", fault])
    assert result.exit_code != 0
    assert "kind:start:duration:magnitude" in result.output


def test_chaos_rejects_invalid_fault_params() -> None:
    result = CliRunner().invoke(main, ["chaos", SPEC, "--fault", "meltdown:40:30:0.5"])
    assert result.exit_code != 0
    assert "invalid fault" in result.output


def test_chaos_rejects_too_few_steps() -> None:
    result = CliRunner().invoke(
        main,
        ["chaos", SPEC, "--fault", "coupling_drop:40:30:0.5", "--steps", "50"],
    )
    assert result.exit_code != 0
    assert "steps must exceed" in result.output
