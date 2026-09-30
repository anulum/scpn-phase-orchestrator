# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding actuator admission contracts

"""Cross-actuator admission, CLI refusal and repaired command projection."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.actuation.mapper import ActuationMapper, ControlAction
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.types import ActuatorMapping, BindingSpec
from scpn_phase_orchestrator.binding.validator import validate_binding_spec
from scpn_phase_orchestrator.runtime.cli import main


def _binding(actuators: list[ActuatorMapping]) -> BindingSpec:
    """Load the real two-layer domain with the supplied actuator declarations."""
    root = Path(__file__).resolve().parents[1]
    return replace(
        load_binding_spec(root / "domainpacks/minimal_domain/binding_spec.yaml"),
        actuators=actuators,
    )


def _write_binding(spec: BindingSpec, path: Path) -> BindingSpec:
    """Round-trip actuator declarations through the production JSON loader."""
    payload = asdict(spec)
    payload["drivers"] = {
        "physical": spec.drivers.physical,
        "informational": spec.drivers.informational,
        "symbolic": spec.drivers.symbolic,
        **(spec.drivers.extra or {}),
    }
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return load_binding_spec(path)


def _project_commands(spec: BindingSpec, knob: str, expected: float) -> None:
    """Verify bounded commands reach the actual scoped actuator mapper."""
    action = ControlAction(
        knob=knob,
        scope="global",
        value=4.0,
        ttl_s=2.0,
        justification="binding admission recovery",
    )
    original = replace(action)
    projector = ActionProjector.from_actuator_mappings(iter(spec.actuators))
    projected = projector.project(action, previous_value=0.5)
    assert projected == replace(action, value=expected)
    assert action == original
    mapper = ActuationMapper(spec.actuators)
    assert mapper.validate_action(projected)
    assert mapper.map_actions([projected]) == [
        {
            "actuator": actuator.name,
            "knob": knob,
            "scope": "global",
            "value": expected,
            "ttl_s": 2.0,
        }
        for actuator in spec.actuators
        if actuator.knob == knob
    ]


@pytest.mark.parametrize("knob", ["K", "alpha", "zeta", "Psi"])
@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("conflict", ["bounds", "rate"])
def test_conflicting_actuators_refuse_before_runtime_and_recover(
    tmp_path: Path, knob: str, reverse_order: bool, conflict: str
) -> None:
    """Both APIs and spo validate reject ambiguous knobs in either record order."""
    first = ActuatorMapping(
        name="global_control",
        knob=knob,
        scope="global",
        limits=(0.0, 2.0),
        rate_limit_per_step=0.25,
    )
    second = replace(first, name="layer_control", scope="layer_0")
    if conflict == "bounds":
        second = replace(second, limits=(0.0, 1.0))
    else:
        second = replace(second, rate_limit_per_step=0.5)
    records = [first, second]
    if reverse_order:
        records.reverse()
    spec = _binding(records)
    original = asdict(spec)
    path = tmp_path / "binding.json"
    loaded = _write_binding(spec, path)
    source = path.read_bytes()
    expected = (
        f"actuator {records[1].name!r}: knob {knob!r} limits "
        f"{list(records[1].limits)} differ from actuator {records[0].name!r} "
        f"{list(records[0].limits)}; the runtime projector holds one bound per knob"
        if conflict == "bounds"
        else f"actuator {records[1].name!r}: knob {knob!r} rate_limit_per_step "
        f"{records[1].rate_limit_per_step!r} differs from actuator "
        f"{records[0].name!r} {records[0].rate_limit_per_step!r}; "
        "the runtime projector holds one rate limit per knob"
    )
    for candidate in (spec, loaded):
        assert validate_binding_spec(candidate) == [expected]
        assert validate_binding_spec(candidate) == [expected]
        with pytest.raises(ValueError, match=f"conflicting .* {knob!r}"):
            ActionProjector.from_actuator_mappings(iter(candidate.actuators))
    result = CliRunner().invoke(main, ["validate", str(path)])
    assert result.exit_code == 1
    assert result.output.strip() == f"ERROR: {expected}"
    assert path.read_bytes() == source
    assert asdict(spec) == original

    repaired = replace(
        spec,
        actuators=[
            replace(record, limits=(0.0, 2.0), rate_limit_per_step=0.25)
            for record in records
        ],
    )
    loaded_repair = _write_binding(repaired, path)
    assert validate_binding_spec(repaired) == []
    assert validate_binding_spec(loaded_repair) == []
    result = CliRunner().invoke(main, ["validate", str(path)])
    assert result.exit_code == 0
    assert result.output.startswith("Valid\n")
    _project_commands(loaded_repair, knob, expected=0.75)
    assert asdict(spec) == original


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    ("rates", "expected"),
    [
        pytest.param((None, None, None), 2.0, id="no-rate-declaration"),
        pytest.param((None, 0.25, None), 0.75, id="one-explicit-rate"),
        pytest.param((0.25, None, 0.25), 0.75, id="equal-explicit-rates"),
        pytest.param((None, 0.0, 0.0), 0.5, id="explicit-zero-is-a-rate"),
    ],
)
def test_optional_rate_declarations_admit_and_reach_scoped_commands(
    tmp_path: Path,
    reverse_order: bool,
    rates: tuple[float | None, float | None, float | None],
    expected: float,
) -> None:
    """Absent declarations neither conflict with nor override a declared rate."""
    actuators = [
        ActuatorMapping(
            name=name,
            knob="K",
            scope=scope,
            limits=(0.0, 2.0),
            rate_limit_per_step=rate,
        )
        for name, scope, rate in zip(
            ("global_control", "lower_control", "upper_control"),
            ("global", "layer_0", "layer_1"),
            rates,
            strict=True,
        )
    ]
    if reverse_order:
        actuators.reverse()
    spec = _binding(actuators)
    original = asdict(spec)
    path = tmp_path / "binding.json"
    loaded = _write_binding(spec, path)
    source = path.read_bytes()
    assert validate_binding_spec(spec) == []
    assert validate_binding_spec(loaded) == []
    result = CliRunner().invoke(main, ["validate", str(path)])
    assert result.exit_code == 0
    _project_commands(loaded, "K", expected)
    action = ControlAction("K", "layer_1", 4.0, 2.0, "scoped recovery")
    projected = ActionProjector.from_actuator_mappings(loaded.actuators).project(
        action, previous_value=0.5
    )
    assert ActuationMapper(loaded.actuators).map_actions([projected]) == [
        {
            "actuator": "upper_control",
            "knob": "K",
            "scope": "layer_1",
            "value": expected,
            "ttl_s": 2.0,
        }
    ]
    assert path.read_bytes() == source
    assert asdict(spec) == original


def test_independent_knobs_do_not_share_bounds_or_slew_limits(tmp_path: Path) -> None:
    """Per-knob declarations remain independent after JSON and CLI admission."""
    spec = _binding(
        [
            ActuatorMapping("coupling", "K", "global", (0.0, 2.0), 0.25),
            ActuatorMapping("decay", "alpha", "layer_0", (0.0, 1.0), 0.125),
            ActuatorMapping("drive", "zeta", "layer_1", (0.0, 3.0), None),
            ActuatorMapping("phase", "Psi", "global", (-2.0, 5.0), 0.0),
        ]
    )
    path = tmp_path / "binding.json"
    loaded = _write_binding(spec, path)
    assert validate_binding_spec(spec) == []
    assert validate_binding_spec(loaded) == []
    assert CliRunner().invoke(main, ["validate", str(path)]).exit_code == 0
    for knob, expected in (("K", 0.75), ("alpha", 0.625), ("zeta", 3.0), ("Psi", 0.5)):
        _project_commands(loaded, knob, expected)


def test_all_actuator_disagreements_are_reported_in_declaration_order(
    tmp_path: Path,
) -> None:
    """Report every bound and rate conflict against the first declaration."""
    spec = _binding(
        [
            ActuatorMapping("anchor", "K", "global", (0.0, 2.0), 0.25),
            ActuatorMapping("lower", "K", "layer_0", (0.0, 1.0), None),
            ActuatorMapping("upper", "K", "layer_1", (0.0, 3.0), 0.5),
            ActuatorMapping("duplicate", "K", "global", (0.0, 2.0), 0.0),
        ]
    )
    expected = [
        "actuator 'lower': knob 'K' limits [0.0, 1.0] differ from actuator "
        "'anchor' [0.0, 2.0]; the runtime projector holds one bound per knob",
        "actuator 'upper': knob 'K' limits [0.0, 3.0] differ from actuator "
        "'anchor' [0.0, 2.0]; the runtime projector holds one bound per knob",
        "actuator 'upper': knob 'K' rate_limit_per_step 0.5 differs from actuator "
        "'anchor' 0.25; the runtime projector holds one rate limit per knob",
        "actuator 'duplicate': knob 'K' rate_limit_per_step 0.0 differs from "
        "actuator 'anchor' 0.25; the runtime projector holds one rate limit per knob",
    ]
    original = asdict(spec)
    path = tmp_path / "binding.json"
    loaded = _write_binding(spec, path)
    source = path.read_bytes()
    assert validate_binding_spec(spec) == expected
    assert validate_binding_spec(loaded) == expected
    result = CliRunner().invoke(main, ["validate", str(path)])
    assert result.exit_code == 1
    assert result.output.splitlines() == [f"ERROR: {message}" for message in expected]
    assert path.read_bytes() == source
    assert asdict(spec) == original
    repaired = replace(
        spec,
        actuators=[
            replace(actuator, limits=(0.0, 2.0), rate_limit_per_step=0.25)
            for actuator in spec.actuators
        ],
    )
    loaded_repair = _write_binding(repaired, path)
    assert validate_binding_spec(loaded_repair) == []
    assert CliRunner().invoke(main, ["validate", str(path)]).exit_code == 0
    _project_commands(loaded_repair, "K", expected=0.75)
