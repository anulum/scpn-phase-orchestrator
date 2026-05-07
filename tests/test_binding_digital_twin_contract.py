# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin binding contract tests

from __future__ import annotations

from pathlib import Path
from typing import get_type_hints

import pytest

from scpn_phase_orchestrator.binding import (
    DigitalTwinBindingContract,
    build_digital_twin_binding_contract,
    load_binding_spec,
)
from scpn_phase_orchestrator.binding.types import BindingSpec


def test_digital_twin_contract_builder_is_typed() -> None:
    hints = get_type_hints(build_digital_twin_binding_contract)

    assert hints["spec"] is BindingSpec
    assert hints["return"] is DigitalTwinBindingContract


def test_digital_twin_contract_serialises_binding_and_channel_contract() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")

    contract = build_digital_twin_binding_contract(spec)
    record = contract.to_audit_record()

    assert record["contract_version"] == "spo-digital-twin-binding/v1"
    assert record["binding"] == {
        "name": "digital_twin_nchannel",
        "version": "0.1.0",
        "safety_tier": "research",
    }
    assert record["timing"] == {"sample_period_s": 0.01, "control_period_s": 0.1}
    assert [layer["name"] for layer in record["layers"]] == [
        "machine_cells",
        "process_line",
        "twin_supervisor",
    ]
    assert [actuator["name"] for actuator in record["actuators"]] == [
        "line_phase_lag",
        "twin_coupling",
    ]
    channel_algebra = record["channel_algebra"]
    assert channel_algebra["declared_channels"] == [
        "I",
        "P",
        "Quality",
        "S",
        "Thermal",
        "TwinResidual",
    ]
    assert channel_algebra["derived_channels"] == ["TwinResidual"]
    assert "TwinResidual" not in channel_algebra["coupling_participating_channels"]
    assert {capability["name"] for capability in record["sync_capabilities"]} == {
        "state_snapshot",
        "phase_observation",
        "control_action_proposal",
        "audit_replay",
    }
    assert len(record["contract_hash"]) == 64


def test_digital_twin_contract_hash_is_deterministic_and_sensitive() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")

    default_contract = build_digital_twin_binding_contract(spec)
    repeated_contract = build_digital_twin_binding_contract(spec)
    custom_contract = build_digital_twin_binding_contract(
        spec,
        sync_capabilities=("state_snapshot",),
    )

    assert default_contract.contract_hash == repeated_contract.contract_hash
    assert default_contract.to_json() == repeated_contract.to_json()
    assert default_contract.contract_hash != custom_contract.contract_hash
    assert custom_contract.sync_capabilities[0].direction == "twin_to_spo"
    assert custom_contract.sync_capabilities[0].payload == "state_vector"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"contract_version": ""}, "contract_version must be a non-empty string"),
        (
            {"sync_capabilities": ()},
            "sync_capabilities must contain at least one capability",
        ),
        (
            {"sync_capabilities": ("",)},
            "sync capability must be a non-empty string",
        ),
    ],
)
def test_digital_twin_contract_rejects_invalid_contract_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    spec = load_binding_spec(
        Path("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    )

    with pytest.raises(ValueError, match=message):
        build_digital_twin_binding_contract(spec, **kwargs)  # type: ignore[arg-type]
