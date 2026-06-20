# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — binding/digital_twin package wiring tests

"""Wiring contract for the ``binding.digital_twin`` package.

Asserts that every responsibility submodule imports under its dotted path and
that the package ``__init__`` re-exports each public symbol as the *same object*
defined in its owning submodule, so the split preserves the original flat
``binding.digital_twin`` import surface with zero consumer churn.
"""

from __future__ import annotations

import importlib

import pytest

import scpn_phase_orchestrator.binding.digital_twin as digital_twin

PACKAGE = "scpn_phase_orchestrator.binding.digital_twin"

# Full dotted paths spelled out so the module-linkage guard
# (tools/check_test_module_linkage.py) finds each submodule's import path.
SUBMODULES = (
    "scpn_phase_orchestrator.binding.digital_twin._shared",
    "scpn_phase_orchestrator.binding.digital_twin.contract",
    "scpn_phase_orchestrator.binding.digital_twin.envelope",
    "scpn_phase_orchestrator.binding.digital_twin.evidence",
    "scpn_phase_orchestrator.binding.digital_twin.adapter_memory",
    "scpn_phase_orchestrator.binding.digital_twin.adapter_grpc",
    "scpn_phase_orchestrator.binding.digital_twin.adapter_kafka",
    "scpn_phase_orchestrator.binding.digital_twin.adapter_hardware",
    "scpn_phase_orchestrator.binding.digital_twin.adapter_rest",
)

# Owning submodule for each public symbol re-exported by the package __init__.
PUBLIC_SYMBOL_MODULE = {
    "DigitalTwinAdapterCompatibility": "contract",
    "DigitalTwinAdapterManifest": "contract",
    "DigitalTwinBindingContract": "contract",
    "DigitalTwinLayerContract": "contract",
    "DigitalTwinOperatorEvidence": "evidence",
    "DigitalTwinSyncCapability": "contract",
    "DigitalTwinSyncEnvelope": "envelope",
    "DigitalTwinSyncGrpcAdapter": "adapter_grpc",
    "DigitalTwinSyncGrpcResponse": "adapter_grpc",
    "DigitalTwinSyncHardwareAdapter": "adapter_hardware",
    "DigitalTwinSyncHardwareResponse": "adapter_hardware",
    "DigitalTwinSyncJsonlReport": "envelope",
    "DigitalTwinSyncKafkaAdapter": "adapter_kafka",
    "DigitalTwinSyncKafkaResponse": "adapter_kafka",
    "DigitalTwinSyncMemoryAdapter": "adapter_memory",
    "DigitalTwinSyncRestAdapter": "adapter_rest",
    "DigitalTwinSyncRestResponse": "adapter_rest",
    "DigitalTwinTransportValidation": "envelope",
    "build_digital_twin_adapter_manifest": "contract",
    "build_digital_twin_binding_contract": "contract",
    "build_digital_twin_operator_evidence": "evidence",
    "build_digital_twin_sync_envelope": "envelope",
    "read_digital_twin_sync_jsonl": "envelope",
    "validate_digital_twin_sync_envelope": "envelope",
    "write_digital_twin_sync_jsonl": "envelope",
}


@pytest.mark.parametrize("dotted_path", SUBMODULES)
def test_submodule_imports_under_dotted_path(dotted_path: str) -> None:
    assert dotted_path.startswith(f"{PACKAGE}.")
    module = importlib.import_module(dotted_path)
    assert module.__name__ == dotted_path
    assert (module.__doc__ or "").strip(), f"{dotted_path} has no module docstring"


def test_public_symbol_module_table_covers_dunder_all() -> None:
    assert set(PUBLIC_SYMBOL_MODULE) == set(digital_twin.__all__)
    assert len(digital_twin.__all__) == len(set(digital_twin.__all__))


@pytest.mark.parametrize("symbol,module_name", sorted(PUBLIC_SYMBOL_MODULE.items()))
def test_public_symbol_reexport_is_owning_module_object(
    symbol: str, module_name: str
) -> None:
    owner = importlib.import_module(f"{PACKAGE}.{module_name}")
    assert getattr(digital_twin, symbol) is getattr(owner, symbol)


def test_every_reexport_owner_is_a_real_submodule() -> None:
    owners = set(PUBLIC_SYMBOL_MODULE.values())
    submodule_names = {path.rsplit(".", 1)[1] for path in SUBMODULES}
    assert owners <= submodule_names
