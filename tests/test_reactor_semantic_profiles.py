# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor semantic-profile registry tests

"""Public invariants for reactor ownership and semantic ingress profiles."""

from __future__ import annotations

from dataclasses import replace

import pytest

import scpn_phase_orchestrator.reactor_semantics as semantics
from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_REGISTRY,
    DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY,
    REACTOR_FAMILY_ASSIGNMENT_MAP_SHA256,
    REACTOR_SEMANTIC_PROFILE_REGISTRY_VERSION,
    ReactorSemanticProfile,
    ReactorSemanticProfileRegistry,
    SemanticIngressState,
)


def test_profiles_cover_all_34_configurations_and_exact_device_assignments() -> None:
    registry = DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY
    expected = {
        "beam_target": "SCPN-BEAM-TARGET-CORE",
        "colliding_beam": "SCPN-BEAM-TARGET-CORE",
        "conventional_tokamak": "SCPN-TOKAMAK-CORE",
        "cusp": "SCPN-MAGNETIC-CUSP-CORE",
        "dense_plasma_focus": "SCPN-DENSE-PLASMA-FOCUS-CORE",
        "field_reversed_configuration": "SCPN-FRC-CORE",
        "frc_compression_mif": "SCPN-MIF-CORE",
        "fusion_fission_hybrid": "SCPN-FUSION-FISSION-HYBRID-CORE",
        "gas_dynamic_mirror": "SCPN-MIRROR-CORE",
        "gridded_iec": "SCPN-IEC-CORE",
        "heliotron": "SCPN-STELLARATOR-CORE",
        "ion_beam_icf": "SCPN-ICF-BEAM-CORE",
        "laser_icf_direct_drive": "SCPN-ICF-LASER-CORE",
        "laser_icf_fast_or_shock_ignition": "SCPN-ICF-LASER-CORE",
        "laser_icf_indirect_drive": "SCPN-ICF-LASER-CORE",
        "levitated_dipole": "SCPN-LEVITATED-DIPOLE-CORE",
        "maglif": "SCPN-MIF-MAGLIF-CORE",
        "mechanical_or_liquid_liner_mif": "SCPN-MIF-LINER-CORE",
        "plasma_jet_mif": "SCPN-MIF-PLASMA-JET-CORE",
        "polywell": "SCPN-IEC-CORE",
        "projectile_or_impact_icf": "SCPN-ICF-IMPACT-CORE",
        "pulsed_electron_beam_icf": "SCPN-ICF-BEAM-CORE",
        "reversed_field_pinch": "SCPN-RFP-CORE",
        "scpn.reactor_systems:lattice_confinement_fusion": ("SCPN-LATTICE-FUSION-CORE"),
        "scpn.reactor_systems:muon_catalysed_fusion": "SCPN-MUON-FUSION-CORE",
        "sheared_flow_z_pinch": "SCPN-Z-PINCH-CORE",
        "simple_magnetic_mirror": "SCPN-MIRROR-CORE",
        "spheromak": "SCPN-SPHEROMAK-CORE",
        "spherical_tokamak": "SCPN-TOKAMAK-CORE",
        "stellarator": "SCPN-STELLARATOR-CORE",
        "tandem_mirror": "SCPN-MIRROR-CORE",
        "theta_pinch": "SCPN-THETA-PINCH-CORE",
        "torsatron": "SCPN-STELLARATOR-CORE",
        "z_pinch": "SCPN-Z-PINCH-CORE",
    }

    assert registry.version == REACTOR_SEMANTIC_PROFILE_REGISTRY_VERSION
    assert registry.assignment_map_sha256 == REACTOR_FAMILY_ASSIGNMENT_MAP_SHA256
    assert set(registry.profiles) == set(DEFAULT_REACTOR_REGISTRY.configurations)
    assert {
        key: profile.device_project for key, profile in registry.profiles.items()
    } == expected
    assert len(registry.digest) == 64


def test_only_exercised_adapters_are_advertised() -> None:
    registry = DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY
    verified = {
        key: profile
        for key, profile in registry.profiles.items()
        if profile.ingress_state is SemanticIngressState.VERIFIED_REVIEW_ADAPTER
    }

    assert set(verified) == {"conventional_tokamak", "frc_compression_mif"}
    assert verified["conventional_tokamak"].producer_project == "SCPN-FUSION-CORE"
    assert verified["frc_compression_mif"].producer_project == "SCPN-MIF-CORE"
    for profile in verified.values():
        module_name, _, name = profile.adapter_api.rpartition(".")  # type: ignore[union-attr]
        assert getattr(__import__(module_name, fromlist=[name]), name) is getattr(
            semantics, name
        )
        assert profile.authority == "review_only"
        assert profile.actionable is False
        assert profile.control_adapter_contract is None
        assert profile.control_intent_profile is None
        assert profile.machine_protection_final_veto is True

    unavailable = set(registry.profiles) - set(verified)
    assert len(unavailable) == 32
    for configuration in unavailable:
        profile = registry.profiles[configuration]
        assert profile.ingress_state is SemanticIngressState.NOT_DECLARED
        assert profile.producer_project is None
        assert profile.adapter_api is None
        assert profile.semantic_profile is None


def test_profile_resolution_honours_configuration_aliases_without_remapping_owner() -> (
    None
):
    registry = DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY

    assert registry.resolve("frc").configuration == "field_reversed_configuration"
    assert registry.resolve("frc").device_project == "SCPN-FRC-CORE"
    assert registry.resolve("frc_compression_mif").device_project == "SCPN-MIF-CORE"
    with pytest.raises(ValueError, match="unregistered"):
        registry.resolve("unknown_reactor")


def test_registry_refuses_partial_or_actionable_bindings_and_registry_drift() -> None:
    unavailable = DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.resolve("stellarator")
    with pytest.raises(ValueError, match="cannot advertise"):
        replace(unavailable, producer_project="SCPN-FUSION-CORE")
    with pytest.raises(ValueError, match="complete adapter binding"):
        replace(
            unavailable,
            ingress_state=SemanticIngressState.VERIFIED_REVIEW_ADAPTER,
        )
    with pytest.raises(ValueError, match="review-only"):
        replace(unavailable, actionable=True)
    with pytest.raises(ValueError, match="final veto"):
        replace(unavailable, machine_protection_final_veto=False)
    with pytest.raises(ValueError, match="not built into SPO"):
        ReactorSemanticProfile(
            configuration="laboratory.example:novel_reactor",
            device_project="SCPN-NOVEL-CORE",
            ingress_state=SemanticIngressState.NOT_DECLARED,
        )
    with pytest.raises(ValueError, match="SemanticIngressState"):
        replace(unavailable, ingress_state="not_declared")  # type: ignore[arg-type]
    future_contracts = replace(
        unavailable,
        control_adapter_contract="scpn-control.device-adapter.v1",
        control_intent_profile="spo.control-intent.research.v1",
    )
    assert future_contracts.control_adapter_contract is not None
    assert future_contracts.control_intent_profile is not None
    with pytest.raises(ValueError, match="exact SPO reactor registry"):
        ReactorSemanticProfileRegistry(
            version="1.0.0",
            reactor_registry_version=DEFAULT_REACTOR_REGISTRY.version,
            reactor_registry_digest="0" * 64,
            assignment_map_sha256=REACTOR_FAMILY_ASSIGNMENT_MAP_SHA256,
            profiles=DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.profiles,
        )


def test_registry_refuses_incomplete_configuration_coverage() -> None:
    profiles = dict(DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.profiles)
    profiles.pop("z_pinch")

    with pytest.raises(ValueError, match="coverage mismatch"):
        ReactorSemanticProfileRegistry(
            version="1.0.0",
            reactor_registry_version=DEFAULT_REACTOR_REGISTRY.version,
            reactor_registry_digest=DEFAULT_REACTOR_REGISTRY.digest,
            assignment_map_sha256=REACTOR_FAMILY_ASSIGNMENT_MAP_SHA256,
            profiles=profiles,
        )

    mismatched = dict(DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.profiles)
    mismatched["z_pinch"] = mismatched["theta_pinch"]
    with pytest.raises(ValueError, match="profile key"):
        ReactorSemanticProfileRegistry(
            version="1.0.0",
            reactor_registry_version=DEFAULT_REACTOR_REGISTRY.version,
            reactor_registry_digest=DEFAULT_REACTOR_REGISTRY.digest,
            assignment_map_sha256=REACTOR_FAMILY_ASSIGNMENT_MAP_SHA256,
            profiles=mismatched,
        )


def test_profile_record_is_complete_deterministic_and_non_actuating() -> None:
    profile = ReactorSemanticProfile(
        configuration="z_pinch",
        device_project="SCPN-Z-PINCH-CORE",
        ingress_state=SemanticIngressState.NOT_DECLARED,
    )

    assert profile.to_record() == {
        "actionable": False,
        "adapter_api": None,
        "authority": "review_only",
        "configuration": "z_pinch",
        "control_adapter_contract": None,
        "control_intent_profile": None,
        "device_project": "SCPN-Z-PINCH-CORE",
        "handoff_schema": None,
        "ingress_state": "not_declared",
        "machine_protection_final_veto": True,
        "producer_project": None,
        "semantic_profile": None,
        "semantic_profile_version": None,
        "source_schema": None,
    }
