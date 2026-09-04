# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor-semantic registry tests

"""Public-surface tests for the family-neutral reactor registry."""

from __future__ import annotations

from dataclasses import replace

import pytest

from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_REGISTRY,
    REACTOR_REGISTRY_V1_0_0,
    ConfinementFamily,
    ReactorConfiguration,
    ReactorConfigurationRegistry,
    build_reactor_reference_portfolio,
    canonical_json,
    resolve_reactor_registry_release,
)


def test_current_registry_spans_reactor_families_without_tokamak_default() -> None:
    registry = DEFAULT_REACTOR_REGISTRY

    assert len(registry.configurations) == 34
    assert registry.resolve("conventional_tokamak").confinement_family is (
        ConfinementFamily.MAGNETIC_CLOSED
    )
    assert registry.resolve("stellarator").identifier == "stellarator"
    assert registry.resolve("frc").identifier == "field_reversed_configuration"
    assert registry.resolve("tandem_mirror").confinement_family is (
        ConfinementFamily.MAGNETIC_OPEN
    )
    assert registry.resolve("z_pinch").confinement_family is (
        ConfinementFamily.SELF_MAGNETIC
    )
    assert registry.resolve("laser_icf_indirect_drive").confinement_family is (
        ConfinementFamily.INERTIAL
    )
    assert registry.resolve("maglif").confinement_family is (
        ConfinementFamily.MAGNETO_INERTIAL
    )
    assert registry.resolve("gridded_iec").confinement_family is (
        ConfinementFamily.ELECTROSTATIC
    )
    assert registry.resolve("fusion_fission_hybrid").confinement_family is (
        ConfinementFamily.HYBRID
    )
    assert (
        registry.resolve(
            "scpn.reactor_systems:lattice_confinement_fusion"
        ).confinement_family
        is ConfinementFamily.EXTENSION
    )
    assert (
        registry.resolve(
            "scpn.reactor_systems:muon_catalysed_fusion"
        ).confinement_family
        is ConfinementFamily.EXTENSION
    )
    assert len(registry.digest) == 64
    assert registry.to_record()["configurations"][0]["identifier"] == "beam_target"


def test_registry_1_0_remains_exactly_resolvable_for_producer_custody() -> None:
    assert REACTOR_REGISTRY_V1_0_0.version == "1.0.0"
    assert len(REACTOR_REGISTRY_V1_0_0.configurations) == 32
    assert REACTOR_REGISTRY_V1_0_0.digest == (
        "786d9542ce76c56dd7748fa948b17efed6c073525e527ce90e6d5e29a2d00090"
    )
    assert (
        resolve_reactor_registry_release(
            REACTOR_REGISTRY_V1_0_0.version,
            REACTOR_REGISTRY_V1_0_0.digest,
        )
        is REACTOR_REGISTRY_V1_0_0
    )


def test_registry_extensions_are_namespaced_immutable_and_deterministic() -> None:
    extension = ReactorConfiguration(
        "laboratory.example:novel_mirror",
        ConfinementFamily.EXTENSION,
        "declared experimental topology",
    )
    extended = DEFAULT_REACTOR_REGISTRY.register(
        extension,
        aliases=("laboratory.example:mirror_v1",),
    )

    assert extension.to_record()["identifier"] == extension.identifier
    assert extended.resolve("laboratory.example:mirror_v1") is extension
    assert extension.identifier not in DEFAULT_REACTOR_REGISTRY.configurations
    assert extended.digest != DEFAULT_REACTOR_REGISTRY.digest

    base = build_reactor_reference_portfolio()[0].context
    context = replace(
        base,
        configuration=extension.identifier,
        confinement_family=extension.confinement_family,
        topology=extension.topology,
        registry_digest=extended.digest,
    )
    assert context.validate_registry(extended) is context
    assert extension.identifier in canonical_json(context, registry=extended)
    with pytest.raises(ValueError, match="registry_digest"):
        context.validate_registry(DEFAULT_REACTOR_REGISTRY)


@pytest.mark.parametrize(
    "configuration, aliases, match",
    [
        (
            ReactorConfiguration(
                "unqualified_extension",
                ConfinementFamily.EXTENSION,
                "extension",
            ),
            (),
            "namespaced",
        ),
        (
            ReactorConfiguration(
                "laboratory.example:extension",
                ConfinementFamily.EXTENSION,
                "extension",
            ),
            ("frc",),
            "already registered",
        ),
        (
            ReactorConfiguration(
                "laboratory.example:extension",
                ConfinementFamily.EXTENSION,
                "extension",
            ),
            ("plain_alias",),
            "namespaced",
        ),
    ],
)
def test_registry_refuses_unsafe_extensions(
    configuration: ReactorConfiguration,
    aliases: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        DEFAULT_REACTOR_REGISTRY.register(configuration, aliases=aliases)


def test_registry_refuses_unknown_and_inconsistent_records() -> None:
    with pytest.raises(ValueError, match="unregistered"):
        DEFAULT_REACTOR_REGISTRY.resolve("unknown_configuration")

    configuration = ReactorConfiguration(
        "laboratory.example:item",
        ConfinementFamily.EXTENSION,
        "extension",
    )
    with pytest.raises(ValueError, match="registry key"):
        ReactorConfigurationRegistry(
            "1.0.0",
            {"laboratory.example:wrong": configuration},
            {},
        )
    with pytest.raises(ValueError, match="targets unknown"):
        ReactorConfigurationRegistry(
            "1.0.0",
            {configuration.identifier: configuration},
            {"laboratory.example:alias": "laboratory.example:missing"},
        )
