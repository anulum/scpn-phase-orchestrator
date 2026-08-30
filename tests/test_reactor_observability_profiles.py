# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-reactor observability profile tests

"""Fail-closed invariants for candidate signal meanings across reactors."""

from __future__ import annotations

from dataclasses import replace

import pytest

from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY,
    DEFAULT_REACTOR_REGISTRY,
    REACTOR_OBSERVABILITY_PROFILE_REGISTRY_VERSION,
    ObservabilityClass,
    ReactorObservabilityProfileRegistry,
    SemanticCarrier,
    UnmetEvidenceDisposition,
)


def test_catalogue_covers_every_configuration_without_claiming_evidence() -> None:
    registry = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY

    assert registry.version == REACTOR_OBSERVABILITY_PROFILE_REGISTRY_VERSION
    assert len(registry.candidates) == 21
    assert len(registry.digest) == 64
    for configuration in DEFAULT_REACTOR_REGISTRY.configurations:
        candidates = registry.for_configuration(configuration)
        assert candidates
        assert all(configuration in item.configurations for item in candidates)
        assert all(item.authority == "review_only" for item in candidates)
        assert all(item.actionable is False for item in candidates)
        assert all(item.evidence_claimed is False for item in candidates)


def test_catalogue_preserves_distinct_reactor_meanings() -> None:
    registry = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
    tokamak = {
        item.candidate_id for item in registry.for_configuration("conventional_tokamak")
    }

    assert tokamak == {
        "closed.equilibrium_profiles",
        "closed.recurrent_transient",
        "closed.resolved_mhd_mode",
        "model.synthetic_oscillator_coordinate",
    }
    assert registry.resolve("closed.equilibrium_profiles").admissible_carriers == (
        SemanticCarrier.BOUNDED_FEATURE,
        SemanticCarrier.CATEGORICAL_STATE,
    )
    assert registry.resolve("closed.resolved_mhd_mode").admissible_carriers == (
        SemanticCarrier.COMPLEX_MODE,
        SemanticCarrier.CYCLIC_PHASE,
        SemanticCarrier.FIELD_PHASE,
    )
    assert registry.resolve("inertial.driver_timing").unmet_evidence is (
        UnmetEvidenceDisposition.EVENT_TIMESTAMPS_OR_PROTOCOL
    )
    assert registry.resolve(
        "model.synthetic_oscillator_coordinate"
    ).admissible_carriers == (SemanticCarrier.NUMERICAL_PHASE,)
    with pytest.raises(ValueError, match="unknown signal candidate"):
        registry.resolve("reactor.generic_phase")


def test_alias_resolution_does_not_collapse_frc_and_mif() -> None:
    registry = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
    frc = {item.candidate_id for item in registry.for_configuration("frc")}
    compressed = {
        item.candidate_id for item in registry.for_configuration("frc_compression_mif")
    }

    assert "closed.resolved_mhd_mode" in frc
    assert "magneto_inertial.driver_arrival" not in frc
    assert "magneto_inertial.driver_arrival" in compressed
    assert "closed.resolved_mhd_mode" not in compressed
    with pytest.raises(ValueError, match="unregistered"):
        registry.for_configuration("unknown_reactor")


def test_candidate_records_are_deterministic_and_non_operational() -> None:
    candidate = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.resolve(
        "beam.rf_bunch_phase"
    )
    record = candidate.to_record()

    assert record["observability_class"] == "direct_cyclic"
    assert record["admissible_carriers"] == ["cyclic_phase", "field_phase"]
    assert record["reference_required"] is True
    assert record["observation_operator_required"] is False
    assert record["repeated_cycle_required"] is False
    assert record["evidence_claimed"] is False
    assert (
        DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.to_record()["evidence_claimed"]
        is False
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"configurations": ()}, "requires reactor configurations"),
        ({"configurations": ("stellarator", "stellarator")}, "unique and sorted"),
        ({"configurations": ("laboratory.example:unknown",)}, "unknown reactor"),
        ({"observability_class": "derived_cyclic"}, "ObservabilityClass"),
        ({"unmet_evidence": "unobservable_phase"}, "UnmetEvidenceDisposition"),
        (
            {
                "admissible_carriers": (
                    SemanticCarrier.COMPLEX_MODE,
                    SemanticCarrier.COMPLEX_MODE,
                )
            },
            "unique",
        ),
        ({"admissible_carriers": ()}, "do not match"),
        ({"required_evidence": ()}, "requires evidence requirements"),
        ({"required_evidence": ("clock_epoch", "clock_epoch")}, "must be unique"),
        (
            {"unmet_evidence": UnmetEvidenceDisposition.PRESERVE_NONCYCLIC},
            "disposition",
        ),
        ({"observation_operator_required": False}, "observation operator"),
        ({"reference_required": False}, "reference requirement"),
        ({"repeated_cycle_required": True}, "repetition gating"),
        ({"authority": "action"}, "non-actuating"),
        ({"actionable": True}, "non-actuating"),
        ({"evidence_claimed": True}, "non-actuating"),
    ],
)
def test_candidate_refuses_invalid_structure_and_authority(
    values: dict[str, object], message: str
) -> None:
    candidate = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.resolve(
        "closed.resolved_mhd_mode"
    )

    with pytest.raises(ValueError, match=message):
        replace(candidate, **values)  # type: ignore[arg-type]


def test_unobservable_class_has_no_carrier_and_no_usable_phase() -> None:
    candidate = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.resolve(
        "closed.resolved_mhd_mode"
    )
    unobservable = replace(
        candidate,
        observability_class=ObservabilityClass.UNOBSERVABLE,
        admissible_carriers=(),
        required_evidence=("observability_failure_reason",),
        unmet_evidence=UnmetEvidenceDisposition.NO_USABLE_PHASE,
        reference_required=False,
        observation_operator_required=False,
    )

    assert unobservable.admissible_carriers == ()
    assert unobservable.unmet_evidence is UnmetEvidenceDisposition.NO_USABLE_PHASE


def test_registry_refuses_empty_bad_keys_incomplete_coverage_and_drift() -> None:
    registry = DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
    kwargs = {
        "version": registry.version,
        "reactor_registry_version": registry.reactor_registry_version,
        "reactor_registry_digest": registry.reactor_registry_digest,
    }
    with pytest.raises(ValueError, match="requires candidates"):
        ReactorObservabilityProfileRegistry(candidates={}, **kwargs)

    wrong_key = dict(registry.candidates)
    candidate = wrong_key.pop("closed.equilibrium_profiles")
    wrong_key["closed.wrong_key"] = candidate
    with pytest.raises(ValueError, match="candidate key"):
        ReactorObservabilityProfileRegistry(candidates=wrong_key, **kwargs)

    incomplete = {
        key: item
        for key, item in registry.candidates.items()
        if key != "hybrid.source_blanket_response"
        and key != "model.synthetic_oscillator_coordinate"
    }
    with pytest.raises(ValueError, match="coverage mismatch"):
        ReactorObservabilityProfileRegistry(candidates=incomplete, **kwargs)

    with pytest.raises(ValueError, match="exact reactor registry"):
        ReactorObservabilityProfileRegistry(
            candidates=registry.candidates,
            version=registry.version,
            reactor_registry_version=registry.reactor_registry_version,
            reactor_registry_digest="0" * 64,
        )
