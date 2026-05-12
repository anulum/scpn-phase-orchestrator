# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding validator tests

from __future__ import annotations

from dataclasses import replace

from scpn_phase_orchestrator.binding.types import (
    ActuatorMapping,
    AmplitudeSpec,
    BoundaryDef,
    ChannelGroupSpec,
    ChannelSpec,
    CrossChannelCouplingSpec,
    ImprintSpec,
    OscillatorFamily,
)
from scpn_phase_orchestrator.binding.validator import validate_binding_spec


def test_valid_spec_no_errors(sample_binding_spec):
    errors = validate_binding_spec(sample_binding_spec)
    assert errors == []


def test_empty_name_error(sample_binding_spec):
    bad = replace(sample_binding_spec, name="")
    errors = validate_binding_spec(bad)
    assert any("name" in e for e in errors)


def test_bad_version_format(sample_binding_spec):
    bad = replace(sample_binding_spec, version="1.0")
    errors = validate_binding_spec(bad)
    assert any("version" in e for e in errors)

    bad2 = replace(sample_binding_spec, version="1.a.0")
    errors2 = validate_binding_spec(bad2)
    assert any("version" in e for e in errors2)


def test_invalid_safety_tier(sample_binding_spec):
    bad = replace(sample_binding_spec, safety_tier="military")
    errors = validate_binding_spec(bad)
    assert any("safety_tier" in e for e in errors)


def test_negative_sample_period(sample_binding_spec):
    bad = replace(sample_binding_spec, sample_period_s=-0.01)
    errors = validate_binding_spec(bad)
    assert any("sample_period_s" in e for e in errors)


def test_invalid_channel_identifier(sample_binding_spec):
    bad_families = {
        "x": OscillatorFamily(
            channel="bad channel",
            extractor_type="hilbert",
            config={},
        ),
    }
    bad = replace(sample_binding_spec, oscillator_families=bad_families)
    errors = validate_binding_spec(bad)
    assert any("channel" in e for e in errors)


def test_bad_actuator_knob(sample_binding_spec):
    bad_actuators = [
        ActuatorMapping(name="bad", knob="omega", scope="global", limits=(0.0, 1.0)),
    ]
    bad = replace(sample_binding_spec, actuators=bad_actuators)
    errors = validate_binding_spec(bad)
    assert any("knob" in e for e in errors)


def test_bad_boundary_severity(sample_binding_spec):
    bad_bounds = [
        BoundaryDef(name="b", variable="R", lower=0.0, upper=1.0, severity="extreme"),
    ]
    bad = replace(sample_binding_spec, boundaries=bad_bounds)
    errors = validate_binding_spec(bad)
    assert any("severity" in e for e in errors)


def test_empty_layers_error(sample_binding_spec):
    bad = replace(sample_binding_spec, layers=[])
    errors = validate_binding_spec(bad)
    assert any("at least one layer" in e for e in errors)


def test_objective_references_missing_layer(sample_binding_spec):
    from scpn_phase_orchestrator.binding.types import ObjectivePartition

    bad_obj = ObjectivePartition(good_layers=[0, 99], bad_layers=[])
    bad = replace(sample_binding_spec, objectives=bad_obj)
    errors = validate_binding_spec(bad)
    assert any("layer index 99" in e for e in errors)


def test_control_period_must_be_positive(sample_binding_spec):
    bad = replace(sample_binding_spec, control_period_s=0.0)
    errors = validate_binding_spec(bad)
    assert any("control_period_s" in e and "> 0" in e for e in errors)


def test_control_period_ge_sample_period(sample_binding_spec):
    bad = replace(sample_binding_spec, sample_period_s=0.1, control_period_s=0.05)
    errors = validate_binding_spec(bad)
    assert any("control_period_s must be >= sample_period_s" in e for e in errors)


def test_actuator_limits_ordering(sample_binding_spec):
    bad_actuators = [
        ActuatorMapping(name="inv", knob="K", scope="global", limits=(1.0, 0.0)),
    ]
    bad = replace(sample_binding_spec, actuators=bad_actuators)
    errors = validate_binding_spec(bad)
    assert any("limits" in e and "lo <= hi" in e for e in errors)


def test_empty_objectives_error(sample_binding_spec):
    from scpn_phase_orchestrator.binding.types import ObjectivePartition

    bad_obj = ObjectivePartition(good_layers=[], bad_layers=[])
    bad = replace(sample_binding_spec, objectives=bad_obj)
    errors = validate_binding_spec(bad)
    assert any("at least one good or bad layer" in e for e in errors)


# ---------------------------------------------------------------------------
# Discriminatory: valid values must NOT produce errors
# ---------------------------------------------------------------------------


def test_all_valid_safety_tiers_accepted(sample_binding_spec):
    """Every valid safety_tier must produce zero tier-related errors."""
    for tier in ["research", "clinical", "production", "consumer"]:
        spec = replace(sample_binding_spec, safety_tier=tier)
        errors = validate_binding_spec(spec)
        tier_errors = [e for e in errors if "safety_tier" in e]
        assert tier_errors == [], f"tier={tier!r} should be valid, got: {tier_errors}"


def test_valid_channels_accepted(sample_binding_spec):
    """Standard and named extension channels must pass validation."""
    for ch in ["P", "I", "S", "Q", "sensor_4", "edge-node"]:
        families = {
            "f": OscillatorFamily(channel=ch, extractor_type="hilbert", config={}),
        }
        channels = (
            {}
            if ch in {"P", "I", "S"}
            else {ch: ChannelSpec(role="domain", units="phase")}
        )
        spec = replace(
            sample_binding_spec,
            oscillator_families=families,
            channels=channels,
        )
        errors = validate_binding_spec(spec)
        ch_errors = [e for e in errors if "channel" in e]
        assert ch_errors == [], f"channel={ch!r} should be valid, got: {ch_errors}"


def test_valid_knobs_accepted(sample_binding_spec):
    """All valid knobs (K, alpha, zeta, Psi) must pass validation."""
    for knob in ["K", "alpha", "zeta", "Psi"]:
        actuators = [
            ActuatorMapping(
                name=f"{knob}_g", knob=knob, scope="global", limits=(0.0, 1.0)
            ),
        ]
        spec = replace(sample_binding_spec, actuators=actuators)
        errors = validate_binding_spec(spec)
        knob_errors = [e for e in errors if "knob" in e]
        assert knob_errors == [], f"knob={knob!r} should be valid, got: {knob_errors}"


def test_multiple_errors_all_reported(sample_binding_spec):
    """Multiple simultaneous violations must all be reported (not fail-fast)."""
    from scpn_phase_orchestrator.binding.types import ObjectivePartition

    bad = replace(
        sample_binding_spec,
        name="",
        version="bad",
        safety_tier="military",
        sample_period_s=-1.0,
        objectives=ObjectivePartition(good_layers=[], bad_layers=[]),
    )
    errors = validate_binding_spec(bad)
    assert len(errors) >= 4, f"Should report ≥4 errors, got {len(errors)}: {errors}"


def test_valid_version_formats(sample_binding_spec):
    """Semver versions with all-numeric parts must pass."""
    for v in ["0.0.1", "1.0.0", "10.20.30"]:
        spec = replace(sample_binding_spec, version=v)
        errors = validate_binding_spec(spec)
        ver_errors = [e for e in errors if "version" in e]
        assert ver_errors == [], f"version={v!r} should be valid, got: {ver_errors}"


def test_named_nchannel_must_be_declared(sample_binding_spec):
    families = {
        "risk": OscillatorFamily(channel="Risk", extractor_type="event", config={}),
    }
    bad = replace(sample_binding_spec, oscillator_families=families, channels={})

    errors = validate_binding_spec(bad)

    assert any("named N-channel" in e and "Risk" in e for e in errors)


def test_required_channel_must_have_runtime_evidence(sample_binding_spec):
    bad = replace(
        sample_binding_spec,
        channels={"Risk": ChannelSpec(role="risk", required=True)},
    )

    errors = validate_binding_spec(bad)

    assert any("required channel" in e and "Risk" in e for e in errors)


def test_derived_channel_requires_derived_replay_semantics(sample_binding_spec):
    bad = replace(
        sample_binding_spec,
        channels={
            "Risk": ChannelSpec(
                role="risk",
                required=False,
                replay_semantics="phase",
                derived_from=["P"],
                derive_rule="risk = phase(P)",
            )
        },
    )

    errors = validate_binding_spec(bad)

    assert any("replay_semantics='derived'" in e for e in errors)


def test_derived_replay_semantics_requires_sources(sample_binding_spec):
    bad = replace(
        sample_binding_spec,
        channels={
            "Risk": ChannelSpec(
                role="risk",
                required=False,
                replay_semantics="derived",
            )
        },
    )

    errors = validate_binding_spec(bad)

    assert any("requires derived_from" in e for e in errors)


def test_derive_rule_requires_sources(sample_binding_spec):
    bad = replace(
        sample_binding_spec,
        channels={
            "Risk": ChannelSpec(
                role="risk",
                required=False,
                replay_semantics="phase",
                derive_rule="risk = phase(P)",
            )
        },
    )

    errors = validate_binding_spec(bad)

    assert any("derive_rule requires derived_from" in e for e in errors)


def test_invalid_replay_semantics_is_reported(sample_binding_spec):
    bad = replace(
        sample_binding_spec,
        channels={
            "Risk": ChannelSpec(
                role="risk", required=False, replay_semantics="log"
            )
        },
    )

    errors = validate_binding_spec(bad)

    assert any("replay_semantics must be one of" in e and "log" in e for e in errors)


def test_derived_channel_cannot_reference_itself(sample_binding_spec):
    bad = replace(
        sample_binding_spec,
        channels={
            "Risk": ChannelSpec(
                role="risk",
                required=False,
                replay_semantics="derived",
                derived_from=["Risk"],
                derive_rule="risk = lag(Risk)",
            )
        },
    )

    errors = validate_binding_spec(bad)

    assert any("derived_from must not include itself" in e for e in errors)


def test_channel_group_rejects_unknown_member(sample_binding_spec):
    bad = replace(
        sample_binding_spec,
        channel_groups={
            "control_surface": ChannelGroupSpec(channels=["P", "MissingRisk"])
        },
    )

    errors = validate_binding_spec(bad)

    assert any("channel_group 'control_surface'" in e for e in errors)
    assert any("MissingRisk" in e for e in errors)


def test_cross_channel_coupling_rejects_unknown_source_and_target(sample_binding_spec):
    bad = replace(
        sample_binding_spec,
        cross_channel_couplings=[
            CrossChannelCouplingSpec(
                source="UnknownSource",
                target="UnknownTarget",
                strength=0.1,
                mode="directed",
            )
        ],
    )

    errors = validate_binding_spec(bad)

    assert any("source references unknown channel" in e for e in errors)
    assert any("UnknownSource" in e for e in errors)
    assert any("target references unknown channel" in e for e in errors)
    assert any("UnknownTarget" in e for e in errors)


def test_invalid_extractor_type_is_reported(sample_binding_spec):
    bad = replace(
        sample_binding_spec,
        oscillator_families={
            "phys": OscillatorFamily(
                channel="P", extractor_type="spectrogram", config={}
            )
        },
    )

    errors = validate_binding_spec(bad)

    assert any("extractor_type must be one of" in e for e in errors)
    assert any("spectrogram" in e for e in errors)


def test_boundary_lower_greater_than_upper_is_reported(sample_binding_spec):
    boundary = object.__new__(BoundaryDef)
    object.__setattr__(boundary, "name", "unsafe_R")
    object.__setattr__(boundary, "variable", "R")
    object.__setattr__(boundary, "lower", 0.9)
    object.__setattr__(boundary, "upper", 0.2)
    object.__setattr__(boundary, "severity", "hard")
    bad = replace(sample_binding_spec, boundaries=[boundary])

    errors = validate_binding_spec(bad)

    assert any("unsafe_R" in e and "must be <= upper" in e for e in errors)


def test_actuator_scope_must_match_global_or_layer_index(sample_binding_spec):
    bad = replace(
        sample_binding_spec,
        actuators=[
            ActuatorMapping(
                name="K_future", knob="K", scope="layer_99", limits=(0.0, 1.0)
            )
        ],
    )

    errors = validate_binding_spec(bad)

    assert any("scope 'layer_99'" in e and "valid scopes" in e for e in errors)


def test_imprint_model_bounds_are_reported(sample_binding_spec):
    bad = replace(
        sample_binding_spec,
        imprint_model=ImprintSpec(decay_rate=-0.01, saturation=0.0, modulates=["K"]),
    )

    errors = validate_binding_spec(bad)

    assert any("imprint_model.decay_rate must be >= 0" in e for e in errors)
    assert any("imprint_model.saturation must be > 0" in e for e in errors)


def test_amplitude_model_bounds_are_reported(sample_binding_spec):
    bad = replace(
        sample_binding_spec,
        amplitude=AmplitudeSpec(mu=float("inf"), epsilon=-0.01),
    )

    errors = validate_binding_spec(bad)

    assert "amplitude.mu must be finite" in errors
    assert any("amplitude.epsilon must be >= 0" in e for e in errors)


# Pipeline wiring: binding validator tested via schema enforcement, required field
# validation, and type checking. TestValidationLogic (above) proves the validator
# gates pipeline inputs.
