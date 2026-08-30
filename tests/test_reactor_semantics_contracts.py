# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor-semantic contract tests

"""Fail-closed tests for public U0 reactor semantic contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from scpn_phase_orchestrator.reactor_semantics import (
    ACTION_OWNER,
    REVIEW_ONLY_AUTHORITY,
    SEMANTIC_OWNER,
    ClockKind,
    EvidenceClass,
    PhaseRelationType,
    QualityAssessment,
    QualityState,
    RegimeEstimate,
    RegimeState,
    RelationInterpretation,
    SemanticCarrier,
    Uncertainty,
    ValidityState,
    ValidityWindow,
    build_phase_relation,
    build_reactor_reference_portfolio,
    validate_observable_sequence,
)


def _phase_pair():
    source = build_reactor_reference_portfolio()[0].semantics[0]
    target = replace(source, phase_id="u0.a1.target", phase_rad=0.7)
    return source, target


def test_phase_relation_accepts_only_explicitly_compatible_phase_records() -> None:
    source, target = _phase_pair()
    relation = build_phase_relation(
        source,
        target,
        relation_id="u0.a1.relation",
        relation_type=PhaseRelationType.SAME_MODE,
        interpretation=RelationInterpretation.CONTEXT_DEPENDENT,
        identification_method="cross_spectral_review",
        evidence_class=EvidenceClass.SCAFFOLD,
    )

    assert relation.harmonic_ratio == (1, 1)
    assert relation.source_phase_id == source.phase_id

    with pytest.raises(ValueError, match="reference frames"):
        build_phase_relation(
            source,
            replace(target, reference_frame="u0.a1.other_frame"),
            relation_id="u0.a1.bad_frame",
            relation_type=PhaseRelationType.SAME_MODE,
            interpretation=RelationInterpretation.AMBIGUOUS,
            identification_method="review",
            evidence_class=EvidenceClass.SCAFFOLD,
        )
    with pytest.raises(ValueError, match="clock domains"):
        build_phase_relation(
            source,
            replace(target, clock_domain="u0.a1.other_clock"),
            relation_id="u0.a1.bad_clock",
            relation_type=PhaseRelationType.SAME_MODE,
            interpretation=RelationInterpretation.AMBIGUOUS,
            identification_method="review",
            evidence_class=EvidenceClass.SCAFFOLD,
        )
    with pytest.raises(ValueError, match="clock domains"):
        build_phase_relation(
            source,
            replace(target, clock_kind=ClockKind.MODEL_TICK),
            relation_id="u0.a1.bad_clock_kind",
            relation_type=PhaseRelationType.SAME_MODE,
            interpretation=RelationInterpretation.AMBIGUOUS,
            identification_method="review",
            evidence_class=EvidenceClass.SCAFFOLD,
        )
    with pytest.raises(ValueError, match="different harmonics"):
        build_phase_relation(
            source,
            replace(target, mode_harmonic=(3, 2)),
            relation_id="u0.a1.bad_harmonic",
            relation_type=PhaseRelationType.SAME_MODE,
            interpretation=RelationInterpretation.AMBIGUOUS,
            identification_method="review",
            evidence_class=EvidenceClass.SCAFFOLD,
        )


def test_noncyclic_features_and_event_counts_cannot_become_angles() -> None:
    phase = build_reactor_reference_portfolio()[0].semantics[1]

    for carrier in (
        SemanticCarrier.BOUNDED_FEATURE,
        SemanticCarrier.CATEGORICAL_STATE,
        SemanticCarrier.PROTOCOL_PHASE,
    ):
        with pytest.raises(ValueError, match="non-phase carrier"):
            replace(phase, carrier_type=carrier)

    with pytest.raises(ValueError, match="reference_signal"):
        replace(
            phase,
            carrier_type=SemanticCarrier.EVENT_CYCLE,
            phenomenon="event_count",
            reference_signal=None,
        )


def test_low_observability_and_unusable_evidence_fail_closed() -> None:
    phase = build_reactor_reference_portfolio()[0].semantics[1]
    unobservable = replace(
        phase,
        phase_rad=None,
        observability=0.01,
        observability_threshold=0.10,
        validity=ValidityWindow(
            ValidityState.UNOBSERVABLE,
            valid_from_ns=900_000,
            valid_until_ns=1_100_000,
            reasons=("below declared diagnostic threshold",),
        ),
    )

    assert unobservable.is_usable is False
    assert unobservable.phase_rad is None
    with pytest.raises(ValueError, match="UNOBSERVABLE"):
        replace(phase, observability=0.01, observability_threshold=0.10)
    with pytest.raises(ValueError, match="cannot publish phase_rad"):
        replace(unobservable, phase_rad=0.2)
    with pytest.raises(ValueError, match="unknown or invalid quality"):
        replace(
            phase,
            quality=QualityAssessment(QualityState.UNKNOWN, ("missing_signal",)),
        )


def test_special_phase_carriers_enforce_their_evidence_semantics() -> None:
    complex_mode = build_reactor_reference_portfolio()[0].semantics[0]
    numerical = build_reactor_reference_portfolio()[2].semantics[2]

    with pytest.raises(ValueError, match="zero-amplitude"):
        replace(complex_mode, amplitude=0.0)
    with pytest.raises(ValueError, match="cannot claim observed"):
        replace(numerical, evidence_class=EvidenceClass.OBSERVED)


@pytest.mark.parametrize(
    "validity_state",
    [
        ValidityState.UNKNOWN,
        ValidityState.STALE,
        ValidityState.OUT_OF_DISTRIBUTION,
        ValidityState.INVALID,
    ],
)
def test_nonusable_regime_states_remain_unknown(validity_state: ValidityState) -> None:
    regime = build_reactor_reference_portfolio()[0].regime
    validity = ValidityWindow(
        validity_state,
        valid_from_ns=900_000,
        valid_until_ns=1_100_000,
        reasons=(validity_state.value,),
    )
    unknown = replace(regime, state=RegimeState.UNKNOWN, validity=validity)

    assert unknown.semantic_owner == SEMANTIC_OWNER
    assert unknown.action_owner == ACTION_OWNER
    assert unknown.authority == REVIEW_ONLY_AUTHORITY
    with pytest.raises(ValueError, match="UNKNOWN regime"):
        replace(regime, validity=validity)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("semantic_owner", "SCPN-CONTROL", "semantic owner"),
        ("action_owner", "SCPN-PHASE-ORCHESTRATOR", "action owner"),
        ("authority", "actuate", "review-only"),
    ],
)
def test_regime_ownership_cannot_be_reassigned(
    field: str,
    value: str,
    match: str,
) -> None:
    regime: RegimeEstimate = build_reactor_reference_portfolio()[0].regime
    with pytest.raises(ValueError, match=match):
        replace(regime, **{field: value})


def test_uncertainty_accepts_signed_bounds_but_refuses_invalid_ranges() -> None:
    uncertainty = Uncertainty(0.1, 0.95, lower_bound=-1.0, upper_bound=2.0)
    assert Uncertainty.from_record(uncertainty.to_record()) == uncertainty
    with pytest.raises(ValueError, match="lower_bound"):
        Uncertainty(0.1, 0.95, lower_bound=2.0, upper_bound=-1.0)


def test_observable_sequence_requires_one_usable_strictly_monotonic_clock() -> None:
    first = build_reactor_reference_portfolio()[0].observable
    second = replace(
        first,
        clock=replace(first.clock, timestamp_ns=first.clock.timestamp_ns + 1),
    )

    assert validate_observable_sequence((first, second)) == (first, second)
    with pytest.raises(ValueError, match="must not be empty"):
        validate_observable_sequence(())
    with pytest.raises(ValueError, match="strictly monotonic"):
        validate_observable_sequence((first, first))
    with pytest.raises(ValueError, match="mixes stream"):
        validate_observable_sequence(
            (
                first,
                replace(
                    second,
                    clock=replace(second.clock, kind=ClockKind.MODEL_TICK),
                ),
            )
        )
    stale = replace(
        second,
        validity=ValidityWindow(
            ValidityState.STALE,
            valid_from_ns=900_000,
            valid_until_ns=1_100_000,
            reasons=("stale",),
        ),
    )
    with pytest.raises(ValueError, match="unusable evidence"):
        validate_observable_sequence((first, stale))
