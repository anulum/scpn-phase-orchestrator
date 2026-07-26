# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio schema-B evidence tests

"""Focused contracts for float-free Studio schema-B evidence emission."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest

pytest.importorskip("scpn_studio_platform")

from scpn_studio_platform.evidence import (  # noqa: E402
    ClaimStatus,
    EvidenceKind,
    EvidenceLevel,
    Freshness,
    Substrate,
    validate_studio_bundle,
)
from scpn_studio_platform.seal import canonicalize, content_digest  # noqa: E402

import scpn_phase_orchestrator.studio.evidence_bundles as evidence_bundles  # noqa: E402
from scpn_phase_orchestrator.studio.evidence_bundles import (  # noqa: E402
    StudioEvidenceEmission,
    build_studio_evidence_emissions,
    render_studio_evidence_emissions_json,
)
from scpn_phase_orchestrator.studio.live_feed import (  # noqa: E402
    LIVE_FEED_EVIDENCE_SCHEMAS,
)

_TIMESTAMP = "2026-07-26T12:00:00Z"


def _snapshot() -> dict[str, object]:
    """Return one representative numerical-model runtime snapshot."""
    return {
        "step": 3,
        "R_global": 0.75,
        "regime": "sync",
        "layers": [
            {"name": "p", "R": 0.7, "psi": 0.1},
            {"name": "i", "R": 0.8, "psi": -0.2},
        ],
        "n_oscillators": 4,
        "amplitude_mode": True,
        "mean_amplitude": 1.25,
    }


def _contains_float(value: object) -> bool:
    """Return whether a nested JSON-like value contains a Python float."""
    if isinstance(value, float):
        return True
    if isinstance(value, Mapping):
        return any(_contains_float(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_float(item) for item in value)
    return False


def _emissions() -> tuple[StudioEvidenceEmission, ...]:
    """Build the canonical focused-test emissions."""
    return build_studio_evidence_emissions(
        _snapshot(),
        studio_version="1.0.0",
        activity_timestamp=_TIMESTAMP,
        operator="test-operator",
    )


def test_emits_every_advertised_schema_as_an_admitted_boundary_bundle() -> None:
    """The schema-A evidence surface has one real schema-B producer per family."""
    emissions = _emissions()

    assert tuple(emission.bundle.schema for emission in emissions) == (
        "studio.runtime-state.v1",
        "studio.phase-coherence.v1",
        "studio.regime-state.v1",
    )
    assert tuple(emission.bundle.schema for emission in emissions) == (
        LIVE_FEED_EVIDENCE_SCHEMAS
    )
    assert tuple(emission.bundle.evidence_kind for emission in emissions) == (
        EvidenceKind.MEASURED,
        EvidenceKind.MEASURED,
        EvidenceKind.PRODUCER_ASSERTED,
    )
    for emission in emissions:
        assert emission.bundle.evidence_level is EvidenceLevel.TAXONOMY
        assert emission.bundle.claim_boundary.status is ClaimStatus.BOUNDED_MODEL
        assert emission.bundle.claim_boundary.validity_domain is not None
        assert emission.bundle.substrate is Substrate.NUMERICAL_MODEL
        assert emission.bundle.freshness is Freshness.TRACEABLE_UNCHECKED
        assert emission.bundle.renders_as_validated is False
        assert emission.verdict.admitted is True
        assert emission.verdict.mode == "boundary"
        assert emission.verdict.rejections == ()
        assert validate_studio_bundle(emission.bundle.to_dict()) == emission.verdict


def test_artifacts_are_float_free_and_content_addressed() -> None:
    """Every seal input uses strings for bounded numbers and pins exact bytes."""
    runtime, coherence, regime = _emissions()

    assert runtime.artifact["mean_amplitude"] == "1.25"
    assert coherence.artifact["r_global"] == "0.75"
    assert coherence.artifact["layers"] == [
        {"name": "p", "psi": "0.1", "r": "0.7"},
        {"name": "i", "psi": "-0.2", "r": "0.8"},
    ]
    assert regime.artifact["regime"] == "sync"
    for emission in (runtime, coherence, regime):
        assert not _contains_float(emission.artifact)
        assert not _contains_float(emission.bundle.to_dict())
        assert not _contains_float(emission.to_dict())
        assert content_digest(emission.artifact) == emission.bundle.entity.digest


def test_returned_artifact_is_a_copy_not_the_digest_source() -> None:
    """Caller mutation cannot detach a bundle digest from its canonical artifact."""
    emission = _emissions()[0]
    mutated = emission.artifact

    mutated["step"] = 999

    assert emission.artifact["step"] == 3
    assert content_digest(emission.artifact) == emission.bundle.entity.digest


def test_canonical_renderer_is_deterministic_and_float_free() -> None:
    """Equal inputs produce byte-identical cross-language-safe JSON."""
    first = render_studio_evidence_emissions_json(
        _snapshot(), studio_version="1.0.0", activity_timestamp=_TIMESTAMP
    )
    second = render_studio_evidence_emissions_json(
        _snapshot(), studio_version="1.0.0", activity_timestamp=_TIMESTAMP
    )

    assert first == second
    assert first.endswith("\n")
    assert first[:-1].encode() == canonicalize(json.loads(first))
    assert not _contains_float(json.loads(first))


def test_amplitude_field_is_omitted_when_runtime_has_no_amplitude_state() -> None:
    """Phase-only snapshots do not invent an amplitude observation."""
    snapshot = _snapshot()
    snapshot["amplitude_mode"] = False
    snapshot.pop("mean_amplitude")

    runtime = build_studio_evidence_emissions(
        snapshot,
        studio_version="1.0.0",
        activity_timestamp=_TIMESTAMP,
    )[0]

    assert "mean_amplitude" not in runtime.artifact


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("studio_version", "", "studio_version must be a non-empty string"),
        ("activity_timestamp", "", "activity_timestamp must be a non-empty string"),
        ("operator", " ", "operator must be a non-empty string"),
    ],
)
def test_empty_provenance_metadata_fails_closed(
    keyword: str,
    value: str,
    match: str,
) -> None:
    """PROV identifiers cannot silently cross the federation gate empty."""
    kwargs = {
        "studio_version": "1.0.0",
        "activity_timestamp": _TIMESTAMP,
        "operator": "test-operator",
        keyword: value,
    }

    with pytest.raises(ValueError, match=match):
        build_studio_evidence_emissions(_snapshot(), **kwargs)


def test_invalid_runtime_snapshot_fails_before_evidence_emission() -> None:
    """The producer reuses the live-feed snapshot validation boundary."""
    snapshot = _snapshot()
    snapshot["R_global"] = float("nan")

    with pytest.raises(ValueError, match="R_global must be finite"):
        build_studio_evidence_emissions(
            snapshot,
            studio_version="1.0.0",
            activity_timestamp=_TIMESTAMP,
        )


def test_tampered_float_artifact_is_rejected_even_when_json_canonicalizes() -> None:
    """The producer's stricter seal contract rejects an otherwise valid JSON float."""
    emission = _emissions()[0]
    tampered = {"schema": emission.bundle.schema, "value": 1.0}

    with pytest.raises(ValueError, match=r"JSON float at \$\.value"):
        replace(emission, _artifact_bytes=canonicalize(tampered))


def test_tampered_artifact_digest_is_rejected() -> None:
    """Canonical artifact bytes cannot be swapped beneath a valid bundle."""
    emission = _emissions()[0]
    tampered = {**emission.artifact, "step": 4}

    with pytest.raises(ValueError, match="artifact digest does not match"):
        replace(emission, _artifact_bytes=canonicalize(tampered))


def test_emission_rejects_noncanonical_bytes_and_schema_drift() -> None:
    """Canonical bytes and the artifact/bundle schema identity are load-bearing."""
    emission = _emissions()[0]

    with pytest.raises(ValueError, match="canonical serialization"):
        replace(emission, _artifact_bytes=b" " + canonicalize(emission.artifact))

    wrong_schema = {**emission.artifact, "schema": "studio.other.v1"}
    with pytest.raises(ValueError, match="schemas must match"):
        replace(emission, _artifact_bytes=canonicalize(wrong_schema))


def test_emission_rejects_nonadmitted_or_promoted_verdicts() -> None:
    """An emission cannot bypass admission or promote itself above boundary."""
    emission = _emissions()[0]
    rejected = replace(
        emission.verdict,
        admitted=False,
        rejections=("synthetic rejection",),
    )
    promoted = replace(emission.verdict, mode="validated")

    with pytest.raises(ValueError, match="must pass the Platform federation gate"):
        replace(emission, verdict=rejected)
    with pytest.raises(ValueError, match="must remain a boundary claim"):
        replace(emission, verdict=promoted)


def test_private_artifact_guards_fail_closed_on_malformed_internal_shapes() -> None:
    """Defensive reducers reject shapes that validated runtime data cannot contain."""
    with pytest.raises(ValueError, match=r"layers\[0\] must be a mapping"):
        evidence_bundles._cases_for_artifact(
            {"schema": "studio.phase-coherence.v1", "layers": [1]}
        )
    with pytest.raises(ValueError, match="runtime layers must be a sequence"):
        evidence_bundles._runtime_layers({"layers": "bad"})
    with pytest.raises(ValueError, match=r"runtime layers\[0\] must be a mapping"):
        evidence_bundles._runtime_layers({"layers": [1]})
    with pytest.raises(ValueError, match="runtime layers must not be empty"):
        evidence_bundles._runtime_layers({"layers": []})


def test_private_scalar_guards_reject_impossible_post_validation_values() -> None:
    """Internal scalar reducers preserve the same strict runtime value contract."""
    with pytest.raises(ValueError, match="step must be a non-negative integer"):
        evidence_bundles._required_runtime_int({"step": True}, "step")
    with pytest.raises(ValueError, match="amplitude_mode must be a bool"):
        evidence_bundles._required_runtime_bool({"amplitude_mode": 1}, "amplitude_mode")
    with pytest.raises(ValueError, match="r must be a finite real number"):
        evidence_bundles._float_text(True, "r")
    with pytest.raises(ValueError, match="r must be finite"):
        evidence_bundles._float_text(float("inf"), "r")


def test_float_free_guard_rejects_nontext_keys_and_non_json_objects() -> None:
    """Seal trees admit only string-keyed JSON values, never arbitrary objects."""
    with pytest.raises(ValueError, match="key at \\$ must be text"):
        evidence_bundles._assert_float_free({1: "bad"})
    with pytest.raises(ValueError, match=r"unsupported object at \$"):
        evidence_bundles._assert_float_free(object())
