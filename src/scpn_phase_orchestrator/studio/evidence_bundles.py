# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio schema-B evidence emission

"""Emit float-free Studio schema-B evidence from a SPO runtime snapshot.

The live Studio feed advertises three evidence families: runtime state, phase
coherence, and regime state. This module makes those declarations load-bearing
by reducing a validated runtime snapshot to an immutable canonical artifact,
binding its digest into :class:`scpn_studio_platform.evidence.EvidenceBundle`,
and passing the wire bundle through the Platform federation gate.

The producer never promotes a simulator observation to reference validation.
Every bundle is admitted only as a boundary claim on the ``numerical-model``
substrate. Seal-bound artifacts and bundle wires contain no JSON floats:
bounded numeric observations cross the wire as shortest-round-trip strings, so
Python and JavaScript cannot disagree over integer-looking float encodings.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import cast

from scpn_studio_platform.evidence import (
    AdmissionDecision,
    CaseResult,
    ClaimBoundary,
    ClaimStatus,
    EvidenceBundle,
    EvidenceKind,
    EvidenceLevel,
    FederationVerdict,
    Freshness,
    NumericProvenance,
    ProvActivity,
    ProvAgent,
    ProvEntity,
    Substrate,
    ValidityDomain,
    validate_studio_bundle,
)
from scpn_studio_platform.seal import canonicalize, content_digest

from scpn_phase_orchestrator.studio.live_feed import (
    PHASE_COHERENCE_SCHEMA,
    REGIME_STATE_SCHEMA,
    RUNTIME_SCHEMA,
    RUNTIME_STATE_SCHEMA,
    STUDIO_ID,
    runtime_summary,
)

JsonScalar = str | int | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_REGENERATED_BY = (
    "scpn_phase_orchestrator.studio.evidence_bundles.build_studio_evidence_emissions"
)


@dataclass(frozen=True, slots=True)
class StudioEvidenceEmission:
    """One immutable artifact, its schema-B bundle, and federation verdict.

    Parameters
    ----------
    bundle : EvidenceBundle
        Platform schema-B bundle whose entity digest pins the artifact bytes.
    verdict : FederationVerdict
        Era-v2 Platform admission result for the bundle wire.

    Notes
    -----
    Canonical artifact bytes are held privately so callers cannot mutate a
    dictionary after its digest has been bound into ``bundle.entity.digest``.
    The :attr:`artifact` property returns a fresh decoded mapping.
    """

    bundle: EvidenceBundle
    verdict: FederationVerdict
    _artifact_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        """Reject inconsistent or non-canonical emission components."""
        artifact = self.artifact
        _assert_float_free(artifact)
        if canonicalize(artifact) != self._artifact_bytes:
            raise ValueError("artifact bytes must use Platform canonical serialization")
        if artifact.get("schema") != self.bundle.schema:
            raise ValueError("artifact and evidence bundle schemas must match")
        if content_digest(artifact) != self.bundle.entity.digest:
            raise ValueError("artifact digest does not match bundle entity digest")
        if not self.verdict.admitted or self.verdict.rejections:
            raise ValueError("evidence bundle must pass the Platform federation gate")
        if self.verdict.mode != "boundary":
            raise ValueError("SPO runtime evidence must remain a boundary claim")
        _assert_float_free(self.bundle.to_dict())

    @property
    def artifact(self) -> dict[str, JsonValue]:
        """Return a fresh copy of the canonical float-free evidence artifact.

        Returns
        -------
        dict[str, JsonValue]
            Decoded artifact data. Mutating it cannot invalidate the stored
            content digest because the canonical source bytes remain private.
        """
        return cast("dict[str, JsonValue]", json.loads(self._artifact_bytes))

    def to_dict(self) -> dict[str, object]:
        """Return the artifact and schema-B bundle as a seal-safe wire record.

        Returns
        -------
        dict[str, object]
            Mapping with ``artifact`` and ``evidence_bundle`` fields and no
            JSON floating-point values.
        """
        record: dict[str, object] = {
            "artifact": self.artifact,
            "evidence_bundle": self.bundle.to_dict(),
        }
        _assert_float_free(record)
        return record


def build_studio_evidence_emissions(
    snapshot: Mapping[str, object],
    *,
    studio_version: str,
    activity_timestamp: str,
    operator: str = "local-operator",
) -> tuple[StudioEvidenceEmission, ...]:
    """Build and federation-gate the three live SPO evidence emissions.

    Parameters
    ----------
    snapshot : Mapping[str, object]
        Raw mapping from ``runtime.server.SimulationState.snapshot``.
    studio_version : str
        Exact SPO package version responsible for the emission.
    activity_timestamp : str
        ISO-8601 timestamp recorded as both start and end of the snapshot
        reduction. It is explicit so deterministic replays remain deterministic.
    operator : str, optional
        Opaque local operator or tenant identity for the PROV agent.

    Returns
    -------
    tuple[StudioEvidenceEmission, ...]
        Runtime-state, phase-coherence, and regime-state emissions, in the same
        stable order as the schema-A manifest's evidence types.

    Raises
    ------
    ValueError
        If snapshot validation fails, metadata is empty, a seal-bound float is
        found, or the Platform gate does not admit a boundary-only bundle.
    """
    _require_text(studio_version, "studio_version")
    _require_text(activity_timestamp, "activity_timestamp")
    _require_text(operator, "operator")
    runtime = runtime_summary(snapshot)
    artifacts = _artifacts(runtime)
    specifications = (
        (
            "simulate",
            EvidenceKind.MEASURED,
            artifacts[0],
            "One validated SPO numerical-model runtime snapshot only; no "
            "temporal stability, safety, hardware, or actuation claim.",
        ),
        (
            "analyse",
            EvidenceKind.MEASURED,
            artifacts[1],
            "Finite order parameters from one validated SPO numerical-model "
            "snapshot; no reference-validation, stability, safety, hardware, "
            "or actuation claim.",
        ),
        (
            "supervise",
            EvidenceKind.PRODUCER_ASSERTED,
            artifacts[2],
            "Producer-reported regime label from one SPO numerical-model "
            "snapshot; no independent validation, forecast, safety, hardware, "
            "or actuation claim.",
        ),
    )
    return tuple(
        _build_emission(
            artifact=artifact,
            verb=verb,
            evidence_kind=kind,
            validity_note=note,
            studio_version=studio_version,
            activity_timestamp=activity_timestamp,
            operator=operator,
        )
        for verb, kind, artifact, note in specifications
    )


def render_studio_evidence_emissions_json(
    snapshot: Mapping[str, object],
    *,
    studio_version: str,
    activity_timestamp: str,
    operator: str = "local-operator",
) -> str:
    """Render canonical float-free JSON for all live evidence emissions.

    Parameters
    ----------
    snapshot : Mapping[str, object]
        Raw mapping from ``runtime.server.SimulationState.snapshot``.
    studio_version : str
        Exact SPO package version responsible for the emission.
    activity_timestamp : str
        ISO-8601 timestamp recorded in every PROV activity.
    operator : str, optional
        Opaque local operator or tenant identity for the PROV agent.

    Returns
    -------
    str
        Platform-canonical compact JSON array followed by one newline.
    """
    emissions = build_studio_evidence_emissions(
        snapshot,
        studio_version=studio_version,
        activity_timestamp=activity_timestamp,
        operator=operator,
    )
    records = [emission.to_dict() for emission in emissions]
    _assert_float_free(records)
    canonical = cast("bytes", canonicalize(records))
    return canonical.decode("utf-8") + "\n"


def _build_emission(
    *,
    artifact: dict[str, JsonValue],
    verb: str,
    evidence_kind: EvidenceKind,
    validity_note: str,
    studio_version: str,
    activity_timestamp: str,
    operator: str,
) -> StudioEvidenceEmission:
    """Bind one canonical artifact into an admitted boundary-only bundle."""
    _assert_float_free(artifact)
    artifact_bytes = canonicalize(artifact)
    digest = content_digest(artifact)
    schema = cast("str", artifact["schema"])
    cases = _cases_for_artifact(artifact)
    bundle = EvidenceBundle(
        schema=schema,
        entity=ProvEntity(
            entity_id=(
                f"{STUDIO_ID}:{schema.removeprefix('studio.').removesuffix('.v1')}:"
                f"{digest.removeprefix('sha256:')[:16]}"
            ),
            digest=digest,
        ),
        activity=ProvActivity(
            verb=verb,
            studio=STUDIO_ID,
            started=activity_timestamp,
            ended=activity_timestamp,
            regenerated_by=_REGENERATED_BY,
        ),
        agent=ProvAgent(studio_version=studio_version, operator=operator),
        evidence_level=EvidenceLevel.TAXONOMY,
        evidence_kind=evidence_kind,
        claim_boundary=ClaimBoundary(
            status=ClaimStatus.BOUNDED_MODEL,
            admission=AdmissionDecision.ADMITTED,
            validity_domain=ValidityDomain(note=validity_note),
        ),
        substrate=Substrate.NUMERICAL_MODEL,
        freshness=Freshness.TRACEABLE_UNCHECKED,
        numeric_provenance=NumericProvenance(active_backend="spo-runtime-snapshot"),
        cases=cases,
    )
    verdict = validate_studio_bundle(bundle.to_dict())
    return StudioEvidenceEmission(
        bundle=bundle,
        verdict=verdict,
        _artifact_bytes=artifact_bytes,
    )


def _artifacts(runtime: Mapping[str, object]) -> tuple[dict[str, JsonValue], ...]:
    """Reduce a validated runtime summary into three float-free artifacts."""
    step = _required_runtime_int(runtime, "step")
    layers = _runtime_layers(runtime)
    runtime_artifact: dict[str, JsonValue] = {
        "schema": RUNTIME_STATE_SCHEMA,
        "snapshot_schema": RUNTIME_SCHEMA,
        "step": step,
        "n_oscillators": _required_runtime_int(runtime, "n_oscillators"),
        "amplitude_mode": _required_runtime_bool(runtime, "amplitude_mode"),
        "layer_count": len(layers),
        "layer_names": [cast("str", layer["name"]) for layer in layers],
    }
    if "mean_amplitude" in runtime:
        runtime_artifact["mean_amplitude"] = _float_text(
            runtime["mean_amplitude"], "mean_amplitude"
        )
    coherence_artifact: dict[str, JsonValue] = {
        "schema": PHASE_COHERENCE_SCHEMA,
        "step": step,
        "r_global": _float_text(runtime.get("r_global"), "r_global"),
        "layers": [
            {
                "name": cast("str", layer["name"]),
                "r": _float_text(layer.get("r"), f"layers[{index}].r"),
                "psi": _float_text(layer.get("psi"), f"layers[{index}].psi"),
            }
            for index, layer in enumerate(layers)
        ],
    }
    regime_artifact: dict[str, JsonValue] = {
        "schema": REGIME_STATE_SCHEMA,
        "step": step,
        "regime": _required_runtime_text(runtime, "regime"),
    }
    artifacts = (runtime_artifact, coherence_artifact, regime_artifact)
    for artifact in artifacts:
        _assert_float_free(artifact)
    return artifacts


def _cases_for_artifact(artifact: Mapping[str, JsonValue]) -> tuple[CaseResult, ...]:
    """Return integer-only case rows that preserve an artifact's review scope."""
    schema = cast("str", artifact["schema"])
    if schema == PHASE_COHERENCE_SCHEMA:
        layers = cast("list[JsonValue]", artifact["layers"])
        cases: list[CaseResult] = []
        for index, layer in enumerate(layers, start=1):
            if not isinstance(layer, Mapping):
                raise ValueError(
                    f"phase artifact layers[{index - 1}] must be a mapping"
                )
            name = layer.get("name")
            _require_text(name, f"phase artifact layers[{index - 1}].name")
            cases.append(
                CaseResult(
                    operation_family=f"phase-layer:{name}",
                    dimension=index,
                    status=ClaimStatus.BOUNDED_MODEL.value,
                )
            )
        return tuple(cases)
    if schema == RUNTIME_STATE_SCHEMA:
        names = cast("list[JsonValue]", artifact["layer_names"])
        return tuple(
            CaseResult(
                operation_family=f"runtime-layer:{name}",
                dimension=index,
                status=ClaimStatus.BOUNDED_MODEL.value,
            )
            for index, name in enumerate(names, start=1)
        )
    return (
        CaseResult(
            operation_family=f"regime:{artifact['regime']}",
            dimension=cast("int", artifact["step"]),
            status=ClaimStatus.BOUNDED_MODEL.value,
        ),
    )


def _runtime_layers(runtime: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return already-validated runtime layer mappings with defensive checks."""
    value = runtime.get("layers")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("runtime layers must be a sequence")
    layers: list[Mapping[str, object]] = []
    for index, layer in enumerate(value):
        if not isinstance(layer, Mapping):
            raise ValueError(f"runtime layers[{index}] must be a mapping")
        _require_text(layer.get("name"), f"runtime layers[{index}].name")
        layers.append(layer)
    if not layers:
        raise ValueError("runtime layers must not be empty")
    return layers


def _required_runtime_text(runtime: Mapping[str, object], key: str) -> str:
    """Return a required non-empty runtime text field."""
    value = runtime.get(key)
    _require_text(value, key)
    return cast("str", value)


def _required_runtime_int(runtime: Mapping[str, object], key: str) -> int:
    """Return a required non-negative runtime integer field."""
    value = runtime.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _required_runtime_bool(runtime: Mapping[str, object], key: str) -> bool:
    """Return a required runtime boolean field."""
    value = runtime.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a bool")
    return value


def _float_text(value: object, label: str) -> str:
    """Return a finite real as its shortest-round-trip decimal string."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return repr(result)


def _require_text(value: object, label: str) -> None:
    """Reject an empty or non-text metadata value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _assert_float_free(value: object, *, path: str = "$") -> None:
    """Reject JSON floats or unsupported values anywhere in a seal-bound tree."""
    if isinstance(value, float):
        raise ValueError(f"seal-bound artifact contains a JSON float at {path}")
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"seal-bound artifact key at {path} must be text")
            _assert_float_free(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_float_free(item, path=f"{path}[{index}]")
        return
    raise ValueError(
        f"seal-bound artifact contains unsupported {type(value).__name__} at {path}"
    )


__all__ = [
    "StudioEvidenceEmission",
    "build_studio_evidence_emissions",
    "render_studio_evidence_emissions_json",
]
