# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STUDIO federation CapabilityManifest

"""Build the SCPN STUDIO federation `CapabilityManifest` for SPO.

This emits the schema-A capability manifest the SCPN STUDIO Hub ingests
(`scpn-studio-platform`), describing the verbs SPO exposes and their honesty
attributes (safety tier, side effect, timing, fidelity, proof, produces,
backends). It is distinct from `tools/capability_manifest.py`, which generates
the repository's static public inventory.

Honesty is built in by construction: SPO is review-only, so no verb is
``live-hardware``; SPO exports PRISM/TLA+ models but does not discharge them to a
proven depth, so no verb claims a formal ``proof``; and the safety tier is
``research`` rather than ``certified``/``production`` because SPO is a
research-grade toolkit, not a certified product. The platform SDK is an optional
dependency (the ``studio`` extra); importing this module without it raises a
clear error.

The manifest declares the federated UI remote through ``ui_module``: the panel
built in ``studio-web/`` and pull-deployed to the studio's own space. Its
federation name, exposed module, and remote-entry URL
(:data:`STUDIO_FEDERATION_NAME`, :data:`STUDIO_EXPOSED_MODULE`,
:data:`STUDIO_REMOTE_ENTRY`) are the contract with
``studio-web/module-federation.config.ts`` and must match it exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scpn_phase_orchestrator.studio.live_feed import LIVE_FEED_EVIDENCE_SCHEMAS

try:
    from scpn_studio_platform.manifest import (
        CapabilityManifest,
        TransportProfile,
        UiModule,
        content_digest,
    )
    from scpn_studio_platform.verbs import (
        Fidelity,
        SafetyTier,
        SideEffect,
        Timing,
        TimingClass,
        Verb,
    )

    _HAS_STUDIO_SDK = True
except ImportError:  # pragma: no cover - exercised via the import-guard test
    _HAS_STUDIO_SDK = False

if TYPE_CHECKING:
    from scpn_studio_platform.manifest import CapabilityManifest as _CapabilityManifest

__all__ = ["build_capability_manifest", "manifest_dict"]

_PLATFORM_SDK = "0.3.0"
_PROTOCOL_VERSION = "1"
_POLYGLOT = ("rust", "webgpu", "mojo", "julia", "go", "python")
_NUMPY_ONLY = ("python",)

# The federated UI remote (built in ``studio-web/``). These three values are the
# contract with the JavaScript remote and MUST match
# ``studio-web/module-federation.config.ts``: the federation container name, the
# exposed panel module, and the pull-deployed remote-entry URL under the studio's
# own space (``base=/studios/scpn-phase-orchestrator/``).
STUDIO_FEDERATION_NAME = "scpn_phase_orchestrator"
STUDIO_EXPOSED_MODULE = "./SpoStudioPanel"
STUDIO_REMOTE_ENTRY = (
    "https://www.anulum.org/studios/scpn-phase-orchestrator/remoteEntry.js"
)


def _require_sdk() -> None:
    """Raise a clear error if the optional studio SDK is not installed."""
    if not _HAS_STUDIO_SDK:
        raise RuntimeError(
            "the SCPN STUDIO platform SDK is required to build the federation "
            "manifest: pip install 'scpn-phase-orchestrator[studio]' "
            "(scpn-studio-platform>=0.3)"
        )


def _verbs() -> tuple[Verb, ...]:
    """SPO's verbs with honest federation attributes.

    Every verb is review-only (``read-only`` or ``simulated`` — never
    ``live-hardware``) and carries no formal ``proof`` (SPO exports PRISM/TLA+
    models but does not discharge them).
    """
    batch = Timing(timing_class=TimingClass.BATCH)
    research = SafetyTier.RESEARCH
    read_only = SideEffect.READ_ONLY
    simulated = SideEffect.SIMULATED

    return (
        Verb(
            name="bind",
            safety_tier=research,
            side_effect=read_only,
            timing=batch,
            produces=("binding_spec",),
            backends=_NUMPY_ONLY,
        ),
        Verb(
            name="simulate",
            safety_tier=research,
            side_effect=simulated,
            timing=batch,
            fidelity=Fidelity.FIRST_PRINCIPLES,
            produces=("upde_state", "order_parameter"),
            backends=_POLYGLOT,
        ),
        Verb(
            name="analyse",
            safety_tier=research,
            side_effect=read_only,
            timing=batch,
            produces=("coherence", "lyapunov", "transfer_entropy", "twin_confidence"),
            backends=_POLYGLOT,
        ),
        Verb(
            name="supervise",
            safety_tier=research,
            side_effect=read_only,
            timing=batch,
            fidelity=Fidelity.REDUCED_ORDER,
            produces=("control_action_proposal", "regime"),
            backends=_NUMPY_ONLY,
        ),
        Verb(
            name="project",
            safety_tier=research,
            side_effect=simulated,
            timing=batch,
            produces=("projected_action",),
            backends=_NUMPY_ONLY,
        ),
        Verb(
            name="forecast",
            safety_tier=research,
            side_effect=read_only,
            timing=batch,
            fidelity=Fidelity.REDUCED_ORDER,
            produces=("regime_forecast",),
            backends=_NUMPY_ONLY,
        ),
        Verb(
            name="certify-conformal",
            safety_tier=research,
            side_effect=read_only,
            timing=batch,
            produces=("conformal_admission",),
            backends=_NUMPY_ONLY,
        ),
        Verb(
            name="audit",
            safety_tier=research,
            side_effect=read_only,
            timing=batch,
            produces=("audit_record",),
            backends=_NUMPY_ONLY,
        ),
        Verb(
            name="replay",
            safety_tier=research,
            side_effect=read_only,
            timing=batch,
            produces=("replay_verdict",),
            backends=_NUMPY_ONLY,
        ),
        Verb(
            name="assure",
            safety_tier=research,
            side_effect=read_only,
            timing=batch,
            produces=("assurance_bundle",),
            backends=_NUMPY_ONLY,
        ),
        Verb(
            name="autotune",
            safety_tier=research,
            side_effect=simulated,
            timing=batch,
            produces=("binding_spec",),
            backends=_NUMPY_ONLY,
        ),
    )


def _verb_fingerprint(verb: Verb) -> str:
    """Stable fingerprint of a verb's honesty attributes for the content digest.

    Covers every attribute that distinguishes the verb's contract, so any change
    to a tier, side effect, fidelity, timing, output, or backend changes the
    manifest digest.
    """
    fidelity = verb.fidelity.value if verb.fidelity is not None else ""
    proof = verb.proof.method.value if verb.proof is not None else ""
    return "|".join(
        (
            verb.name,
            verb.safety_tier.value,
            verb.side_effect.value,
            verb.timing.timing_class.value,
            fidelity,
            proof,
            ",".join(verb.produces),
            ",".join(verb.backends),
        )
    )


def _manifest_content_digest(
    verbs: tuple[Verb, ...],
    evidence_types: tuple[str, ...],
) -> str:
    """Return the digest for SPO's advertised Studio federation surface.

    Parameters
    ----------
    verbs : tuple[Verb, ...]
        The manifest verb contracts to fingerprint.
    evidence_types : tuple[str, ...]
        The versioned evidence schemas SPO emits through the live Studio feed.

    Returns
    -------
    str
        A ``sha256:`` digest over the advertised verb and evidence schema surface.
    """
    entries = {f"verb:{verb.name}": _verb_fingerprint(verb).encode() for verb in verbs}
    entries.update(
        {
            f"evidence:{index}:{schema}": schema.encode()
            for index, schema in enumerate(evidence_types)
        }
    )
    return str(content_digest(entries))


def build_capability_manifest(*, studio_version: str) -> _CapabilityManifest:
    """Construct SPO's STUDIO federation `CapabilityManifest`.

    Parameters
    ----------
    studio_version : str
        The SPO package version stamped into the manifest.

    Returns
    -------
    CapabilityManifest
        The schema-A manifest describing SPO's verbs and honesty attributes.

    Raises
    ------
    RuntimeError
        If the ``scpn-studio-platform`` SDK is not installed.
    """
    _require_sdk()
    verbs = _verbs()
    evidence_types = LIVE_FEED_EVIDENCE_SCHEMAS
    digest = _manifest_content_digest(verbs, evidence_types)
    return CapabilityManifest(
        studio="scpn-phase-orchestrator",
        studio_version=studio_version,
        platform_sdk=_PLATFORM_SDK,
        content_digest=digest,
        protocol_version=_PROTOCOL_VERSION,
        transport_profile=TransportProfile.LOCAL_FIRST,
        verbs=verbs,
        evidence_types=evidence_types,
        external_reference_datasets=(),
        ui_module=UiModule(
            remote_entry=STUDIO_REMOTE_ENTRY,
            exposes=(STUDIO_EXPOSED_MODULE,),
            federation=STUDIO_FEDERATION_NAME,
        ),
        contract_era="v1",
        enumeration="language-agnostic",
    )


def manifest_dict(*, studio_version: str) -> dict[str, object]:
    """Return the federation manifest as a JSON-serialisable dict.

    Parameters
    ----------
    studio_version : str
        The SPO package version stamped into the manifest.

    Returns
    -------
    dict[str, object]
        ``CapabilityManifest.to_dict()`` output.
    """
    result: dict[str, object] = build_capability_manifest(
        studio_version=studio_version
    ).to_dict()
    return result
