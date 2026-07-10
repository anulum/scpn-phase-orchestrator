# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STUDIO federation SDK fixture tests

"""SDK-independent contract tests for SPO's STUDIO federation manifest."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import sys
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

import pytest

from scpn_phase_orchestrator.studio.live_feed import LIVE_FEED_EVIDENCE_SCHEMAS

_EXPECTED_VERBS = {
    "bind",
    "simulate",
    "analyse",
    "supervise",
    "project",
    "forecast",
    "certify-conformal",
    "audit",
    "replay",
    "assure",
    "autotune",
}


@dataclass(frozen=True)
class _EnumValue:
    value: str


class _TransportProfile:
    LOCAL_FIRST = _EnumValue("local-first")


class _Fidelity:
    FIRST_PRINCIPLES = _EnumValue("first-principles")
    REDUCED_ORDER = _EnumValue("reduced-order")


class _SafetyTier:
    RESEARCH = _EnumValue("research")


class _SideEffect:
    READ_ONLY = _EnumValue("read-only")
    SIMULATED = _EnumValue("simulated")
    LIVE_HARDWARE = _EnumValue("live-hardware")


class _TimingClass:
    BATCH = _EnumValue("batch")


class _ProofMethod:
    BMC = _EnumValue("bmc")


@dataclass(frozen=True)
class _Timing:
    timing_class: _EnumValue


@dataclass(frozen=True)
class _VerbProof:
    method: _EnumValue
    engine: str
    engine_version: str
    depth: int
    non_vacuity_checked: bool


@dataclass(frozen=True)
class _Verb:
    name: str
    safety_tier: _EnumValue
    side_effect: _EnumValue
    timing: _Timing
    fidelity: _EnumValue | None = None
    proof: _VerbProof | None = None
    produces: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()


@dataclass(frozen=True)
class _UiModule:
    remote_entry: str
    exposes: tuple[str, ...]
    federation: str = "module-federation-2"

    def to_dict(self) -> dict[str, object]:
        """Return the schema-A ``ui_module`` block."""
        return {
            "remote_entry": self.remote_entry,
            "exposes": list(self.exposes),
            "federation": self.federation,
        }


@dataclass(frozen=True)
class _CapabilityManifest:
    studio: str
    studio_version: str
    platform_sdk: str
    content_digest: str
    protocol_version: str
    transport_profile: _EnumValue
    verbs: tuple[_Verb, ...]
    evidence_types: tuple[str, ...]
    external_reference_datasets: tuple[str, ...]
    ui_module: _UiModule | None
    contract_era: str
    enumeration: str

    def to_dict(self) -> dict[str, object]:
        """Return the schema-A JSON representation consumed by STUDIO Hub."""
        return {
            "studio": self.studio,
            "studio_version": self.studio_version,
            "platform_sdk": self.platform_sdk,
            "content_digest": self.content_digest,
            "protocol_version": self.protocol_version,
            "transport_profile": self.transport_profile.value,
            "verbs": [
                {
                    "verb": verb.name,
                    "safety_tier": verb.safety_tier.value,
                    "side_effect": verb.side_effect.value,
                    "timing": verb.timing.timing_class.value,
                    "fidelity": None if verb.fidelity is None else verb.fidelity.value,
                    "proof": None if verb.proof is None else verb.proof.method.value,
                    "produces": list(verb.produces),
                    "backends": list(verb.backends),
                }
                for verb in self.verbs
            ],
            "evidence_types": list(self.evidence_types),
            "external_reference_datasets": list(self.external_reference_datasets),
            "ui_module": None if self.ui_module is None else self.ui_module.to_dict(),
            "contract_era": self.contract_era,
            "enumeration": self.enumeration,
        }


def _content_digest(entries: Mapping[str, bytes]) -> str:
    """Return the Studio SDK-compatible deterministic content digest."""
    digest = hashlib.sha256()
    for name in sorted(entries):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entries[name])
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _install_sdk_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install an in-process Studio SDK ABI fixture for import-time binding."""
    package = types.ModuleType("scpn_studio_platform")
    manifest = types.ModuleType("scpn_studio_platform.manifest")
    verbs = types.ModuleType("scpn_studio_platform.verbs")

    manifest.__dict__.update(
        {
            "CapabilityManifest": _CapabilityManifest,
            "TransportProfile": _TransportProfile,
            "UiModule": _UiModule,
            "content_digest": _content_digest,
        }
    )
    verbs.__dict__.update(
        {
            "Fidelity": _Fidelity,
            "ProofMethod": _ProofMethod,
            "SafetyTier": _SafetyTier,
            "SideEffect": _SideEffect,
            "Timing": _Timing,
            "TimingClass": _TimingClass,
            "Verb": _Verb,
            "VerbProof": _VerbProof,
        }
    )

    monkeypatch.setitem(sys.modules, "scpn_studio_platform", package)
    monkeypatch.setitem(sys.modules, "scpn_studio_platform.manifest", manifest)
    monkeypatch.setitem(sys.modules, "scpn_studio_platform.verbs", verbs)


@contextmanager
def _manifest_module_with_sdk_fixture() -> Iterator[ModuleType]:
    """Yield the manifest module rebound against the Studio SDK fixture."""
    monkeypatch = pytest.MonkeyPatch()
    import scpn_phase_orchestrator.studio.federation_manifest as federation_manifest

    try:
        _install_sdk_fixture(monkeypatch)
        yield importlib.reload(federation_manifest)
    finally:
        monkeypatch.undo()
        importlib.reload(federation_manifest)


def _manifest_verbs(payload: dict[str, object]) -> list[dict[str, object]]:
    """Return the typed verb payloads from a manifest dictionary."""
    return cast("list[dict[str, object]]", payload["verbs"])


def test_manifest_builds_schema_a_contract_with_sdk_fixture() -> None:
    """The public builder emits the full schema-A contract without real SDK IO."""
    with _manifest_module_with_sdk_fixture() as federation_manifest:
        manifest = federation_manifest.build_capability_manifest(studio_version="1.2.3")
        payload = federation_manifest.manifest_dict(studio_version="1.2.3")

    assert manifest.studio == "scpn-phase-orchestrator"
    assert manifest.studio_version == "1.2.3"
    assert manifest.transport_profile.value == "local-first"
    assert manifest.evidence_types == LIVE_FEED_EVIDENCE_SCHEMAS
    assert manifest.content_digest.startswith("sha256:")
    assert payload["studio"] == manifest.studio
    assert payload["platform_sdk"] == "0.3.0"
    assert payload["protocol_version"] == "1"
    assert payload["contract_era"] == "v1"
    assert payload["enumeration"] == "language-agnostic"
    assert payload["ui_module"] == {
        "remote_entry": (
            "https://www.anulum.org/studios/scpn-phase-orchestrator/remoteEntry.js"
        ),
        "exposes": ["./SpoStudioPanel"],
        "federation": "scpn_phase_orchestrator",
    }
    assert payload["evidence_types"] == list(LIVE_FEED_EVIDENCE_SCHEMAS)
    assert {verb["verb"] for verb in _manifest_verbs(payload)} == _EXPECTED_VERBS


def test_manifest_sdk_fixture_preserves_review_only_honesty_contract() -> None:
    """Every advertised SPO verb remains research-tier and non-actuating."""
    with _manifest_module_with_sdk_fixture() as federation_manifest:
        payload = federation_manifest.manifest_dict(studio_version="0.0.0")

    for verb in _manifest_verbs(payload):
        assert verb["safety_tier"] == "research"
        assert verb["side_effect"] != "live-hardware"
        assert verb["timing"] == "batch"
        assert verb["proof"] is None
        assert isinstance(verb["produces"], list)
        assert isinstance(verb["backends"], list)


def test_manifest_digest_changes_when_evidence_schema_changes() -> None:
    """Evidence feed schema changes alter the advertised federation digest."""
    with _manifest_module_with_sdk_fixture() as federation_manifest:
        verbs = federation_manifest.build_capability_manifest(
            studio_version="0.0.0"
        ).verbs
        baseline = federation_manifest._manifest_content_digest(
            verbs,
            LIVE_FEED_EVIDENCE_SCHEMAS,
        )
        changed = federation_manifest._manifest_content_digest(
            verbs,
            (*LIVE_FEED_EVIDENCE_SCHEMAS, "spo.extra-evidence.v1"),
        )

    assert changed != baseline


def test_manifest_digest_tracks_formal_proof_attributes() -> None:
    """A proof-bearing future verb changes the manifest fingerprint."""
    plain = _Verb(
        name="verify",
        safety_tier=_SafetyTier.RESEARCH,
        side_effect=_SideEffect.READ_ONLY,
        timing=_Timing(timing_class=_TimingClass.BATCH),
    )
    proved = _Verb(
        name="verify",
        safety_tier=_SafetyTier.RESEARCH,
        side_effect=_SideEffect.READ_ONLY,
        timing=_Timing(timing_class=_TimingClass.BATCH),
        proof=_VerbProof(
            method=_ProofMethod.BMC,
            engine="nuXmv",
            engine_version="2.0",
            depth=20,
            non_vacuity_checked=True,
        ),
    )

    with _manifest_module_with_sdk_fixture() as federation_manifest:
        plain_fingerprint = federation_manifest._verb_fingerprint(plain)
        proved_fingerprint = federation_manifest._verb_fingerprint(proved)

    assert proved_fingerprint != plain_fingerprint
    assert "|bmc|" in proved_fingerprint


def test_manifest_import_guard_fails_closed_without_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production import guard still fails closed when the SDK is absent."""
    import scpn_phase_orchestrator.studio.federation_manifest as federation_manifest

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """Block only the optional Studio SDK import path."""
        if name.startswith("scpn_studio_platform"):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for module_name in (
        "scpn_studio_platform",
        "scpn_studio_platform.manifest",
        "scpn_studio_platform.verbs",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    try:
        reloaded = importlib.reload(federation_manifest)
        with pytest.raises(RuntimeError, match="STUDIO platform SDK is required"):
            reloaded.manifest_dict(studio_version="0.0.0")
    finally:
        monkeypatch.undo()
        importlib.reload(federation_manifest)
