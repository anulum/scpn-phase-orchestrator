# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STUDIO federation manifest tests

"""Tests for the SPO STUDIO federation `CapabilityManifest` generator: the
manifest is well-formed, the digest is deterministic and attribute-sensitive,
and the honesty invariants hold (review-only, no formal proof, research tier).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import scpn_phase_orchestrator.studio.federation_manifest as fm
from scpn_phase_orchestrator.studio.live_feed import LIVE_FEED_EVIDENCE_SCHEMAS

pytest.importorskip("scpn_studio_platform")

from scpn_studio_platform.manifest import resolve_pipeline  # noqa: E402
from scpn_studio_platform.verbs import (  # noqa: E402  (after importorskip)
    ProofMethod,
    SafetyTier,
    SideEffect,
    Timing,
    TimingClass,
    Verb,
    VerbProof,
)

_STUDIO_PLATFORM_SRC = (
    Path(__file__).resolve().parents[1].parent / "SCPN-STUDIO-PLATFORM" / "src"
)
_VALIDATE_WITH_SIBLING_PLATFORM = """
import json
import sys

from scpn_studio_platform.manifest.federation import validate_studio_manifest

verdict = validate_studio_manifest(json.loads(sys.stdin.read()))
print(
    json.dumps(
        {
            "admitted": verdict.admitted,
            "contract_era": verdict.contract_era,
            "rejections": list(verdict.rejections),
            "warnings": list(verdict.warnings),
        },
        sort_keys=True,
    )
)
ok = verdict.admitted and verdict.contract_era == "v1" and not verdict.warnings
raise SystemExit(0 if ok else 1)
"""
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
_EXPECTED_CONSUMES = {
    "simulate": ("binding_spec",),
    "analyse": ("upde_state",),
    "supervise": ("coherence", "twin_confidence"),
    "project": ("control_action_proposal",),
    "forecast": ("upde_state",),
    "certify-conformal": ("regime_forecast",),
    "audit": ("projected_action",),
    "replay": ("audit_record",),
    "assure": ("audit_record", "replay_verdict"),
}


def test_manifest_is_well_formed() -> None:
    manifest = fm.build_capability_manifest(studio_version="1.2.3")
    assert manifest.studio == "scpn-phase-orchestrator"
    assert manifest.studio_version == "1.2.3"
    assert manifest.transport_profile.value == "local-first"
    assert manifest.content_digest.startswith("sha256:")
    assert manifest.platform_sdk == ">=0.11,<0.12"
    assert {v.name for v in manifest.verbs} == _EXPECTED_VERBS
    assert manifest.evidence_types == LIVE_FEED_EVIDENCE_SCHEMAS
    assert {
        verb.name: verb.consumes for verb in manifest.verbs if verb.consumes
    } == _EXPECTED_CONSUMES


def test_manifest_resolves_its_internal_hard_functional_edges() -> None:
    """Every declared SPO input has a producer in the same schema-A manifest."""
    manifest = fm.build_capability_manifest(studio_version="1.2.3")
    resolution = resolve_pipeline((manifest,))

    assert resolution.admitted
    assert resolution.unresolved_upstreams == ()
    assert resolution.era_conflicts == ()
    edges = {
        (edge.upstream_verb, edge.downstream_verb, edge.wire_format)
        for edge in resolution.edges
    }
    assert ("bind", "simulate", "binding_spec") in edges
    assert ("simulate", "analyse", "upde_state") in edges
    assert ("analyse", "supervise", "coherence") in edges
    assert ("supervise", "project", "control_action_proposal") in edges
    assert ("audit", "replay", "audit_record") in edges
    assert ("replay", "assure", "replay_verdict") in edges


def test_manifest_declares_the_federated_ui_remote() -> None:
    manifest = fm.build_capability_manifest(studio_version="1.2.3")
    ui_module = manifest.ui_module
    assert ui_module is not None
    # The three fields are the wire contract with module-federation.config.ts.
    assert (
        ui_module.federation == fm.STUDIO_FEDERATION_NAME == "scpn_phase_orchestrator"
    )
    assert ui_module.exposes == (fm.STUDIO_EXPOSED_MODULE,) == ("./SpoStudioPanel",)
    assert ui_module.remote_entry == fm.STUDIO_REMOTE_ENTRY
    # Pull-deployed under the studio's own federation space, stable filename.
    assert ui_module.remote_entry.startswith(
        "https://www.anulum.org/studios/scpn-phase-orchestrator/"
    )
    assert ui_module.remote_entry.endswith("/remoteEntry.js")


def test_ui_module_matches_the_javascript_federation_config() -> None:
    """The Python ui_module contract equals the committed TS remote config."""
    config = (
        Path(__file__).resolve().parents[1]
        / "studio-web"
        / "module-federation.config.ts"
    ).read_text(encoding="utf-8")
    assert f'FEDERATION_NAME = "{fm.STUDIO_FEDERATION_NAME}"' in config
    assert f'PANEL_EXPOSE_KEY = "{fm.STUDIO_EXPOSED_MODULE}"' in config
    assert 'filename: "remoteEntry.js"' in config


def test_manifest_is_admitted_by_studio_platform_federation_gate() -> None:
    """The emitted schema-A wire manifest passes the Hub's platform validator."""
    assert _STUDIO_PLATFORM_SRC.is_dir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_STUDIO_PLATFORM_SRC)
    completed = subprocess.run(
        [sys.executable, "-c", _VALIDATE_WITH_SIBLING_PLATFORM],
        input=json.dumps(fm.manifest_dict(studio_version="1.2.3")),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert (
        completed.stdout.strip()
        == '{"admitted": true, "contract_era": "v1", "rejections": [], "warnings": []}'
    )


def test_digest_is_deterministic_and_version_independent() -> None:
    a = fm.build_capability_manifest(studio_version="1.0.0").content_digest
    b = fm.build_capability_manifest(studio_version="9.9.9").content_digest
    # The digest fingerprints the manifest surface, not the package version.
    assert a == b


def test_digest_tracks_live_feed_evidence_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evidence schema changes alter the manifest content digest."""
    baseline = fm.build_capability_manifest(studio_version="1.0.0").content_digest
    monkeypatch.setattr(
        fm,
        "LIVE_FEED_EVIDENCE_SCHEMAS",
        (*LIVE_FEED_EVIDENCE_SCHEMAS, "spo.extra-evidence.v1"),
    )
    changed = fm.build_capability_manifest(studio_version="1.0.0").content_digest
    assert changed != baseline


def test_digest_tracks_hard_functional_inputs() -> None:
    """A consumes-contract change alters the manifest content digest."""
    verbs = fm._verbs()
    baseline = fm._manifest_content_digest(verbs, LIVE_FEED_EVIDENCE_SCHEMAS)
    changed_verbs = tuple(
        replace(verb, consumes=(*verb.consumes, "external_state"))
        if verb.name == "analyse"
        else verb
        for verb in verbs
    )
    changed = fm._manifest_content_digest(
        changed_verbs,
        LIVE_FEED_EVIDENCE_SCHEMAS,
    )

    assert changed != baseline


def test_honesty_invariants_hold() -> None:
    manifest = fm.build_capability_manifest(studio_version="0.0.0")
    for verb in manifest.verbs:
        # SPO is review-only: nothing actuates live hardware.
        assert verb.side_effect is not SideEffect.LIVE_HARDWARE
        # SPO exports formal models but does not discharge them to a proof.
        assert verb.proof is None
        # SPO is a research-grade toolkit, not a certified/production product.
        assert verb.safety_tier is SafetyTier.RESEARCH


def test_manifest_dict_round_trips() -> None:
    payload = fm.manifest_dict(studio_version="0.0.0")
    assert payload["studio"] == "scpn-phase-orchestrator"
    assert {v["verb"] for v in payload["verbs"]} == _EXPECTED_VERBS
    assert payload["transport_profile"] == "local-first"
    verbs = {verb["verb"]: verb for verb in payload["verbs"]}
    assert verbs["simulate"]["consumes"] == ["binding_spec"]
    assert verbs["simulate"]["produces"] == [
        "upde_state",
        "order_parameter",
        "studio.runtime-state.v1",
    ]
    assert verbs["analyse"]["produces"][-1] == "studio.phase-coherence.v1"
    assert verbs["supervise"]["produces"][-1] == "studio.regime-state.v1"


def test_fingerprint_includes_a_formal_proof_method() -> None:
    # White-box: a future verb that carries a discharged proof must change its
    # fingerprint (covers the proof-present branch).
    proved = Verb(
        name="verify",
        safety_tier=SafetyTier.RESEARCH,
        side_effect=SideEffect.READ_ONLY,
        timing=Timing(timing_class=TimingClass.BATCH),
        proof=VerbProof(
            method=ProofMethod.BMC,
            engine="nuXmv",
            engine_version="2.0",
            depth=20,
            non_vacuity_checked=True,
        ),
    )
    fingerprint = fm._verb_fingerprint(proved)
    assert "bmc" in fingerprint
    assert "verify" in fingerprint


def test_build_requires_the_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fm, "_HAS_STUDIO_SDK", False)
    with pytest.raises(RuntimeError, match="STUDIO platform SDK is required"):
        fm.build_capability_manifest(studio_version="0.0.0")
