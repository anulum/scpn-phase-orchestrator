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
from pathlib import Path

import pytest

import scpn_phase_orchestrator.studio.federation_manifest as fm
from scpn_phase_orchestrator.studio.live_feed import LIVE_FEED_EVIDENCE_SCHEMAS

pytest.importorskip("scpn_studio_platform")

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


def test_manifest_is_well_formed() -> None:
    manifest = fm.build_capability_manifest(studio_version="1.2.3")
    assert manifest.studio == "scpn-phase-orchestrator"
    assert manifest.studio_version == "1.2.3"
    assert manifest.transport_profile.value == "local-first"
    assert manifest.content_digest.startswith("sha256:")
    assert {v.name for v in manifest.verbs} == _EXPECTED_VERBS
    assert manifest.evidence_types == LIVE_FEED_EVIDENCE_SCHEMAS


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
