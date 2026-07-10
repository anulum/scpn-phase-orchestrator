# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — architecture Studio manifest tests

"""Regression tests for the public architecture STUDIO manifest map."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import scpn_phase_orchestrator.studio.federation_manifest as federation_manifest

pytest.importorskip("scpn_studio_platform")

ARCHITECTURE_MANIFEST = Path("docs/architecture/manifest.json")


def _architecture_payload() -> dict[str, object]:
    """Return the public architecture manifest JSON payload."""
    return cast(
        "dict[str, object]",
        json.loads(ARCHITECTURE_MANIFEST.read_text(encoding="utf-8")),
    )


def _schema_a_core() -> dict[str, object]:
    """Return the public schema-A core block from the architecture manifest."""
    return cast("dict[str, object]", _architecture_payload()["schema_a_core"])


def test_architecture_schema_a_tracks_studio_federation_manifest() -> None:
    """The public architecture manifest mirrors the production Studio manifest."""
    core = _schema_a_core()
    payload = federation_manifest.manifest_dict(studio_version="0.0.0")

    assert core["status"] == "fleet-validator-admitted"
    assert core["project"] == payload["studio"]
    assert core["platform_sdk"] == payload["platform_sdk"]
    assert core["protocol_version"] == payload["protocol_version"]
    assert core["transport_profile"] == payload["transport_profile"]
    assert core["evidence_types"] == payload["evidence_types"]
    # The architecture manifest mirrors the serialised schema-A ui_module block.
    assert core["ui_module"] == payload["ui_module"]
    assert core["contract_era"] == payload["contract_era"]
    assert core["enumeration"] == payload["enumeration"]
    assert core["content_digest"] == payload["content_digest"]
    assert core["verbs"] == [verb["verb"] for verb in payload["verbs"]]


def test_architecture_map_names_live_studio_feed_boundary() -> None:
    """The architecture map keeps the live feed and platform admission boundary."""
    payload = _architecture_payload()
    architecture_map = cast("dict[str, object]", payload["architecture_map"])
    lanes = cast("list[dict[str, object]]", architecture_map["lanes"])
    studio_lane = next(
        lane for lane in lanes if lane["name"] == "studio+reporting+visualization"
    )
    boundaries = cast("list[str]", architecture_map["boundaries"])

    assert "studio.control-feed.v1 JSON" in cast("list[str]", studio_lane["outputs"])
    assert "spo.studio-runtime-snapshot.v1" in cast("str", studio_lane["note"])
    assert any(
        "admitted by the current STUDIO Platform federation gate" in boundary
        for boundary in boundaries
    )
