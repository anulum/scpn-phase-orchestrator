# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio live feed tests

"""Tests for the SPO ``studio.control-feed.v1`` live feed."""

from __future__ import annotations

import json

import pytest

from scpn_phase_orchestrator.studio import build_studio_control_feed
from scpn_phase_orchestrator.studio.live_feed import (
    FEED_SCHEMA,
    RUNTIME_SCHEMA,
    STUDIO_ID,
    render_studio_control_feed_json,
    runtime_summary,
)


def _snapshot() -> dict[str, object]:
    """Return a representative runtime server snapshot."""
    return {
        "step": 3,
        "R_global": 0.75,
        "regime": "sync",
        "layers": [
            {"name": "p", "R": 0.7, "psi": 0.1},
            {"name": "i", "R": 0.8, "psi": 0.2},
        ],
        "n_oscillators": 4,
        "amplitude_mode": False,
    }


def test_build_studio_control_feed_uses_control_feed_envelope() -> None:
    """The live feed mirrors the shared Studio control-feed envelope."""
    feed = build_studio_control_feed(_snapshot(), studio_version="1.2.3")

    assert feed["feed_schema"] == FEED_SCHEMA
    assert feed["studio"] == STUDIO_ID
    assert feed["studio_version"] == "1.2.3"
    assert str(feed["content_digest"]).startswith("sha256:")
    assert {verb["name"] for verb in feed["verbs"]} >= {"simulate", "replay"}
    assert {claim["schema"] for claim in feed["claims"]} == {
        "spo.runtime-state.v1",
        "spo.phase-coherence.v1",
        "spo.regime-state.v1",
    }
    assert feed["runtime"] == {
        "schema": RUNTIME_SCHEMA,
        "step": 3,
        "r_global": 0.75,
        "regime": "sync",
        "n_oscillators": 4,
        "amplitude_mode": False,
        "layers": [
            {"name": "p", "r": 0.7, "psi": 0.1},
            {"name": "i", "r": 0.8, "psi": 0.2},
        ],
    }


def test_render_studio_control_feed_json_is_deterministic() -> None:
    """The JSON renderer produces stable sorted output."""
    rendered = render_studio_control_feed_json(_snapshot(), studio_version="1.2.3")
    parsed = json.loads(rendered)

    assert rendered.endswith("\n")
    assert parsed == build_studio_control_feed(_snapshot(), studio_version="1.2.3")


def test_runtime_summary_accepts_amplitude_mode() -> None:
    """Amplitude-mode snapshots carry a finite mean-amplitude field."""
    snapshot = {**_snapshot(), "amplitude_mode": True, "mean_amplitude": 1.25}

    runtime = runtime_summary(snapshot)

    assert runtime["amplitude_mode"] is True
    assert runtime["mean_amplitude"] == 1.25


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("R_global", 1.5, "R_global must be in"),
        ("R_global", True, "R_global must be a finite real"),
        ("step", -1, "step must be a non-negative integer"),
        ("regime", "", "regime must be a non-empty string"),
        ("layers", [], "layers must not be empty"),
    ],
)
def test_runtime_summary_rejects_invalid_snapshot_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    """Malformed live snapshots fail before reaching Studio."""
    snapshot = {**_snapshot(), field: value}

    with pytest.raises(ValueError, match=match):
        runtime_summary(snapshot)
