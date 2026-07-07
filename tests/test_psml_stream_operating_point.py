# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed grid streaming operating-point evidence tests

"""Integrity tests for the committed grid streaming operating-point evidence.

The committed `grid_modal_stream_operating_point.json` measures how much of the offline
per-window grid modal-growth skill survives a live-stream deployment at a matched stream
false alarm, on the same PSML corpus. These tests guard the committed
derived artefact without the raw PSML data: they recompute its content hash from the
committed payload alone (no raw re-run, so no cross-platform float drift), pin the
digest, and assert the honest reality — streaming skill is far below the per-window
figure, the naive per-window threshold is useless on a stream, and the fit-quality gate
is the operating point that holds the target false alarm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

_ARTEFACT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "real_data"
    / "psml_modal_growth"
    / "grid_modal_stream_operating_point.json"
)

#: Recomputing the hash from the committed payload alone proves it was not hand-edited.
_PINNED_HASH = "3e2d74b7970d9f4aa78cda431f90d0bb4c6dc9cadd9366e45c79474f2d31f8f2"


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Return the committed sealed streaming operating-point record."""
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def test_content_hash_recomputes(payload: dict[str, Any]) -> None:
    body = {key: value for key, value in payload.items() if key != "content_hash"}
    assert payload["content_hash"] == canonical_record_hash(body)


def test_content_hash_is_pinned(payload: dict[str, Any]) -> None:
    assert payload["content_hash"] == _PINNED_HASH


def test_offline_reference_is_the_per_window_flagship(payload: dict[str, Any]) -> None:
    offline = payload["offline_per_window"]
    assert offline["led"] == 36
    assert offline["n_transitions"] == 90


def test_naive_streaming_is_useless(payload: dict[str, Any]) -> None:
    # at the per-window threshold the stream alarms on almost everything, incl. a large
    # fraction of damped scenarios — an operationally useless false-alarm rate
    naive = payload["naive_stream_at_per_window_threshold"]
    assert naive["led"] >= 80  # leads almost all transitions ...
    assert naive["stream_false_alarm"] > 0.6  # ... but false-alarms on most damped


def test_streaming_skill_is_below_per_window_and_gated(payload: dict[str, Any]) -> None:
    # the honest streaming operating point (fit-quality gate) leads far fewer than the
    # offline per-window detector, but holds the target false alarm
    assert "r2gate" in payload["verdict"]
    assert "24%" in payload["verdict"]
    assert "40%" in payload["verdict"]
    features = {row["feature"] for row in payload["search"]}
    assert features == {"focal", "r2gate"}
    # the fit-quality gate (window 2 s, persistence 2) holds the held-out false alarm at
    # the target where the plain focal rate drifts above it
    by_key = {
        (row["feature"], row["window_seconds"], row["persistence"]): row
        for row in payload["search"]
    }
    gated = by_key[("r2gate", 2.0, 2)]
    focal = by_key[("focal", 2.0, 2)]
    assert gated["held_out_false_alarm"] <= payload["target_stream_false_alarm"] * 1.2
    assert focal["held_out_false_alarm"] > gated["held_out_false_alarm"]
