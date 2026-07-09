# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed CAP Kuramoto-variant evidence tests

"""Integrity tests for the committed CAP Kuramoto-variant audit evidence.

``examples/real_data/cap_kuramoto_variants/`` is produced by
``bench/cap_kuramoto_variants.py`` from the same four-recording PhysioNet CAP
panel used by the multi-channel staging audit. These tests guard the committed
artefacts without the raw EDF/text files (citation-only, not redistributed):
they recompute each content seal, assert the sealed records are internally
consistent with the aggregate, and check that all seven detectors are audited on
every recording at the matched false-alarm operating point.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "real_data"
    / "cap_kuramoto_variants"
)
_AGGREGATE_PATH = _EVIDENCE_DIR / "cap_kuramoto_variants.json"

#: The four CAP panel recordings, in audit order.
_RECORDINGS = ("n1", "n2", "brux2", "narco2")

#: Every detector audited by ``bench/cap_kuramoto_variants.py``: three
#: established detectors plus the four new variants.
_DETECTORS = (
    "normalized_delta_envelope",
    "multi_channel_delta_kuramoto",
    "snr_weighted_delta_kuramoto",
    "amplitude_gated_delta_kuramoto",
    "sustained_delta_kuramoto",
    "adaptive_channel_kuramoto",
    "coherent_sustained_kuramoto",
)

#: The four detectors introduced by this study.
_NEW_VARIANTS = (
    "amplitude_gated_delta_kuramoto",
    "sustained_delta_kuramoto",
    "adaptive_channel_kuramoto",
    "coherent_sustained_kuramoto",
)


@pytest.fixture(scope="module")
def aggregate() -> dict[str, Any]:
    """Return the committed cross-subject aggregate comparison record."""
    return json.loads(_AGGREGATE_PATH.read_text(encoding="utf-8"))


def _load_audit(recording_id: str, detector_name: str) -> dict[str, Any]:
    """Load a sealed audit record from the committed artefacts."""
    path = _EVIDENCE_DIR / recording_id / f"{recording_id}_{detector_name}_audit.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_summary(recording_id: str, detector_name: str) -> dict[str, Any]:
    """Load a sealed audit summary from the committed artefacts."""
    path = _EVIDENCE_DIR / recording_id / f"{recording_id}_{detector_name}_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("recording_id", _RECORDINGS)
@pytest.mark.parametrize("detector_name", _DETECTORS)
def test_audit_content_seal_recomputes(
    recording_id: str,
    detector_name: str,
) -> None:
    """Every sealed audit record's ``content_hash`` matches its canonical payload."""
    audit = _load_audit(recording_id, detector_name)
    payload = copy.deepcopy(audit)
    sealed = payload.pop("content_hash")
    assert canonical_record_hash(payload) == sealed


@pytest.mark.parametrize("recording_id", _RECORDINGS)
@pytest.mark.parametrize("detector_name", _DETECTORS)
def test_audit_reports_matched_false_alarm_protocol(
    recording_id: str,
    detector_name: str,
) -> None:
    """Each sealed audit uses the documented matched-FA permutation protocol."""
    audit = _load_audit(recording_id, detector_name)["audit"]
    assert audit["detector_name"] == detector_name
    assert audit["n_events"] > 0
    assert audit["n_nulls"] > 0
    assert audit["target_false_alarm"] == pytest.approx(0.10, abs=1.0e-6)
    assert 0.0 <= audit["achieved_false_alarm"] <= 0.20
    assert 0.0 <= audit["detection_rate"] <= 1.0
    assert isinstance(audit["beats_chance"], bool)
    assert audit["significance"]["seed"] == 42
    assert audit["significance"]["n_permutations"] == 10_000


@pytest.mark.parametrize("recording_id", _RECORDINGS)
@pytest.mark.parametrize("detector_name", _DETECTORS)
def test_summary_matches_sealed_audit(
    recording_id: str,
    detector_name: str,
) -> None:
    """Each summary carries the same identifiers and verdict as its sealed record."""
    audit = _load_audit(recording_id, detector_name)
    summary = _load_summary(recording_id, detector_name)
    assert summary["audit_content_hash"] == audit["content_hash"]
    assert summary["corpus_id"] == audit["corpus_id"]
    assert summary["detection_rate"] == audit["audit"]["detection_rate"]
    assert summary["p_value"] == audit["audit"]["significance"]["p_value"]
    assert summary["beats_chance"] == audit["audit"]["beats_chance"]


def test_aggregate_spans_panel_and_detectors(aggregate: dict[str, Any]) -> None:
    """The aggregate covers the four-recording panel and all seven detectors."""
    assert aggregate["benchmark"] == "cap_kuramoto_variants"
    assert aggregate["corpus"] == "PhysioNet CAP Sleep Database"
    assert aggregate["target_false_alarm"] == pytest.approx(0.10, abs=1.0e-6)
    assert aggregate["n_recordings"] == 4
    assert aggregate["recording_ids"] == list(_RECORDINGS)
    for detector in _DETECTORS:
        stats = aggregate[detector]
        assert 0.0 <= stats["mean_detection_rate"] <= 1.0
        assert 0.0 <= stats["fraction_beats_chance"] <= 1.0


@pytest.mark.parametrize("recording_id", _RECORDINGS)
@pytest.mark.parametrize("detector_name", _DETECTORS)
def test_aggregate_per_recording_matches_sealed_audit(
    aggregate: dict[str, Any],
    recording_id: str,
    detector_name: str,
) -> None:
    """Each per-recording summary in the aggregate matches its sealed audit."""
    per_rec = {r["recording_id"]: r for r in aggregate["per_recording"]}
    summary = per_rec[recording_id]["detectors"][detector_name]
    audit = _load_audit(recording_id, detector_name)
    assert summary["audit_content_hash"] == audit["content_hash"]
    assert summary["detection_rate"] == audit["audit"]["detection_rate"]


def test_aggregate_records_a_data_driven_recommendation(
    aggregate: dict[str, Any],
) -> None:
    """The aggregate names a preferred variant chosen by mean detection rate."""
    rec = aggregate["recommendation"]
    assert rec["preferred_variant"] in _DETECTORS
    best = max(_DETECTORS, key=lambda d: aggregate[d]["mean_detection_rate"])
    assert rec["preferred_variant"] == best
    assert isinstance(rec["refine"], bool)
    assert len(rec["rationale"]) > 0


def test_new_variants_are_all_audited(aggregate: dict[str, Any]) -> None:
    """The four new variants are present on every recording in the aggregate."""
    for rec in aggregate["per_recording"]:
        for variant in _NEW_VARIANTS:
            assert variant in rec["detectors"]
