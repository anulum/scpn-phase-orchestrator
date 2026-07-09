# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed regime-adaptive ensemble evidence tests

"""Integrity tests for the committed cross-corpus regime-adaptive ensemble audit.

``examples/real_data/regime_adaptive_ensemble/`` is produced by
``bench/regime_adaptive_ensemble.py`` across the combined CAP + Sleep-EDF
manifest. These tests recompute every content seal and pin the study's honest
outcome: the channel-count regime router returns one of its two component scores
on every recording, and — because the coherence advantage is recording-specific
rather than a clean montage-size effect — it does **not** beat the plain delta
envelope, which remains the most robust single detector cross-corpus.
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
    / "regime_adaptive_ensemble"
)
_AGGREGATE_PATH = _EVIDENCE_DIR / "regime_adaptive_ensemble.json"

_RECORDINGS = ("n1", "n2", "brux2", "narco2", "SC4001E0")

_DETECTORS = (
    "normalized_delta_envelope",
    "multi_channel_delta_kuramoto",
    "snr_weighted_delta_kuramoto",
    "amplitude_gated_delta_kuramoto",
    "sustained_delta_kuramoto",
    "adaptive_channel_kuramoto",
    "coherent_sustained_kuramoto",
    "regime_adaptive_montage",
    "regime_adaptive_full",
)

_ROUTERS = ("regime_adaptive_montage", "regime_adaptive_full")


@pytest.fixture(scope="module")
def aggregate() -> dict[str, Any]:
    """Return the committed cross-corpus aggregate record."""
    return json.loads(_AGGREGATE_PATH.read_text(encoding="utf-8"))


def _load_audit(recording_id: str, detector_name: str) -> dict[str, Any]:
    """Load a sealed audit record from the committed artefacts."""
    path = _EVIDENCE_DIR / recording_id / f"{recording_id}_{detector_name}_audit.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _per_recording(aggregate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the per-recording fragments by recording id."""
    return {r["recording_id"]: r for r in aggregate["per_recording"]}


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


def test_aggregate_spans_cross_corpus_manifest(aggregate: dict[str, Any]) -> None:
    """The aggregate covers all five recordings and all nine detectors."""
    assert aggregate["benchmark"] == "regime_adaptive_ensemble"
    assert aggregate["n_recordings"] == 5
    assert set(aggregate["recording_ids"]) == set(_RECORDINGS)
    for detector in _DETECTORS:
        assert detector in aggregate


@pytest.mark.parametrize("recording_id", _RECORDINGS)
@pytest.mark.parametrize("router", _ROUTERS)
def test_router_returns_one_of_its_component_scores(
    aggregate: dict[str, Any],
    recording_id: str,
    router: str,
) -> None:
    """Each router returns exactly the envelope, coherent-sustained, or mean-R score."""
    rec = _per_recording(aggregate)[recording_id]["detectors"]
    router_dr = rec[router]["detection_rate"]
    component_drs = {
        rec["normalized_delta_envelope"]["detection_rate"],
        rec["coherent_sustained_kuramoto"]["detection_rate"],
        rec["multi_channel_delta_kuramoto"]["detection_rate"],
    }
    assert router_dr in component_drs


def test_envelope_is_the_most_robust_single_detector(
    aggregate: dict[str, Any],
) -> None:
    """The plain delta envelope has the best cross-corpus mean detection rate."""
    means = {
        d: aggregate[d]["mean_detection_rate"] for d in _DETECTORS if d not in _ROUTERS
    }
    assert max(means, key=lambda d: means[d]) == "normalized_delta_envelope"
    assert (
        aggregate["recommendation"]["preferred_variant"] == "normalized_delta_envelope"
    )


@pytest.mark.parametrize("router", _ROUTERS)
def test_channel_count_router_does_not_beat_the_envelope(
    aggregate: dict[str, Any],
    router: str,
) -> None:
    """The honest negative: the channel-count router does not beat the envelope.

    It sits between the coherence detector and the envelope — at least as good as
    coherent-sustained alone, but not better than the envelope — because the
    coherence advantage is recording-specific, not a clean montage-size effect.
    """
    router_mean = aggregate[router]["mean_detection_rate"]
    envelope_mean = aggregate["normalized_delta_envelope"]["mean_detection_rate"]
    coherent_mean = aggregate["coherent_sustained_kuramoto"]["mean_detection_rate"]
    assert router_mean <= envelope_mean
    assert router_mean >= coherent_mean


@pytest.mark.parametrize("detector_name", _DETECTORS)
def test_aggregate_per_recording_matches_sealed_audit(
    aggregate: dict[str, Any],
    detector_name: str,
) -> None:
    """Each per-recording summary in the aggregate matches its sealed audit."""
    per_rec = _per_recording(aggregate)
    for recording_id in _RECORDINGS:
        summary = per_rec[recording_id]["detectors"][detector_name]
        audit = _load_audit(recording_id, detector_name)
        assert summary["audit_content_hash"] == audit["content_hash"]
        assert summary["detection_rate"] == audit["audit"]["detection_rate"]
