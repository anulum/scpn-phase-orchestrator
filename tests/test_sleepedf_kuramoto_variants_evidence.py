# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed Sleep-EDF Kuramoto-variant transfer evidence

"""Integrity tests for the committed Sleep-EDF Kuramoto-variant transfer audit.

``examples/real_data/sleepedf_kuramoto_variants/`` is produced by
``bench/sleepedf_kuramoto_variants.py`` from PhysioNet Sleep-EDF recording
``SC4001E0`` using its two EEG derivations. These tests guard the committed
artefacts without the raw EDF files (citation-only): they recompute each content
seal and pin the study's central transfer finding — pure phase-coherence
detectors carry no signal on a two-channel montage, while the amplitude-gated
variants beat chance.
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
    / "sleepedf_kuramoto_variants"
)
_AGGREGATE_PATH = _EVIDENCE_DIR / "sleepedf_kuramoto_variants.json"
_RECORDING = "SC4001E0"

_DETECTORS = (
    "normalized_delta_envelope",
    "multi_channel_delta_kuramoto",
    "snr_weighted_delta_kuramoto",
    "amplitude_gated_delta_kuramoto",
    "sustained_delta_kuramoto",
    "adaptive_channel_kuramoto",
    "coherent_sustained_kuramoto",
)

#: Detectors that beat chance on the two-channel montage (the amplitude-gated
#: family plus the delta envelope baseline).
_BEATS_CHANCE = (
    "normalized_delta_envelope",
    "amplitude_gated_delta_kuramoto",
    "coherent_sustained_kuramoto",
)

#: Pure-coherence detectors that carry no N3-vs-Wake signal on two channels.
_DEAD_ON_TWO_CHANNELS = (
    "multi_channel_delta_kuramoto",
    "snr_weighted_delta_kuramoto",
    "sustained_delta_kuramoto",
    "adaptive_channel_kuramoto",
)


@pytest.fixture(scope="module")
def aggregate() -> dict[str, Any]:
    """Return the committed Sleep-EDF transfer aggregate record."""
    return json.loads(_AGGREGATE_PATH.read_text(encoding="utf-8"))


def _load_audit(detector_name: str) -> dict[str, Any]:
    """Load a sealed audit record from the committed artefacts."""
    path = _EVIDENCE_DIR / _RECORDING / f"{_RECORDING}_{detector_name}_audit.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("detector_name", _DETECTORS)
def test_audit_content_seal_recomputes(detector_name: str) -> None:
    """Every sealed audit record's ``content_hash`` matches its canonical payload."""
    audit = _load_audit(detector_name)
    payload = copy.deepcopy(audit)
    sealed = payload.pop("content_hash")
    assert canonical_record_hash(payload) == sealed


@pytest.mark.parametrize("detector_name", _DETECTORS)
def test_audit_uses_matched_false_alarm_protocol(detector_name: str) -> None:
    """Each sealed audit uses the documented matched-FA permutation protocol."""
    audit = _load_audit(detector_name)["audit"]
    assert audit["detector_name"] == detector_name
    assert audit["n_events"] == 220
    assert audit["n_nulls"] == 1997
    assert audit["target_false_alarm"] == pytest.approx(0.10, abs=1.0e-6)
    assert audit["significance"]["seed"] == 42
    assert audit["significance"]["n_permutations"] == 10_000


def test_aggregate_spans_single_recording_and_detectors(
    aggregate: dict[str, Any],
) -> None:
    """The aggregate covers the single Sleep-EDF recording and all seven detectors."""
    assert aggregate["benchmark"] == "sleepedf_kuramoto_variants"
    assert aggregate["corpus"] == "PhysioNet Sleep-EDF Expanded"
    assert aggregate["n_recordings"] == 1
    assert aggregate["recording_ids"] == [_RECORDING]
    for detector in _DETECTORS:
        assert detector in aggregate


def test_envelope_dominates_the_sparse_montage(aggregate: dict[str, Any]) -> None:
    """The delta envelope is near-perfect on the two-channel Sleep-EDF montage."""
    assert aggregate["normalized_delta_envelope"]["mean_detection_rate"] > 0.9


@pytest.mark.parametrize("detector_name", _DEAD_ON_TWO_CHANNELS)
def test_pure_coherence_detectors_carry_no_signal(
    aggregate: dict[str, Any],
    detector_name: str,
) -> None:
    """Pure-coherence variants do not beat chance on two channels (the finding)."""
    audit = _load_audit(detector_name)["audit"]
    assert audit["beats_chance"] is False
    assert audit["detection_rate"] == pytest.approx(0.0, abs=1.0e-6)
    assert aggregate[detector_name]["fraction_beats_chance"] == pytest.approx(
        0.0, abs=1.0e-6
    )


@pytest.mark.parametrize("detector_name", _BEATS_CHANCE)
def test_amplitude_gated_family_transfers(
    aggregate: dict[str, Any],
    detector_name: str,
) -> None:
    """The amplitude-gated variants (and the envelope) beat chance (the finding)."""
    audit = _load_audit(detector_name)["audit"]
    assert audit["beats_chance"] is True
    assert audit["significance"]["p_value"] < 0.05
    assert aggregate[detector_name]["fraction_beats_chance"] == pytest.approx(
        1.0, abs=1.0e-6
    )


@pytest.mark.parametrize("detector_name", _DETECTORS)
def test_aggregate_per_recording_matches_sealed_audit(
    aggregate: dict[str, Any],
    detector_name: str,
) -> None:
    """Each per-recording summary in the aggregate matches its sealed audit."""
    summary = aggregate["per_recording"][0]["detectors"][detector_name]
    audit = _load_audit(detector_name)
    assert summary["audit_content_hash"] == audit["content_hash"]
    assert summary["detection_rate"] == audit["audit"]["detection_rate"]
    assert summary["beats_chance"] == audit["audit"]["beats_chance"]
