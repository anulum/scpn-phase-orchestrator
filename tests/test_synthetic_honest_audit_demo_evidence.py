# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — synthetic honest-audit harness demo evidence tests

"""Integrity tests for the synthetic honest-audit harness demo artefacts.

`examples/real_data/synthetic_honest_audit_demo/` is produced by
`bench/synthetic_honest_audit_demo.py` from a fully synthetic, deterministic
corpus. These tests guard the committed artefacts: they recompute the content
seals and assert the expected honest-audit result.
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
    / "synthetic_honest_audit_demo"
)

_RECORDINGS = ("run_a", "run_b", "run_c")
_DETECTORS = ("lag1_autocorrelation", "window_mean_control")

#: Content hashes of the committed sealed audit records.
_AUDIT_CONTENT_HASHES: dict[str, dict[str, str]] = {
    "run_a": {
        "lag1_autocorrelation": (
            "6a430c4bead215d13f4a4b3eed4eaebc093dcbcc72d41ad0c478cb7d57b9cd34"
        ),
        "window_mean_control": (
            "3ff8d537f21c7447b0b4aca180a50c31c0bc3678d1c5d04867c064632fab9752"
        ),
    },
    "run_b": {
        "lag1_autocorrelation": (
            "4f273f71afc2c442ca85ffe0d88a57bc03c7f6e9a227bff059a171725bdd8b15"
        ),
        "window_mean_control": (
            "014c54db8742e50ba30355fbafb683889b3181fb496be495626ed23c7d051ce9"
        ),
    },
    "run_c": {
        "lag1_autocorrelation": (
            "a675dc9068b89ad993b781872f974afa59e2e0d8f32edbfda9de5cc37e42698e"
        ),
        "window_mean_control": (
            "8c08ae0d4af7765d1e734978ac1363122b054b1ec269ffd2e3553c73cacbc670"
        ),
    },
}


@pytest.fixture(scope="module")
def aggregate() -> dict[str, Any]:
    """Return the committed aggregate comparison record."""
    path = _EVIDENCE_DIR / "synthetic_honest_audit_demo.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_audit(recording_id: str, detector_name: str) -> dict[str, Any]:
    """Load one sealed audit record."""
    path = _EVIDENCE_DIR / recording_id / f"{recording_id}_{detector_name}_audit.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("recording_id", _RECORDINGS)
@pytest.mark.parametrize("detector_name", _DETECTORS)
def test_audit_content_seal_recomputes(
    recording_id: str,
    detector_name: str,
) -> None:
    """Every sealed audit record's content hash matches its canonical payload."""
    audit = _load_audit(recording_id, detector_name)
    payload = copy.deepcopy(audit)
    sealed = payload.pop("content_hash")
    assert canonical_record_hash(payload) == sealed


@pytest.mark.parametrize("recording_id", _RECORDINGS)
@pytest.mark.parametrize("detector_name", _DETECTORS)
def test_audit_content_hash_matches_committed_value(
    recording_id: str,
    detector_name: str,
) -> None:
    """Committed sealed audit records have not drifted."""
    audit = _load_audit(recording_id, detector_name)
    assert audit["content_hash"] == _AUDIT_CONTENT_HASHES[recording_id][detector_name]


def test_aggregate_spans_three_recordings(aggregate: dict[str, Any]) -> None:
    """The aggregate spans the three synthetic recordings."""
    assert aggregate["benchmark"] == "synthetic_honest_audit_demo"
    assert aggregate["corpus"] == "Synthetic AR(1) critical-slowing-down corpus"
    assert aggregate["n_recordings"] == 3
    assert aggregate["recording_ids"] == list(_RECORDINGS)
    assert aggregate["target_false_alarm"] == pytest.approx(0.10, abs=1.0e-6)


def test_aggregate_cross_subject_stats(aggregate: dict[str, Any]) -> None:
    """The aggregate reports the expected per-detector statistics."""
    auto = aggregate["lag1_autocorrelation"]
    mean = aggregate["window_mean_control"]
    assert auto["mean_detection_rate"] == pytest.approx(1.0, abs=1.0e-6)
    assert auto["mean_achieved_false_alarm"] == pytest.approx(0.10, abs=1.0e-6)
    assert auto["fraction_beats_chance"] == pytest.approx(1.0, abs=1.0e-6)
    assert mean["mean_detection_rate"] == pytest.approx(0.0, abs=1.0e-6)
    assert mean["mean_achieved_false_alarm"] == pytest.approx(0.0, abs=1.0e-6)
    assert mean["fraction_beats_chance"] == pytest.approx(0.0, abs=1.0e-6)


def test_aggregate_recommendation(aggregate: dict[str, Any]) -> None:
    """The data-driven recommendation prefers the skilful detector."""
    rec = aggregate["recommendation"]
    assert rec["preferred_variant"] == "lag1_autocorrelation"
    assert rec["refine"] is True


@pytest.mark.parametrize("recording_id", _RECORDINGS)
def test_autocorrelation_detector_beats_chance(
    recording_id: str,
) -> None:
    """The autocorrelation detector beats chance on every recording."""
    audit = _load_audit(recording_id, "lag1_autocorrelation")
    assert audit["audit"]["beats_chance"] is True
    assert audit["audit"]["detection_rate"] == pytest.approx(1.0, abs=1.0e-6)


@pytest.mark.parametrize("recording_id", _RECORDINGS)
def test_window_mean_control_does_not_beat_chance(
    recording_id: str,
) -> None:
    """The window-mean control does not beat chance on any recording."""
    audit = _load_audit(recording_id, "window_mean_control")
    assert audit["audit"]["beats_chance"] is False
    assert audit["audit"]["detection_rate"] == pytest.approx(0.0, abs=1.0e-6)
