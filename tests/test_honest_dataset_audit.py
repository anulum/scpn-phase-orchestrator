# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — honest dataset audit harness tests

"""Tests for the reusable honest-audit harness.

The harness is exercised with tiny synthetic datasets so the tests run in
milliseconds. It must still produce sealed audit records whose content hashes
recompute correctly and whose aggregate statistics match the expected values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bench.honest_dataset_audit import (
    AuditConfig,
    RecordingSpec,
    compute_aggregate,
    default_recommendation,
    file_sha256,
    run_audit,
    run_honest_audit,
    write_aggregate,
    write_audit_files,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash


@pytest.fixture
def tiny_tmp_path(tmp_path: Path) -> Path:
    """Return a temporary directory for this test."""
    return tmp_path


def test_file_sha256_matches_known_digest(tiny_tmp_path: Path) -> None:
    """file_sha256 agrees with a straightforward hashlib computation."""
    path = tiny_tmp_path / "hello.txt"
    payload = b"hello honest audit"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert file_sha256(path) == expected


def test_run_audit_produces_consistent_seal() -> None:
    """run_audit returns a sealed record whose content hash recomputes."""
    rng = np.random.default_rng(42)
    scores = np.round(rng.random(80), 6)
    labels = ["event"] * 20 + ["null"] * 60
    config = AuditConfig(n_permutations=1_000, seed=7)

    audit_record, summary = run_audit(
        scores=scores,
        labels=labels,
        detector_name="synthetic",
        corpus_id="synthetic-corpus",
        config=config,
        event_label="event",
        null_label="null",
    )

    payload = dict(audit_record)
    sealed = payload.pop("content_hash")
    assert canonical_record_hash(payload) == sealed
    assert summary["detector_name"] == "synthetic"
    assert summary["n_events"] == 20
    assert summary["n_nulls"] == 60
    assert summary["corpus_id"] == "synthetic-corpus"
    assert summary["audit_content_hash"] == sealed
    assert summary["permutation_seed"] == 7
    assert summary["n_permutations"] == 1_000


def test_run_audit_requires_both_classes() -> None:
    """run_audit raises when one of the event/null classes is missing."""
    scores = np.zeros(10)
    config = AuditConfig()
    with pytest.raises(ValueError, match="no event epochs"):
        run_audit(scores, ["null"] * 10, "d", "c", config)
    with pytest.raises(ValueError, match="no null epochs"):
        run_audit(scores, ["event"] * 10, "d", "c", config)


def test_write_audit_files_creates_expected_outputs(tiny_tmp_path: Path) -> None:
    """write_audit_files writes audit + summary JSON with the correct names."""
    rng = np.random.default_rng(1)
    scores = np.round(rng.random(40), 6)
    labels = ["event"] * 10 + ["null"] * 30
    config = AuditConfig(n_permutations=500, seed=1)
    audit_record, summary = run_audit(
        scores, labels, "foo", "c", config, event_label="event", null_label="null"
    )

    write_audit_files(
        output_dir=tiny_tmp_path,
        prefix="rec1",
        detector_name="foo",
        audit_record=audit_record,
        summary=summary,
    )

    audit_path = tiny_tmp_path / "rec1_foo_audit.json"
    summary_path = tiny_tmp_path / "rec1_foo_summary.json"
    assert audit_path.exists()
    assert summary_path.exists()
    loaded_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded_audit["content_hash"] == audit_record["content_hash"]
    assert loaded_summary["audit_content_hash"] == summary["audit_content_hash"]


def test_compute_aggregate_and_default_recommendation() -> None:
    """compute_aggregate produces per-detector stats and a sensible recommendation."""
    records = [
        {
            "recording_id": "a",
            "detectors": {
                "det1": {
                    "detection_rate": 0.8,
                    "achieved_false_alarm": 0.09,
                    "p_value": 0.001,
                    "beats_chance": True,
                    "target_false_alarm": 0.10,
                },
                "det2": {
                    "detection_rate": 0.5,
                    "achieved_false_alarm": 0.11,
                    "p_value": 0.05,
                    "beats_chance": False,
                    "target_false_alarm": 0.10,
                },
            },
        },
        {
            "recording_id": "b",
            "detectors": {
                "det1": {
                    "detection_rate": 0.6,
                    "achieved_false_alarm": 0.10,
                    "p_value": 0.01,
                    "beats_chance": True,
                    "target_false_alarm": 0.10,
                },
                "det2": {
                    "detection_rate": 0.4,
                    "achieved_false_alarm": 0.10,
                    "p_value": 0.10,
                    "beats_chance": False,
                    "target_false_alarm": 0.10,
                },
            },
        },
    ]

    aggregate = compute_aggregate(
        records=records,
        benchmark="synthetic",
        corpus="synthetic corpus",
        detector_names=["det1", "det2"],
    )

    assert aggregate["benchmark"] == "synthetic"
    assert aggregate["corpus"] == "synthetic corpus"
    assert aggregate["target_false_alarm"] == pytest.approx(0.10, abs=1.0e-6)
    assert aggregate["n_recordings"] == 2
    assert aggregate["recording_ids"] == ["a", "b"]
    assert aggregate["det1"]["mean_detection_rate"] == pytest.approx(0.7, abs=1.0e-6)
    assert aggregate["det2"]["mean_detection_rate"] == pytest.approx(0.45, abs=1.0e-6)
    assert aggregate["recommendation"]["preferred_variant"] == "det1"


def test_default_recommendation_stops_when_no_detector_beats_chance() -> None:
    """If the best detector never beats chance, the recommendation is to stop."""
    records = [
        {
            "recording_id": "a",
            "detectors": {
                "det1": {
                    "detection_rate": 0.1,
                    "achieved_false_alarm": 0.10,
                    "p_value": 0.5,
                    "beats_chance": False,
                    "target_false_alarm": 0.10,
                },
            },
        }
    ]
    stats_by_detector = {
        "det1": {
            "mean_detection_rate": 0.1,
            "std_detection_rate": 0.0,
            "mean_achieved_false_alarm": 0.10,
            "geometric_mean_p_value": 0.5,
            "fraction_beats_chance": 0.0,
        }
    }
    rec = default_recommendation(records, stats_by_detector)
    assert rec["refine"] is False
    assert rec["preferred_variant"] == "det1"


def test_run_honest_audit_end_to_end(tiny_tmp_path: Path) -> None:
    """run_honest_audit produces sealed audits and an aggregate for a toy dataset."""

    def loader(spec: RecordingSpec) -> dict[str, Any]:
        return {"n_epochs": 40}

    def label_extractor(obj: dict[str, Any]) -> list[str]:
        return ["event"] * 10 + ["null"] * 30

    def good_detector(obj: dict[str, Any]) -> np.ndarray:
        rng = np.random.default_rng(99)
        return np.round(rng.random(40) + np.concatenate([np.ones(10), np.zeros(30)]), 6)

    def bad_detector(obj: dict[str, Any]) -> np.ndarray:
        rng = np.random.default_rng(100)
        return np.round(rng.random(40), 6)

    manifest = [
        RecordingSpec("rec1", {"txt": tiny_tmp_path / "rec1.txt"}),
        RecordingSpec("rec2", {"txt": tiny_tmp_path / "rec2.txt"}),
    ]
    for spec in manifest:
        spec.paths["txt"].write_text("dummy", encoding="utf-8")

    config = AuditConfig(n_permutations=500, seed=2)
    aggregate = run_honest_audit(
        manifest=manifest,
        loader=loader,
        detectors={"good": good_detector, "bad": bad_detector},
        label_extractor=label_extractor,
        output_dir=tiny_tmp_path / "out",
        config=config,
        benchmark="toy",
        corpus="toy corpus",
        event_label="event",
        null_label="null",
    )

    assert aggregate["benchmark"] == "toy"
    assert aggregate["corpus"] == "toy corpus"
    assert aggregate["n_recordings"] == 2
    assert aggregate["recommendation"]["preferred_variant"] == "good"

    out = tiny_tmp_path / "out"
    aggregate_path = out / "toy.json"
    assert aggregate_path.exists()
    for rec in manifest:
        rec_dir = out / rec.recording_id
        assert rec_dir.exists()
        assert (rec_dir / f"{rec.recording_id}_good_audit.json").exists()
        assert (rec_dir / f"{rec.recording_id}_bad_audit.json").exists()


def test_write_aggregate_uses_benchmark_as_filename(tiny_tmp_path: Path) -> None:
    """write_aggregate writes ``{benchmark}.json`` under the output directory."""
    aggregate = {"benchmark": "my-bench", "n_recordings": 0, "per_recording": []}
    write_aggregate(tiny_tmp_path, aggregate)
    path = tiny_tmp_path / "my-bench.json"
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["benchmark"] == "my-bench"
