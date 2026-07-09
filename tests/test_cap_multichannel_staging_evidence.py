# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed CAP multi-channel staging evidence tests

"""Integrity tests for the committed CAP multi-channel N3-vs-Wake evidence.

`examples/real_data/cap_multichannel_staging/` is produced by
`bench/cap_multichannel_n3_vs_wake.py` from a public PhysioNet CAP Sleep
Database recording. These tests guard the committed artefacts without the raw
EDF/text files (which are citation-only and not redistributed): they recompute
the content seals, pin the source-file digests, and assert the documented
honest-audit results for all three detectors.
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
    / "cap_multichannel_staging"
)

_N1_DIR = _EVIDENCE_DIR / "n1"
_N2_DIR = _EVIDENCE_DIR / "n2"
_BRUX2_DIR = _EVIDENCE_DIR / "brux2"
_NARCO2_DIR = _EVIDENCE_DIR / "narco2"

_ENVELOPE_AUDIT_PATH = _N1_DIR / "cap_n1_delta_envelope_audit.json"
_ENVELOPE_SUMMARY_PATH = _N1_DIR / "cap_n1_delta_envelope_summary.json"
_KURAMOTO_AUDIT_PATH = _N1_DIR / "cap_n1_multichannel_kuramoto_audit.json"
_KURAMOTO_SUMMARY_PATH = _N1_DIR / "cap_n1_multichannel_kuramoto_summary.json"
_SNR_KURAMOTO_AUDIT_PATH = _N1_DIR / "cap_n1_snr_weighted_kuramoto_audit.json"
_SNR_KURAMOTO_SUMMARY_PATH = _N1_DIR / "cap_n1_snr_weighted_kuramoto_summary.json"
_COMPARISON_PATH = _N1_DIR / "cap_n1_detector_comparison.json"
_AGGREGATE_PATH = _EVIDENCE_DIR / "cap_multichannel_aggregate.json"
_DIAGNOSTIC_PATH = _EVIDENCE_DIR / "cap_kuramoto_diagnostic.json"

#: SHA-256 of the raw CAP EDF used to generate the committed evidence.
_EDF_SHA256 = "25d4e62a7a90d8245c71c7f168e34756238bb6ac7b1d43dcc75818d0331d43f4"
#: SHA-256 of the raw REMlogic text annotation file used to generate the evidence.
_TXT_SHA256 = "913ffd33166376f7eb46c946377e5b4017ad38809b28757763cb5bfd71004143"
#: Content hashes of the committed sealed audit records.
_ENVELOPE_CONTENT_HASH = (
    "ce4ecc10453673da472bcdc8ff508d40dcd371f8cfc03e8146764a4c58d0931a"
)
_KURAMOTO_CONTENT_HASH = (
    "52db1dcb9c5f46523deefaf47a11c0a4f848c6d22fed7a245c848171e6634352"
)
_SNR_KURAMOTO_CONTENT_HASH = (
    "c5da64619849a85cb68435769efa80757918e46d1527fe8bdf4bd4eb9948deab"
)

#: Cross-subject aggregate recording panel.
_CROSS_SUBJECT_RECORDINGS = ("n1", "n2", "brux2", "narco2")

#: Source-file digests for the citation-only raw CAP recordings.
_SOURCE_DIGESTS: dict[str, dict[str, str]] = {
    "n1": {
        "edf_sha256": (
            "25d4e62a7a90d8245c71c7f168e34756238bb6ac7b1d43dcc75818d0331d43f4"
        ),
        "txt_sha256": (
            "913ffd33166376f7eb46c946377e5b4017ad38809b28757763cb5bfd71004143"
        ),
    },
    "n2": {
        "edf_sha256": (
            "2bea9302fb7a50a10558cddcb9a6666da55799e051b90fc5f4cd661943b68782"
        ),
        "txt_sha256": (
            "933b6aeedec614036af82e168e8bb314b8592b30315d31c004f5d058f3ae16a7"
        ),
    },
    "brux2": {
        "edf_sha256": (
            "141487dad3975c0c24a693e6c19ba4809558750bb880299bb21f12a985286646"
        ),
        "txt_sha256": (
            "db4cf667cc2680e86e688cc154c645acdc209240162b68323b61e9a13ca0b73d"
        ),
    },
    "narco2": {
        "edf_sha256": (
            "a5b964a42ade6b45434fd99d598ad6515f4f066e93e7a8ffa0054e19d6f0461e"
        ),
        "txt_sha256": (
            "08e2ccc3842447452dc39b814cfa87e4c0446df2661f8503ae12d1fc4bd0f606"
        ),
    },
}

#: Committed content hashes of the sealed audit records.
_AUDIT_CONTENT_HASHES: dict[str, dict[str, str]] = {
    "n1": {
        "normalized_delta_envelope": (
            "ce4ecc10453673da472bcdc8ff508d40dcd371f8cfc03e8146764a4c58d0931a"
        ),
        "multi_channel_delta_kuramoto": (
            "52db1dcb9c5f46523deefaf47a11c0a4f848c6d22fed7a245c848171e6634352"
        ),
        "snr_weighted_delta_kuramoto": (
            "c5da64619849a85cb68435769efa80757918e46d1527fe8bdf4bd4eb9948deab"
        ),
    },
    "n2": {
        "normalized_delta_envelope": (
            "e8407f52542ed2203a9cb318bbf2d7642a206c53e06178ced92225dbd8d2eedd"
        ),
        "multi_channel_delta_kuramoto": (
            "76a0c3fc9e489baac8da7afee2d52d3a8d2c55a040ae2fdbb770d44c3b71f1fc"
        ),
        "snr_weighted_delta_kuramoto": (
            "b9bac09b97f153810e56fcf11c6b4f469f096ccff427d9bf20a38928767b91ab"
        ),
    },
    "brux2": {
        "normalized_delta_envelope": (
            "7b75e93aa95912a851bbe1c23556701b15821f2a69052e34a5720d841cfdf2b3"
        ),
        "multi_channel_delta_kuramoto": (
            "eed2d7fda5b1f19d01e4b7a04a57d72c10c0b2a5f5e0de12655a323230aa9e8f"
        ),
        "snr_weighted_delta_kuramoto": (
            "8d396df69c914a5138e47c9e50f6810b5766098ada5dc5cffa82e7d423002d5b"
        ),
    },
    "narco2": {
        "normalized_delta_envelope": (
            "55b7ef52e5a1413f7f0168520917883eb680603ca8650cecfcdc6286ee6a8ec3"
        ),
        "multi_channel_delta_kuramoto": (
            "41f95df6c01f72724abe9a241157534ab3206fd96b89d71e38024447782beb33"
        ),
        "snr_weighted_delta_kuramoto": (
            "8a579b39ebee1ad3319443ed765c8fe4bf504691fcf7508a8687802e6e0bc62e"
        ),
    },
}


@pytest.fixture(scope="module")
def comparison() -> dict[str, Any]:
    """Return the committed detector comparison record."""
    return json.loads(_COMPARISON_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def envelope_audit() -> dict[str, Any]:
    """Return the committed delta-envelope sealed audit record."""
    return json.loads(_ENVELOPE_AUDIT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def envelope_summary() -> dict[str, Any]:
    """Return the committed delta-envelope summary record."""
    return json.loads(_ENVELOPE_SUMMARY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kuramoto_audit() -> dict[str, Any]:
    """Return the committed multi-channel Kuramoto sealed audit record."""
    return json.loads(_KURAMOTO_AUDIT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kuramoto_summary() -> dict[str, Any]:
    """Return the committed multi-channel Kuramoto summary record."""
    return json.loads(_KURAMOTO_SUMMARY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def snr_kuramoto_audit() -> dict[str, Any]:
    """Return the committed SNR-weighted Kuramoto sealed audit record."""
    return json.loads(_SNR_KURAMOTO_AUDIT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def snr_kuramoto_summary() -> dict[str, Any]:
    """Return the committed SNR-weighted Kuramoto summary record."""
    return json.loads(_SNR_KURAMOTO_SUMMARY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def aggregate() -> dict[str, Any]:
    """Return the committed cross-subject aggregate comparison record."""
    return json.loads(_AGGREGATE_PATH.read_text(encoding="utf-8"))


def _recording_dir(recording_id: str) -> Path:
    """Return the evidence subdirectory for ``recording_id``."""
    return _EVIDENCE_DIR / recording_id


def _load_audit(recording_id: str, detector_name: str) -> dict[str, Any]:
    """Load a sealed audit record from the committed artefacts."""
    suffix_map = {
        "normalized_delta_envelope": "delta_envelope_audit",
        "multi_channel_delta_kuramoto": "multichannel_kuramoto_audit",
        "snr_weighted_delta_kuramoto": "snr_weighted_kuramoto_audit",
    }
    suffix = suffix_map[detector_name]
    path = _recording_dir(recording_id) / f"cap_{recording_id}_{suffix}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _test_audit_content_seal_recomputes(audit: dict[str, Any]) -> None:
    """The record's ``content_hash`` matches its canonical payload."""
    payload = copy.deepcopy(audit)
    sealed = payload.pop("content_hash")
    assert canonical_record_hash(payload) == sealed


def test_envelope_audit_content_seal_recomputes(
    envelope_audit: dict[str, Any],
) -> None:
    """The delta-envelope record's content hash is internally consistent."""
    _test_audit_content_seal_recomputes(envelope_audit)


def test_kuramoto_audit_content_seal_recomputes(
    kuramoto_audit: dict[str, Any],
) -> None:
    """The multi-channel Kuramoto record's content hash is internally consistent."""
    _test_audit_content_seal_recomputes(kuramoto_audit)


def test_snr_kuramoto_audit_content_seal_recomputes(
    snr_kuramoto_audit: dict[str, Any],
) -> None:
    """The SNR-weighted Kuramoto record's content hash is internally consistent."""
    _test_audit_content_seal_recomputes(snr_kuramoto_audit)


def test_envelope_audit_content_hash_matches_committed_value(
    envelope_audit: dict[str, Any],
) -> None:
    """The committed delta-envelope record's content hash has not drifted."""
    assert envelope_audit["content_hash"] == _ENVELOPE_CONTENT_HASH


def test_kuramoto_audit_content_hash_matches_committed_value(
    kuramoto_audit: dict[str, Any],
) -> None:
    """The committed multi-channel Kuramoto record's content hash has not drifted."""
    assert kuramoto_audit["content_hash"] == _KURAMOTO_CONTENT_HASH


def test_snr_kuramoto_audit_content_hash_matches_committed_value(
    snr_kuramoto_audit: dict[str, Any],
) -> None:
    """The committed SNR-weighted Kuramoto record's content hash has not drifted."""
    assert snr_kuramoto_audit["content_hash"] == _SNR_KURAMOTO_CONTENT_HASH


@pytest.mark.parametrize("recording_id", _CROSS_SUBJECT_RECORDINGS)
@pytest.mark.parametrize(
    "detector_name",
    [
        "normalized_delta_envelope",
        "multi_channel_delta_kuramoto",
        "snr_weighted_delta_kuramoto",
    ],
)
def test_cross_subject_audit_content_seal_recomputes(
    recording_id: str,
    detector_name: str,
) -> None:
    """Every sealed audit record's content hash matches its canonical payload."""
    audit = _load_audit(recording_id, detector_name)
    _test_audit_content_seal_recomputes(audit)


@pytest.mark.parametrize("recording_id", _CROSS_SUBJECT_RECORDINGS)
@pytest.mark.parametrize(
    "detector_name",
    [
        "normalized_delta_envelope",
        "multi_channel_delta_kuramoto",
        "snr_weighted_delta_kuramoto",
    ],
)
def test_cross_subject_audit_content_hash_matches_committed_value(
    recording_id: str,
    detector_name: str,
) -> None:
    """Committed sealed audit records have not drifted across the panel."""
    audit = _load_audit(recording_id, detector_name)
    assert audit["content_hash"] == _AUDIT_CONTENT_HASHES[recording_id][detector_name]


def test_comparison_pins_source_file_digests(comparison: dict[str, Any]) -> None:
    """The comparison pins the SHA-256 digests of the citation-only source files."""
    assert comparison["source_files"]["edf_sha256"] == _EDF_SHA256
    assert comparison["source_files"]["txt_sha256"] == _TXT_SHA256


def test_summaries_match_audit_records(
    envelope_audit: dict[str, Any],
    envelope_summary: dict[str, Any],
    kuramoto_audit: dict[str, Any],
    kuramoto_summary: dict[str, Any],
    snr_kuramoto_audit: dict[str, Any],
    snr_kuramoto_summary: dict[str, Any],
) -> None:
    """Each summary carries the same audit identifiers and hash as its sealed record."""
    assert envelope_summary["audit_content_hash"] == envelope_audit["content_hash"]
    assert envelope_summary["corpus_id"] == envelope_audit["corpus_id"]
    assert envelope_summary["captured_at"] == envelope_audit["captured_at"]
    assert kuramoto_summary["audit_content_hash"] == kuramoto_audit["content_hash"]
    assert kuramoto_summary["corpus_id"] == kuramoto_audit["corpus_id"]
    assert kuramoto_summary["captured_at"] == kuramoto_audit["captured_at"]
    assert (
        snr_kuramoto_summary["audit_content_hash"] == snr_kuramoto_audit["content_hash"]
    )
    assert snr_kuramoto_summary["corpus_id"] == snr_kuramoto_audit["corpus_id"]
    assert snr_kuramoto_summary["captured_at"] == snr_kuramoto_audit["captured_at"]


def test_comparison_counts_and_channels(comparison: dict[str, Any]) -> None:
    """The comparison carries the documented epoch counts and channel set."""
    assert comparison["n_epochs"] == 1154
    assert comparison["n_n3"] == 321
    assert comparison["n_wake"] == 39
    assert comparison["channels"] == [
        "F1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "F2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
    ]


def test_envelope_audit_verdict_matches_documented_result(
    envelope_audit: dict[str, Any],
) -> None:
    """The sealed delta-envelope audit reports the documented honest result."""
    a = envelope_audit["audit"]
    assert a["detector_name"] == "normalized_delta_envelope"
    assert a["n_events"] == 321
    assert a["n_nulls"] == 39
    assert a["target_false_alarm"] == pytest.approx(0.10, abs=1.0e-6)
    assert a["achieved_false_alarm"] == pytest.approx(0.077, abs=1.0e-3)
    assert a["detection_rate"] == pytest.approx(0.380, abs=1.0e-3)
    assert a["beats_chance"] is True
    assert a["significance"]["p_value"] < 0.001
    assert a["significance"]["seed"] == 42
    assert a["significance"]["n_permutations"] == 10_000


def test_kuramoto_audit_verdict_matches_documented_result(
    kuramoto_audit: dict[str, Any],
) -> None:
    """The sealed multi-channel Kuramoto audit reports the documented honest result."""
    a = kuramoto_audit["audit"]
    assert a["detector_name"] == "multi_channel_delta_kuramoto"
    assert a["n_events"] == 321
    assert a["n_nulls"] == 39
    assert a["target_false_alarm"] == pytest.approx(0.10, abs=1.0e-6)
    assert a["achieved_false_alarm"] == pytest.approx(0.077, abs=1.0e-3)
    assert a["detection_rate"] == pytest.approx(0.231, abs=1.0e-3)
    assert a["beats_chance"] is True
    assert a["significance"]["p_value"] == pytest.approx(0.017, abs=1.0e-3)
    assert a["significance"]["seed"] == 42
    assert a["significance"]["n_permutations"] == 10_000


def test_snr_kuramoto_audit_verdict_matches_documented_result(
    snr_kuramoto_audit: dict[str, Any],
) -> None:
    """The sealed SNR-weighted Kuramoto audit reports the documented honest result."""
    a = snr_kuramoto_audit["audit"]
    assert a["detector_name"] == "snr_weighted_delta_kuramoto"
    assert a["n_events"] == 321
    assert a["n_nulls"] == 39
    assert a["target_false_alarm"] == pytest.approx(0.10, abs=1.0e-6)
    assert a["achieved_false_alarm"] == pytest.approx(0.077, abs=1.0e-3)
    assert a["detection_rate"] == pytest.approx(0.218, abs=1.0e-3)
    assert a["beats_chance"] is True
    assert a["significance"]["p_value"] == pytest.approx(0.025, abs=1.0e-3)
    assert a["significance"]["seed"] == 42
    assert a["significance"]["n_permutations"] == 10_000


def test_envelope_summary_counts_and_scores(
    envelope_summary: dict[str, Any],
) -> None:
    """The delta-envelope summary carries documented epoch counts and mean scores."""
    assert envelope_summary["n_events"] == 321
    assert envelope_summary["n_nulls"] == 39
    assert envelope_summary["score_mean_events"] == pytest.approx(0.762, abs=1.0e-3)
    assert envelope_summary["score_mean_nulls"] == pytest.approx(0.702, abs=1.0e-3)


def test_kuramoto_summary_counts_and_scores(
    kuramoto_summary: dict[str, Any],
) -> None:
    """The multi-channel Kuramoto summary carries the documented counts and scores."""
    assert kuramoto_summary["n_events"] == 321
    assert kuramoto_summary["n_nulls"] == 39
    assert kuramoto_summary["score_mean_events"] == pytest.approx(0.629, abs=1.0e-3)
    assert kuramoto_summary["score_mean_nulls"] == pytest.approx(0.604, abs=1.0e-3)


def test_snr_kuramoto_summary_counts_and_scores(
    snr_kuramoto_summary: dict[str, Any],
) -> None:
    """The SNR-weighted Kuramoto summary carries the documented counts and scores."""
    assert snr_kuramoto_summary["n_events"] == 321
    assert snr_kuramoto_summary["n_nulls"] == 39
    assert snr_kuramoto_summary["score_mean_events"] == pytest.approx(0.629, abs=1.0e-3)
    assert snr_kuramoto_summary["score_mean_nulls"] == pytest.approx(0.605, abs=1.0e-3)


# --------------------------------------------------------------------------- #
# Cross-subject aggregate integrity                                             #
# --------------------------------------------------------------------------- #


def test_aggregate_has_four_recordings(aggregate: dict[str, Any]) -> None:
    """The aggregate spans the documented four-recording panel."""
    assert aggregate["benchmark"] == "cap_multichannel_n3_vs_wake"
    assert aggregate["corpus"] == "PhysioNet CAP Sleep Database"
    assert aggregate["target_false_alarm"] == pytest.approx(0.10, abs=1.0e-6)
    assert aggregate["n_recordings"] == 4
    assert aggregate["recording_ids"] == list(_CROSS_SUBJECT_RECORDINGS)


def test_aggregate_pins_source_file_digests(aggregate: dict[str, Any]) -> None:
    """The aggregate pins the SHA-256 digests of every citation-only source file."""
    for rec in aggregate["per_recording"]:
        recording_id = rec["recording_id"]
        expected = _SOURCE_DIGESTS[recording_id]
        assert rec["source_files"]["edf_sha256"] == expected["edf_sha256"]
        assert rec["source_files"]["txt_sha256"] == expected["txt_sha256"]


def test_aggregate_cross_subject_stats(aggregate: dict[str, Any]) -> None:
    """The aggregate reports the documented per-detector cross-subject statistics."""
    env = aggregate["normalized_delta_envelope"]
    kur = aggregate["multi_channel_delta_kuramoto"]
    snr = aggregate["snr_weighted_delta_kuramoto"]
    assert env["mean_detection_rate"] == pytest.approx(0.525, abs=1.0e-3)
    assert env["std_detection_rate"] == pytest.approx(0.360, abs=1.0e-3)
    assert env["mean_achieved_false_alarm"] == pytest.approx(0.092, abs=1.0e-3)
    assert env["geometric_mean_p_value"] == pytest.approx(0.001, abs=1.0e-3)
    assert env["fraction_beats_chance"] == pytest.approx(0.75, abs=1.0e-6)
    assert kur["mean_detection_rate"] == pytest.approx(0.184, abs=1.0e-3)
    assert kur["std_detection_rate"] == pytest.approx(0.070, abs=1.0e-3)
    assert kur["mean_achieved_false_alarm"] == pytest.approx(0.092, abs=1.0e-3)
    assert kur["geometric_mean_p_value"] == pytest.approx(0.015, abs=1.0e-3)
    assert kur["fraction_beats_chance"] == pytest.approx(0.75, abs=1.0e-6)
    assert snr["mean_detection_rate"] == pytest.approx(0.175, abs=1.0e-3)
    assert snr["std_detection_rate"] == pytest.approx(0.075, abs=1.0e-3)
    assert snr["mean_achieved_false_alarm"] == pytest.approx(0.092, abs=1.0e-3)
    assert snr["geometric_mean_p_value"] == pytest.approx(0.016, abs=1.0e-3)
    assert snr["fraction_beats_chance"] == pytest.approx(0.75, abs=1.0e-6)


def test_aggregate_recommendation(aggregate: dict[str, Any]) -> None:
    """The data-driven recommendation is recorded and justified."""
    rec = aggregate["recommendation"]
    assert rec["refine_kuramoto"] is False
    assert rec["preferred_variant"] == "normalized_delta_envelope"
    assert "SNR-weighting" in rec["rationale"] or "does not improve" in rec["rationale"]


@pytest.mark.parametrize("recording_id", _CROSS_SUBJECT_RECORDINGS)
@pytest.mark.parametrize(
    "detector_name",
    [
        "normalized_delta_envelope",
        "multi_channel_delta_kuramoto",
        "snr_weighted_delta_kuramoto",
    ],
)
def test_aggregate_per_recording_verdict_matches_committed_value(
    aggregate: dict[str, Any],
    recording_id: str,
    detector_name: str,
) -> None:
    """Each per-recording detector summary in the aggregate matches its sealed audit."""
    per_rec = {r["recording_id"]: r for r in aggregate["per_recording"]}
    summary = per_rec[recording_id]["detectors"][detector_name]
    audit = _load_audit(recording_id, detector_name)
    assert summary["audit_content_hash"] == audit["content_hash"]
    assert summary["detection_rate"] == audit["audit"]["detection_rate"]
    assert summary["p_value"] == audit["audit"]["significance"]["p_value"]
    assert summary["beats_chance"] == audit["audit"]["beats_chance"]


# --------------------------------------------------------------------------- #
# Kuramoto diagnostic integrity                                                 #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def diagnostic() -> dict[str, Any]:
    """Return the committed Kuramoto diagnostic record."""
    return json.loads(_DIAGNOSTIC_PATH.read_text(encoding="utf-8"))


def test_diagnostic_covers_four_recordings(diagnostic: dict[str, Any]) -> None:
    """The diagnostic spans the same four-recording panel."""
    assert diagnostic["benchmark"] == "cap_kuramoto_diagnostic"
    assert diagnostic["n_recordings"] == 4
    assert diagnostic["recording_ids"] == list(_CROSS_SUBJECT_RECORDINGS)


def test_diagnostic_detector_rates_match_aggregate(
    diagnostic: dict[str, Any],
    aggregate: dict[str, Any],
) -> None:
    """Per-recording detection rates in the diagnostic match the audited values."""
    agg_by_id = {r["recording_id"]: r for r in aggregate["per_recording"]}
    for rec in diagnostic["per_recording"]:
        rec_id = rec["recording_id"]
        agg = agg_by_id[rec_id]
        assert rec["detector_results"]["envelope_detection_rate"] == pytest.approx(
            agg["detectors"]["normalized_delta_envelope"]["detection_rate"],
            abs=1.0e-6,
        )
        assert rec["detector_results"]["kuramoto_detection_rate"] == pytest.approx(
            agg["detectors"]["multi_channel_delta_kuramoto"]["detection_rate"],
            abs=1.0e-6,
        )


def test_diagnostic_records_top_predictor_and_recommendation(
    diagnostic: dict[str, Any],
) -> None:
    """The diagnostic names a top predictor and a concrete next variant."""
    assert diagnostic["correlations"]["top_predictor"] in {
        "n_channels",
        "delta_snr",
        "delta_env_mean",
        "delta_env_std",
        "phase_circvar",
        "kuramoto_r_mean",
        "kuramoto_r_std",
        "hf_power_ratio",
        "signal_kurtosis",
    }
    rec = diagnostic["recommendation"]
    assert rec["next_variant"] in {
        "snr_weighted_kuramoto",
        "artifact_aware_channel_selection",
        "adaptive_channel_selection",
        "robust_temporal_aggregation",
    }
    assert len(rec["rationale"]) > 0
    assert rec["recording_where_kuramoto_works"] == "n2"
