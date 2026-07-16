# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — tests for detector meta-analysis

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_phase_orchestrator.evaluation.detector_meta_analysis import (
    EvidenceRow,
    benjamini_hochberg_by_domain,
    build_report,
    discover_aggregate_jsons,
    extract_evidence,
    main,
    rank_per_domain,
    run_analysis,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_REAL_DATA = REPO_ROOT / "examples" / "real_data"


@pytest.fixture
def project_examples_dir():
    return EXAMPLES_REAL_DATA


def test_discover_aggregate_jsons_finds_committed_aggregates(project_examples_dir):
    paths = discover_aggregate_jsons(project_examples_dir)
    names = {p.name for p in paths}
    expected = {
        "cap_multichannel_aggregate.json",
        "synthetic_honest_audit_demo.json",
        "early_warning_leadtime_eeg_results.json",
        "early_warning_leadtime_cardiac_results.json",
        "early_warning_leadtime_climate_results.json",
        "early_warning_leadtime_grid_results.json",
    }
    assert expected.issubset(names)
    assert len(paths) >= len(expected)


def test_extract_honest_audit_cap_aggregate(project_examples_dir):
    path = (
        project_examples_dir
        / "cap_multichannel_staging"
        / "cap_multichannel_aggregate.json"
    )
    rows = extract_evidence(path)
    assert rows
    by_detector = {row.detector: row for row in rows}
    assert set(by_detector) == {
        "normalized_delta_envelope",
        "multi_channel_delta_kuramoto",
        "snr_weighted_delta_kuramoto",
    }
    assert by_detector["normalized_delta_envelope"].detection_rate == pytest.approx(
        0.525456, abs=1e-6
    )
    assert by_detector["normalized_delta_envelope"].beats_chance is True
    assert (
        by_detector["snr_weighted_delta_kuramoto"].detection_rate
        < by_detector["multi_channel_delta_kuramoto"].detection_rate
    )


def test_extract_honest_audit_synthetic_demo(project_examples_dir):
    path = (
        project_examples_dir
        / "synthetic_honest_audit_demo"
        / "synthetic_honest_audit_demo.json"
    )
    rows = extract_evidence(path)
    by_detector = {row.detector: row for row in rows}
    assert by_detector["lag1_autocorrelation"].detection_rate == pytest.approx(1.0)
    assert by_detector["lag1_autocorrelation"].beats_chance is True
    assert by_detector["window_mean_control"].detection_rate == pytest.approx(0.0)


def test_extract_leadtime_nested_schema(project_examples_dir):
    path = (
        project_examples_dir
        / "chb01_seizures"
        / "early_warning_leadtime_eeg_results.json"
    )
    rows = extract_evidence(path)
    by_detector = {row.detector: row for row in rows}
    assert "critical_slowing_down" in by_detector
    assert by_detector["critical_slowing_down"].detection_rate == pytest.approx(
        2 / 6, abs=1e-9
    )
    assert by_detector["transition_entropy"].detection_rate == pytest.approx(0.0)


def test_extract_leadtime_flat_climate_schema(project_examples_dir):
    path = (
        project_examples_dir
        / "dakos_climate_transitions"
        / "early_warning_leadtime_climate_results.json"
    )
    rows = extract_evidence(path)
    assert len(rows) == 1
    row = rows[0]
    assert row.detector == "critical_slowing_down"
    assert row.detection_rate == pytest.approx(1 / 6, abs=1e-9)
    assert row.domain == "dakos_climate_transitions"


def test_extract_csd_variant_synthetic_aggregate(project_examples_dir):
    path = (
        project_examples_dir
        / "csd_variant_synthetic"
        / "csd_variant_synthetic_results.json"
    )
    rows = extract_evidence(path)
    by_detector = {row.detector: row for row in rows}
    assert "critical_slowing_down_baseline" in by_detector
    assert "critical_slowing_down_multiscale" in by_detector
    assert "critical_slowing_down_surrogate" in by_detector
    assert (
        by_detector["critical_slowing_down_multiscale"].domain
        == "csd_variant_synthetic"
    )


def test_rank_per_domain_uses_competition_ranking():
    rows = [
        EvidenceRow("d1", "a", 0.9, 0.01, True, "f.json"),
        EvidenceRow("d1", "b", 0.8, 0.02, True, "f.json"),
        EvidenceRow("d1", "c", 0.8, 0.01, True, "f.json"),
        EvidenceRow("d2", "a", 0.5, 0.5, False, "f.json"),
    ]
    rankings = rank_per_domain(rows)
    d1 = rankings["d1"]
    assert d1[0][0] == 1
    assert d1[0][1].detector == "a"
    # b and c tie on detection rate; p-value breaks the tie.
    assert d1[1][0] == 2
    assert d1[1][1].detector == "c"
    assert d1[2][0] == 3
    assert d1[2][1].detector == "b"
    assert rankings["d2"][0][0] == 1


def test_rank_overall_prefers_low_mean_rank_and_more_wins(project_examples_dir):
    rows, rankings, overall, _, _ = run_analysis(project_examples_dir)
    assert rows
    assert overall
    detectors = [entry["detector"] for entry in overall]
    assert "critical_slowing_down" in detectors
    csd = next(e for e in overall if e["detector"] == "critical_slowing_down")
    assert csd["wins"] >= 2
    assert csd["mean_rank"] <= 2.0


def test_build_report_contains_expected_sections(project_examples_dir):
    rows, rankings, overall, source_paths, unsupported = run_analysis(
        project_examples_dir
    )
    report = build_report(rows, rankings, overall, source_paths, unsupported)
    for heading in (
        "# Cross-Domain Detector Meta-Analysis Report",
        "## Data sources",
        "## Per-domain rankings",
        "## Cross-domain overall ranking",
        "## Ranked refinement backlog",
    ):
        assert heading in report
    assert "critical_slowing_down" in report
    assert "normalized_delta_envelope" in report
    assert "SNR-weighted Kuramoto did not improve" in report


def _row(domain: str, detector: str, p_value: float) -> EvidenceRow:
    return EvidenceRow(
        domain=domain,
        detector=detector,
        detection_rate=0.5,
        p_value=p_value,
        beats_chance=p_value < 0.05,
        source_file=f"{domain}_aggregate.json",
    )


def test_benjamini_hochberg_by_domain_corrects_within_each_domain():
    rows = [
        _row("d1", "a", 0.001),
        _row("d1", "b", 0.5),
        _row("d2", "c", 0.02),
    ]
    adjusted = benjamini_hochberg_by_domain(rows)
    # d1 family of two: 0.001 · 2/1 = 0.002; 0.5 unchanged.
    assert adjusted[("d1", "a")] == pytest.approx(0.002)
    assert adjusted[("d1", "b")] == pytest.approx(0.5)
    # d2 is a family of one, so its correction leaves the raw value.
    assert adjusted[("d2", "c")] == pytest.approx(0.02)


def test_benjamini_hochberg_by_domain_handles_no_rows():
    assert benjamini_hochberg_by_domain([]) == {}


def test_build_report_shows_the_multiplicity_corrected_column(project_examples_dir):
    rows, rankings, overall, source_paths, unsupported = run_analysis(
        project_examples_dir
    )
    report = build_report(rows, rankings, overall, source_paths, unsupported)
    assert "BH-adj p" in report
    assert "Benjamini–Hochberg" in report
    # The per-domain table header carries the raw and the adjusted p side by side.
    assert "| Rank | Detector | Detection rate | p-value | BH-adj p " in report


def test_main_generates_report(tmp_path, project_examples_dir):
    output = tmp_path / "report.md"
    rc = main(["--root", str(project_examples_dir), "--output", str(output)])
    assert rc == 0
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "## Cross-domain overall ranking" in text


def test_unsupported_aggregates_are_recorded(tmp_path):
    root = tmp_path / "examples" / "real_data"
    domain = root / "unknown_domain"
    domain.mkdir(parents=True)
    (domain / "unknown_aggregate.json").write_text(
        json.dumps({"some_unrelated_key": [1, 2, 3]}), encoding="utf-8"
    )
    rows, rankings, overall, source_paths, unsupported = run_analysis(root)
    assert not rows
    assert not rankings
    assert not overall
    assert len(source_paths) == 1
    assert len(unsupported) == 1
    report = build_report(rows, rankings, overall, source_paths, unsupported)
    assert "### Unsupported artefacts" in report
    assert "unknown_aggregate.json" in report
