# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — bulk DNB early-warning capstone tests

"""Tests for the GSE2565 bulk DNB selection-controlled surrogate capstone.

The SOFT and series-matrix parsers, the rise score, the selection-controlled surrogate
null, the rank p-value, and the sealed evaluation are exercised on synthetic GEO-format
text with a planted DNB module — so every path, including the raw-file ingestion and the
end-to-end sealing, runs without the citation-only corpus. Gzip and plain inputs, the
missing-table and empty-timepoint guards, and both verdict branches are covered.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from bench.early_warning_dnb_bulk import (
    ArmResult,
    _gse2565_verdict,
    _open_maybe_gzip,
    _record_sample,
    arm_rising_frames,
    dnb_rise_score,
    evaluate_arm,
    evaluate_gse2565,
    load_expression_matrix,
    main,
    parse_sample_groups,
    surrogate_rank_pvalue,
    surrogate_rise_scores,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

_TIMEPOINTS = ("0.5", "1", "4", "8")


def _planted_frames(
    rng: np.random.Generator, *, n_probes: int = 12
) -> list[np.ndarray]:
    """Return four rising-limb frames with a module co-varying toward the transition."""
    frames = []
    for step in range(4):
        frame = rng.normal(5.0, 0.2, size=(4, n_probes))
        factor = rng.normal(0.0, 0.3 + 0.9 * step, size=4)
        for gene in (0, 1, 2):
            frame[:, gene] = 5.0 + factor + rng.normal(0.0, 0.05, size=4)
        frames.append(frame)
    return frames


def _write_synthetic_gse2565(
    directory: Path, *, planted: bool = True
) -> tuple[Path, Path]:
    """Write a tiny synthetic SOFT + series-matrix pair, returning their paths."""
    rng = np.random.default_rng(7)
    n_probes = 16
    reps = 3
    groups = ("CG", "Air")
    samples: list[tuple[str, str, str]] = []  # (accession, group, timepoint)
    columns: list[np.ndarray] = []
    counter = 0
    for group in groups:
        for step, timepoint in enumerate(_TIMEPOINTS):
            for _ in range(reps):
                counter += 1
                accession = f"GSM{counter:04d}"
                sample = rng.normal(200.0, 20.0, size=n_probes)
                if planted and group == "CG":
                    shared = rng.normal(0.0, 40.0 * step, size=1)
                    sample[:4] = 200.0 + shared + rng.normal(0.0, 5.0, size=4)
                samples.append((accession, group, timepoint))
                columns.append(np.abs(sample))
    matrix = np.column_stack(columns)  # probes x samples

    soft_lines = []
    for accession, group, timepoint in samples:
        soft_lines.append(f"^SAMPLE = {accession}\n")
        soft_lines.append("!Sample_description = phosgene study\n")
        soft_lines.append(f"!Sample_description = {group}\n")
        soft_lines.append("!Sample_description = 1\n")
        soft_lines.append(f"!Sample_description = {timepoint}\n")
        soft_lines.append("!Sample_description = chip replicate A\n")
    soft_path = directory / "family.soft"
    soft_path.write_text("".join(soft_lines), encoding="utf-8")

    header = "\t".join(['"ID_REF"'] + [f'"{a}"' for a, _, _ in samples])
    body = [header]
    for probe in range(n_probes):
        row = "\t".join(
            [f'"probe_{probe}"'] + [f"{value:.3f}" for value in matrix[probe]]
        )
        body.append(row)
    matrix_lines = ["!series_matrix_table_begin", *body, "!series_matrix_table_end", ""]
    matrix_path = directory / "series_matrix.txt"
    matrix_path.write_text("\n".join(matrix_lines), encoding="utf-8")
    return matrix_path, soft_path


# --------------------------------------------------------------------------- #
# Ingestion                                                                    #
# --------------------------------------------------------------------------- #


def test_parse_sample_groups_recovers_group_and_timepoint(tmp_path: Path) -> None:
    _, soft = _write_synthetic_gse2565(tmp_path)
    groups = parse_sample_groups(soft)
    assert groups["GSM0001"] == ("CG", "0.5")
    assert ("Air", "8") in groups.values()


def test_load_expression_matrix_reads_probes_and_log2(tmp_path: Path) -> None:
    matrix_path, _ = _write_synthetic_gse2565(tmp_path)
    accessions, probe_ids, matrix = load_expression_matrix(matrix_path)
    assert accessions[0] == "GSM0001"
    assert probe_ids[0] == "probe_0"
    assert matrix.shape == (16, 24)
    assert np.all(matrix > 0.0)  # log2 of positive intensities


def test_load_expression_matrix_reads_gzip(tmp_path: Path) -> None:
    matrix_path, _ = _write_synthetic_gse2565(tmp_path)
    gz_path = tmp_path / "series_matrix.txt.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write(matrix_path.read_text(encoding="utf-8"))
    _, _, matrix = load_expression_matrix(gz_path)
    assert matrix.shape == (16, 24)


def test_load_expression_matrix_rejects_a_file_without_a_table(tmp_path: Path) -> None:
    bad = tmp_path / "no_table.txt"
    bad.write_text("!Series_title = nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no !series_matrix_table_begin"):
        load_expression_matrix(bad)


def test_open_maybe_gzip_handles_plain_text(tmp_path: Path) -> None:
    plain = tmp_path / "plain.txt"
    plain.write_text("hello\n", encoding="utf-8")
    with _open_maybe_gzip(plain) as handle:
        assert handle.read().strip() == "hello"


def test_record_sample_ignores_an_incomplete_block() -> None:
    groups: dict[str, tuple[str, str]] = {}
    _record_sample(groups, "GSM9", ["only", "two"])  # fewer than four descriptions
    assert groups == {}
    _record_sample(groups, None, ["a", "b", "c", "d"])  # no accession
    assert groups == {}


def test_arm_rising_frames_builds_top_probe_frames(tmp_path: Path) -> None:
    matrix_path, soft = _write_synthetic_gse2565(tmp_path)
    accessions, _, matrix = load_expression_matrix(matrix_path)
    sample_groups = parse_sample_groups(soft)
    frames, top = arm_rising_frames(
        matrix, accessions, sample_groups, group="CG", n_top_probes=8
    )
    assert len(frames) == 4
    assert all(frame.shape == (3, 8) for frame in frames)
    assert len(top) == 8


def test_arm_rising_frames_rejects_a_missing_timepoint(tmp_path: Path) -> None:
    matrix_path, soft = _write_synthetic_gse2565(tmp_path)
    accessions, _, matrix = load_expression_matrix(matrix_path)
    sample_groups = parse_sample_groups(soft)
    with pytest.raises(ValueError, match="no CG samples at timepoint 99"):
        arm_rising_frames(
            matrix,
            accessions,
            sample_groups,
            group="CG",
            timepoints=("0.5", "99"),
            n_top_probes=8,
        )


# --------------------------------------------------------------------------- #
# Selection-controlled surrogate test                                         #
# --------------------------------------------------------------------------- #


def test_dnb_rise_score_returns_a_slope_and_module() -> None:
    rng = np.random.default_rng(1)
    frames = _planted_frames(rng)
    slope, module = dnb_rise_score(frames, candidate_fraction=0.4, min_module=3)
    assert slope > 0.0  # a planted rising module
    assert len(module) >= 2


def test_surrogate_rise_scores_reselect_per_surrogate() -> None:
    rng = np.random.default_rng(2)
    frames = _planted_frames(rng)
    scores = surrogate_rise_scores(
        frames, n_surrogates=30, seed=0, candidate_fraction=0.4, min_module=3
    )
    assert len(scores) == 30
    assert all(np.isfinite(scores))


def test_surrogate_rise_scores_tolerate_a_degenerate_shuffle() -> None:
    # Two samples per timepoint and a strong single-timepoint spike: many shuffles leave
    # no gene with a rising standard deviation, so module selection fails and the
    # surrogate contributes a zero slope rather than raising.
    frames = [
        np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]),
        np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]),
        np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]),
        np.array([[9.0, 1.0, 5.0], [1.0, 9.0, 5.0]]),
    ]
    scores = surrogate_rise_scores(
        frames, n_surrogates=40, seed=0, candidate_fraction=0.5, min_module=2
    )
    assert 0.0 in scores  # at least one degenerate shuffle yielded no trend


def test_surrogate_rank_pvalue_counts_the_tail() -> None:
    assert surrogate_rank_pvalue(1.0, [0.0, 0.5, 2.0]) == pytest.approx(2 / 4)
    assert surrogate_rank_pvalue(3.0, [0.0, 0.5, 2.0]) == pytest.approx(1 / 4)


def test_surrogate_rank_pvalue_rejects_an_empty_null() -> None:
    with pytest.raises(ValueError, match="surrogates must not be empty"):
        surrogate_rank_pvalue(1.0, [])


def test_evaluate_arm_produces_a_result_record() -> None:
    rng = np.random.default_rng(3)
    frames = _planted_frames(rng)
    probe_ids = [f"probe_{index}" for index in range(12)]
    top = list(range(12))
    result = evaluate_arm(
        frames,
        probe_ids,
        top,
        group="CG",
        n_surrogates=30,
        seed=0,
        candidate_fraction=0.4,
        min_module=3,
    )
    assert isinstance(result, ArmResult)
    record = result.to_record()
    assert record["group"] == "CG"
    assert 0.0 < record["p_value"] <= 1.0
    assert all(probe.startswith("probe_") for probe in record["module_probes"])


# --------------------------------------------------------------------------- #
# Verdict                                                                      #
# --------------------------------------------------------------------------- #


def test_verdict_reports_a_silence() -> None:
    verdict = _gse2565_verdict({"p_value": 0.38})
    assert "does not reach significance" in verdict


def test_verdict_reports_a_reached_significance() -> None:
    verdict = _gse2565_verdict({"p_value": 0.01})
    assert "beats chance" in verdict


# --------------------------------------------------------------------------- #
# End-to-end sealing                                                          #
# --------------------------------------------------------------------------- #


def test_evaluate_gse2565_seals_both_arms(tmp_path: Path) -> None:
    matrix_path, soft = _write_synthetic_gse2565(tmp_path)
    payload = evaluate_gse2565(
        matrix_path, soft, n_surrogates=40, seed=0, n_top_probes=10
    )
    assert payload["benchmark"] == "early_warning_dnb_gse2565"
    arms = payload["arms"]
    assert isinstance(arms, list)
    assert {arm["group"] for arm in arms} == {"CG", "Air"}
    sealed = payload.pop("content_hash")
    assert sealed == canonical_record_hash(payload)


def test_evaluate_gse2565_is_reproducible(tmp_path: Path) -> None:
    matrix_path, soft = _write_synthetic_gse2565(tmp_path)
    first = evaluate_gse2565(
        matrix_path, soft, n_surrogates=40, seed=0, n_top_probes=10
    )
    second = evaluate_gse2565(
        matrix_path, soft, n_surrogates=40, seed=0, n_top_probes=10
    )
    assert first == second


def test_main_writes_the_sealed_artefact(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    matrix_path, soft = _write_synthetic_gse2565(data)
    # main() reads the citation-only file names; gzip the synthetic content into them.
    for source, name in (
        (matrix_path, "GSE2565_series_matrix.txt.gz"),
        (soft, "GSE2565_family.soft.gz"),
    ):
        with gzip.open(data / name, "wt", encoding="utf-8") as handle:
            handle.write(source.read_text(encoding="utf-8"))
    out = tmp_path / "out"
    main(data, out)
    payload = json.loads(
        (out / "early_warning_dnb_gse2565.json").read_text(encoding="utf-8")
    )
    sealed = payload.pop("content_hash")
    assert sealed == canonical_record_hash(payload)
