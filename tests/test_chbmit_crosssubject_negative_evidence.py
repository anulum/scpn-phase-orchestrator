# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed CHB-MIT cross-subject negative integrity tests

"""Integrity tests for the committed CHB-MIT cross-subject generalisation record.

`examples/real_data/chbmit_crosssubject_kuramoto/chbmit_crosssubject_kuramoto.json`
is the sealed-by-provenance output of `bench/chbmit_crosssubject_validation.py`: a
leave-one-subject-out test of whether the global top-k PLV Kuramoto detector
generalises beyond the single subject it was tuned on. Its recorded answer is the
decisive **negative** the honest cross-domain transfer surface exists to honour —
both mean AUCs sit at chance (≈ 0.50), so the detector does **not** generalise
across subjects.

These tests guard that committed answer without the raw CHB-MIT EDFs (which are
citation-only and not redistributed). They cannot re-run the detector, so they do
the honest thing available on the derived record: recompute the aggregate AUCs
from the per-subject rows so a hand-edited headline cannot drift from its own
evidence, and assert the aggregate remains at chance — a guard against the record
being quietly flipped to claim a generalisation it never showed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_RECORD_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "real_data"
    / "chbmit_crosssubject_kuramoto"
    / "chbmit_crosssubject_kuramoto.json"
)

#: Half-open band around 0.5 within which a mean AUC counts as "at chance" — wide
#: enough for the honest recorded spread (0.499 / 0.509), tight enough that a real
#: generalisation claim (AUC well above 0.6) would trip the guard.
_CHANCE_LOW = 0.40
_CHANCE_HIGH = 0.60


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    """Return the committed CHB-MIT cross-subject generalisation record."""
    return json.loads(_RECORD_PATH.read_text(encoding="utf-8"))


def test_record_describes_the_leave_one_subject_out_benchmark(
    record: dict[str, Any],
) -> None:
    """The record identifies the LOSO CHB-MIT cross-subject question and corpus."""
    assert record["benchmark"] == "chbmit_crosssubject_kuramoto"
    assert "CHB-MIT" in record["corpus"]
    assert "across subjects" in record["question"]
    assert record["target_false_alarm"] == pytest.approx(0.1)


def test_subject_count_is_internally_consistent(record: dict[str, Any]) -> None:
    """``n_subjects`` matches both the subject list and the per-subject rows."""
    per_subject = record["per_subject"]
    assert record["n_subjects"] == len(record["subjects"])
    assert record["n_subjects"] == len(per_subject)
    assert [row["subject"] for row in per_subject] == record["subjects"]


def test_each_subject_row_is_a_valid_matched_false_alarm_audit(
    record: dict[str, Any],
) -> None:
    """Every per-subject AUC, detection rate, and calibrated ``k`` is in range."""
    k_grid = record["k_grid"]
    for row in record["per_subject"]:
        assert row["calibrated_k"] in k_grid
        assert row["n_preictal_epochs"] > 0
        for arm in ("topk_plv", "mean_r"):
            assert 0.0 <= row[arm]["auc"] <= 1.0
            assert 0.0 <= row[arm]["detection_rate"] <= 1.0


def test_aggregate_aucs_recompute_from_the_per_subject_rows(
    record: dict[str, Any],
) -> None:
    """The recorded mean AUCs match the mean of the per-subject AUCs.

    A hand-edited headline that no longer matches its own per-subject evidence
    trips this guard.
    """
    per_subject = record["per_subject"]
    n = len(per_subject)
    topk_mean = sum(row["topk_plv"]["auc"] for row in per_subject) / n
    mean_r_mean = sum(row["mean_r"]["auc"] for row in per_subject) / n
    assert record["mean_auc_topk_plv"] == pytest.approx(topk_mean, abs=1e-6)
    assert record["mean_auc_mean_r"] == pytest.approx(mean_r_mean, abs=1e-6)


def test_topk_plv_beats_mean_r_count_recomputes(record: dict[str, Any]) -> None:
    """The recorded minority-win count matches the per-subject comparison."""
    per_subject = record["per_subject"]
    beats = sum(
        1 for row in per_subject if row["topk_plv"]["auc"] > row["mean_r"]["auc"]
    )
    assert record["topk_plv_beats_mean_r_subjects"] == beats
    # The honest headline: top-k PLV does not beat the mean-R baseline on a
    # majority of held-out subjects.
    assert beats < record["n_subjects"] / 2


def test_both_aggregate_aucs_stay_at_chance(record: dict[str, Any]) -> None:
    """Both mean AUCs remain at chance — the detector does not generalise.

    This is the decisive negative. A future edit lifting either mean AUC to a
    genuine-skill level (well above 0.6) would be a cross-subject generalisation
    claim, and must not slip in silently against this recorded null.
    """
    assert _CHANCE_LOW <= record["mean_auc_topk_plv"] <= _CHANCE_HIGH
    assert _CHANCE_LOW <= record["mean_auc_mean_r"] <= _CHANCE_HIGH
