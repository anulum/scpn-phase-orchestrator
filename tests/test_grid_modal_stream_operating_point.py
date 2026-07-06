# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid streaming operating-point assembly tests

"""Unit tests for the grid streaming operating-point pure logic.

The raw-corpus search is an I/O shell (exercised by the sealed-artefact integrity test);
these tests pin the sustained-score reduction, the matched-false-alarm operating point,
the honest verdict, and the hash-sealed payload assembly with synthetic inputs, so the
record and its content hash are reproducible without any data.
"""

from __future__ import annotations

from typing import Any

from bench.grid_modal_stream_operating_point import (
    BENCHMARK,
    matched_operating_point,
    stream_operating_point_payload,
    stream_operating_point_verdict,
    sustained_score,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash


def test_sustained_score_with_persistence_one_is_the_maximum() -> None:
    assert sustained_score([0.1, 0.9, 0.4], persistence=1) == 0.9


def test_sustained_score_requires_consecutive_windows() -> None:
    # the best run of two consecutive windows is (0.5, 0.7) -> min 0.5
    assert sustained_score([0.9, 0.1, 0.5, 0.7], persistence=2) == 0.5


def test_sustained_score_is_neg_inf_with_too_few_windows() -> None:
    assert sustained_score([0.5], persistence=2) == float("-inf")


def test_matched_operating_point_calibrates_and_counts() -> None:
    dev_null = [0.0] * 9 + [5.0]  # 10% of the dev nulls are high
    result = matched_operating_point(
        dev_null,
        [3.0, 3.0, 0.0],  # two dev transitions above a ~near-zero threshold
        [0.0] * 10,
        [4.0, 0.0],  # one held-out transition high
        target_fa=0.1,
    )
    assert result["n_dev"] == 3
    assert result["n_held_out"] == 2
    assert result["dev_led"] == 2
    assert result["held_out_led"] == 1
    assert result["held_out_false_alarm"] == 0.0


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "window_seconds": 2.0,
            "step_seconds": 0.5,
            "feature": "focal",
            "persistence": 2,
            "threshold": 1.5,
            "dev_led": 9,
            "n_dev": 45,
            "held_out_led": 9,
            "n_held_out": 45,
            "held_out_false_alarm": 0.18,
        },
        {
            "window_seconds": 2.0,
            "step_seconds": 0.5,
            "feature": "r2gate",
            "persistence": 2,
            "threshold": 1.7,
            "dev_led": 10,
            "n_dev": 45,
            "held_out_led": 11,
            "n_held_out": 45,
            "held_out_false_alarm": 0.10,
        },
    ]


def test_verdict_picks_the_fa_holding_development_best_and_contrasts_offline() -> None:
    # focal has more dev leads (9) but its held-out FA drifts to 18% (disqualified at a
    # matched 10% target); the r2gate row holds FA at 10%, so it is the operating point
    verdict = stream_operating_point_verdict(
        _rows(), offline_led=36, offline_n=90, target_fa=0.1
    )
    assert "11/45" in verdict
    assert "r2gate" in verdict
    assert "36/90" in verdict
    assert "40%" in verdict


def test_verdict_falls_back_when_no_configuration_holds_the_target() -> None:
    # both rows drift far above the target -> fall back to the overall development-best
    rows = [{**row, "held_out_false_alarm": 0.5} for row in _rows()]
    verdict = stream_operating_point_verdict(
        rows, offline_led=36, offline_n=90, target_fa=0.1
    )
    assert "r2gate" in verdict  # r2gate has the most dev leads (10 > 9)
    assert "50%" in verdict  # the drifted false alarm is reported honestly


def test_payload_seals_and_recomputes() -> None:
    payload = stream_operating_point_payload(
        offline={"led": 36, "n_transitions": 90, "per_window_false_alarm": 0.09},
        naive_stream={
            "window_seconds": 2.0,
            "step_seconds": 0.5,
            "threshold": 1.32,
            "led": 82,
            "n_transitions": 90,
            "stream_false_alarm": 0.73,
        },
        rows=_rows(),
        corpus={"source": "PSML", "n_transitions": 90, "n_nulls": 186},
        target_fa=0.1,
    )
    assert payload["benchmark"] == BENCHMARK
    assert set(payload) == {
        "benchmark",
        "question",
        "corpus",
        "target_stream_false_alarm",
        "offline_per_window",
        "naive_stream_at_per_window_threshold",
        "search",
        "verdict",
        "content_hash",
    }
    assert "r2gate" in payload["verdict"]  # type: ignore[operator]
    body = {key: value for key, value in payload.items() if key != "content_hash"}
    assert payload["content_hash"] == canonical_record_hash(body)
