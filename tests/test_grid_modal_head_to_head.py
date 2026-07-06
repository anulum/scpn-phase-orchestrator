# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid modal-vs-generic head-to-head assembly tests

"""Unit tests for the grid modal-vs-generic head-to-head payload and verdict.

The raw-corpus ingestion is an I/O shell (exercised end-to-end by the sealed-artefact
integrity test); these tests pin the *pure* logic — the verdict sentence in both the
modal-wins and modal-loses branches, and the hash-sealed payload assembly — with
synthetic significance records, so the assembled record and its content hash are
reproducible without any data.
"""

from __future__ import annotations

from typing import Any

from bench.grid_modal_head_to_head import (
    BENCHMARK,
    head_to_head_payload,
    modal_versus_generic_verdict,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash


def _significance(observed_led: int, *, n: int, p_value: float) -> dict[str, Any]:
    """Return a permutation-significance-shaped record."""
    return {
        "observed_led": observed_led,
        "n_transitions": n,
        "expected_led": n * 0.1,
        "p_value": p_value,
    }


def _modal_record(observed_led: int, *, n: int, p_value: float) -> dict[str, Any]:
    """Return a ModalGrowthSignificance-shaped audit record."""
    return {
        "detector": "modal_envelope_growth_rate_focal",
        "aggregation": "focal",
        "recency_top": 3.0,
        "score_threshold": 0.5,
        "achieved_false_alarm": 0.09,
        "significance": _significance(observed_led, n=n, p_value=p_value),
    }


_GENERIC = {
    "critical_slowing_down": _significance(8, n=90, p_value=0.676),
    "synchronisation": _significance(4, n=90, p_value=0.977),
    "transition_entropy": _significance(13, n=90, p_value=0.152),
    "ensemble": _significance(7, n=90, p_value=0.788),
}


def test_verdict_reports_the_modal_win() -> None:
    verdict = modal_versus_generic_verdict(
        _significance(40, n=90, p_value=0.0001), _GENERIC
    )
    assert "40/90" in verdict
    assert "beating" in verdict
    # the best generic member (most leads) is named
    assert "transition_entropy" in verdict
    assert "13/90" in verdict


def test_verdict_reports_no_modal_win_when_generic_leads_more() -> None:
    verdict = modal_versus_generic_verdict(
        _significance(5, n=90, p_value=0.4), _GENERIC
    )
    assert "does not beat" in verdict
    assert "transition_entropy" in verdict


def test_verdict_reports_no_win_when_lead_is_not_significant() -> None:
    # more leads than the best generic, but not significant -> still not a claimed win
    verdict = modal_versus_generic_verdict(
        _significance(14, n=90, p_value=0.2), _GENERIC
    )
    assert "does not beat" in verdict


def test_payload_seals_and_recomputes() -> None:
    payload = head_to_head_payload(
        modal_record=_modal_record(40, n=90, p_value=0.0001),
        generic_records=_GENERIC,
        corpus={"source": "PSML", "n_transitions": 90, "n_nulls": 88},
        operating_point={"aggregation": "focal", "recency_top": 3.0},
        held_out_validation={
            "n_transitions": 45,
            "observed_led": 24,
            "p_value": 0.0002,
        },
        target_fa=0.1,
        n_permutations=10000,
        seed=0,
    )
    assert payload["benchmark"] == BENCHMARK
    assert set(payload) == {
        "benchmark",
        "question",
        "corpus",
        "operating_point",
        "target_false_alarm",
        "n_permutations",
        "seed",
        "modal",
        "generic_suite",
        "held_out_validation",
        "verdict",
        "content_hash",
    }
    assert "beating" in payload["verdict"]  # type: ignore[operator]
    body = {key: value for key, value in payload.items() if key != "content_hash"}
    assert payload["content_hash"] == canonical_record_hash(body)


def test_payload_is_deterministic() -> None:
    kwargs: dict[str, Any] = {
        "modal_record": _modal_record(40, n=90, p_value=0.0001),
        "generic_records": _GENERIC,
        "corpus": {"source": "PSML", "n_transitions": 90, "n_nulls": 88},
        "operating_point": {"aggregation": "focal", "recency_top": 3.0},
        "held_out_validation": {
            "n_transitions": 45,
            "observed_led": 24,
            "p_value": 0.0002,
        },
        "target_fa": 0.1,
        "n_permutations": 10000,
        "seed": 0,
    }
    assert (
        head_to_head_payload(**kwargs)["content_hash"]
        == head_to_head_payload(**kwargs)["content_hash"]
    )
