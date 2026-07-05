# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SCPN-vs-Dakos head-to-head runner tests

"""Tests for the SCPN-vs-Dakos head-to-head runner's aggregate reader.

The per-domain segment builders and ``main`` are an I/O shell over the citation-only
corpora and the already-tested capstone ingestion; the one piece of pure logic is
``scpn_best_detector``, which reads a committed aggregate. It handles both the
single-series capstone shape (one significance record) and the multi-node shape (one
per detector), so both are exercised here on synthetic aggregate mappings, plus the
committed four-domain comparison artefact is checked for the honest side-by-side result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench.head_to_head_ar1_kendall import scpn_best_detector

_COMPARISON = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "real_data"
    / "head_to_head_ar1_kendall"
    / "head_to_head_ar1_kendall.json"
)


def test_best_detector_reads_a_single_series_aggregate() -> None:
    # A single-series capstone records one significance record directly.
    aggregate = {
        "permutation_significance": {
            "observed_led": 1,
            "n_transitions": 6,
            "p_value": 0.51,
        }
    }
    best = scpn_best_detector(aggregate)
    assert best["detector"] == "critical_slowing_down"
    assert best["observed_led"] == 1
    assert best["n_transitions"] == 6
    assert best["p_value"] == 0.51


def test_best_detector_picks_the_leading_member_of_a_multi_node_aggregate() -> None:
    # A multi-node capstone records one significance record per detector; the member
    # with the most leads wins, ties broken by the smaller p-value.
    aggregate = {
        "permutation_significance": {
            "critical_slowing_down": {
                "observed_led": 3,
                "n_transitions": 12,
                "p_value": 0.20,
            },
            "synchronisation": {
                "observed_led": 0,
                "n_transitions": 12,
                "p_value": 1.0,
            },
            "transition_entropy": {
                "observed_led": 2,
                "n_transitions": 12,
                "p_value": 0.40,
            },
        }
    }
    best = scpn_best_detector(aggregate)
    assert best["detector"] == "critical_slowing_down"
    assert best["observed_led"] == 3


def test_committed_comparison_records_the_honest_head_to_head() -> None:
    """The artefact shows no detector significant; the EEG AR(1)-τ trend is closest."""
    payload: dict[str, Any] = json.loads(_COMPARISON.read_text(encoding="utf-8"))
    by_domain = {c["domain"]: c for c in payload["comparisons"]}
    assert set(by_domain) == {"palaeoclimate", "grid", "cardiac", "eeg"}
    for comparison in payload["comparisons"]:
        scpn_p = comparison["scpn"]["p_value"]
        dakos_p = comparison["dakos_ar1_kendall"]["significance"]["p_value"]
        assert scpn_p > 0.05  # SCPN reaches no significance in any domain
        assert dakos_p > 0.05  # nor does the Dakos competitor
    # The Dakos AR(1)-Kendall-τ trend on scalp EEG is the strongest signal anywhere.
    eeg_dakos = by_domain["eeg"]["dakos_ar1_kendall"]["significance"]
    assert eeg_dakos["observed_led"] == 3
    assert eeg_dakos["p_value"] < 0.10
