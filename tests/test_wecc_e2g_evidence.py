# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed WECC 240-bus E2.G evidence tests

"""Integrity tests for the committed WECC 240-bus E2.G cross-dataset evidence.

`examples/real_data/wecc_240_osl/` seals the E2.G statistical leg of the
PSML-certified modal-growth detector on the 13 forced-oscillation cases of
the 2021 IEEE-NASPI OSL contest (WECC 240-bus synthetic PMU). These tests
guard the committed derived artefact without the citation-only raw data:
they recompute the content hash to prove the record was not hand-edited, pin
the raw-source digests, and assert the honest headline — the frozen
operating point does not port (zero crossings anywhere), the early-warning
lead is not significant, and the detection branch stays descriptive with its
chance bound sealed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from bench.early_warning_leadtime_wecc import DOCUMENTED_MODES
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

_DIR = Path(__file__).resolve().parents[1] / "examples" / "real_data" / "wecc_240_osl"

#: Raw-source digests of each case's ``BusVolMag.txt`` (TOFU-pinned at
#: acquisition, 2026-08-26; the archive digest is in the README).
_SOURCE_SHA256 = {
    "WECC_case1": ("94a5f89ef6d051c81b313b758b8736dc87f7086cfa101b58b9e6fa519de0b7b8"),
    "WECC_case2": ("e3b4e1e6918a2e4ac92d48cf86df927256ed0be080c634dfa61ffea4db6a398c"),
    "WECC_case3": ("d9a9988de751cec90c98dffa72baeab21961da59ec37ab41409889f41d614fdf"),
    "WECC_case4": ("b2d455485f920bfd589691447f9affe879922130b734c2700b7f301c3ceeebf5"),
    "WECC_case5": ("2f30dc65dd9c94bf602897e07edf4c933a70880d27d7ae865127bd37fe927712"),
    "WECC_case6": ("8239bcf4e85e3e3594250406ab5622b09fe254316a2b264607eac721315fd1a1"),
    "WECC_case7": ("fc4ec6c05b6bb416372288c4df7def5bfaeaa685783c20005a577568a58936d6"),
    "WECC_case8": ("c735189b15db55fedc45eaf660971da7a08a7ea74e46919a0d9f61fc94dff192"),
    "WECC_case9": ("76b048ed88c27a2c29ce3b8473d68f10a8168f663d285792d13973e68df0c3c8"),
    "WECC_case10": ("32d321534729b90917753a67bae54e25485bcd98bca61bb531eb7187aa2b8d8a"),
    "WECC_case11": ("00478d9adc42bb5e2b4415e60c091b88038b1c250d4c433ccda9201fe8163e0f"),
    "WECC_case12": ("25396a74f7bd9fb0c77327709e4964b7628f2a4cfd03b710ec44d2b64ddecb40"),
    "WECC_case13": ("b5fde906e1742246aec530819a9fdeae8c1e5d4c4d767d42ea690942365e8d07"),
}

_CONTENT_HASH = "f77be580e06b532e2471b452ac72856f87a5df0444c2d38b782b4be306982b66"


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """The committed sealed payload, parsed once per module."""
    path = _DIR / "wecc_240_osl_modal_growth_cross_dataset.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_content_hash_recomputes_from_the_payload(payload: dict[str, Any]) -> None:
    """The sealed hash must recompute from the committed record alone."""
    record = copy.deepcopy(payload)
    sealed = record.pop("content_hash")
    assert sealed == _CONTENT_HASH
    assert canonical_record_hash(record) == sealed


def test_tampered_payload_is_rejected(payload: dict[str, Any]) -> None:
    """Any edit to the record must break the seal."""
    record = copy.deepcopy(payload)
    record.pop("content_hash")
    record["local_calibration"]["led"] = [True] * 13
    assert canonical_record_hash(record) != _CONTENT_HASH


def test_source_digests_chain_to_the_raw_exports(payload: dict[str, Any]) -> None:
    """Every contest case pins its raw ``BusVolMag.txt`` digest."""
    transitions = {entry["case"]: entry for entry in payload["corpus"]["transitions"]}
    assert set(transitions) == set(_SOURCE_SHA256) == set(DOCUMENTED_MODES)
    for case, digest in _SOURCE_SHA256.items():
        assert transitions[case]["source_sha256"] == digest


def test_frozen_transfer_headline_is_honest(payload: dict[str, Any]) -> None:
    """G-a seals the conservative-direction non-portability: zero crossings."""
    assert len(payload["frozen_transfer"]) == 13
    for entry in payload["frozen_transfer"]:
        assert entry["null_crossings"] == 0
        assert entry["transition_crossings"] == 0
        assert entry["n_null"] == 47
    assert sum(entry["n_null"] for entry in payload["frozen_transfer"]) == 611


def test_early_warning_headline_is_honest(payload: dict[str, Any]) -> None:
    """G-b seals two led cases, positive leads, held FA, and p above 0.05."""
    calibration = payload["local_calibration"]
    assert calibration["led"].count(True) == 2
    leads = [value for value in calibration["lead_seconds"] if value is not None]
    assert len(leads) == 2
    assert all(value > 0.0 for value in leads)
    assert calibration["achieved_false_alarm"] <= calibration["target_false_alarm"]
    assert calibration["n_null"] == 130
    significance = calibration["significance"]
    assert significance["observed_led"] == 2
    assert significance["n_transitions"] == 13
    assert significance["p_value"] > 0.05  # no early-warning significance claim


def test_instant_onsets_are_disclosed(payload: dict[str, Any]) -> None:
    """9 of 13 cases seal an empty early-warning region at the forcing start."""
    transitions = payload["corpus"]["transitions"]
    instant = [
        entry
        for entry in transitions
        if abs(entry["onset_seconds"] - 30.0) < 0.05
        and entry["n_transition_windows"] == 0
    ]
    assert len(instant) == 9
    assert all(entry["onset_resolved"] for entry in transitions)


def test_detection_branch_stays_descriptive(payload: dict[str, Any]) -> None:
    """The detection branch seals its caveat and chance bound, no p claim."""
    secondary = payload["detection_secondary"]
    assert "DESCRIPTIVE ONLY" in secondary["significance_caveat"]
    assert secondary["detected"].count(True) == 12
    bounds = secondary["chance_detection_upper_bound"]
    assert len(bounds) == 13
    expected_by_chance = sum(entry["p_chance_upper"] for entry in bounds)
    assert expected_by_chance > 12.0  # observed 12/13 is chance-compatible
    latencies = [value for value in secondary["latency_seconds"] if value is not None]
    assert len(latencies) == 12
    assert sorted(latencies)[len(latencies) // 2] <= 4.0  # fast when it fires
