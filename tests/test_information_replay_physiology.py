# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physiology replay monitor tests

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.information_replay_physiology import (
    build_physiology_integrated_information_replays,
)


def test_replays_are_deterministic() -> None:
    """Physiology replay records are stable for identical configured inputs."""
    first = build_physiology_integrated_information_replays(n_samples=192, n_bins=9)
    second = build_physiology_integrated_information_replays(n_samples=192, n_bins=9)

    assert first == second


def test_replay_records_are_json_safe_and_non_actuating() -> None:
    """Records are JSON serialisable and preserve the non-actuating proxy boundary."""
    records = build_physiology_integrated_information_replays(n_samples=192, n_bins=8)
    dumped = json.dumps(records)
    loaded = json.loads(dumped)

    assert isinstance(loaded, list)
    assert len(loaded) == len(records)
    assert all(record["non_actuating"] is True for record in loaded)
    assert all(
        record["claim_boundary"] == "engineering_proxy_not_theoretical_iit"
        for record in loaded
    )
    assert all(
        isinstance(record["minimum_partition"], list)
        and len(record["minimum_partition"]) == 2
        for record in loaded
    )
    assert all(
        isinstance(part, list)
        for record in loaded
        for part in record["minimum_partition"]
    )


def test_replay_corpus_has_minimum_size_and_required_fields() -> None:
    """Corpus should expose both physiology domains and all required fields."""
    records = build_physiology_integrated_information_replays()
    by_name = {record["case_name"]: record for record in records}
    required_names = {
        "cardiac_respiratory_lock",
        "cardiac_respiratory_recovery",
        "eeg_sleep_spindle",
        "eeg_sleep_baseline",
    }
    required_fields = (
        "domain",
        "case_name",
        "description",
        "n_oscillators",
        "n_samples",
        "n_bins",
        "phi",
        "normalised_phi",
        "total_integration",
        "minimum_partition",
        "expected_relationship",
        "claim_boundary",
        "non_actuating",
    )

    assert len(records) >= 2
    assert required_names.issubset(by_name.keys())

    for record in records:
        assert record["domain"] == "physiology"
        assert all(field in record for field in required_fields)
        assert isinstance(record["n_oscillators"], int) and record["n_oscillators"] >= 2
        assert isinstance(record["n_samples"], int) and record["n_samples"] >= 2
        assert isinstance(record["phi"], float)
        assert isinstance(record["normalised_phi"], float)
        assert isinstance(record["total_integration"], float)
        assert isinstance(record["expected_relationship"], str)
        assert 0.0 <= record["normalised_phi"] <= 1.0
        assert record["claim_boundary"] == "engineering_proxy_not_theoretical_iit"


def test_expected_coherent_vs_fragmented_ordering_is_present() -> None:
    """Coherent replay cases should be stronger than fragmented/recovery baselines."""
    records = build_physiology_integrated_information_replays(n_samples=192, n_bins=10)
    by_name = {record["case_name"]: record for record in records}

    assert (
        by_name["cardiac_respiratory_lock"]["phi"]
        > by_name["cardiac_respiratory_recovery"]["phi"]
    )
    assert (
        "cardiac_respiratory_lock > cardiac_respiratory_recovery"
        in by_name["cardiac_respiratory_lock"]["expected_relationship"]
    )

    assert by_name["eeg_sleep_spindle"]["phi"] > by_name["eeg_sleep_baseline"]["phi"]
    assert (
        "eeg_sleep_spindle > eeg_sleep_baseline"
        in by_name["eeg_sleep_spindle"]["expected_relationship"]
    )

    assert np.isfinite(by_name["cardiac_respiratory_lock"]["phi"])
    assert np.isfinite(by_name["eeg_sleep_spindle"]["phi"])


def test_build_replays_rejects_invalid_parameters() -> None:
    """Bad corpus inputs fail closed with ValueError."""
    with pytest.raises(ValueError, match="n_samples"):
        build_physiology_integrated_information_replays(n_samples=31)

    with pytest.raises(ValueError, match="n_bins"):
        build_physiology_integrated_information_replays(n_bins=1)

    with pytest.raises(ValueError, match="n_samples"):
        build_physiology_integrated_information_replays(n_samples=31.0)
