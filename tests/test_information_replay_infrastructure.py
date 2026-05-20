# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Infrastructure replay monitor tests

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.information_replay_infrastructure import (
    build_infrastructure_integrated_information_replays,
)


def test_replays_are_deterministic() -> None:
    """The infrastructure replay corpus must be stable for identical inputs."""
    first = build_infrastructure_integrated_information_replays(
        n_samples=192,
        n_bins=9,
    )
    second = build_infrastructure_integrated_information_replays(
        n_samples=192,
        n_bins=9,
    )

    assert first == second


def test_replay_records_are_json_safe_and_non_actuating() -> None:
    """Records must remain JSON-safe and explicitly non-actuating."""
    records = build_infrastructure_integrated_information_replays(
        n_samples=192,
        n_bins=8,
    )
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


def test_corpus_contains_minimum_required_size_and_fields() -> None:
    """Corpus should include infrastructure cases with required audit fields."""
    records = build_infrastructure_integrated_information_replays()

    assert len(records) >= 2
    by_name = {record["case_name"]: record for record in records}
    required_names = {
        "power_grid_islanding",
        "power_grid_resynchronisation",
        "traffic_spillback_fragmentation",
        "traffic_platoon_recovery",
    }
    assert required_names.issubset(by_name.keys())

    for record in records:
        assert record["domain"] == "infrastructure"
        assert record["n_samples"] == 256
        assert isinstance(record["n_oscillators"], int)
        assert record["n_oscillators"] >= 2
        assert isinstance(record["n_bins"], int)
        assert record["n_bins"] == 8
        assert isinstance(record["phi"], float)
        assert record["normalised_phi"] >= 0.0
        assert record["total_integration"] >= 0.0
        assert record["claim_boundary"] == "engineering_proxy_not_theoretical_iit"
        assert record["non_actuating"] is True


def test_build_replays_reject_invalid_parameters() -> None:
    """Invalid corpus size or bin settings should fail closed."""
    with pytest.raises(ValueError, match="n_samples"):
        build_infrastructure_integrated_information_replays(n_samples=31)
    with pytest.raises(ValueError, match="n_bins"):
        build_infrastructure_integrated_information_replays(n_bins=1)
    with pytest.raises(ValueError, match="n_samples"):
        build_infrastructure_integrated_information_replays(n_samples=31.0)


def test_recovered_infrastructure_cases_exceed_fragmented_base() -> None:
    """Recovered/re-synchronised cases should score above fragmented/islanding cases."""
    records = build_infrastructure_integrated_information_replays(
        n_samples=192,
        n_bins=10,
    )
    by_name = {record["case_name"]: record for record in records}

    assert by_name["power_grid_resynchronisation"]["phi"] > by_name[
        "power_grid_islanding"
    ]["phi"]
    assert by_name["traffic_platoon_recovery"]["phi"] > by_name[
        "traffic_spillback_fragmentation"
    ]["phi"]
    assert np.isfinite(by_name["power_grid_resynchronisation"]["phi"])
    assert np.isfinite(by_name["traffic_platoon_recovery"]["phi"])
    assert "power_grid_resynchronisation > power_grid_islanding" in by_name[
        "power_grid_resynchronisation"
    ]["expected_relationship"]
    assert "traffic_platoon_recovery > traffic_spillback_fragmentation" in by_name[
        "traffic_platoon_recovery"
    ]["expected_relationship"]
