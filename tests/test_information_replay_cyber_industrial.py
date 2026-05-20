# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cyber-industrial replay monitor tests

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.information_replay_cyber_industrial import (
    build_cyber_industrial_integrated_information_replays,
)


def test_replays_are_deterministic() -> None:
    """The function must produce identical records for identical inputs."""
    first = build_cyber_industrial_integrated_information_replays(
        n_samples=192,
        n_bins=9,
    )
    second = build_cyber_industrial_integrated_information_replays(
        n_samples=192,
        n_bins=9,
    )

    assert first == second


def test_replay_records_are_json_safe_and_non_actuating() -> None:
    """Records are serializable and encode the audit boundary as a proxy-only claim."""
    records = build_cyber_industrial_integrated_information_replays(
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


def test_corpus_contains_minimum_required_cases_and_fields() -> None:
    """Corpus must include both disruption and recovery phases for both domains."""
    records = build_cyber_industrial_integrated_information_replays()
    names = [record["case_name"] for record in records]
    by_name = {record["case_name"]: record for record in records}

    assert len(records) >= 2
    assert set(names) >= {
        "cyber_disruption",
        "cyber_recontainment",
        "spc_fragmentation",
        "spc_recovery",
    }

    for record in records:
        assert record["domain"] == "cyber_industrial"
        assert record["n_samples"] == 256
        assert isinstance(record["n_oscillators"], int)
        assert record["n_oscillators"] >= 2
        assert record["n_bins"] == 8
        assert isinstance(record["phi"], float)
        assert isinstance(record["normalised_phi"], float)
        assert isinstance(record["total_integration"], float)
        assert isinstance(record["expected_relationship"], str)

    assert by_name["cyber_recontainment"]["phi"] > by_name["cyber_disruption"]["phi"]
    assert by_name["spc_recovery"]["phi"] > by_name["spc_fragmentation"]["phi"]


def test_build_replays_rejects_invalid_parameters() -> None:
    """Invalid trajectory size or bins fail closed."""
    with pytest.raises(ValueError, match="n_samples"):
        build_cyber_industrial_integrated_information_replays(n_samples=31)

    with pytest.raises(ValueError, match="n_bins"):
        build_cyber_industrial_integrated_information_replays(n_bins=1)

    with pytest.raises(ValueError, match="n_samples"):
        build_cyber_industrial_integrated_information_replays(n_samples=31.0)


def test_expected_relationship_is_reflective_of_phi_ordering() -> None:
    """Expected relationship strings must align with measured integration ordering."""
    records = build_cyber_industrial_integrated_information_replays(
        n_samples=192,
        n_bins=10,
    )
    by_name = {record["case_name"]: record for record in records}

    assert "recontainment" in by_name["cyber_recontainment"]["expected_relationship"]
    assert by_name["cyber_recontainment"]["phi"] > by_name["cyber_disruption"]["phi"]
    assert (
        "spc_recovery > spc_fragmentation"
        in by_name["spc_recovery"]["expected_relationship"]
    )
    assert by_name["spc_recovery"]["phi"] > by_name["spc_fragmentation"]["phi"]
    assert np.isfinite(by_name["cyber_recontainment"]["phi"])
    assert np.isfinite(by_name["spc_recovery"]["phi"])
