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
    _validate_replay_records,
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
        assert 0.0 <= record["normalised_phi"] <= 1.0
        assert 0.0 <= record["phi"] <= record["total_integration"]
        assert isinstance(record["expected_relationship"], str)

    assert by_name["cyber_recontainment"]["phi"] > by_name["cyber_disruption"]["phi"]
    assert by_name["spc_recovery"]["phi"] > by_name["spc_fragmentation"]["phi"]


def test_build_replays_rejects_invalid_parameters() -> None:
    """Invalid trajectory size or bins fail closed."""
    with pytest.raises(ValueError, match="n_samples"):
        build_cyber_industrial_integrated_information_replays(n_samples=31)

    with pytest.raises(ValueError, match="n_bins"):
        build_cyber_industrial_integrated_information_replays(n_bins=1)

    with pytest.raises(ValueError, match="n_bins"):
        build_cyber_industrial_integrated_information_replays(n_bins=True)

    with pytest.raises(ValueError, match="n_samples"):
        build_cyber_industrial_integrated_information_replays(n_samples=31.0)


def test_build_replays_accepts_numpy_integer_parameters() -> None:
    """NumPy integer aliases should preserve valid integer replay contracts."""
    records = build_cyber_industrial_integrated_information_replays(
        n_samples=np.int64(192),
        n_bins=np.int64(8),
    )

    assert len(records) == 4
    assert all(record["n_samples"] == 192 for record in records)
    assert all(record["n_bins"] == 8 for record in records)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("phi", -0.1, "phi"),
        ("phi", np.nan, "phi"),
        ("normalised_phi", 1.1, "normalised_phi"),
        ("normalised_phi", True, "normalised_phi"),
        ("total_integration", -0.1, "total_integration"),
    ],
)
def test_validate_records_rejects_invalid_metric_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    """Replay record metrics must remain finite and physically bounded."""
    record = dict(
        build_cyber_industrial_integrated_information_replays(n_samples=192)[0],
    )
    record[field] = value

    with pytest.raises(ValueError, match=match):
        _validate_replay_records((record, record))


@pytest.mark.parametrize(
    "partition",
    [
        [[0], [0, 1, 2, 3, 4, 5]],
        [[0], [1, 2]],
        [[0, 0], [1, 2, 3, 4, 5]],
        [[0], [1, 2, True]],
        [[0], []],
    ],
)
def test_validate_records_rejects_invalid_minimum_partition(
    partition: list[list[object]],
) -> None:
    """Replay records must carry a valid bipartition over all oscillators."""
    records = tuple(
        dict(record)
        for record in build_cyber_industrial_integrated_information_replays(
            n_samples=192,
        )
    )
    bad = dict(records[0])
    bad["minimum_partition"] = partition

    with pytest.raises(ValueError, match="minimum_partition"):
        _validate_replay_records((bad, *records[1:]))


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
