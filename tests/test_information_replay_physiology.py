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
    _validate_replay_records,
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

    with pytest.raises(ValueError, match="n_bins"):
        build_physiology_integrated_information_replays(n_bins=True)

    with pytest.raises(ValueError, match="n_samples"):
        build_physiology_integrated_information_replays(n_samples=np.bool_(True))

    with pytest.raises(ValueError, match="n_bins"):
        build_physiology_integrated_information_replays(n_bins=np.bool_(True))


def test_build_replays_accepts_numpy_integer_parameters() -> None:
    records = build_physiology_integrated_information_replays(
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
        ("normalised_phi", np.bool_(True), "normalised_phi"),
        ("normalised_phi", 0.5 + 0.0j, "normalised_phi"),
        (
            "normalised_phi",
            np.asarray(complex(0.5, 0.0), dtype=object),
            "real-valued",
        ),
        ("total_integration", -0.1, "total_integration"),
        (
            "total_integration",
            np.asarray(complex(0.5, 0.0), dtype=object),
            "real-valued",
        ),
        ("n_oscillators", True, "n_oscillators"),
        ("n_samples", np.bool_(True), "n_samples"),
        ("n_bins", np.bool_(True), "n_bins"),
    ],
)
def test_validate_records_rejects_invalid_metric_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    record = dict(build_physiology_integrated_information_replays(n_samples=192)[0])
    record[field] = value

    with pytest.raises(ValueError, match=match):
        _validate_replay_records((record, record))


@pytest.mark.parametrize(
    "partition",
    [
        [[0], [0, 1, 2, 3]],
        [[0], [1, 2]],
        [[0, 0], [1, 2, 3]],
        [[0], [1, 2, True]],
        [[0], [1, 2, np.bool_(True)]],
        [[0], [1, 2, complex(3.0, 0.0)]],
        [[0], []],
    ],
)
def test_validate_records_rejects_invalid_minimum_partition(
    partition: list[list[object]],
) -> None:
    records = tuple(
        dict(record)
        for record in build_physiology_integrated_information_replays(n_samples=192)
    )
    bad = dict(records[0])
    bad["minimum_partition"] = partition

    with pytest.raises(ValueError, match="minimum_partition"):
        _validate_replay_records((bad, *records[1:]))


def test_validate_records_rejects_object_complex_partition_as_real_integer() -> None:
    records = tuple(
        dict(record)
        for record in build_physiology_integrated_information_replays(n_samples=192)
    )
    bad = dict(records[0])
    bad["minimum_partition"] = [[0], [1, 2, complex(3.0, 0.0)]]

    with pytest.raises(ValueError, match="real integer"):
        _validate_replay_records((bad, *records[1:]))


def _valid_records() -> list[dict[str, object]]:
    return [
        dict(record)
        for record in build_physiology_integrated_information_replays(n_samples=192)
    ]


def _by_name() -> dict[str, dict[str, object]]:
    return {str(record["case_name"]): record for record in _valid_records()}


def test_validate_records_rejects_single_record_corpus() -> None:
    with pytest.raises(ValueError, match="at least two records"):
        _validate_replay_records((_valid_records()[0],))


def test_validate_records_rejects_missing_required_field() -> None:
    records = _valid_records()
    del records[0]["phi"]
    with pytest.raises(ValueError, match="missing fields"):
        _validate_replay_records(tuple(records))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("domain", "neurology", "domain=physiology"),
        ("claim_boundary", "theoretical_iit", "invalid claim boundary"),
        ("non_actuating", False, "must be non-actuating"),
        ("minimum_partition", "not-a-list", "minimum_partition must be a list pair"),
        ("n_oscillators", 1, "n_oscillators must be an integer >= 2"),
        ("n_oscillators", complex(2.0, 0.0), "n_oscillators must be a real integer"),
    ],
)
def test_validate_records_rejects_record_field(field, value, match) -> None:
    records = _valid_records()
    records[0][field] = value
    with pytest.raises(ValueError, match=match):
        _validate_replay_records(tuple(records))


def test_validate_records_rejects_phi_exceeding_total_integration() -> None:
    records = _valid_records()
    records[0]["phi"] = 10.0
    records[0]["total_integration"] = 1.0
    with pytest.raises(ValueError, match="phi must not exceed total_integration"):
        _validate_replay_records(tuple(records))


def test_validate_records_rejects_negative_partition_index() -> None:
    records = _valid_records()
    records[0]["minimum_partition"] = [[0], [-1]]
    with pytest.raises(ValueError, match="non-negative indices"):
        _validate_replay_records(tuple(records))


def test_validate_records_rejects_incomplete_case_corpus() -> None:
    by_name = _by_name()
    with pytest.raises(ValueError, match="missing replay cases"):
        _validate_replay_records(
            (
                by_name["cardiac_respiratory_lock"],
                by_name["cardiac_respiratory_recovery"],
            )
        )


def test_validate_records_rejects_cardiac_lock_below_recovery() -> None:
    by_name = _by_name()
    by_name["cardiac_respiratory_lock"]["phi"] = 0.0
    with pytest.raises(ValueError, match="cardiac-respiratory lock above recovery"):
        _validate_replay_records(tuple(by_name.values()))


def test_validate_records_rejects_sleep_spindle_below_baseline() -> None:
    by_name = _by_name()
    by_name["eeg_sleep_spindle"]["phi"] = 0.0
    with pytest.raises(ValueError, match="sleep-spindle phase coupling above baseline"):
        _validate_replay_records(tuple(by_name.values()))
