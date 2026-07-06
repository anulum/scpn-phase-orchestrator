# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — cross-system external-validation benchmark tests

"""Tests for the cross-system eigenvalue-ground-truth benchmark's pure core.

The correlation, the per-system record, the verdict, and the payload seal are exercised
on synthetic sweeps with planted correlations; the ANDES sweep generation is pragma-
excluded, so only the tested pure functions are measured.
"""

from __future__ import annotations

import pytest

from bench.grid_eigenvalue_external_validation import (
    correlation,
    external_validation_payload,
    external_validation_verdict,
    system_record,
)

_TRUE = [1.0, 2.0, 3.0, 4.0, 5.0]


def _system(name: str, *, focal: list[float], mean: list[float]) -> dict:
    return system_record(
        name=name,
        case=f"{name}/case.xlsx",
        loads=[0.8, 0.9, 1.0, 1.1, 1.2],
        true_sigma=_TRUE,
        detector_sigma={"focal": focal, "mean": mean, "spatial": mean},
    )


# --------------------------------------------------------------------------- #
# correlation                                                                 #
# --------------------------------------------------------------------------- #


def test_correlation_recovers_a_perfect_positive_trend() -> None:
    record = correlation(_TRUE, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert record["pearson"] == pytest.approx(1.0)
    assert record["spearman"] == pytest.approx(1.0)
    assert record["n"] == 5


def test_correlation_recovers_a_perfect_negative_trend() -> None:
    record = correlation(_TRUE, [5.0, 4.0, 3.0, 2.0, 1.0])
    assert record["pearson"] == pytest.approx(-1.0)
    assert record["spearman"] == pytest.approx(-1.0)


def test_correlation_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        correlation(_TRUE, [1.0, 2.0])


def test_correlation_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least 3 points"):
        correlation([1.0, 2.0], [1.0, 2.0])


# --------------------------------------------------------------------------- #
# system_record                                                               #
# --------------------------------------------------------------------------- #


def test_system_record_carries_a_correlation_per_aggregation() -> None:
    record = _system("ieee39", focal=[5, 4, 3, 2, 1], mean=[1, 2, 3, 4, 5])
    assert record["name"] == "ieee39"
    assert record["n"] == 5
    aggs = record["aggregations"]
    assert aggs["mean"]["correlation"]["spearman"] == pytest.approx(1.0)
    assert aggs["focal"]["correlation"]["spearman"] == pytest.approx(-1.0)


def test_system_record_rejects_a_missing_aggregation() -> None:
    with pytest.raises(ValueError, match="missing aggregation 'spatial'"):
        system_record(
            name="x",
            case="x.xlsx",
            loads=_TRUE,
            true_sigma=_TRUE,
            detector_sigma={"focal": _TRUE, "mean": _TRUE},
        )


# --------------------------------------------------------------------------- #
# verdict + payload                                                           #
# --------------------------------------------------------------------------- #


def test_verdict_reports_generalisation_and_focal_non_transfer() -> None:
    systems = [
        _system("ieee39", focal=[5, 4, 3, 2, 1], mean=[1, 2, 3, 4, 5]),
        _system("kundur", focal=[5, 4, 3, 2, 1], mean=[1, 2, 3, 4, 5]),
    ]
    verdict = external_validation_verdict(systems)
    assert "generalises" in verdict
    assert "does not transfer" in verdict


def test_verdict_flags_uneven_focal_transfer() -> None:
    # one system's focal correlates positively -> focal does not uniformly fail
    systems = [
        _system("a", focal=[1, 2, 3, 4, 5], mean=[1, 2, 3, 4, 5]),
        _system("b", focal=[5, 4, 3, 2, 1], mean=[1, 2, 3, 4, 5]),
    ]
    assert "transfers unevenly" in external_validation_verdict(systems)


def test_verdict_reports_non_generalisation() -> None:
    # a system where both coherent aggregations anti-correlate -> does not generalise
    systems = [_system("a", focal=[1, 2, 3, 4, 5], mean=[5, 4, 3, 2, 1])]
    assert "does not generalise" in external_validation_verdict(systems)


def test_payload_seals_a_reproducible_hash() -> None:
    systems = [_system("ieee39", focal=[5, 4, 3, 2, 1], mean=[1, 2, 3, 4, 5])]
    payload = external_validation_payload(systems=systems, andes_version="2.0.0")
    assert payload["benchmark"] == "grid_eigenvalue_external_validation"
    assert payload["andes_version"] == "2.0.0"

    from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

    sealed = dict(payload)
    stored = sealed.pop("content_hash")
    assert stored == canonical_record_hash(sealed)
