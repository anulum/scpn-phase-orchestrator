# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Recurrence benchmark contract tests

"""Benchmark contract tests for the recurrence polyglot parity gate."""

from __future__ import annotations

import json

from benchmarks.recurrence_benchmark import (
    BACKEND_ORDER,
    benchmark_recurrence_polyglot_parity_gate,
)


def test_polyglot_parity_gate_records_every_declared_backend() -> None:
    out = benchmark_recurrence_polyglot_parity_gate(
        t=24,
        d=3,
        epsilon=0.8,
        calls=1,
        seed=19,
    )
    records = json.loads(str(out["backend_records_json"]))

    assert out["suite"] == "recurrence_polyglot_parity_gate"
    assert out["backend_count"] == len(BACKEND_ORDER)
    assert out["python_reference_present"] == 1
    assert out["all_available_passed"] == 1
    assert out["acceptance_passed"] == 1
    assert out["parity_pass_count"] == out["available_backend_count"]
    assert 0.0 <= float(out["reference_recurrence_rate"]) <= 1.0
    assert 0.0 <= float(out["reference_determinism"]) <= 1.0
    assert 0.0 <= float(out["reference_laminarity"]) <= 1.0
    assert [record["backend"] for record in records] == list(BACKEND_ORDER)

    python_record = next(record for record in records if record["backend"] == "python")
    assert python_record["status"] == "available"
    assert python_record["mismatch_count"] == 0
    assert python_record["self_cross_mismatch_count"] == 0
    assert python_record["max_abs_error"] == 0
    assert python_record["recurrence_invariants_passed"] is True
    assert python_record["self_cross_equals_recurrence"] is True
    assert python_record["parity_passed"] is True

    for record in records:
        if record["status"] == "available":
            assert record["ms_per_call"] is not None
            assert record["recurrence_sha256"] is not None
            assert record["cross_recurrence_sha256"] is not None
            assert record["self_cross_sha256"] is not None
            assert record["mismatch_count"] == 0
            assert record["self_cross_mismatch_count"] == 0
            assert record["max_abs_error"] == 0
            assert record["recurrence_invariants_passed"] is True
            assert record["self_cross_equals_recurrence"] is True
            continue
        assert record["status"] == "unavailable"
        assert record["unavailable_reason"]
        assert record["recurrence_sha256"] is None
        assert record["cross_recurrence_sha256"] is None
        assert record["self_cross_sha256"] is None
        assert record["parity_passed"] is False


def test_polyglot_parity_gate_rejects_invalid_controls() -> None:
    for kwargs in (
        {"t": True},
        {"t": 1},
        {"d": 0},
        {"epsilon": True},
        {"epsilon": -0.1},
        {"calls": 0},
        {"seed": -1},
    ):
        try:
            benchmark_recurrence_polyglot_parity_gate(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid controls were accepted: {kwargs!r}")
