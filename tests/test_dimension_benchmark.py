# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Dimension benchmark contract tests

"""Benchmark contract tests for the fractal-dimension polyglot parity gate."""

from __future__ import annotations

import json

from benchmarks.dimension_benchmark import (
    BACKEND_ORDER,
    benchmark_dimension_polyglot_parity_gate,
)


def test_polyglot_parity_gate_records_every_declared_backend() -> None:
    out = benchmark_dimension_polyglot_parity_gate(
        t=24,
        d=3,
        n_k=8,
        calls=1,
        seed=19,
    )
    records = json.loads(str(out["backend_records_json"]))

    assert out["suite"] == "dimension_polyglot_parity_gate"
    assert out["backend_count"] == len(BACKEND_ORDER)
    assert out["python_reference_present"] == 1
    assert out["all_available_passed"] == 1
    assert out["acceptance_passed"] == 1
    assert out["parity_pass_count"] == out["available_backend_count"]
    assert 0.0 <= float(out["reference_ci_min"]) <= 1.0
    assert 0.0 <= float(out["reference_ci_max"]) <= 1.0
    assert 0.0 <= float(out["reference_kaplan_yorke"]) <= 5.0
    assert [record["backend"] for record in records] == list(BACKEND_ORDER)

    python_record = next(record for record in records if record["backend"] == "python")
    assert python_record["status"] == "available"
    assert python_record["ci_max_abs_error"] == 0.0
    assert python_record["ky_abs_error"] == 0.0
    assert python_record["max_abs_error"] == 0.0
    assert python_record["ci_monotonic"] is True
    assert python_record["ci_unit_interval"] is True
    assert python_record["ky_bounds_passed"] is True
    assert python_record["parity_passed"] is True

    for record in records:
        if record["status"] == "available":
            assert record["ms_per_call"] is not None
            assert record["correlation_integral_sha256"] is not None
            assert record["kaplan_yorke_sha256"] is not None
            assert record["ci_max_abs_error"] <= record["tolerance"]
            assert record["ky_abs_error"] <= record["tolerance"]
            assert record["max_abs_error"] <= record["tolerance"]
            assert record["ci_monotonic"] is True
            assert record["ci_unit_interval"] is True
            assert record["ky_bounds_passed"] is True
            continue
        assert record["status"] == "unavailable"
        assert record["unavailable_reason"]
        assert record["correlation_integral_sha256"] is None
        assert record["kaplan_yorke_sha256"] is None
        assert record["parity_passed"] is False


def test_polyglot_parity_gate_rejects_invalid_controls() -> None:
    for kwargs in (
        {"t": True},
        {"t": 1},
        {"d": 0},
        {"n_k": 1},
        {"calls": 0},
        {"seed": -1},
    ):
        try:
            benchmark_dimension_polyglot_parity_gate(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid controls were accepted: {kwargs!r}")
