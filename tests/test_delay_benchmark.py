# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delayed Kuramoto benchmark contract tests

from __future__ import annotations

import json

from benchmarks.delay_benchmark import (
    BENCHMARK_EVIDENCE_KIND,
    bench_at,
    benchmark_delay_polyglot_parity_gate,
)


def test_delay_polyglot_parity_gate_reports_declared_backend_slots() -> None:
    out = benchmark_delay_polyglot_parity_gate(
        n=16, delay_steps=3, n_steps=80, calls=1, seed=2026
    )
    records = json.loads(str(out["backend_records_json"]))
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))

    assert out["suite"] == "delay_polyglot_parity_gate"
    assert out["backend_count"] == 5
    assert out["python_reference_present"] == 1
    assert out["all_available_passed"] == 1
    assert out["reference_contracts_passed"] == 1
    assert out["acceptance_passed"] == 1
    assert out["benchmark_evidence_kind"] == BENCHMARK_EVIDENCE_KIND
    assert out["isolation_method"] == "none"
    assert out["production_timing_claim"] == 0
    assert len(str(out["benchmark_sha256"])) == 64
    assert len(str(out["reference_phases_sha256"])) == 64
    assert float(out["wall_time_s"]) > 0.0
    assert float(out["steps_per_second"]) > 0.0
    assert float(out["pure_rotation_max_abs_error"]) <= 1.0e-9
    assert thresholds["backend_order"] == ["rust", "mojo", "julia", "go", "python"]
    assert thresholds["require_all_available_parity"] is True
    assert thresholds["require_pure_rotation_limit"] is True
    assert thresholds["require_phases_in_range"] is True
    assert thresholds["production_timing_claim"] is False

    assert [record["backend"] for record in records] == thresholds["backend_order"]
    for record in records:
        if record["status"] == "available":
            assert record["parity_passed"] is True
            assert record["reference_contracts_passed"] is True
            assert record["max_abs_error"] <= record["tolerance"]
        else:
            assert record["status"] == "unavailable"
            assert record["unavailable_reason"]
            assert record["parity_passed"] is False


def test_delay_polyglot_parity_gate_rejects_invalid_controls() -> None:
    for kwargs in (
        {"n": True},
        {"n": 0},
        {"delay_steps": 0},
        {"delay_steps": True},
        {"n_steps": 0},
        {"calls": 0},
        {"calls": False},
        {"seed": -1},
    ):
        try:
            benchmark_delay_polyglot_parity_gate(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid delay benchmark controls accepted: {kwargs!r}")


def test_delay_wallclock_benchmark_reports_per_backend_timings() -> None:
    row = bench_at(8, 2, 20, 1, seed=2026)
    assert row["n"] == 8
    assert row["delay_steps"] == 2
    assert row["n_steps"] == 20
    assert row["calls"] == 1
    for backend in row["available"]:
        assert float(row[f"{backend}_ms_per_call"]) >= 0.0
