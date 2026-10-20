# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delayed Kuramoto benchmark contract tests

"""Verify numerical comparison records and benchmark control contracts."""

from __future__ import annotations

import json
from typing import TypedDict

from benchmarks.delay_benchmark import (
    BENCHMARK_EVIDENCE_KIND,
    bench_at,
    benchmark_delay_polyglot_parity_gate,
)


class _InvalidCounts(TypedDict, total=False):
    """Integer control keywords whose boundary values are intentionally invalid."""

    n: int
    delay_steps: int
    n_steps: int
    calls: int
    seed: int


def test_delay_polyglot_parity_gate_reports_declared_backend_slots() -> None:
    """Report all declared owners with truthful parity and contract outcomes."""
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
    wall_time = out["wall_time_s"]
    step_rate = out["steps_per_second"]
    rotation_error = out["pure_rotation_max_abs_error"]
    assert isinstance(wall_time, float) and wall_time > 0.0
    assert isinstance(step_rate, float) and step_rate > 0.0
    assert isinstance(rotation_error, float) and rotation_error <= 1e-9
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
    """Refuse invalid benchmark dimensions, counts and seeds."""
    invalid: tuple[_InvalidCounts, ...] = (
        {"n": True},
        {"n": 0},
        {"delay_steps": 0},
        {"delay_steps": True},
        {"n_steps": 0},
        {"calls": 0},
        {"calls": False},
        {"seed": -1},
    )
    for kwargs in invalid:
        try:
            benchmark_delay_polyglot_parity_gate(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid delay benchmark controls accepted: {kwargs!r}")


def test_delay_wallclock_benchmark_reports_per_backend_timings() -> None:
    """Report positive diagnostic timings only for actual measured owners."""
    row = bench_at(8, 2, 20, 1, seed=2026)
    assert row["n"] == 8
    assert row["delay_steps"] == 2
    assert row["n_steps"] == 20
    assert row["calls"] == 1
    available = row["available"]
    assert isinstance(available, list)
    for backend in available:
        assert isinstance(backend, str)
        milliseconds = row[f"{backend}_ms_per_call"]
        assert isinstance(milliseconds, float) and milliseconds >= 0.0
