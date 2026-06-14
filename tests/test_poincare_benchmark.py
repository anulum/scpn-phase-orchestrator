# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Poincaré benchmark contract tests

from __future__ import annotations

import json

from benchmarks.poincare_benchmark import (
    BENCHMARK_EVIDENCE_KIND,
    bench_at,
    benchmark_poincare_polyglot_parity_gate,
)


def test_poincare_polyglot_parity_gate_reports_declared_backend_slots() -> None:
    out = benchmark_poincare_polyglot_parity_gate(
        n_steps=240, n_osc=4, calls=1, seed=2026
    )
    records = json.loads(str(out["backend_records_json"]))
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))

    assert out["suite"] == "poincare_polyglot_parity_gate"
    assert out["backend_count"] == 5
    assert out["python_reference_present"] == 1
    assert out["all_available_passed"] == 1
    assert out["reference_contracts_passed"] == 1
    assert out["acceptance_passed"] == 1
    assert out["benchmark_evidence_kind"] == BENCHMARK_EVIDENCE_KIND
    assert out["isolation_method"] == "none"
    assert out["production_timing_claim"] == 0
    assert len(str(out["benchmark_sha256"])) == 64
    assert float(out["wall_time_s"]) > 0.0
    assert float(out["steps_per_second"]) > 0.0
    # Geometric contracts: section crossings on the plane, phase crossings
    # recover the section phase exactly (the wrong interpolation gave ~0.09).
    assert float(out["section_plane_residual"]) <= 1.0e-9
    assert float(out["phase_value_residual"]) <= 1.0e-9
    assert int(out["reference_section_count"]) > 0
    assert int(out["reference_phase_count"]) > 0
    assert thresholds["backend_order"] == ["rust", "mojo", "julia", "go", "python"]
    assert thresholds["require_section_on_plane"] is True
    assert thresholds["require_phase_recovers_section"] is True
    assert thresholds["require_strictly_increasing_times"] is True
    assert thresholds["require_nonzero_crossings"] is True
    assert thresholds["require_matching_crossing_counts"] is True
    assert thresholds["production_timing_claim"] is False

    assert [record["backend"] for record in records] == thresholds["backend_order"]
    for record in records:
        if record["status"] == "available":
            assert record["parity_passed"] is True
            assert record["reference_contracts_passed"] is True
            assert record["count_match"] is True
            assert record["max_abs_error"] <= record["tolerance"]
            assert record["section_crossings_sha256"] is not None
            assert record["phase_crossings_sha256"] is not None
            assert record["ms_per_call"] is not None
        else:
            assert record["status"] == "unavailable"
            assert record["unavailable_reason"]
            assert record["parity_passed"] is False
            assert record["section_crossings_sha256"] is None


def test_poincare_polyglot_parity_gate_rejects_invalid_controls() -> None:
    for kwargs in (
        {"n_steps": True},
        {"n_steps": 4},
        {"n_osc": 1},
        {"calls": 0},
        {"calls": False},
        {"seed": -1},
    ):
        try:
            benchmark_poincare_polyglot_parity_gate(**kwargs)
        except ValueError:
            continue
        raise AssertionError(
            f"invalid Poincaré benchmark controls accepted: {kwargs!r}"
        )


def test_poincare_wallclock_benchmark_reports_per_backend_timings() -> None:
    row = bench_at(64, 3, 1, seed=2026)
    assert row["n_steps"] == 64
    assert row["n_osc"] == 3
    assert row["calls"] == 1
    for backend in row["available"]:
        assert float(row[f"{backend}_ms_per_call"]) >= 0.0
