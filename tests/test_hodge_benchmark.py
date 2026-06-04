# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge benchmark contract tests

from __future__ import annotations

import json

from benchmarks.hodge_benchmark import (
    BENCHMARK_EVIDENCE_KIND,
    bench_at,
    benchmark_hodge_polyglot_parity_gate,
)


def test_hodge_polyglot_parity_gate_reports_declared_backend_slots() -> None:
    out = benchmark_hodge_polyglot_parity_gate(n=8, calls=1, seed=2026)
    records = json.loads(str(out["backend_records_json"]))
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))

    assert out["suite"] == "hodge_polyglot_parity_gate"
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
    assert float(out["reconstruction_max_abs_error"]) <= 1.0e-10
    assert float(out["harmonic_max_abs_error"]) <= 1.0e-10
    assert float(out["phase_shift_max_abs_error"]) <= 1.0e-10
    assert float(out["symmetric_curl_max_abs_error"]) <= 1.0e-10
    assert float(out["antisymmetric_gradient_max_abs_error"]) <= 1.0e-10
    assert float(out["antisymmetric_curl_sum_abs_error"]) <= 1.0e-10
    assert float(out["two_node_curl_max_abs_error"]) <= 1.0e-10
    assert float(out["scale_covariance_max_abs_error"]) <= 1.0e-10
    assert thresholds["backend_order"] == ["rust", "mojo", "julia", "go", "python"]
    assert thresholds["require_reconstruction_contract"] is True
    assert thresholds["require_global_phase_shift_invariance"] is True
    assert thresholds["require_symmetric_zero_curl"] is True
    assert thresholds["require_antisymmetric_zero_gradient"] is True
    assert thresholds["require_two_node_antisymmetric_closed_form"] is True
    assert thresholds["require_scale_covariance"] is True
    assert thresholds["production_timing_claim"] is False

    assert [record["backend"] for record in records] == thresholds["backend_order"]
    for record in records:
        if record["status"] == "available":
            assert record["parity_passed"] is True
            assert record["reference_contracts_passed"] is True
            assert record["max_abs_error"] <= record["tolerance"]
            assert record["gradient_sha256"] is not None
            assert record["curl_sha256"] is not None
            assert record["harmonic_sha256"] is not None
            assert record["bundle_sha256"] is not None
            assert record["ms_per_call"] is not None
        else:
            assert record["status"] == "unavailable"
            assert record["unavailable_reason"]
            assert record["parity_passed"] is False
            assert record["gradient_sha256"] is None


def test_hodge_polyglot_parity_gate_rejects_invalid_controls() -> None:
    for kwargs in (
        {"n": True},
        {"n": 1},
        {"calls": 0},
        {"calls": False},
        {"seed": -1},
    ):
        try:
            benchmark_hodge_polyglot_parity_gate(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid Hodge benchmark controls accepted: {kwargs!r}")


def test_hodge_wallclock_benchmark_is_labelled_non_isolated_regression_evidence() -> (
    None
):
    records = bench_at([4], calls=1, seed=2026)

    assert len(records) == 1
    record = records[0]
    assert record["suite"] == "hodge_wallclock_local"
    assert record["n"] == 4
    assert record["calls"] == 1
    assert record["benchmark_evidence_kind"] == BENCHMARK_EVIDENCE_KIND
    assert record["isolation_method"] == "none"
    assert record["production_timing_claim"] == 0
    assert float(record["seconds"]) > 0.0
    assert float(record["ms_per_call"]) > 0.0
    assert float(record["steps_per_second"]) > 0.0
