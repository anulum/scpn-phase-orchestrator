# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Embedding benchmark contract tests

from __future__ import annotations

import json

from benchmarks.embedding_benchmark import (
    BENCHMARK_EVIDENCE_KIND,
    bench_at,
    benchmark_embedding_polyglot_parity_gate,
)


def test_embedding_polyglot_parity_gate_reports_declared_backend_slots() -> None:
    out = benchmark_embedding_polyglot_parity_gate(n=96, calls=1, seed=2026)
    records = json.loads(str(out["backend_records_json"]))
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))

    assert out["suite"] == "embedding_polyglot_parity_gate"
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
    assert float(out["exact_indexing_max_abs_error"]) <= 1.0e-12
    assert float(out["time_shift_row_max_abs_error"]) <= 1.0e-12
    assert float(out["nearest_neighbor_line_distance_error"]) <= 1.0e-12
    assert int(out["nearest_neighbor_line_index_contract"]) == 1
    assert float(out["constant_signal_mi_abs_error"]) <= 1.0e-12
    assert int(out["zero_lag_mi_exceeds_distant_lag"]) == 1
    assert thresholds["backend_order"] == ["rust", "mojo", "julia", "go", "python"]
    assert thresholds["require_exact_delay_indexing"] is True
    assert thresholds["require_time_shift_row_consistency"] is True
    assert thresholds["require_constant_signal_zero_mutual_information"] is True
    assert thresholds["require_zero_lag_mi_exceeds_distant_lag"] is True
    assert thresholds["require_nearest_neighbor_self_exclusion"] is True
    assert thresholds["require_public_dispatch_parity"] is True
    assert thresholds["production_timing_claim"] is False

    assert [record["backend"] for record in records] == thresholds["backend_order"]
    for record in records:
        if record["status"] == "available":
            assert record["delay_parity_passed"] is True
            assert record["mi_parity_passed"] is True
            assert record["nn_parity_passed"] is True
            assert record["public_dispatch_parity_passed"] is True
            assert record["contracts_passed"] is True
            assert record["delay_max_abs_error"] <= thresholds["delay_tolerance"]
            assert record["delay_sha256"] is not None
            assert record["ms_per_call"] is not None
            if record["mi_supported"]:
                assert record["mi_abs_error"] <= thresholds["mi_tolerance"]
                assert record["mi_sha256"] is not None
            if record["nn_supported"]:
                assert record["nn_distance_max_abs_error"] <= thresholds["nn_tolerance"]
                assert record["nn_index_exact"] is True
                assert record["nn_distance_sha256"] is not None
                assert record["nn_index_sha256"] is not None
        else:
            assert record["status"] == "unavailable"
            assert record["unavailable_reason"]
            assert record["delay_parity_passed"] is False
            assert record["delay_sha256"] is None


def test_embedding_polyglot_parity_gate_rejects_invalid_controls() -> None:
    for kwargs in (
        {"n": True},
        {"n": 63},
        {"calls": 0},
        {"calls": False},
        {"seed": -1},
    ):
        try:
            benchmark_embedding_polyglot_parity_gate(**kwargs)
        except ValueError:
            continue
        raise AssertionError(
            f"invalid embedding benchmark controls accepted: {kwargs!r}"
        )


def test_embedding_wallclock_benchmark_is_labelled_non_isolated_regression() -> None:
    records = bench_at([64], calls=1, seed=2026)

    assert len(records) == 1
    record = records[0]
    assert record["suite"] == "embedding_wallclock_local"
    assert record["n"] == 64
    assert record["calls"] == 1
    assert record["benchmark_evidence_kind"] == BENCHMARK_EVIDENCE_KIND
    assert record["isolation_method"] == "none"
    assert record["production_timing_claim"] == 0
    assert float(record["seconds"]) > 0.0
    assert float(record["ms_per_call"]) > 0.0
    assert float(record["steps_per_second"]) > 0.0
