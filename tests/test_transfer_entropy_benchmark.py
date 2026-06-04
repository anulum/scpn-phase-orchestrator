# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Transfer entropy benchmark gate tests

"""Behavioural tests for ``benchmarks/transfer_entropy_benchmark.py``."""

from __future__ import annotations

import json

from benchmarks.transfer_entropy_benchmark import (
    bench_at,
    benchmark_transfer_entropy_polyglot_parity_gate,
)


def test_transfer_entropy_polyglot_parity_gate_reports_contracts() -> None:
    out = benchmark_transfer_entropy_polyglot_parity_gate(
        n=64,
        calls=1,
        seed=2026,
        n_bins=16,
    )
    records = json.loads(str(out["backend_records_json"]))
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))

    assert out["suite"] == "transfer_entropy_polyglot_parity_gate"
    assert out["backend_count"] == 5
    assert out["python_reference_present"] == 1
    assert out["all_available_passed"] == 1
    assert out["reference_contracts_passed"] == 1
    assert out["acceptance_passed"] == 1
    assert float(out["reference_forward_te"]) > float(out["reference_reverse_te"])
    assert (
        float(out["reference_direction_margin"])
        > thresholds["min_causal_direction_margin"]
    )
    assert float(out["scalar_matrix_forward_error"]) <= 1.0e-12
    assert float(out["scalar_matrix_reverse_error"]) <= 1.0e-12
    assert float(out["diagonal_max_abs"]) <= 1.0e-12
    assert float(out["matrix_min_value"]) >= -1.0e-12
    assert float(out["matrix_max_entropy_excess"]) <= 1.0e-12
    assert float(out["phase_wrap_scalar_abs_error"]) <= 1.0e-12
    assert float(out["phase_wrap_matrix_max_abs_error"]) <= 1.0e-12
    assert float(out["short_series_abs_error"]) <= 1.0e-12
    assert len(str(out["reference_forward_te_sha256"])) == 64
    assert len(str(out["reference_reverse_te_sha256"])) == 64
    assert len(str(out["reference_matrix_sha256"])) == 64
    assert len(str(out["benchmark_sha256"])) == 64
    assert out["benchmark_evidence_kind"] == "local_regression_non_isolated"
    assert out["isolation_method"] == "none"
    assert out["production_timing_claim"] == 0
    assert float(out["steps_per_second"]) > 0.0
    assert [record["backend"] for record in records] == [
        "rust",
        "mojo",
        "julia",
        "go",
        "python",
    ]
    assert thresholds == {
        "backend_order": ["rust", "mojo", "julia", "go", "python"],
        "max_mojo_abs_error": 1.0e-9,
        "max_native_abs_error": 1.0e-12,
        "max_reference_contract_abs_error": 1.0e-12,
        "min_causal_direction_margin": 1.0e-2,
        "production_timing_claim": False,
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_causal_direction_preservation": True,
        "require_entropy_bound": True,
        "require_matrix_scalar_consistency": True,
        "require_phase_wrapping_invariance": True,
        "require_public_dispatch_parity": True,
        "require_python_reference": True,
        "require_zero_diagonal_matrix": True,
    }

    for record in records:
        if record["status"] == "available":
            assert record["parity_passed"] is True
            assert record["public_dispatch_parity_passed"] is True
            assert record["contracts_passed"] is True
            assert record["causal_direction_preserved"] is True
            assert record["forward_te"] > record["reverse_te"]
            assert record["scalar_forward_abs_error"] <= record["tolerance"]
            assert record["scalar_reverse_abs_error"] <= record["tolerance"]
            assert record["matrix_max_abs_error"] <= record["matrix_tolerance"]
            assert record["forward_te_sha256"] is not None
            assert record["reverse_te_sha256"] is not None
            assert record["matrix_sha256"] is not None
            assert record["ms_per_call"] is not None
        else:
            assert record["status"] == "unavailable"
            assert record["unavailable_reason"]
            assert record["parity_passed"] is False


def test_transfer_entropy_polyglot_parity_gate_rejects_invalid_controls() -> None:
    invalid_kwargs = (
        {"n": True},
        {"n": 47},
        {"calls": 0},
        {"seed": -1},
        {"n_bins": True},
        {"n_bins": 1},
    )
    for kwargs in invalid_kwargs:
        try:
            benchmark_transfer_entropy_polyglot_parity_gate(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid controls were accepted: {kwargs!r}")


def test_transfer_entropy_legacy_wallclock_row_keeps_evidence_boundary() -> None:
    row = bench_at(64, 1, seed=2026, n_bins=16)

    assert row["n"] == 64
    assert row["calls"] == 1
    assert row["seed"] == 2026
    assert row["n_bins"] == 16
    assert row["boundary_contract"] == "exact_numpy_histogram_estimator_validated"
    assert row["benchmark_evidence_kind"] == "local_regression_non_isolated"
    assert row["isolation_method"] == "none"
    assert row["production_timing_claim"] == 0
    assert "python" in row["available"]
    for backend in row["available"]:
        assert row[f"{backend}_ms_per_call"] > 0.0
