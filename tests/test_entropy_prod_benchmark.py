# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Entropy production benchmark gate tests

"""Behavioural tests for ``benchmarks/entropy_prod_benchmark.py``."""

from __future__ import annotations

import json

from benchmarks.entropy_prod_benchmark import (
    bench_at,
    benchmark_entropy_production_polyglot_parity_gate,
)


def test_entropy_production_polyglot_parity_gate_reports_contracts() -> None:
    out = benchmark_entropy_production_polyglot_parity_gate(
        n=8,
        calls=1,
        seed=2026,
        alpha=0.5,
        dt=0.01,
    )
    records = json.loads(str(out["backend_records_json"]))
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))

    assert out["suite"] == "entropy_production_polyglot_parity_gate"
    assert out["backend_count"] == 5
    assert out["python_reference_present"] == 1
    assert out["all_available_passed"] == 1
    assert out["reference_contracts_passed"] == 1
    assert out["acceptance_passed"] == 1
    assert float(out["reference_rate"]) >= 0.0
    assert float(out["minimum_observed_rate"]) >= -1.0e-12
    assert float(out["manual_formula_abs_error"]) <= 1.0e-12
    assert float(out["fixed_point_abs_error"]) <= 1.0e-12
    assert float(out["zero_dt_abs_error"]) <= 1.0e-12
    assert float(out["dt_scaling_abs_error"]) <= 1.0e-12
    assert float(out["phase_shift_abs_error"]) <= 1.0e-12
    assert float(out["permutation_abs_error"]) <= 1.0e-12
    assert float(out["alpha_quadratic_abs_error"]) <= 1.0e-12
    assert len(str(out["reference_rate_sha256"])) == 64
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
        "production_timing_claim": False,
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_alpha_quadratic_scaling": True,
        "require_dt_linear_scaling": True,
        "require_exact_formula_parity": True,
        "require_fixed_point_zero": True,
        "require_non_negative_rate": True,
        "require_permutation_invariance": True,
        "require_phase_shift_invariance": True,
        "require_public_dispatch_parity": True,
        "require_python_reference": True,
    }

    for record in records:
        if record["status"] == "available":
            assert record["parity_passed"] is True
            assert record["public_dispatch_parity_passed"] is True
            assert record["contracts_passed"] is True
            assert record["non_negative_rate"] is True
            assert record["rate"] >= -1.0e-12
            assert record["rate_abs_error"] <= record["tolerance"]
            assert record["public_rate_abs_error"] <= record["tolerance"]
            assert record["rate_sha256"] is not None
            assert record["ms_per_call"] is not None
        else:
            assert record["status"] == "unavailable"
            assert record["unavailable_reason"]
            assert record["parity_passed"] is False


def test_entropy_production_polyglot_parity_gate_rejects_invalid_controls() -> None:
    invalid_kwargs = (
        {"n": True},
        {"n": 1},
        {"calls": 0},
        {"seed": -1},
        {"alpha": True},
        {"alpha": 0.0},
        {"dt": False},
        {"dt": 0.0},
    )
    for kwargs in invalid_kwargs:
        try:
            benchmark_entropy_production_polyglot_parity_gate(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid controls were accepted: {kwargs!r}")


def test_entropy_production_legacy_wallclock_row_keeps_evidence_boundary() -> None:
    row = bench_at(8, 1, seed=2026, alpha=0.5, dt=0.01)

    assert row["n"] == 8
    assert row["calls"] == 1
    assert row["seed"] == 2026
    assert row["alpha"] == 0.5
    assert row["dt"] == 0.01
    assert row["boundary_contract"] == "exact_overdamped_kuramoto_dissipation_validated"
    assert row["benchmark_evidence_kind"] == "local_regression_non_isolated"
    assert row["isolation_method"] == "none"
    assert row["production_timing_claim"] == 0
    assert "python" in row["available"]
    for backend in row["available"]:
        assert row[f"{backend}_ms_per_call"] > 0.0
