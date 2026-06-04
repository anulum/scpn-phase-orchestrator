# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spectral benchmark contract tests

"""Benchmark contract tests for the spectral polyglot parity gate."""

from __future__ import annotations

import json

import pytest

from benchmarks.spectral_benchmark import benchmark_spectral_polyglot_parity_gate


def test_polyglot_parity_gate_records_every_declared_backend() -> None:
    out = benchmark_spectral_polyglot_parity_gate(n=8, calls=1, seed=19)
    records = json.loads(str(out["backend_records_json"]))
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))

    assert out["suite"] == "spectral_polyglot_parity_gate"
    assert out["backend_count"] == 5
    assert out["python_reference_present"] == 1
    assert out["all_available_passed"] == 1
    assert out["acceptance_passed"] == 1
    assert out["laplacian_contracts_passed"] == 1
    assert float(out["reference_fiedler_value"]) > 0.0
    assert float(out["reference_spectral_gap"]) >= 0.0
    assert float(out["laplacian_row_sum_error"]) <= 1.0e-12
    assert float(out["laplacian_symmetry_error"]) <= 1.0e-12
    assert float(out["laplacian_psd_floor"]) >= -1.0e-10
    assert float(out["laplacian_zero_mode_abs"]) <= 1.0e-10
    assert float(out["fiedler_orthogonality_abs"]) <= 1.0e-8
    assert float(out["uniform_path_abs_error"]) <= 1.0e-10
    assert float(out["complete_graph_abs_error"]) <= 1.0e-10
    assert abs(float(out["complete_graph_spectral_gap"])) <= 1.0e-10
    assert len(str(out["reference_fiedler_vector_sha256"])) == 64
    assert len(str(out["benchmark_sha256"])) == 64
    assert thresholds["backend_order"] == [
        "rust",
        "mojo",
        "julia",
        "go",
        "python",
    ]
    assert thresholds["require_laplacian_psd_row_sum_contract"] is True
    assert thresholds["require_uniform_path_exact_lambda2"] is True
    assert thresholds["require_complete_graph_exact_lambda2"] is True

    python_record = next(record for record in records if record["backend"] == "python")
    assert python_record["status"] == "available"
    assert python_record["max_abs_error"] == 0.0
    assert python_record["fiedler_value_abs_error"] == 0.0
    assert python_record["spectral_gap_abs_error"] == 0.0
    assert python_record["fiedler_direction_abs_error"] == 0.0
    assert python_record["parity_passed"] is True

    for record in records:
        if record["status"] == "available":
            assert record["ms_per_call"] is not None
            assert record["fiedler_vector_sha256"] is not None
            assert record["fiedler_value_sha256"] is not None
            assert record["spectral_gap_sha256"] is not None
            assert record["max_abs_error"] <= record["tolerance"]
        else:
            assert record["unavailable_reason"]
            assert record["fiedler_vector_sha256"] is None
            assert record["fiedler_value_sha256"] is None
            assert record["spectral_gap_sha256"] is None


def test_polyglot_parity_gate_rejects_invalid_controls() -> None:
    for kwargs in (
        {"n": True},
        {"n": 2},
        {"calls": 0},
        {"calls": 1.5},
        {"seed": -1},
    ):
        with pytest.raises(ValueError):
            benchmark_spectral_polyglot_parity_gate(**kwargs)
