# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — NPE benchmark contract tests

"""Benchmark contract tests for the NPE polyglot parity gate."""

from __future__ import annotations

import json

from benchmarks.npe_benchmark import benchmark_npe_polyglot_parity_gate


def test_polyglot_parity_gate_records_every_declared_backend() -> None:
    out = benchmark_npe_polyglot_parity_gate(n=8, calls=1, seed=19)
    records = json.loads(str(out["backend_records_json"]))

    assert out["suite"] == "npe_polyglot_parity_gate"
    assert out["backend_count"] == 5
    assert out["python_reference_present"] == 1
    assert out["all_available_passed"] == 1
    assert out["acceptance_passed"] == 1
    assert 0.0 <= float(out["reference_npe"]) <= 1.0
    assert [record["backend"] for record in records] == [
        "rust",
        "mojo",
        "julia",
        "go",
        "python",
    ]

    python_record = next(record for record in records if record["backend"] == "python")
    assert python_record["status"] == "available"
    assert python_record["max_abs_error"] == 0.0
    assert python_record["max_distance_abs_error"] == 0.0
    assert python_record["npe_abs_error"] == 0.0
    assert python_record["parity_passed"] is True

    for record in records:
        if record["status"] == "available":
            assert record["ms_per_call"] is not None
            assert record["distance_matrix_sha256"] is not None
            assert record["npe_sha256"] is not None
            assert record["max_abs_error"] <= record["tolerance"]
        else:
            assert record["unavailable_reason"]
            assert record["distance_matrix_sha256"] is None
            assert record["npe_sha256"] is None
