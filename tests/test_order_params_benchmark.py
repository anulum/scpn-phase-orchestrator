# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Order-parameter benchmark contract tests

"""Benchmark contract tests for the order-parameter polyglot parity gate."""

from __future__ import annotations

import json

from benchmarks.order_params_benchmark import (
    BACKEND_ORDER,
    benchmark_order_parameter_polyglot_parity_gate,
)


def test_polyglot_parity_gate_records_every_declared_backend() -> None:
    out = benchmark_order_parameter_polyglot_parity_gate(n=12, calls=1, seed=19)
    records = json.loads(str(out["backend_records_json"]))

    assert out["suite"] == "order_parameter_polyglot_parity_gate"
    assert out["backend_count"] == len(BACKEND_ORDER)
    assert out["python_reference_present"] == 1
    assert out["all_available_passed"] == 1
    assert out["acceptance_passed"] == 1
    assert out["parity_pass_count"] == out["available_backend_count"]
    assert 0.0 <= float(out["reference_r"]) <= 1.0
    assert 0.0 <= float(out["reference_plv"]) <= 1.0
    assert 0.0 <= float(out["reference_layer_coherence"]) <= 1.0
    assert [record["backend"] for record in records] == list(BACKEND_ORDER)

    python_record = next(record for record in records if record["backend"] == "python")
    assert python_record["status"] == "available"
    assert python_record["max_abs_error"] == 0.0
    assert python_record["r_abs_error"] == 0.0
    assert python_record["psi_abs_error"] == 0.0
    assert python_record["plv_abs_error"] == 0.0
    assert python_record["layer_abs_error"] == 0.0
    assert python_record["parity_passed"] is True

    for record in records:
        if record["status"] == "available":
            assert record["ms_per_call"] is not None
            assert record["order_parameter_sha256"] is not None
            assert record["plv_sha256"] is not None
            assert record["layer_coherence_sha256"] is not None
            assert record["max_abs_error"] <= record["tolerance"]
        else:
            assert record["unavailable_reason"]
            assert record["order_parameter_sha256"] is None
            assert record["plv_sha256"] is None
            assert record["layer_coherence_sha256"] is None


def test_polyglot_parity_gate_rejects_invalid_controls() -> None:
    for kwargs in (
        {"n": True},
        {"n": 1},
        {"calls": 0},
        {"seed": -1},
    ):
        try:
            benchmark_order_parameter_polyglot_parity_gate(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid controls were accepted: {kwargs!r}")
