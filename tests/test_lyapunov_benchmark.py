# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov benchmark contract tests

from __future__ import annotations

import json

from benchmarks.lyapunov_benchmark import (
    BACKEND_ORDER,
    benchmark_lyapunov_polyglot_parity_gate,
)


def test_polyglot_parity_gate_records_every_declared_backend() -> None:
    record = benchmark_lyapunov_polyglot_parity_gate(
        n=3,
        n_steps=40,
        qr_interval=10,
        calls=1,
        zeta=0.4,
        psi=0.2,
    )
    backend_records = json.loads(str(record["backend_records_json"]))

    assert record["backend_count"] == len(BACKEND_ORDER)
    assert [item["backend"] for item in backend_records] == list(BACKEND_ORDER)
    assert record["python_reference_present"] == 1
    assert record["all_available_passed"] == 1
    assert record["acceptance_passed"] == 1
    assert record["parity_pass_count"] == record["available_backend_count"]

    python_record = next(
        item for item in backend_records if item["backend"] == "python"
    )
    assert python_record["status"] == "available"
    assert python_record["max_abs_error"] == 0.0
    assert python_record["parity_passed"] is True

    for item in backend_records:
        if item["status"] == "available":
            assert item["ms_per_call"] is not None
            assert item["spectrum_sha256"] is not None
            assert item["max_abs_error"] <= item["tolerance"]
            continue
        assert item["status"] == "unavailable"
        assert item["unavailable_reason"]
        assert item["spectrum_sha256"] is None
