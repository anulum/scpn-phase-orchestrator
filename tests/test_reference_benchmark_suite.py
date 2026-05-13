# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reference benchmark suite smoke tests

from __future__ import annotations

from benchmarks.reference_suite import (
    BENCHMARK_COMMAND,
    benchmark_auto_binding_proposal_quality,
    benchmark_kuramoto_reference,
    benchmark_petri_reachability,
    benchmark_stuart_landau_reference,
    build_benchmark_metadata,
    run_reference_suite,
)


def test_kuramoto_reference_benchmark_shape() -> None:
    out = benchmark_kuramoto_reference(n_oscillators=8, n_steps=20, dt=0.01)
    assert out["suite"] == "kuramoto_reference_strogatz_2000"
    assert out["n_oscillators"] == 8
    assert out["n_steps"] == 20
    assert 0.0 <= float(out["final_order_parameter"]) <= 1.0
    assert float(out["steps_per_second"]) > 0.0


def test_stuart_landau_reference_benchmark_shape() -> None:
    out = benchmark_stuart_landau_reference(n_oscillators=8, n_steps=20, dt=0.01)
    assert out["suite"] == "stuart_landau_reference_pikovsky_2001"
    assert out["n_oscillators"] == 8
    assert out["n_steps"] == 20
    assert float(out["final_mean_amplitude"]) > 0.0
    assert float(out["steps_per_second"]) > 0.0


def test_petri_reachability_benchmark_shape() -> None:
    out = benchmark_petri_reachability(n_steps=20)
    assert out["suite"] == "petri_net_reachability"
    assert out["n_steps"] == 20
    assert int(out["reachable_markings"]) >= 2
    assert float(out["steps_per_second"]) > 0.0


def test_auto_binding_proposal_quality_benchmark_shape() -> None:
    out = benchmark_auto_binding_proposal_quality()

    assert out["suite"] == "auto_binding_synthetic_quality"
    assert out["fixture_count"] == 2
    assert out["validation_error_count"] == 0
    assert out["extractor_coverage"] == 1.0
    assert float(out["expected_edge_recall"]) >= 0.5
    assert float(out["steps_per_second"]) > 0.0


def test_reference_suite_aggregates_all_benchmarks() -> None:
    out = run_reference_suite(snapshot_date="2026-05-06")
    assert set(out.keys()) == {"metadata", "benchmarks"}
    assert out["metadata"]["snapshot_date"] == "2026-05-06"
    assert set(out["benchmarks"].keys()) == {
        "auto_binding",
        "kuramoto",
        "stuart_landau",
        "petri_reachability",
    }


def test_reference_suite_metadata_labels_reproduction_context() -> None:
    metadata = build_benchmark_metadata(snapshot_date="2026-05-06")

    assert metadata["suite_version"] == "reference_suite_v1"
    assert metadata["snapshot_date"] == "2026-05-06"
    assert metadata["command"] == BENCHMARK_COMMAND
    assert metadata["backend"] == "python_numpy"
    assert metadata["python_version"]
    assert metadata["numpy_version"]
    assert metadata["platform"]
