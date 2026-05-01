# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reference benchmark suite smoke tests

from __future__ import annotations

from benchmarks.reference_suite import (
    benchmark_kuramoto_reference,
    benchmark_petri_reachability,
    benchmark_stuart_landau_reference,
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


def test_reference_suite_aggregates_all_benchmarks() -> None:
    out = run_reference_suite()
    assert set(out.keys()) == {"kuramoto", "stuart_landau", "petri_reachability"}
