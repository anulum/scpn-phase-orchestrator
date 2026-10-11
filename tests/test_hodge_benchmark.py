# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge benchmark contract tests

"""Real public Hodge comparison, input refusal and timing evidence contracts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks import hodge_benchmark as hodge_benchmark_module
from benchmarks.hodge_benchmark import (
    BENCHMARK_EVIDENCE_KIND,
    bench_at,
    benchmark_hodge_polyglot_parity_gate,
)
from scpn_phase_orchestrator.coupling.hodge import HodgeResult, hodge_decomposition


def test_hodge_polyglot_parity_gate_reports_declared_backend_slots() -> None:
    """Every measured owner must satisfy physical flow and empty-network contracts."""
    out = benchmark_hodge_polyglot_parity_gate(n=8, calls=1, seed=2026)
    records = json.loads(str(out["backend_records_json"]))
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))

    assert out["suite"] == "hodge_polyglot_parity_gate"
    assert out["backend_count"] == 5
    assert out["python_reference_present"] == 1
    assert out["all_available_passed"] == 1
    assert out["reference_contracts_passed"] == 1
    assert out["acceptance_passed"] == 1
    assert out["empty_network_contract_passed"] == 1
    assert out["empty_coupling_shape_refused"] == 1
    assert out["empty_triangle_refused"] == 1
    assert out["required_backends_present"] == 1
    assert json.loads(str(out["required_backends_json"])) == []
    assert out["benchmark_evidence_kind"] == BENCHMARK_EVIDENCE_KIND
    assert out["isolation_method"] == "none"
    assert out["production_timing_claim"] == 0
    assert len(str(out["benchmark_sha256"])) == 64
    assert float(out["wall_time_s"]) > 0.0
    assert float(out["steps_per_second"]) > 0.0
    assert float(out["reconstruction_max_abs_error"]) <= 1.0e-10
    assert float(out["antisymmetry_max_abs_error"]) <= 1.0e-10
    assert float(out["orthogonality_max_abs_error"]) <= 1.0e-10
    assert float(out["curl_divergence_max_abs_error"]) <= 1.0e-10
    assert float(out["harmonic_divergence_max_abs_error"]) <= 1.0e-10
    assert float(out["phase_shift_max_abs_error"]) <= 1.0e-10
    assert float(out["scale_covariance_max_abs_error"]) <= 1.0e-10
    assert float(out["cycle_curl_max_abs_error"]) <= 1.0e-10
    assert float(out["cycle_harmonic_max_abs"]) > 1.0e-3
    assert float(out["cycle_harmonic_max_abs_error"]) <= 1.0e-10
    assert int(out["cycle_betti_one"]) == 1
    assert thresholds["backend_order"] == ["rust", "mojo", "julia", "go", "python"]
    assert thresholds["require_reconstruction_contract"] is True
    assert thresholds["require_antisymmetric_flow_components"] is True
    assert thresholds["require_component_orthogonality"] is True
    assert thresholds["require_divergence_free_curl_and_harmonic"] is True
    assert thresholds["require_global_phase_shift_invariance"] is True
    assert thresholds["require_scale_covariance"] is True
    assert thresholds["require_cycle_betti_one"] is True
    assert thresholds["require_nontrivial_cycle_harmonic"] is True
    assert thresholds["production_timing_claim"] is False
    assert thresholds["require_empty_network_validation"] is True

    assert [record["backend"] for record in records] == thresholds["backend_order"]
    for record in records:
        if record["status"] == "available":
            assert record["parity_passed"] is True
            assert record["empty_network_contract_passed"] == 1
            assert record["empty_coupling_shape_refused"] == 1
            assert record["empty_triangle_refused"] == 1
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
    """Boolean counts and invalid ranges cannot produce a comparison report."""
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


@pytest.mark.parametrize("required", [["unknown"], ["python", "python"]])
def test_hodge_polyglot_gate_rejects_invalid_runtime_requirements(
    required: list[str],
) -> None:
    """Unknown and duplicate runtime requirements cannot qualify a report."""
    with pytest.raises(ValueError, match="required Hodge backend names"):
        benchmark_hodge_polyglot_parity_gate(require_backends=required)


def test_hodge_wallclock_benchmark_is_labelled_non_isolated_regression_evidence() -> (
    None
):
    """Public timing results retain their explicit non-isolated evidence boundary."""
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


@pytest.mark.parametrize("fault", ["validation", "empty_response"])
def test_hodge_comparison_refuses_empty_network_boundary_faults(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """An empty-network boundary fault invalidates a real numerical comparison."""
    original = hodge_decomposition

    def boundary_fault(
        knm: NDArray[np.float64],
        phases: NDArray[np.float64],
        triangles: Sequence[Sequence[int]] | None = None,
        *,
        backend: str | None = None,
    ) -> HodgeResult:
        """Inject only the named empty-network API fault; keep nonempty solvers real."""
        if phases.size == 0 and fault == "validation":
            return original(np.empty((0, 0)), phases, backend=backend)
        result = original(knm, phases, triangles, backend=backend)
        if phases.size == 0:
            return replace(result, gradient=np.ones((1, 1)))
        return result

    monkeypatch.setattr(hodge_benchmark_module, "hodge_decomposition", boundary_fault)
    report = hodge_benchmark_module.benchmark_hodge_polyglot_parity_gate(
        n=4, calls=1, require_backends=("python",)
    )
    assert report["required_backends_present"] == 1
    assert report["reference_contracts_passed"] == 0
    assert report["acceptance_passed"] == 0
    if fault == "validation":
        assert report["empty_coupling_shape_refused"] == 0
        assert report["empty_triangle_refused"] == 0
    else:
        assert report["empty_network_contract_passed"] == 0
