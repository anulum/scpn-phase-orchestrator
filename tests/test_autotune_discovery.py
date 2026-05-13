# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — time-series discovery tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.autotune.discovery import (
    discover_time_series_structure,
    infer_sample_rate_from_time_column,
)


def test_discover_time_series_structure_reports_json_ready_evidence() -> None:
    samples = np.asarray(
        [
            [0.00, 0.00, 1.00],
            [0.20, 0.10, 0.98],
            [0.40, 0.20, 0.92],
            [0.60, 0.31, 0.83],
            [0.78, 0.42, 0.70],
            [0.94, 0.54, 0.54],
            [1.07, 0.66, 0.36],
            [1.17, 0.77, 0.17],
        ],
        dtype=np.float64,
    )

    report = discover_time_series_structure(
        samples,
        columns=("source", "driven", "independent"),
        sample_period_s=0.1,
    )
    audit_record = report.to_audit_record()

    assert audit_record["sindy"]["library"] == "affine_state_derivative"
    assert audit_record["sindy"]["active_terms"] > 0
    assert audit_record["correlation_graph"]["edge_count"] >= 1
    assert audit_record["clustering"]["cluster_count"] >= 1
    assert 0.0 <= report.sindy_sparsity <= 1.0
    assert 0.0 <= report.correlation_graph_density <= 1.0
    assert 0.0 <= report.cluster_coverage <= 1.0


def test_discover_time_series_structure_reports_phase_sindy_edges() -> None:
    times = np.linspace(0.0, 2.4, 25, dtype=np.float64)
    phases = np.column_stack(
        [
            0.8 * times,
            0.8 * times + 0.35 * np.sin(times),
            -0.4 * times + 0.2 * np.cos(times),
        ]
    )

    report = discover_time_series_structure(
        phases,
        columns=("theta_source", "theta_driven", "theta_aux"),
        sample_period_s=float(times[1] - times[0]),
    )
    audit_record = report.to_audit_record()
    phase_sindy = audit_record["phase_sindy"]

    assert phase_sindy["status"] == "fitted"
    assert phase_sindy["library"] == "kuramoto_sine_phase_differences"
    assert phase_sindy["active_terms"] > 0
    assert phase_sindy["coupling_edge_count"] >= 1
    assert "phase_sindy_sparsity" in report.confidence_evidence


def test_discover_time_series_structure_ranks_sindy_libraries() -> None:
    times = np.linspace(0.0, 2.4, 25, dtype=np.float64)
    phases = np.column_stack(
        [
            0.8 * times,
            0.8 * times + 0.35 * np.sin(times),
            -0.4 * times + 0.2 * np.cos(times),
        ]
    )

    report = discover_time_series_structure(
        phases,
        columns=("theta_source", "theta_driven", "theta_aux"),
        sample_period_s=float(times[1] - times[0]),
    )
    selection = report.to_audit_record()["sindy_model_selection"]

    assert selection["method"] == "residual_bic_with_sparsity_tie_break"
    assert selection["candidate_count"] == 2
    assert selection["selected_library"] in {
        "affine_state_derivative",
        "kuramoto_sine_phase_differences",
    }
    candidates = selection["candidates"]
    assert {candidate["library"] for candidate in candidates} == {
        "affine_state_derivative",
        "kuramoto_sine_phase_differences",
    }
    fitted = [candidate for candidate in candidates if candidate["status"] == "fitted"]
    assert fitted
    assert all(np.isfinite(candidate["score"]) for candidate in fitted)
    assert all(candidate["residual_rmse"] >= 0.0 for candidate in fitted)


def test_discover_time_series_structure_reports_lagged_graph_edges() -> None:
    driver = np.asarray([1.0, -1.0, 0.8, -0.8, 0.6, -0.6, 0.4, -0.4], dtype=np.float64)
    response = np.asarray([0.0, 0.9, -0.9, 0.72, -0.72, 0.54, -0.54, 0.36])
    independent = np.linspace(-0.2, 0.2, driver.size, dtype=np.float64)
    samples = np.column_stack([driver, response, independent])

    report = discover_time_series_structure(
        samples,
        columns=("driver", "response", "independent"),
        sample_period_s=0.1,
    )
    learned_graph = report.to_audit_record()["learned_graph"]

    assert learned_graph["status"] == "fitted"
    assert learned_graph["method"] == "lagged_sparse_linear_prediction"
    assert learned_graph["edge_count"] >= 1
    assert {(edge["source"], edge["target"]) for edge in learned_graph["edges"]} >= {
        ("driver", "response")
    }
    assert learned_graph["residual_rmse"] >= 0.0
    assert "learned_graph_density" in report.confidence_evidence


def test_discover_time_series_structure_marks_non_phase_sindy_skipped() -> None:
    samples = np.asarray(
        [[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0]],
        dtype=np.float64,
    )

    report = discover_time_series_structure(
        samples,
        columns=("temperature", "pressure"),
        sample_period_s=1.0,
    )

    assert report.to_audit_record()["phase_sindy"]["status"] == "skipped_non_phase_like"
    selection = report.to_audit_record()["sindy_model_selection"]
    assert selection["selected_library"] == "affine_state_derivative"
    assert selection["candidate_count"] == 2
    assert "phase_sindy_sparsity" not in report.confidence_evidence


def test_infer_sample_rate_from_regular_time_column() -> None:
    rows = [
        {"time": "0.0", "signal": "1.0"},
        {"time": "0.25", "signal": "0.5"},
        {"time": "0.50", "signal": "0.0"},
    ]

    sample_rate_hz, inference = infer_sample_rate_from_time_column(
        rows,
        ("time", "signal"),
    )

    assert sample_rate_hz == pytest.approx(4.0)
    assert inference == "time_column"


def test_infer_sample_rate_rejects_irregular_time_column() -> None:
    rows = [
        {"time": "0.0", "signal": "1.0"},
        {"time": "0.20", "signal": "0.5"},
        {"time": "0.55", "signal": "0.0"},
    ]

    with pytest.raises(ValueError, match="regular sampling interval"):
        infer_sample_rate_from_time_column(rows, ("time", "signal"))
