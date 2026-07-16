# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — time-series discovery tests

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

import scpn_phase_orchestrator.autotune.discovery as discovery_mod
from scpn_phase_orchestrator.autotune.discovery import (
    TimeSeriesDiscoveryConfig,
    TimeSeriesDiscoveryReport,
    _correlation_clusters,
    _regression_quality,
    discover_time_series_structure,
    infer_sample_rate_from_time_column,
)
from tests.typing_contracts import assert_precise_ndarray_hint


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

    # The fitted block must be self-describing for the confidence classifier:
    # a scale-free R^2, the derivative-sample count actually regressed, and the
    # per-node parameter count. These are the quantities an honest tier/posture
    # rule reads to refuse "discovered" on under-determined or poor fits.
    r_squared = phase_sindy["r_squared"]
    assert isinstance(r_squared, float)
    assert np.isfinite(r_squared)
    assert r_squared <= 1.0
    assert phase_sindy["sample_count"] == phases.shape[0] - 1
    assert phase_sindy["node_count"] == phases.shape[1]
    assert report.confidence_evidence["phase_sindy_r_squared"] == r_squared


def test_regression_quality_reports_perfect_fit_as_unit_r_squared() -> None:
    observations = np.asarray([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]], dtype=np.float64)

    quality = _regression_quality(
        observations=observations,
        predictions=observations.copy(),
        active_terms=2,
    )

    assert quality["r_squared"] == pytest.approx(1.0)
    assert quality["residual_rmse"] == pytest.approx(0.0, abs=1e-9)


def test_regression_quality_grants_no_credit_on_variance_free_target() -> None:
    # A constant target has no variance to explain: an honest R^2 is 0.0, not
    # an undefined 0/0 that a naive formula would surface as 1.0.
    observations = np.full((4, 2), 7.0, dtype=np.float64)

    quality = _regression_quality(
        observations=observations,
        predictions=observations.copy(),
        active_terms=1,
    )

    assert quality["r_squared"] == 0.0


def test_regression_quality_keeps_negative_r_squared_below_mean_baseline() -> None:
    # A fit worse than simply predicting the per-column mean must not be
    # clamped to zero — the negative value is the honest "explains nothing"
    # signal the confidence classifier relies on to refuse "discovered".
    observations = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64)
    predictions = np.asarray([[5.0], [5.0], [5.0]], dtype=np.float64)

    quality = _regression_quality(
        observations=observations,
        predictions=predictions,
        active_terms=1,
    )

    assert quality["r_squared"] < 0.0
    assert np.isfinite(quality["r_squared"])


def test_regression_quality_upper_bounds_r_squared_at_unity() -> None:
    observations = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)

    quality = _regression_quality(
        observations=observations,
        predictions=observations.copy(),
        active_terms=1,
    )

    assert quality["r_squared"] <= 1.0


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

    phase_sindy = report.to_audit_record()["phase_sindy"]
    assert phase_sindy["status"] == "skipped_non_phase_like"
    # A skipped fit must not fabricate a quality signal: no R^2, no samples,
    # no nodes were regressed.
    assert phase_sindy["r_squared"] is None
    assert phase_sindy["sample_count"] == 0
    assert phase_sindy["node_count"] == 0
    selection = report.to_audit_record()["sindy_model_selection"]
    assert selection["selected_library"] == "affine_state_derivative"
    assert selection["candidate_count"] == 2
    assert "phase_sindy_sparsity" not in report.confidence_evidence
    assert "phase_sindy_r_squared" not in report.confidence_evidence


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


def test_infer_sample_rate_rejects_degenerate_median_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"time": "0.0", "signal": "1.0"},
        {"time": "0.25", "signal": "0.5"},
        {"time": "0.50", "signal": "0.0"},
    ]

    def zero_median(_values: object) -> float:
        return 0.0

    monkeypatch.setattr(discovery_mod.np, "median", zero_median)
    monkeypatch.setattr(discovery_mod.np, "allclose", lambda *_args, **_kwargs: True)

    with pytest.raises(ValueError, match="could not be inferred"):
        infer_sample_rate_from_time_column(rows, ("time", "signal"))


def test_discovery_config_rejects_boolean_and_complex_threshold_aliases() -> None:
    with pytest.raises(ValueError, match="correlation_threshold"):
        TimeSeriesDiscoveryConfig(correlation_threshold=True)

    with pytest.raises(ValueError, match="sindy_threshold"):
        TimeSeriesDiscoveryConfig(sindy_threshold=1.0 + 0.0j)

    config = TimeSeriesDiscoveryConfig(
        correlation_threshold=np.float64(0.5),
        sindy_threshold=np.float64(0.1),
        phase_sindy_threshold=np.float64(0.2),
        learned_graph_threshold=np.float64(0.3),
    )
    assert type(config.correlation_threshold) is float
    assert type(config.sindy_threshold) is float
    assert type(config.phase_sindy_threshold) is float
    assert type(config.learned_graph_threshold) is float


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"correlation_threshold": -0.1}, "correlation_threshold"),
        ({"correlation_threshold": 1.1}, "correlation_threshold"),
        ({"sindy_threshold": -0.1}, "sindy_threshold"),
        ({"phase_sindy_threshold": -0.1}, "phase_sindy_threshold"),
        ({"learned_graph_threshold": -0.1}, "learned_graph_threshold"),
        ({"learned_graph_threshold": float("inf")}, "learned_graph_threshold"),
    ],
)
def test_discovery_config_rejects_out_of_range_thresholds(
    kwargs: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        TimeSeriesDiscoveryConfig(**kwargs)


def test_discover_time_series_structure_rejects_aliasing_inputs() -> None:
    with pytest.raises(ValueError, match="samples.*boolean"):
        discover_time_series_structure(
            np.asarray([[False, True], [True, False]]),
            columns=("left", "right"),
            sample_period_s=1.0,
        )

    with pytest.raises(ValueError, match="samples.*real-valued"):
        discover_time_series_structure(
            np.asarray([[0.0, 1.0 + 0.0j], [1.0, 2.0]], dtype=object),
            columns=("left", "right"),
            sample_period_s=1.0,
        )

    with pytest.raises(ValueError, match="sample_period_s"):
        discover_time_series_structure(
            np.asarray([[0.0], [1.0]], dtype=np.float64),
            columns=("signal",),
            sample_period_s=True,
        )

    with pytest.raises(ValueError, match="signal column names must be strings"):
        discover_time_series_structure(
            np.asarray([[0.0], [1.0]], dtype=np.float64),
            columns=(1,),
            sample_period_s=1.0,
        )


@pytest.mark.parametrize(
    ("samples", "columns", "sample_period_s", "match"),
    [
        (np.asarray([0.0, 1.0], dtype=np.float64), ("signal",), 1.0, "2-D table"),
        (
            np.asarray([[0.0, 1.0]], dtype=np.float64),
            ("left", "right"),
            1.0,
            "at least two rows",
        ),
        (
            np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64),
            ("left",),
            1.0,
            "column count",
        ),
        (
            np.empty((2, 0), dtype=np.float64),
            (),
            1.0,
            "at least one signal column",
        ),
        (
            np.asarray([[0.0], [1.0]], dtype=np.float64),
            ("signal",),
            -1.0,
            "sample_period_s must be positive",
        ),
        (
            np.asarray([[0.0], [1.0]], dtype=np.float64),
            ("signal",),
            float("nan"),
            "sample_period_s must be finite",
        ),
        (
            np.asarray([["bad"], ["data"]], dtype=object),
            ("signal",),
            1.0,
            "finite real-valued table",
        ),
        (
            np.asarray([[0.0], [float("nan")]], dtype=np.float64),
            ("signal",),
            1.0,
            "only finite values",
        ),
        (
            np.asarray([[0.0], [1.0]], dtype=np.float64),
            (" "),
            1.0,
            "non-empty",
        ),
    ],
)
def test_discover_time_series_structure_rejects_malformed_tables(
    samples: object,
    columns: tuple[object, ...],
    sample_period_s: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        discover_time_series_structure(
            samples,
            columns=columns,
            sample_period_s=sample_period_s,
        )


def test_discover_time_series_structure_reports_skipped_graphs_for_one_signal() -> None:
    samples = np.asarray([[1.0], [1.0], [1.0]], dtype=np.float64)

    report = discover_time_series_structure(
        samples,
        columns=("constant",),
        sample_period_s=1.0,
    )
    audit_record = report.to_audit_record()

    assert audit_record["correlation_graph"]["density"] == 0.0
    assert (
        audit_record["phase_sindy"]["status"] == "requires_at_least_two_phase_columns"
    )
    assert audit_record["learned_graph"]["status"] == "requires_at_least_two_columns"
    assert audit_record["learned_graph"]["density"] == 0.0
    assert "learned_graph_density" not in report.confidence_evidence


def test_discover_time_series_structure_reports_short_phase_and_learned_skips() -> None:
    samples = np.asarray([[0.0, 0.1], [0.2, 0.3]], dtype=np.float64)

    report = discover_time_series_structure(
        samples,
        columns=("theta_a", "theta_b"),
        sample_period_s=1.0,
    )
    audit_record = report.to_audit_record()

    assert audit_record["phase_sindy"]["status"] == "requires_at_least_three_samples"
    assert audit_record["learned_graph"]["status"] == "requires_at_least_three_samples"


def test_discover_time_series_structure_reports_under_sampled_phase_sindy_skip() -> (
    None
):
    samples = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
        ],
        dtype=np.float64,
    )

    report = discover_time_series_structure(
        samples,
        columns=("theta_a", "theta_b", "theta_c", "theta_d"),
        sample_period_s=1.0,
    )

    assert (
        report.to_audit_record()["phase_sindy"]["status"]
        == "requires_at_least_one_derivative_sample_per_feature"
    )


def test_discover_time_series_structure_handles_constant_columns() -> None:
    samples = np.asarray(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )

    report = discover_time_series_structure(
        samples,
        columns=("constant_a", "constant_b"),
        sample_period_s=1.0,
    )
    audit_record = report.to_audit_record()

    assert audit_record["correlation_graph"]["edge_count"] == 0
    assert audit_record["correlation_graph"]["density"] == 0.0
    assert audit_record["sindy"]["equations"] == [
        "d(constant_a)/dt = 0",
        "d(constant_b)/dt = 0",
    ]


def test_infer_sample_rate_rejects_boolean_time_aliases() -> None:
    rows = [
        {"time": False, "signal": "1.0"},
        {"time": True, "signal": "0.5"},
    ]

    with pytest.raises(ValueError, match="non-numeric sample at row 0"):
        infer_sample_rate_from_time_column(rows, ("time", "signal"))


def test_infer_sample_rate_accepts_real_time_values() -> None:
    rows = [
        {"time": 0.0, "signal": "1.0"},
        {"time": 0.5, "signal": "0.5"},
        {"time": 1.0, "signal": "0.0"},
    ]

    sample_rate_hz, inference = infer_sample_rate_from_time_column(
        rows,
        ("time", "signal"),
    )

    assert sample_rate_hz == pytest.approx(2.0)
    assert inference == "time_column"


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([{"time": "0.0", "signal": "1.0"}], "at least two timed samples"),
        (
            [
                {"time": "1.0", "signal": "1.0"},
                {"time": "0.0", "signal": "0.5"},
            ],
            "strictly increasing",
        ),
        (
            [
                {"time": "not-a-number", "signal": "1.0"},
                {"time": "1.0", "signal": "0.5"},
            ],
            "non-numeric sample at row 0",
        ),
        (
            [
                {"time": object(), "signal": "1.0"},
                {"time": "1.0", "signal": "0.5"},
            ],
            "non-numeric sample at row 0",
        ),
        (
            [
                {"time": "nan", "signal": "1.0"},
                {"time": "1.0", "signal": "0.5"},
            ],
            "non-numeric sample at row 0",
        ),
    ],
)
def test_infer_sample_rate_rejects_malformed_time_columns(
    rows: list[dict[str, object]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        infer_sample_rate_from_time_column(rows, ("time", "signal"))


def test_infer_sample_rate_rejects_non_string_fieldnames() -> None:
    rows = [
        {"time": "0.0", "signal": "1.0"},
        {"time": "1.0", "signal": "0.5"},
    ]

    with pytest.raises(ValueError, match="fieldnames must be strings"):
        infer_sample_rate_from_time_column(rows, ("time", 1))


def test_report_cluster_coverage_handles_empty_column_set() -> None:
    report = TimeSeriesDiscoveryReport(
        sample_period_s=1.0,
        sample_count=0,
        columns=(),
        sindy={"sparsity": 1.0},
        phase_sindy={"status": "requires_at_least_two_phase_columns"},
        sindy_model_selection={},
        learned_graph={"status": "requires_at_least_two_columns"},
        correlation_graph={"density": 0.0},
        clustering={"largest_cluster_size": 0},
    )

    assert report.cluster_coverage == 0.0
    assert report.confidence_evidence == {
        "sindy_sparsity": 1.0,
        "correlation_graph_density": 0.0,
        "cluster_coverage": 0.0,
    }


def test_correlation_clusters_ignore_corrupt_edge_payloads() -> None:
    clusters = _correlation_clusters(
        ("left", "right"),
        edges=[
            "not-an-edge",
            {"source": "left", "target": "right"},
            {"source": "left", "target": 3},
        ],
    )

    assert clusters["cluster_count"] == 1
    assert clusters["largest_cluster_size"] == 2
    assert clusters["clusters"] == [["left", "right"]]


def test_discovery_public_array_contracts_are_element_typed() -> None:
    """Guard the V2 typed-array contract for discovery evidence surfaces."""

    samples_hint = get_type_hints(discover_time_series_structure)["samples"]
    assert_precise_ndarray_hint(samples_hint)
    assert "float64" in str(samples_hint)

    report_hints = get_type_hints(TimeSeriesDiscoveryReport)
    assert report_hints["sample_period_s"] is float
    assert report_hints["sample_count"] is int
