# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reference benchmark suite smoke tests

from __future__ import annotations

import json

from benchmarks.reference_suite import (
    BENCHMARK_COMMAND,
    benchmark_auto_binding_proposal_quality,
    benchmark_bayesian_backend_fail_closed,
    benchmark_bayesian_posterior_fit_quality,
    benchmark_kuramoto_reference,
    benchmark_petri_reachability,
    benchmark_replay_policy_candidate_quality,
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
    assert out["fixture_count"] == 4
    assert out["large_fixture_count"] == 4
    assert out["validation_error_count"] == 0
    assert out["extractor_coverage"] == 1.0
    assert float(out["expected_edge_recall"]) == 1.0
    assert out["domain_acceptance_passed"] == 1
    assert out["accepted_domain_count"] == 4
    assert out["failed_domain_count"] == 0
    assert out["min_domain_extractor_coverage"] == 1.0
    assert out["min_domain_expected_edge_recall"] == 1.0
    assert int(out["max_domain_validation_errors"]) == 0
    assert int(out["min_sample_count"]) >= 96
    assert float(out["steps_per_second"]) > 0.0


def test_auto_binding_proposal_quality_reports_domain_thresholds() -> None:
    out = benchmark_auto_binding_proposal_quality()
    thresholds = json.loads(str(out["domain_acceptance_thresholds_json"]))
    results = json.loads(str(out["domain_acceptance_results_json"]))

    assert set(thresholds) == {
        "phase_chain",
        "industrial_sensor_chain",
        "cardiac_rhythm_surrogate",
        "power_grid_surrogate",
    }
    assert [record["domain"] for record in results] == [
        "phase_chain",
        "industrial_sensor_chain",
        "cardiac_rhythm_surrogate",
        "power_grid_surrogate",
    ]
    assert all(record["accepted"] is True for record in results)
    assert all(record["sample_count"] >= 96 for record in results)
    assert all(record["expected_edge_recall"] >= 1.0 for record in results)
    assert all(record["validation_error_count"] == 0 for record in results)


def test_replay_policy_candidate_quality_benchmark_shape() -> None:
    out = benchmark_replay_policy_candidate_quality()

    assert out["suite"] == "replay_policy_candidate_quality"
    assert out["learner_count"] == 3
    assert out["accepted_learner_count"] == 3
    assert out["failed_learner_count"] == 0
    assert out["acceptance_rate"] == 1.0
    assert out["acceptance_passed"] == 1
    assert out["unsafe_acceptance_count"] == 0
    assert out["non_actuating_proposals"] == 1
    assert float(out["min_coherence_improvement"]) >= 0.03
    assert float(out["steps_per_second"]) > 0.0


def test_replay_policy_candidate_quality_reports_thresholds() -> None:
    out = benchmark_replay_policy_candidate_quality()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    results = json.loads(str(out["learner_results_json"]))

    assert thresholds == {
        "max_unsafe_acceptances": 0,
        "min_acceptance_rate": 1.0,
        "min_reward_improvement": 0.03,
        "require_non_actuating": True,
    }
    assert [record["learner_kind"] for record in results] == [
        "ppo_like_replay",
        "sac_like_replay",
        "hybrid_physics_replay",
    ]
    assert all(record["accepted"] is True for record in results)
    assert all(record["non_actuating"] is True for record in results)
    assert all(record["unsafe_selected"] is False for record in results)


def test_bayesian_posterior_fit_quality_benchmark_shape() -> None:
    out = benchmark_bayesian_posterior_fit_quality()

    assert out["suite"] == "bayesian_posterior_fit_quality"
    assert out["sample_count"] >= 96
    assert out["rollout_sample_count"] >= 96
    assert out["acceptance_passed"] == 1
    assert out["finite_audit_record"] == 1
    assert out["zero_diagonal_coupling"] == 1
    assert out["non_negative_coupling"] == 1
    assert float(out["residual_rmse"]) <= 2.5e-3
    assert float(out["omega_mean_abs_error"]) <= 3.0e-2
    assert float(out["knm_mean_abs_error"]) <= 6.0e-2
    assert float(out["credible_interval_width"]) <= 1.0e-2
    assert float(out["steps_per_second"]) > 0.0


def test_bayesian_posterior_fit_quality_reports_thresholds() -> None:
    out = benchmark_bayesian_posterior_fit_quality()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))

    assert thresholds == {
        "max_credible_interval_width": 1.0e-2,
        "max_knm_mean_abs_error": 6.0e-2,
        "max_omega_mean_abs_error": 3.0e-2,
        "max_residual_rmse": 2.5e-3,
        "min_rollout_sample_count": 96,
    }


def test_bayesian_backend_fail_closed_benchmark_shape() -> None:
    out = benchmark_bayesian_backend_fail_closed()

    assert out["suite"] == "bayesian_backend_fail_closed"
    assert out["backend_count"] == 3
    assert out["available_backend_count"] == 1
    assert out["fail_closed_backend_count"] == 2
    assert out["unexpected_reserved_success_count"] == 0
    assert out["numpy_sample_count"] == 16
    assert out["acceptance_passed"] == 1
    assert float(out["steps_per_second"]) > 0.0


def test_bayesian_backend_fail_closed_reports_thresholds_and_records() -> None:
    out = benchmark_bayesian_backend_fail_closed()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    results = json.loads(str(out["backend_results_json"]))

    assert thresholds == {
        "max_unexpected_reserved_successes": 0,
        "min_available_backends": 1,
        "required_fail_closed_backends": ["blackjax", "numpyro"],
    }
    assert [record["backend"] for record in results] == [
        "numpy",
        "numpyro",
        "blackjax",
    ]
    assert results[0]["available"] is True
    assert results[0]["sample_count"] == 16
    assert all(record["fail_closed"] is True for record in results[1:])
    assert all(record["sample_count"] == 0 for record in results[1:])


def test_reference_suite_aggregates_all_benchmarks() -> None:
    out = run_reference_suite(snapshot_date="2026-05-06")
    assert set(out.keys()) == {"metadata", "benchmarks"}
    assert out["metadata"]["snapshot_date"] == "2026-05-06"
    assert set(out["benchmarks"].keys()) == {
        "auto_binding",
        "bayesian_backends",
        "bayesian_posterior",
        "replay_policy",
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
