# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — v1 reference benchmark suite

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np

from scpn_phase_orchestrator.autotune.binding_proposal import (
    propose_binding_from_time_series_csv,
)
from scpn_phase_orchestrator.autotune.learners import (
    generate_hybrid_physics_proposal,
    generate_ppo_like_proposal,
    generate_sac_like_proposal,
)
from scpn_phase_orchestrator.autotune.reward import (
    KnobPolicyCandidate,
    PolicyProposalConfig,
    RewardObservation,
)
from scpn_phase_orchestrator.supervisor.petri_net import (
    Arc,
    Marking,
    PetriNet,
    Place,
    Transition,
)
from scpn_phase_orchestrator.upde.bayesian import (
    BayesianUPDEConfig,
    audit_bayesian_backend_status,
    bayesian_upde_run,
    fit_gaussian_upde_posterior,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "reference_suite.json"
BENCHMARK_COMMAND = "PYTHONPATH=src python benchmarks/reference_suite.py"
REFERENCE_SUITE_VERSION = "reference_suite_v1"


BenchmarkValue = float | int | str
BenchmarkRecord = dict[str, BenchmarkValue]


class AutoBindingAcceptanceThresholds(NamedTuple):
    min_extractor_coverage: float
    min_expected_edge_recall: float
    max_validation_errors: int
    min_sample_count: int
    max_proposed_edge_multiplier: float


class AutoBindingFixture(NamedTuple):
    domain: str
    csv_text: str
    sample_rate_hz: float | None
    expected_edges: frozenset[tuple[str, str]]
    thresholds: AutoBindingAcceptanceThresholds

    @property
    def sample_count(self) -> int:
        return max(0, len(self.csv_text.splitlines()) - 1)


class ReplayLearnerBenchmarkThresholds(NamedTuple):
    min_acceptance_rate: float
    min_reward_improvement: float
    max_unsafe_acceptances: int
    require_non_actuating: bool


class BayesianPosteriorFitThresholds(NamedTuple):
    max_residual_rmse: float
    max_omega_mean_abs_error: float
    max_knm_mean_abs_error: float
    max_credible_interval_width: float
    min_rollout_sample_count: int


class BayesianBackendFailClosedThresholds(NamedTuple):
    min_available_backends: int
    required_fail_closed_backends: frozenset[str]
    max_unexpected_reserved_successes: int


class ReferenceSuiteResult(TypedDict):
    metadata: dict[str, str]
    benchmarks: dict[str, BenchmarkRecord]


def build_benchmark_metadata(*, snapshot_date: str | None = None) -> dict[str, str]:
    return {
        "suite_version": REFERENCE_SUITE_VERSION,
        "snapshot_date": snapshot_date or date.today().isoformat(),
        "command": BENCHMARK_COMMAND,
        "backend": "python_numpy",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def benchmark_kuramoto_reference(
    n_oscillators: int = 64, n_steps: int = 1000, dt: float = 0.01
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_oscillators)
    omegas = np.zeros(n_oscillators)
    knm = np.full((n_oscillators, n_oscillators), 0.4, dtype=float)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros_like(knm)
    engine = UPDEEngine(n_oscillators=n_oscillators, dt=dt, method="rk4")

    t0 = time.perf_counter()
    for _ in range(n_steps):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    elapsed = time.perf_counter() - t0
    final_r, _ = compute_order_parameter(phases)

    return {
        "suite": "kuramoto_reference_strogatz_2000",
        "n_oscillators": n_oscillators,
        "n_steps": n_steps,
        "wall_time_s": elapsed,
        "steps_per_second": n_steps / elapsed,
        "final_order_parameter": float(final_r),
    }


def benchmark_stuart_landau_reference(
    n_oscillators: int = 64, n_steps: int = 1000, dt: float = 0.01
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(7)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_oscillators)
    radius = np.ones(n_oscillators)
    state = np.concatenate((theta, radius))
    omegas = np.full(n_oscillators, 1.0)
    mu = np.full(n_oscillators, 0.5)
    knm = np.full((n_oscillators, n_oscillators), 0.2, dtype=float)
    knm_r = np.full((n_oscillators, n_oscillators), 0.2, dtype=float)
    alpha = np.zeros((n_oscillators, n_oscillators), dtype=float)
    np.fill_diagonal(knm, 0.0)
    np.fill_diagonal(knm_r, 0.0)
    engine = StuartLandauEngine(n_oscillators=n_oscillators, dt=dt, method="rk4")

    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = engine.step(
            state, omegas, mu, knm, knm_r, zeta=0.0, psi=0.0, alpha=alpha, epsilon=1.0
        )
    elapsed = time.perf_counter() - t0
    final_r = float(engine.compute_mean_amplitude(state))

    return {
        "suite": "stuart_landau_reference_pikovsky_2001",
        "n_oscillators": n_oscillators,
        "n_steps": n_steps,
        "wall_time_s": elapsed,
        "steps_per_second": n_steps / elapsed,
        "final_mean_amplitude": final_r,
    }


def benchmark_petri_reachability(n_steps: int = 5000) -> dict[str, float | int | str]:
    net = PetriNet(
        places=[
            Place("nominal"),
            Place("degraded"),
            Place("critical"),
            Place("recovery"),
        ],
        transitions=[
            Transition("n_to_d", inputs=[Arc("nominal")], outputs=[Arc("degraded")]),
            Transition("d_to_c", inputs=[Arc("degraded")], outputs=[Arc("critical")]),
            Transition("c_to_r", inputs=[Arc("critical")], outputs=[Arc("recovery")]),
            Transition("r_to_n", inputs=[Arc("recovery")], outputs=[Arc("nominal")]),
        ],
    )
    marking = Marking(tokens={"nominal": 1})
    visited: set[tuple[tuple[str, int], ...]] = set()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        key = tuple(sorted(marking.tokens.items()))
        visited.add(key)
        marking, _ = net.step(marking, {})
    elapsed = time.perf_counter() - t0

    return {
        "suite": "petri_net_reachability",
        "n_steps": n_steps,
        "wall_time_s": elapsed,
        "steps_per_second": n_steps / elapsed,
        "reachable_markings": len(visited),
    }


def benchmark_auto_binding_proposal_quality() -> dict[str, float | int | str]:
    fixtures = _auto_binding_quality_fixtures()
    total_channels = 0
    covered_extractors = 0
    validation_error_count = 0
    expected_edge_count = 0
    expected_edge_hits = 0
    proposed_edge_count = 0
    accepted_domain_count = 0
    min_domain_extractor_coverage = 1.0
    min_domain_expected_edge_recall = 1.0
    max_domain_validation_errors = 0
    min_sample_count = min(fixture.sample_count for fixture in fixtures)
    domain_results: list[dict[str, float | int | str | bool]] = []

    t0 = time.perf_counter()
    for fixture in fixtures:
        proposal = propose_binding_from_time_series_csv(
            fixture.csv_text,
            sample_rate_hz=fixture.sample_rate_hz,
            project_name=f"{fixture.domain}_benchmark",
        )
        fixture_validation_errors = len(proposal.binding.validation_errors)
        validation_error_count += fixture_validation_errors
        source_columns = _string_records(proposal.binding.provenance["source_columns"])
        extractor_proposals = proposal.binding.provenance[
            "extractor_parameter_proposals"
        ]
        total_channels += len(source_columns)
        fixture_covered_extractors = _extractor_source_coverage(
            source_columns=source_columns,
            extractor_proposals=_mapping_records(extractor_proposals),
        )
        covered_extractors += fixture_covered_extractors
        proposed_edges = _proposed_source_edges(
            proposal.binding.provenance["initial_coupling_proposal"]
        )
        fixture_expected_hits = len(fixture.expected_edges & proposed_edges)
        fixture_expected_edge_recall = fixture_expected_hits / len(
            fixture.expected_edges
        )
        fixture_extractor_coverage = fixture_covered_extractors / len(source_columns)
        fixture_edge_multiplier = len(proposed_edges) / len(fixture.expected_edges)
        fixture_accepted = _auto_binding_fixture_passes_thresholds(
            fixture=fixture,
            extractor_coverage=fixture_extractor_coverage,
            expected_edge_recall=fixture_expected_edge_recall,
            validation_error_count=fixture_validation_errors,
            proposed_edge_multiplier=fixture_edge_multiplier,
        )
        if fixture_accepted:
            accepted_domain_count += 1
        min_domain_extractor_coverage = min(
            min_domain_extractor_coverage, fixture_extractor_coverage
        )
        min_domain_expected_edge_recall = min(
            min_domain_expected_edge_recall, fixture_expected_edge_recall
        )
        max_domain_validation_errors = max(
            max_domain_validation_errors, fixture_validation_errors
        )
        domain_results.append(
            {
                "domain": fixture.domain,
                "sample_count": fixture.sample_count,
                "source_column_count": len(source_columns),
                "validation_error_count": fixture_validation_errors,
                "extractor_coverage": fixture_extractor_coverage,
                "expected_edge_recall": fixture_expected_edge_recall,
                "proposed_edge_count": len(proposed_edges),
                "proposed_edge_multiplier": fixture_edge_multiplier,
                "accepted": fixture_accepted,
            }
        )
        proposed_edge_count += len(proposed_edges)
        expected_edge_count += len(fixture.expected_edges)
        expected_edge_hits += fixture_expected_hits
    elapsed = time.perf_counter() - t0

    return {
        "suite": "auto_binding_synthetic_quality",
        "fixture_count": len(fixtures),
        "large_fixture_count": sum(fixture.sample_count >= 96 for fixture in fixtures),
        "wall_time_s": elapsed,
        "steps_per_second": len(fixtures) / elapsed,
        "validation_error_count": validation_error_count,
        "extractor_coverage": covered_extractors / total_channels,
        "expected_edge_recall": expected_edge_hits / expected_edge_count,
        "proposed_edge_count": proposed_edge_count,
        "domain_acceptance_passed": int(accepted_domain_count == len(fixtures)),
        "accepted_domain_count": accepted_domain_count,
        "failed_domain_count": len(fixtures) - accepted_domain_count,
        "min_domain_extractor_coverage": min_domain_extractor_coverage,
        "min_domain_expected_edge_recall": min_domain_expected_edge_recall,
        "max_domain_validation_errors": max_domain_validation_errors,
        "min_sample_count": min_sample_count,
        "domain_acceptance_thresholds_json": json.dumps(
            _auto_binding_threshold_summary(fixtures), sort_keys=True
        ),
        "domain_acceptance_results_json": json.dumps(domain_results, sort_keys=True),
    }


def benchmark_replay_policy_candidate_quality() -> dict[str, float | int | str]:
    """Benchmark replay-only learner proposals against deterministic gates."""
    thresholds = ReplayLearnerBenchmarkThresholds(
        min_acceptance_rate=1.0,
        min_reward_improvement=0.03,
        max_unsafe_acceptances=0,
        require_non_actuating=True,
    )
    seed_candidate = KnobPolicyCandidate(
        K=0.2,
        alpha=0.0,
        zeta=0.05,
        Psi=0.0,
        channel_weights=(0.8, 0.2),
        cross_channel_gains=(0.05,),
    )
    proposal_config = PolicyProposalConfig(
        min_coherence=0.78,
        min_reward=-0.25,
        max_alternatives=2,
    )
    baseline_observation = _deterministic_replay_observation(seed_candidate)
    baseline_coherence = baseline_observation.coherence

    t0 = time.perf_counter()
    learner_proposals = (
        generate_ppo_like_proposal(
            seed_candidate,
            _deterministic_replay_observation,
            seed_value=17,
            proposal_config=proposal_config,
        ),
        generate_sac_like_proposal(
            seed_candidate,
            _deterministic_replay_observation,
            seed_value=23,
            proposal_config=proposal_config,
        ),
        generate_hybrid_physics_proposal(
            seed_candidate,
            _deterministic_replay_observation,
            critical_coupling_estimate=0.72,
            seed_value=31,
            proposal_config=proposal_config,
        ),
    )
    elapsed = time.perf_counter() - t0

    learner_results: list[dict[str, float | int | str | bool]] = []
    accepted_count = 0
    unsafe_acceptances = 0
    min_reward_improvement = np.inf
    for proposal in learner_proposals:
        policy_proposal = proposal.policy_search.proposal
        selected = policy_proposal.selected
        accepted = policy_proposal.accepted and selected is not None
        accepted_count += int(accepted)
        non_actuating = proposal.actuation_permitted is False
        selected_reward = selected.reward if selected is not None else -np.inf
        selected_coherence = (
            selected.observation.coherence if selected is not None else 0.0
        )
        reward_improvement = selected_coherence - baseline_coherence
        min_reward_improvement = min(min_reward_improvement, reward_improvement)
        selected_unsafe = bool(selected.observation.unsafe) if selected else False
        unsafe_acceptances += int(accepted and selected_unsafe)
        learner_results.append(
            {
                "learner_kind": proposal.learner_kind,
                "accepted": accepted,
                "non_actuating": non_actuating,
                "selected_reward": selected_reward,
                "selected_coherence": selected_coherence,
                "baseline_coherence": baseline_coherence,
                "coherence_improvement": reward_improvement,
                "unsafe_selected": selected_unsafe,
                "candidate_count": len(proposal.policy_search.candidates),
            }
        )

    acceptance_rate = accepted_count / len(learner_proposals)
    threshold_passed = (
        acceptance_rate >= thresholds.min_acceptance_rate
        and min_reward_improvement >= thresholds.min_reward_improvement
        and unsafe_acceptances <= thresholds.max_unsafe_acceptances
        and all(result["non_actuating"] is True for result in learner_results)
    )

    return {
        "suite": "replay_policy_candidate_quality",
        "learner_count": len(learner_proposals),
        "wall_time_s": elapsed,
        "steps_per_second": len(learner_proposals) / elapsed,
        "accepted_learner_count": accepted_count,
        "failed_learner_count": len(learner_proposals) - accepted_count,
        "acceptance_rate": acceptance_rate,
        "baseline_coherence": baseline_coherence,
        "min_coherence_improvement": min_reward_improvement,
        "unsafe_acceptance_count": unsafe_acceptances,
        "non_actuating_proposals": int(
            all(result["non_actuating"] is True for result in learner_results)
        ),
        "acceptance_passed": int(threshold_passed),
        "acceptance_thresholds_json": json.dumps(
            {
                "min_acceptance_rate": thresholds.min_acceptance_rate,
                "min_reward_improvement": thresholds.min_reward_improvement,
                "max_unsafe_acceptances": thresholds.max_unsafe_acceptances,
                "require_non_actuating": thresholds.require_non_actuating,
            },
            sort_keys=True,
        ),
        "learner_results_json": json.dumps(learner_results, sort_keys=True),
    }


def benchmark_bayesian_posterior_fit_quality() -> dict[str, float | int | str]:
    """Benchmark posterior fitting from observed Kuramoto trajectories."""
    thresholds = BayesianPosteriorFitThresholds(
        max_residual_rmse=2.5e-3,
        max_omega_mean_abs_error=3.0e-2,
        max_knm_mean_abs_error=6.0e-2,
        max_credible_interval_width=1.0e-2,
        min_rollout_sample_count=96,
    )
    phases, omega, knm, alpha, dt = _bayesian_posterior_fit_fixture()

    t0 = time.perf_counter()
    fit = fit_gaussian_upde_posterior(
        phases,
        dt=dt,
        alpha=alpha,
        ridge=1e-8,
        coupling_std_floor=2.5e-3,
        omega_std_floor=2.5e-3,
    )
    result = bayesian_upde_run(
        phases[0],
        omega=fit.omega,
        knm=fit.knm,
        alpha=alpha,
        zeta=0.0,
        psi=0.0,
        config=BayesianUPDEConfig(n_samples=128, seed=41, n_steps=32, dt=dt),
    )
    elapsed = time.perf_counter() - t0

    omega_mean_abs_error = float(np.max(np.abs(fit.omega.mean - omega)))
    knm_mean_abs_error = float(np.max(np.abs(fit.knm.mean - knm)))
    credible_interval_width = float(result.r_upper - result.r_lower)
    audit_record = fit.to_audit_record()
    finite_audit = int(_audit_record_is_finite(audit_record))
    zero_diagonal = int(np.allclose(np.diag(fit.knm.mean), 0.0))
    non_negative_coupling = int(np.all(fit.knm.mean >= 0.0))
    acceptance_passed = int(
        fit.residual_rmse <= thresholds.max_residual_rmse
        and omega_mean_abs_error <= thresholds.max_omega_mean_abs_error
        and knm_mean_abs_error <= thresholds.max_knm_mean_abs_error
        and credible_interval_width <= thresholds.max_credible_interval_width
        and result.sample_count >= thresholds.min_rollout_sample_count
        and finite_audit == 1
        and zero_diagonal == 1
        and non_negative_coupling == 1
    )

    return {
        "suite": "bayesian_posterior_fit_quality",
        "sample_count": len(phases),
        "rollout_sample_count": result.sample_count,
        "wall_time_s": elapsed,
        "steps_per_second": len(phases) / elapsed,
        "residual_rmse": fit.residual_rmse,
        "omega_mean_abs_error": omega_mean_abs_error,
        "knm_mean_abs_error": knm_mean_abs_error,
        "credible_interval_width": credible_interval_width,
        "finite_audit_record": finite_audit,
        "zero_diagonal_coupling": zero_diagonal,
        "non_negative_coupling": non_negative_coupling,
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "max_credible_interval_width": (thresholds.max_credible_interval_width),
                "max_knm_mean_abs_error": thresholds.max_knm_mean_abs_error,
                "max_omega_mean_abs_error": thresholds.max_omega_mean_abs_error,
                "max_residual_rmse": thresholds.max_residual_rmse,
                "min_rollout_sample_count": thresholds.min_rollout_sample_count,
            },
            sort_keys=True,
        ),
    }


def benchmark_bayesian_backend_fail_closed() -> dict[str, float | int | str]:
    """Benchmark reserved Bayesian sampler names against fail-closed gates."""
    thresholds = BayesianBackendFailClosedThresholds(
        min_available_backends=1,
        required_fail_closed_backends=frozenset({"numpyro", "blackjax"}),
        max_unexpected_reserved_successes=0,
    )
    phases, omega, knm, alpha, _ = _bayesian_backend_audit_fixture()

    t0 = time.perf_counter()
    statuses = audit_bayesian_backend_status(
        phases,
        omega=omega,
        knm=knm,
        alpha=alpha,
        zeta=0.0,
        psi=0.0,
        config=BayesianUPDEConfig(n_samples=16, seed=83, n_steps=3),
    )
    elapsed = time.perf_counter() - t0

    status_by_backend = {status.backend: status for status in statuses}
    available_backends = sum(status.available for status in statuses)
    fail_closed_backends = {
        status.backend
        for status in statuses
        if status.fail_closed and not status.available
    }
    unexpected_reserved_successes = sum(
        status.available
        for status in statuses
        if status.backend in thresholds.required_fail_closed_backends
    )
    acceptance_passed = int(
        available_backends >= thresholds.min_available_backends
        and thresholds.required_fail_closed_backends <= fail_closed_backends
        and unexpected_reserved_successes
        <= thresholds.max_unexpected_reserved_successes
        and status_by_backend["numpy"].sample_count == 16
    )

    return {
        "suite": "bayesian_backend_fail_closed",
        "backend_count": len(statuses),
        "wall_time_s": elapsed,
        "steps_per_second": len(statuses) / elapsed,
        "available_backend_count": available_backends,
        "fail_closed_backend_count": len(fail_closed_backends),
        "unexpected_reserved_success_count": unexpected_reserved_successes,
        "numpy_sample_count": status_by_backend["numpy"].sample_count,
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "max_unexpected_reserved_successes": (
                    thresholds.max_unexpected_reserved_successes
                ),
                "min_available_backends": thresholds.min_available_backends,
                "required_fail_closed_backends": sorted(
                    thresholds.required_fail_closed_backends
                ),
            },
            sort_keys=True,
        ),
        "backend_results_json": json.dumps(
            [status.to_audit_record() for status in statuses],
            sort_keys=True,
        ),
    }


def _deterministic_replay_observation(
    candidate: KnobPolicyCandidate,
) -> RewardObservation:
    mean_k = float(np.asarray(candidate.K, dtype=np.float64).mean())
    mean_channel_weight = (
        float(np.mean(candidate.channel_weights)) if candidate.channel_weights else 0.0
    )
    coherence = float(
        np.clip(0.68 + 0.44 * mean_k + 0.05 * mean_channel_weight, 0.0, 0.95)
    )
    unsafe = mean_k < 0.0 or abs(mean_k) > 1.2
    return RewardObservation(
        coherence=coherence,
        previous_coherence=0.68,
        unsafe=unsafe,
        regime_changed=False,
    )


def _bayesian_posterior_fit_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    dt = 0.02
    omega = np.array([0.92, 1.03, 1.11], dtype=float)
    knm = np.array(
        [
            [0.0, 0.11, 0.04],
            [0.18, 0.0, 0.07],
            [0.09, 0.14, 0.0],
        ],
        dtype=float,
    )
    alpha = np.zeros_like(knm)
    engine = UPDEEngine(n_oscillators=3, dt=dt, method="rk4")
    phase = np.array([0.1, 0.6, 1.4], dtype=float)
    trajectory = [phase.copy()]
    for _ in range(95):
        phase = engine.step(phase, omega, knm, zeta=0.0, psi=0.0, alpha=alpha)
        trajectory.append(phase.copy())
    return np.asarray(trajectory), omega, knm, alpha, dt


def _bayesian_backend_audit_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    phases = np.array([0.0, 0.4, 1.1, 1.9], dtype=float)
    omega = np.array([0.9, 1.0, 1.08, 1.16], dtype=float)
    knm = np.full((4, 4), 0.18, dtype=float)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros_like(knm)
    return phases, omega, knm, alpha, 0.01


def _audit_record_is_finite(value: object) -> bool:
    if isinstance(value, bool | str):
        return True
    if isinstance(value, int | float):
        return bool(np.isfinite(value))
    if isinstance(value, Mapping):
        return all(_audit_record_is_finite(item) for item in value.values())
    if isinstance(value, Iterable):
        return all(_audit_record_is_finite(item) for item in value)
    return False


def _auto_binding_quality_fixtures() -> tuple[AutoBindingFixture, ...]:
    return (
        AutoBindingFixture(
            domain="phase_chain",
            csv_text=_phase_chain_csv(n_samples=128),
            sample_rate_hz=None,
            expected_edges=frozenset({("theta_source", "theta_driven")}),
            thresholds=AutoBindingAcceptanceThresholds(
                min_extractor_coverage=1.0,
                min_expected_edge_recall=1.0,
                max_validation_errors=0,
                min_sample_count=96,
                max_proposed_edge_multiplier=8.0,
            ),
        ),
        AutoBindingFixture(
            domain="industrial_sensor_chain",
            csv_text=_sensor_chain_csv(n_samples=128),
            sample_rate_hz=10.0,
            expected_edges=frozenset({("source", "driven")}),
            thresholds=AutoBindingAcceptanceThresholds(
                min_extractor_coverage=1.0,
                min_expected_edge_recall=1.0,
                max_validation_errors=0,
                min_sample_count=96,
                max_proposed_edge_multiplier=8.0,
            ),
        ),
        AutoBindingFixture(
            domain="cardiac_rhythm_surrogate",
            csv_text=_cardiac_phase_csv(n_samples=160),
            sample_rate_hz=None,
            expected_edges=frozenset(
                {("pacemaker", "atrium"), ("atrium", "ventricle")}
            ),
            thresholds=AutoBindingAcceptanceThresholds(
                min_extractor_coverage=1.0,
                min_expected_edge_recall=1.0,
                max_validation_errors=0,
                min_sample_count=128,
                max_proposed_edge_multiplier=6.0,
            ),
        ),
        AutoBindingFixture(
            domain="power_grid_surrogate",
            csv_text=_power_grid_phase_csv(n_samples=192),
            sample_rate_hz=None,
            expected_edges=frozenset({("generator", "tie_line"), ("tie_line", "load")}),
            thresholds=AutoBindingAcceptanceThresholds(
                min_extractor_coverage=1.0,
                min_expected_edge_recall=1.0,
                max_validation_errors=0,
                min_sample_count=160,
                max_proposed_edge_multiplier=8.0,
            ),
        ),
    )


def _auto_binding_fixture_passes_thresholds(
    *,
    fixture: AutoBindingFixture,
    extractor_coverage: float,
    expected_edge_recall: float,
    validation_error_count: int,
    proposed_edge_multiplier: float,
) -> bool:
    thresholds = fixture.thresholds
    return (
        extractor_coverage >= thresholds.min_extractor_coverage
        and expected_edge_recall >= thresholds.min_expected_edge_recall
        and validation_error_count <= thresholds.max_validation_errors
        and fixture.sample_count >= thresholds.min_sample_count
        and proposed_edge_multiplier <= thresholds.max_proposed_edge_multiplier
    )


def _auto_binding_threshold_summary(
    fixtures: Iterable[AutoBindingFixture],
) -> dict[str, dict[str, float | int]]:
    return {
        fixture.domain: {
            "min_extractor_coverage": fixture.thresholds.min_extractor_coverage,
            "min_expected_edge_recall": fixture.thresholds.min_expected_edge_recall,
            "max_validation_errors": fixture.thresholds.max_validation_errors,
            "min_sample_count": fixture.thresholds.min_sample_count,
            "max_proposed_edge_multiplier": (
                fixture.thresholds.max_proposed_edge_multiplier
            ),
        }
        for fixture in fixtures
    }


def _extractor_source_coverage(
    *,
    source_columns: tuple[str, ...],
    extractor_proposals: Iterable[Mapping[str, object]],
) -> int:
    covered = set()
    source_column_set = set(source_columns)
    for proposal in extractor_proposals:
        parameters = proposal.get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        source_column = parameters.get("source_column")
        if isinstance(source_column, str) and source_column in source_column_set:
            covered.add(source_column)
    return len(covered)


def _proposed_source_edges(
    initial_coupling_proposal: object,
) -> frozenset[tuple[str, str]]:
    if not isinstance(initial_coupling_proposal, Mapping):
        return frozenset()
    edges = initial_coupling_proposal.get("edges")
    if not isinstance(edges, Iterable) or isinstance(edges, str | bytes):
        return frozenset()
    source_edges: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        source = edge.get("source")
        target = edge.get("target")
        strength = edge.get("strength")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if not isinstance(strength, int | float) or isinstance(strength, bool):
            continue
        if float(strength) <= 0.0:
            continue
        source_edges.add((source, target))
    return frozenset(source_edges)


def _mapping_records(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_records(value: object) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _phase_chain_csv(n_samples: int = 32, dt: float = 0.1) -> str:
    rows = ["time,theta_source,theta_driven,theta_independent"]
    for index in range(n_samples):
        time_s = index * dt
        source = 0.21 * index
        driven = 0.15 * index + 0.18 * np.sin(source)
        independent = 1.1 + 0.09 * index
        rows.append(f"{time_s:.12g},{source:.12g},{driven:.12g},{independent:.12g}")
    return "\n".join(rows)


def _sensor_chain_csv(n_samples: int = 32, dt: float = 0.1) -> str:
    rows = ["time,source,driven,independent"]
    previous_source = 0.0
    for index in range(n_samples):
        time_s = index * dt
        source = np.sin(0.35 * index)
        driven = 0.72 * previous_source + 0.08 * np.cos(0.17 * index)
        independent = np.cos(0.41 * index + 0.3)
        rows.append(f"{time_s:.12g},{source:.12g},{driven:.12g},{independent:.12g}")
        previous_source = source
    return "\n".join(rows)


def _cardiac_phase_csv(n_samples: int = 160, dt: float = 0.02) -> str:
    rows = ["time,pacemaker,atrium,ventricle,artifact"]
    previous_pacemaker = 0.0
    previous_atrium = 0.0
    for index in range(n_samples):
        time_s = index * dt
        pacemaker = 0.19 * index + 0.02 * np.sin(0.07 * index)
        atrium = 0.75 * previous_pacemaker + 0.04 * np.sin(pacemaker)
        ventricle = 0.68 * previous_atrium + 0.03 * np.cos(0.11 * index)
        artifact = np.sin(0.31 * index + 1.7)
        rows.append(
            f"{time_s:.12g},{pacemaker:.12g},{atrium:.12g},"
            f"{ventricle:.12g},{artifact:.12g}"
        )
        previous_pacemaker = pacemaker
        previous_atrium = atrium
    return "\n".join(rows)


def _power_grid_phase_csv(n_samples: int = 192, dt: float = 0.05) -> str:
    rows = ["time,generator,tie_line,load,renewable"]
    previous_generator = 0.0
    previous_tie_line = 0.0
    for index in range(n_samples):
        time_s = index * dt
        generator = np.sin(0.09 * index) + 0.01 * index
        tie_line = 0.62 * previous_generator + 0.05 * np.sin(0.2 * index)
        load = 0.58 * previous_tie_line + 0.05 * np.cos(0.13 * index)
        renewable = np.sin(0.29 * index + 0.4)
        rows.append(
            f"{time_s:.12g},{generator:.12g},{tie_line:.12g},"
            f"{load:.12g},{renewable:.12g}"
        )
        previous_generator = generator
        previous_tie_line = tie_line
    return "\n".join(rows)


def run_reference_suite(*, snapshot_date: str | None = None) -> ReferenceSuiteResult:
    return {
        "metadata": build_benchmark_metadata(snapshot_date=snapshot_date),
        "benchmarks": {
            "auto_binding": benchmark_auto_binding_proposal_quality(),
            "replay_policy": benchmark_replay_policy_candidate_quality(),
            "bayesian_posterior": benchmark_bayesian_posterior_fit_quality(),
            "bayesian_backends": benchmark_bayesian_backend_fail_closed(),
            "kuramoto": benchmark_kuramoto_reference(),
            "stuart_landau": benchmark_stuart_landau_reference(),
            "petri_reachability": benchmark_petri_reachability(),
        },
    }


if __name__ == "__main__":
    results = run_reference_suite()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
