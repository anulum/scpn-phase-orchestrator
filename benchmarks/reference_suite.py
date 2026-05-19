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
from scpn_phase_orchestrator.supervisor.petri_net import (
    Arc,
    Marking,
    PetriNet,
    Place,
    Transition,
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
