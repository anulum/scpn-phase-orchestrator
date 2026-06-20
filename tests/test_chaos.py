# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chaos-engineering resilience tests

"""Tests for chaos-engineering fault injection and resilience scoring.

The fault model, hook application, and resilience metric are exercised directly;
the full experiment runs the real :func:`simulate` pipeline on a bundled
domainpack so the orchestration is covered end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.binding import load_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.runtime.chaos import (
    ChaosFault,
    ChaosSchedule,
    ResilienceMetrics,
    compute_resilience,
    make_chaos_hook,
    run_resilience_experiment,
)
from scpn_phase_orchestrator.runtime.simulation import SimulationScenarioContext

SPEC_PATH = "domainpacks/minimal_domain/binding_spec.yaml"


# ---------------------------------------------------------------------
# Fault and schedule validation
# ---------------------------------------------------------------------


def test_fault_window_and_audit() -> None:
    fault = ChaosFault(
        "frequency_drift", start_step=10, duration_steps=5, magnitude=0.3
    )
    assert fault.end_step == 15
    assert fault.active_at(10)
    assert fault.active_at(14)
    assert not fault.active_at(15)
    assert not fault.active_at(9)
    assert json.loads(json.dumps(fault.to_audit_record())) == fault.to_audit_record()


def test_fault_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        ChaosFault("meltdown", start_step=1, duration_steps=1, magnitude=0.1)


@pytest.mark.parametrize("start", [0, -1, 1.5, True])
def test_fault_rejects_bad_start_step(start: object) -> None:
    with pytest.raises(ValueError, match="start_step"):
        ChaosFault("sensor_noise", start_step=start, duration_steps=1, magnitude=0.1)  # type: ignore[arg-type]


def test_fault_rejects_bad_duration() -> None:
    with pytest.raises(ValueError, match="duration_steps"):
        ChaosFault("sensor_noise", start_step=1, duration_steps=0, magnitude=0.1)


@pytest.mark.parametrize("kind", ["coupling_drop", "drive_dropout"])
def test_fraction_faults_reject_out_of_unit_range(kind: str) -> None:
    with pytest.raises(ValueError, match="magnitude"):
        ChaosFault(kind, start_step=1, duration_steps=1, magnitude=1.5)
    with pytest.raises(ValueError, match="magnitude"):
        ChaosFault(kind, start_step=1, duration_steps=1, magnitude=-0.1)


def test_additive_faults_reject_negative_magnitude() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        ChaosFault("frequency_drift", start_step=1, duration_steps=1, magnitude=-1.0)
    with pytest.raises(ValueError, match="magnitude"):
        ChaosFault(
            "sensor_noise", start_step=1, duration_steps=1, magnitude=float("nan")
        )


def test_additive_fault_rejects_non_finite_magnitude() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ChaosFault(
            "frequency_drift", start_step=1, duration_steps=1, magnitude=float("inf")
        )


def test_fault_rejects_non_real_magnitude() -> None:
    with pytest.raises(ValueError, match="finite real number"):
        ChaosFault("frequency_drift", start_step=1, duration_steps=1, magnitude=True)  # type: ignore[arg-type]


def test_schedule_requires_faults() -> None:
    with pytest.raises(ValueError, match="at least one fault"):
        ChaosSchedule(faults=())


def test_schedule_last_fault_end_and_audit() -> None:
    schedule = ChaosSchedule(
        faults=(
            ChaosFault("coupling_drop", start_step=5, duration_steps=10, magnitude=0.5),
            ChaosFault(
                "frequency_drift", start_step=20, duration_steps=8, magnitude=0.2
            ),
        )
    )
    assert schedule.last_fault_end == 28
    record = schedule.to_audit_record()
    assert record["last_fault_end"] == 28
    assert json.loads(json.dumps(record)) == record


# ---------------------------------------------------------------------
# Hook application (fake scenario context)
# ---------------------------------------------------------------------


def _context(step: int, *, n: int = 4) -> SimulationScenarioContext:
    knm = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    return SimulationScenarioContext(
        spec_name="fixture",
        step=step,
        sample_period_s=0.01,
        phases=np.linspace(0.0, 1.0, n),
        omegas=np.ones(n, dtype=np.float64),
        coupling=CouplingState(
            knm=knm, alpha=np.zeros((n, n)), active_template="t", knm_r=None
        ),
        zeta=1.0,
        psi_target=0.0,
        layer_osc_ranges={0: list(range(n))},
        rng=np.random.default_rng(0),
    )


def test_hook_coupling_drop_scales_relative_to_nominal() -> None:
    schedule = ChaosSchedule(
        faults=(
            ChaosFault("coupling_drop", start_step=2, duration_steps=2, magnitude=0.25),
        )
    )
    hook = make_chaos_hook(schedule)
    hook(_context(0))  # capture nominal at step 0
    ctx = _context(2)
    hook(ctx)
    assert ctx.coupling.knm[0, 1] == pytest.approx(0.75)


def test_hook_frequency_drift_offsets_omegas() -> None:
    schedule = ChaosSchedule(
        faults=(
            ChaosFault(
                "frequency_drift", start_step=1, duration_steps=1, magnitude=0.5
            ),
        )
    )
    hook = make_chaos_hook(schedule)
    hook(_context(0))
    ctx = _context(1)
    hook(ctx)
    assert ctx.omegas[0] == pytest.approx(1.5)


def test_hook_sensor_noise_perturbs_phases() -> None:
    schedule = ChaosSchedule(
        faults=(
            ChaosFault("sensor_noise", start_step=1, duration_steps=1, magnitude=0.1),
        )
    )
    hook = make_chaos_hook(schedule)
    hook(_context(0))
    ctx = _context(1)
    before = ctx.phases.copy()
    hook(ctx)
    assert not np.allclose(ctx.phases, before)


def test_hook_drive_dropout_attenuates_zeta() -> None:
    schedule = ChaosSchedule(
        faults=(
            ChaosFault("drive_dropout", start_step=1, duration_steps=1, magnitude=1.0),
        )
    )
    hook = make_chaos_hook(schedule)
    hook(_context(0))
    ctx = _context(1)
    hook(ctx)
    assert ctx.zeta == pytest.approx(0.0)


def test_hook_inactive_outside_window_leaves_context() -> None:
    schedule = ChaosSchedule(
        faults=(
            ChaosFault(
                "frequency_drift", start_step=5, duration_steps=2, magnitude=1.0
            ),
        )
    )
    hook = make_chaos_hook(schedule)
    hook(_context(0))
    ctx = _context(3)
    hook(ctx)
    assert ctx.omegas[0] == pytest.approx(1.0)


def test_hook_coupling_drop_without_nominal_uses_running_coupling() -> None:
    # Apply a coupling drop at the very first observed step (no captured nominal).
    schedule = ChaosSchedule(
        faults=(
            ChaosFault("coupling_drop", start_step=3, duration_steps=2, magnitude=0.5),
        )
    )
    hook = make_chaos_hook(schedule)
    ctx = _context(3)  # step 0 was never seen, so nominal is unset
    hook(ctx)
    assert ctx.coupling.knm[0, 1] == pytest.approx(0.5)


# ---------------------------------------------------------------------
# Resilience metric
# ---------------------------------------------------------------------


def test_compute_resilience_detects_recovery() -> None:
    nominal = tuple([0.9] * 20)
    perturbed = tuple([0.9] * 5 + [0.4] * 5 + [0.9] * 10)
    metrics = compute_resilience(
        nominal,
        perturbed,
        fault_onset_step=5,
        last_fault_end=10,
        recovery_tolerance=0.01,
    )
    assert metrics.recovered
    assert metrics.recovery_steps == 0
    assert metrics.max_coherence_drop == pytest.approx(0.5)
    assert metrics.final_deviation == pytest.approx(0.0)
    assert len(metrics.metrics_hash) == 64


def test_compute_resilience_reports_no_recovery() -> None:
    nominal = tuple([0.9] * 12)
    perturbed = tuple([0.9] * 4 + [0.3] * 8)
    metrics = compute_resilience(
        nominal,
        perturbed,
        fault_onset_step=4,
        last_fault_end=8,
        recovery_tolerance=0.05,
    )
    assert not metrics.recovered
    assert metrics.recovery_steps is None
    assert metrics.final_deviation == pytest.approx(0.6)


def test_compute_resilience_delayed_recovery_counts_steps() -> None:
    nominal = tuple([0.8] * 10)
    perturbed = tuple([0.8] * 4 + [0.5, 0.6, 0.7, 0.8, 0.8, 0.8])
    metrics = compute_resilience(
        nominal,
        perturbed,
        fault_onset_step=4,
        last_fault_end=5,
        recovery_tolerance=0.02,
    )
    assert metrics.recovered
    assert metrics.recovery_steps == 2


def test_compute_resilience_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_resilience(
            (), (), fault_onset_step=0, last_fault_end=0, recovery_tolerance=0.1
        )


def test_compute_resilience_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="share length"):
        compute_resilience(
            (0.9, 0.9),
            (0.9,),
            fault_onset_step=0,
            last_fault_end=1,
            recovery_tolerance=0.1,
        )


def test_metrics_audit_round_trips() -> None:
    metrics = ResilienceMetrics(True, 3, 0.4, 0.2, 0.1, "abc")
    assert (
        json.loads(json.dumps(metrics.to_audit_record())) == metrics.to_audit_record()
    )


# ---------------------------------------------------------------------
# Full experiment through the real simulation pipeline
# ---------------------------------------------------------------------


def _spec() -> object:
    return load_binding_spec(Path(SPEC_PATH))


def test_experiment_severe_coupling_drop_degrades_coherence() -> None:
    schedule = ChaosSchedule(
        faults=(
            ChaosFault(
                "coupling_drop", start_step=40, duration_steps=30, magnitude=0.85
            ),
        )
    )
    result = run_resilience_experiment(_spec(), schedule, steps=160, seed=7)  # type: ignore[arg-type]
    assert result.metrics.max_coherence_drop > 0.0
    assert result.perturbed_final_r < result.nominal_final_r
    assert result.steps == 160
    assert len(result.result_hash) == 64
    assert json.loads(json.dumps(result.to_audit_record()))["non_actuating"] is True


def test_experiment_mild_drift_recovers() -> None:
    schedule = ChaosSchedule(
        faults=(
            ChaosFault(
                "frequency_drift", start_step=30, duration_steps=10, magnitude=0.05
            ),
        )
    )
    result = run_resilience_experiment(_spec(), schedule, steps=200, seed=3)  # type: ignore[arg-type]
    assert result.metrics.recovered
    assert result.metrics.recovery_steps is not None


def test_experiment_rejects_too_few_steps() -> None:
    schedule = ChaosSchedule(
        faults=(
            ChaosFault(
                "coupling_drop", start_step=40, duration_steps=30, magnitude=0.5
            ),
        )
    )
    with pytest.raises(ValueError, match="steps must exceed"):
        run_resilience_experiment(_spec(), schedule, steps=50, seed=1)  # type: ignore[arg-type]
