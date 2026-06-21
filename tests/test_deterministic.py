# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Deterministic execution mode tests

from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.runtime.deterministic import (
    DeadlineBudget,
    DeadlineExceededError,
    ExecutionTimingReport,
    run_deterministic_loop,
)


class TestDeadlineBudget:
    def test_defaults_effective_wcet_to_period(self) -> None:
        budget = DeadlineBudget(period_s=0.01)
        assert budget.effective_wcet_s == 0.01
        assert budget.miss_policy == "observe"
        assert budget.freeze_gc is True

    def test_explicit_wcet(self) -> None:
        assert DeadlineBudget(period_s=0.01, wcet_s=0.004).effective_wcet_s == 0.004

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"period_s": 0.0}, "period_s"),
            ({"period_s": -1.0}, "period_s"),
            ({"period_s": np.inf}, "period_s"),
            ({"period_s": True}, "period_s"),
            ({"period_s": 0.01, "wcet_s": 0.0}, "wcet_s"),
            ({"period_s": 0.01, "wcet_s": -0.1}, "wcet_s"),
            ({"period_s": 0.01, "wcet_s": np.nan}, "wcet_s"),
            ({"period_s": 0.01, "wcet_s": True}, "wcet_s"),
            ({"period_s": 0.01, "wcet_s": "fast"}, "wcet_s"),
            ({"period_s": 0.01, "miss_policy": "explode"}, "miss_policy"),
            ({"period_s": 0.01, "busy_wait_margin_s": -1.0}, "busy_wait_margin_s"),
            ({"period_s": 0.01, "busy_wait_margin_s": np.inf}, "busy_wait_margin_s"),
        ],
    )
    def test_rejects_invalid(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            DeadlineBudget(**kwargs)


class TestLoopExecution:
    def test_runs_every_step_in_order(self) -> None:
        seen: list[int] = []
        report = run_deterministic_loop(
            seen.append, steps=5, budget=DeadlineBudget(period_s=0.001)
        )
        assert seen == [0, 1, 2, 3, 4]
        assert report.steps == 5
        assert report.latencies_s.shape == (5,)
        assert report.jitters_s.shape == (5,)
        assert np.all(np.isfinite(report.latencies_s))
        assert np.all(np.isfinite(report.jitters_s))

    def test_zero_steps_is_empty(self) -> None:
        report = run_deterministic_loop(
            lambda _i: None, steps=0, budget=DeadlineBudget(period_s=0.001)
        )
        assert report.steps == 0
        assert report.max_latency_s == 0.0
        assert report.mean_latency_s == 0.0
        assert report.max_abs_jitter_s == 0.0
        assert report.deadline_met is True
        assert report.latency_percentile_s(99.0) == 0.0

    @pytest.mark.parametrize("steps", [-1, True, 2.0, "3"])
    def test_rejects_invalid_steps(self, steps: Any) -> None:
        with pytest.raises(ValueError, match="steps"):
            run_deterministic_loop(
                lambda _i: None, steps=steps, budget=DeadlineBudget(period_s=0.001)
            )

    def test_busy_wait_margin_path_completes(self) -> None:
        report = run_deterministic_loop(
            lambda _i: None,
            steps=3,
            budget=DeadlineBudget(period_s=0.002, busy_wait_margin_s=0.0005),
        )
        assert report.steps == 3
        assert np.all(report.latencies_s >= 0.0)

    def test_full_period_spin_margin_skips_sleep(self) -> None:
        # busy_wait_margin == period pushes the spin boundary to the previous
        # step, so each iteration spins to its deadline without sleeping.
        report = run_deterministic_loop(
            lambda _i: None,
            steps=3,
            budget=DeadlineBudget(period_s=0.001, busy_wait_margin_s=0.001),
        )
        assert report.steps == 3

    def test_step_slower_than_period_skips_scheduling_sleep(self) -> None:
        # A step that overruns the period leaves the next boundary already in
        # the past, so the loop never sleeps before the following step.
        report = run_deterministic_loop(
            lambda _i: time.sleep(0.004),
            steps=3,
            budget=DeadlineBudget(period_s=0.001, wcet_s=0.05),
        )
        assert report.steps == 3
        assert report.deadline_met is True


class TestDeadlineMiss:
    def test_overrun_recorded_under_observe(self) -> None:
        def slow(index: int) -> None:
            if index == 1:
                time.sleep(0.02)

        report = run_deterministic_loop(
            slow,
            steps=3,
            budget=DeadlineBudget(period_s=0.05, wcet_s=0.005, miss_policy="observe"),
        )
        assert report.deadline_misses == 1
        assert report.deadline_met is False
        assert report.max_latency_s >= 0.02

    def test_overrun_aborts_under_abort_policy(self) -> None:
        def slow(index: int) -> None:
            time.sleep(0.02)

        with pytest.raises(DeadlineExceededError) as exc:
            run_deterministic_loop(
                slow,
                steps=3,
                budget=DeadlineBudget(period_s=0.05, wcet_s=0.005, miss_policy="abort"),
            )
        assert exc.value.step_index == 0
        assert exc.value.latency_s >= 0.02
        assert exc.value.wcet_s == 0.005

    def test_within_budget_meets_deadline(self) -> None:
        report = run_deterministic_loop(
            lambda _i: None,
            steps=4,
            budget=DeadlineBudget(period_s=0.01, wcet_s=0.009),
        )
        assert report.deadline_met is True
        assert report.deadline_misses == 0


class TestGarbageCollectorControl:
    def test_gc_frozen_during_loop_and_restored(self) -> None:
        assert gc.isenabled()
        states: list[bool] = []
        run_deterministic_loop(
            lambda _i: states.append(gc.isenabled()),
            steps=3,
            budget=DeadlineBudget(period_s=0.001, freeze_gc=True),
        )
        assert states == [False, False, False]
        assert gc.isenabled() is True

    def test_gc_untouched_when_freeze_disabled(self) -> None:
        assert gc.isenabled()
        states: list[bool] = []
        run_deterministic_loop(
            lambda _i: states.append(gc.isenabled()),
            steps=2,
            budget=DeadlineBudget(period_s=0.001, freeze_gc=False),
        )
        assert states == [True, True]
        assert gc.isenabled() is True

    def test_gc_left_disabled_if_already_disabled(self) -> None:
        gc.disable()
        try:
            run_deterministic_loop(
                lambda _i: None,
                steps=2,
                budget=DeadlineBudget(period_s=0.001, freeze_gc=True),
            )
            assert gc.isenabled() is False
        finally:
            gc.enable()

    def test_gc_restored_even_on_abort(self) -> None:
        assert gc.isenabled()
        with pytest.raises(DeadlineExceededError):
            run_deterministic_loop(
                lambda _i: time.sleep(0.02),
                steps=1,
                budget=DeadlineBudget(
                    period_s=0.05, wcet_s=0.001, miss_policy="abort", freeze_gc=True
                ),
            )
        assert gc.isenabled() is True


class TestTimingReport:
    def _report(self) -> ExecutionTimingReport:
        return run_deterministic_loop(
            lambda _i: None, steps=10, budget=DeadlineBudget(period_s=0.001)
        )

    def test_aggregate_properties(self) -> None:
        report = self._report()
        assert report.max_latency_s == pytest.approx(report.latencies_s.max())
        assert report.mean_latency_s == pytest.approx(report.latencies_s.mean())
        assert report.max_abs_jitter_s == pytest.approx(
            float(np.abs(report.jitters_s).max())
        )
        assert report.wall_time_s >= 0.0

    @pytest.mark.parametrize("percentile", [-1.0, 101.0, np.nan, True])
    def test_percentile_rejects_invalid(self, percentile: Any) -> None:
        with pytest.raises(ValueError, match="percentile"):
            self._report().latency_percentile_s(percentile)

    def test_percentile_monotone(self) -> None:
        report = self._report()
        assert report.latency_percentile_s(50.0) <= report.latency_percentile_s(99.0)

    def test_summary_keys(self) -> None:
        summary = self._report().summary()
        for key in (
            "steps",
            "period_s",
            "wcet_s",
            "mean_latency_s",
            "max_latency_s",
            "p99_latency_s",
            "max_abs_jitter_s",
            "deadline_misses",
            "deadline_met",
            "gc_frozen",
            "wall_time_s",
        ):
            assert key in summary


class TestPipelineWiring:
    def test_drives_engine_step_under_deadline_budget(self) -> None:
        """Drive a real UPDE engine step through the deterministic loop."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 8
        engine = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        state = {"phases": rng.uniform(0.0, 2.0 * np.pi, n)}
        omegas = rng.normal(0.0, 0.5, n)
        knm = 0.4 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        def step(_index: int) -> None:
            state["phases"] = engine.step(state["phases"], omegas, knm, 0.0, 0.0, alpha)

        report = run_deterministic_loop(
            step, steps=20, budget=DeadlineBudget(period_s=0.005, wcet_s=0.004)
        )
        assert report.steps == 20
        assert state["phases"].shape == (n,)
        assert np.all(np.isfinite(state["phases"]))
        assert np.all(report.latencies_s >= 0.0)
