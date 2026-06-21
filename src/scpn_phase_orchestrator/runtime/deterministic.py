# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Bounded-jitter hard-deadline step loop

"""Bounded-jitter, hard-deadline execution mode for the control step loop.

A control loop only earns credibility if each step lands on time. This module
runs an arbitrary per-step callable against a fixed period with three guarantees
a plain ``for`` loop cannot give:

* **Bounded jitter** — every step is scheduled at ``t0 + i·period`` on the
  monotonic clock; the loop sleeps to that boundary (optionally finishing the
  last ``busy_wait_margin_s`` with a spin) and records the actual start offset
  so jitter is measured, not assumed.
* **WCET budget** — each step is timed against a worst-case execution-time
  budget; an overrun is a *deadline miss*. ``miss_policy='observe'`` records it
  and continues; ``miss_policy='abort'`` raises :class:`DeadlineExceededError`.
* **No-GC hot path** — the cyclic garbage collector is frozen and disabled for
  the duration of the loop, removing GC pauses from the jitter budget, and
  restored to its prior state afterwards.

The loop is non-actuating and timing-only: it never inspects or mutates the
step's state. The caller closes over its own state in the ``step`` callable, so
this drives the simulation step, a controller tick, or any periodic task without
coupling to a specific engine. Step *results* stay exactly as deterministic as
the callable; this module makes their *timing* bounded, which is the property
hard-real-time control needs.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

#: Per-step callable; receives the zero-based step index and returns nothing.
StepCallable: TypeAlias = Callable[[int], None]

__all__ = [
    "DeadlineBudget",
    "DeadlineExceededError",
    "ExecutionTimingReport",
    "run_deterministic_loop",
]


class DeadlineExceededError(RuntimeError):
    """Raised when a step overruns its WCET budget under ``miss_policy='abort'``.

    Attributes
    ----------
    step_index : int
        The zero-based index of the step that overran.
    latency_s : float
        The measured execution time of that step, in seconds.
    wcet_s : float
        The worst-case execution-time budget that was exceeded, in seconds.
    """

    def __init__(self, step_index: int, latency_s: float, wcet_s: float) -> None:
        self.step_index = step_index
        self.latency_s = latency_s
        self.wcet_s = wcet_s
        super().__init__(
            f"step {step_index} took {latency_s * 1e3:.3f} ms, "
            f"over the {wcet_s * 1e3:.3f} ms WCET budget"
        )


@dataclass(frozen=True)
class DeadlineBudget:
    """Timing budget for a bounded-jitter step loop.

    Attributes
    ----------
    period_s : float
        Target wall-clock period between consecutive step starts, in seconds.
    wcet_s : float
        Worst-case execution-time budget for a single step, in seconds. A step
        whose measured latency exceeds this is a deadline miss. Defaults to the
        full ``period_s`` (a step may use the whole period).
    miss_policy : str
        ``'observe'`` records deadline misses and continues; ``'abort'`` raises
        :class:`DeadlineExceededError` on the first miss.
    freeze_gc : bool
        Freeze (``gc.freeze``) and disable the cyclic garbage collector for the
        loop, restoring the prior state afterwards. Removes GC pauses from the
        jitter budget.
    busy_wait_margin_s : float
        Spin (busy-wait) for the final ``busy_wait_margin_s`` before each
        scheduled boundary instead of sleeping, trading CPU for lower jitter.
        ``0.0`` (default) sleeps the whole remainder.
    """

    period_s: float
    wcet_s: float | None = None
    miss_policy: str = "observe"
    freeze_gc: bool = True
    busy_wait_margin_s: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.period_s, bool) or not isinstance(self.period_s, Real):
            raise ValueError(f"period_s must be a positive real, got {self.period_s!r}")
        if not np.isfinite(self.period_s) or self.period_s <= 0.0:
            raise ValueError(
                f"period_s must be positive and finite, got {self.period_s}"
            )
        if self.wcet_s is not None:
            if isinstance(self.wcet_s, bool) or not isinstance(self.wcet_s, Real):
                raise ValueError(f"wcet_s must be a positive real, got {self.wcet_s!r}")
            if not np.isfinite(self.wcet_s) or self.wcet_s <= 0.0:
                raise ValueError(
                    f"wcet_s must be positive and finite, got {self.wcet_s}"
                )
        if self.miss_policy not in ("observe", "abort"):
            raise ValueError(
                f"miss_policy must be 'observe' or 'abort', got {self.miss_policy!r}"
            )
        if (
            isinstance(self.busy_wait_margin_s, bool)
            or not isinstance(self.busy_wait_margin_s, Real)
            or not np.isfinite(self.busy_wait_margin_s)
            or self.busy_wait_margin_s < 0.0
        ):
            raise ValueError(
                "busy_wait_margin_s must be a finite non-negative real, "
                f"got {self.busy_wait_margin_s!r}"
            )

    @property
    def effective_wcet_s(self) -> float:
        """Return the WCET budget, defaulting to the full period."""
        return float(self.wcet_s) if self.wcet_s is not None else float(self.period_s)


@dataclass(frozen=True)
class ExecutionTimingReport:
    """Timing record of a bounded-jitter step loop.

    Attributes
    ----------
    latencies_s : FloatArray
        Per-step execution time, shape ``(steps,)``.
    jitters_s : FloatArray
        Per-step start offset from the scheduled boundary (signed; positive is
        late), shape ``(steps,)``.
    period_s : float
        The target period.
    wcet_s : float
        The effective WCET budget against which misses were counted.
    deadline_misses : int
        Number of steps whose latency exceeded ``wcet_s``.
    gc_frozen : bool
        Whether the garbage collector was frozen/disabled for the loop.
    wall_time_s : float
        Total wall-clock time of the loop.
    """

    latencies_s: FloatArray = field(repr=False)
    jitters_s: FloatArray = field(repr=False)
    period_s: float
    wcet_s: float
    deadline_misses: int
    gc_frozen: bool
    wall_time_s: float

    @property
    def steps(self) -> int:
        """Number of completed steps."""
        return int(self.latencies_s.shape[0])

    @property
    def max_latency_s(self) -> float:
        """Largest single-step execution time."""
        return float(self.latencies_s.max()) if self.latencies_s.size else 0.0

    @property
    def mean_latency_s(self) -> float:
        """Mean step execution time."""
        return float(self.latencies_s.mean()) if self.latencies_s.size else 0.0

    @property
    def max_abs_jitter_s(self) -> float:
        """Largest absolute start-offset from the scheduled boundary."""
        return float(np.abs(self.jitters_s).max()) if self.jitters_s.size else 0.0

    @property
    def deadline_met(self) -> bool:
        """Whether every step stayed within the WCET budget."""
        return self.deadline_misses == 0

    def latency_percentile_s(self, percentile: float) -> float:
        """Return the latency at ``percentile`` (0-100)."""
        if isinstance(percentile, bool) or not isinstance(percentile, Real):
            raise ValueError(
                f"percentile must be a real in [0, 100], got {percentile!r}"
            )
        value = float(percentile)
        if not np.isfinite(value) or value < 0.0 or value > 100.0:
            raise ValueError(f"percentile must lie in [0, 100], got {percentile!r}")
        if not self.latencies_s.size:
            return 0.0
        return float(np.percentile(self.latencies_s, value))

    def summary(self) -> dict[str, float | int | bool]:
        """Return a flat scalar summary for logging or metric export."""
        return {
            "steps": self.steps,
            "period_s": self.period_s,
            "wcet_s": self.wcet_s,
            "mean_latency_s": self.mean_latency_s,
            "max_latency_s": self.max_latency_s,
            "p99_latency_s": self.latency_percentile_s(99.0),
            "max_abs_jitter_s": self.max_abs_jitter_s,
            "deadline_misses": self.deadline_misses,
            "deadline_met": self.deadline_met,
            "gc_frozen": self.gc_frozen,
            "wall_time_s": self.wall_time_s,
        }


def _validate_steps(steps: object) -> int:
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError(f"steps must be a non-negative integer, got {steps!r}")
    if steps < 0:
        raise ValueError(f"steps must be a non-negative integer, got {steps}")
    return steps


def _sleep_until(target_ns: int, busy_wait_margin_ns: int) -> None:
    """Sleep, then optionally spin, until ``target_ns`` on the monotonic clock."""
    spin_from = target_ns - busy_wait_margin_ns
    remaining = spin_from - time.perf_counter_ns()
    if remaining > 0:
        time.sleep(remaining / 1e9)
    if busy_wait_margin_ns > 0:
        while time.perf_counter_ns() < target_ns:
            pass


def run_deterministic_loop(
    step: StepCallable,
    *,
    steps: int,
    budget: DeadlineBudget,
) -> ExecutionTimingReport:
    """Drive ``step`` for ``steps`` iterations under a bounded-jitter budget.

    Parameters
    ----------
    step : StepCallable
        Per-step callable receiving the zero-based step index. It owns all state
        through its closure; its return value is ignored.
    steps : int
        Number of iterations (``>= 0``).
    budget : DeadlineBudget
        The period, WCET budget, miss policy, GC policy, and spin margin.

    Returns
    -------
    ExecutionTimingReport
        Per-step latencies and jitters plus aggregate timing statistics.

    Raises
    ------
    ValueError
        If ``steps`` is not a non-negative integer.
    DeadlineExceededError
        If a step overruns ``budget.wcet_s`` and ``miss_policy == 'abort'``.
    """
    steps = _validate_steps(steps)
    period_ns = round(budget.period_s * 1e9)
    wcet_s = budget.effective_wcet_s
    margin_ns = round(budget.busy_wait_margin_s * 1e9)
    latencies = np.zeros(steps, dtype=np.float64)
    jitters = np.zeros(steps, dtype=np.float64)
    deadline_misses = 0

    gc_was_enabled = gc.isenabled()
    if budget.freeze_gc:
        gc.collect()
        gc.freeze()
        gc.disable()
    loop_start_ns = time.perf_counter_ns()
    try:
        for index in range(steps):
            scheduled_ns = loop_start_ns + index * period_ns
            if time.perf_counter_ns() < scheduled_ns:
                _sleep_until(scheduled_ns, margin_ns)
            actual_start_ns = time.perf_counter_ns()
            jitters[index] = (actual_start_ns - scheduled_ns) / 1e9
            step(index)
            latency_s = (time.perf_counter_ns() - actual_start_ns) / 1e9
            latencies[index] = latency_s
            if latency_s > wcet_s:
                deadline_misses += 1
                if budget.miss_policy == "abort":
                    raise DeadlineExceededError(index, latency_s, wcet_s)
        wall_time_s = (time.perf_counter_ns() - loop_start_ns) / 1e9
    finally:
        if budget.freeze_gc:
            gc.unfreeze()
            if gc_was_enabled:
                gc.enable()

    return ExecutionTimingReport(
        latencies_s=latencies,
        jitters_s=jitters,
        period_s=float(budget.period_s),
        wcet_s=wcet_s,
        deadline_misses=deadline_misses,
        gc_frozen=budget.freeze_gc,
        wall_time_s=wall_time_s,
    )
