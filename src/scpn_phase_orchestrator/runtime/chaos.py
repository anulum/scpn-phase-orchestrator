# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chaos-engineering resilience injection

"""Chaos-engineering resilience injection for orchestrated phase control.

This module injects realistic, non-actuating perturbations — coupling drops,
frequency drift, sensor noise, and drive dropout — into a controlled simulation
of a binding spec, then measures how the orchestrator recovers. Faults are
applied through the simulation's ``scenario_hook`` boundary, so the heavy compute
stays in the existing multi-language UPDE engine; this module is the
orchestration and resilience-scoring layer on top of it.

A resilience experiment runs the same seeded spec twice — once nominal, once with
the fault schedule — and compares the two order-parameter trajectories to derive
recovery time, peak coherence drop, stability-margin erosion, and final
deviation. The output is review-only evidence: nothing here actuates hardware.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.runtime.simulation import simulate

if TYPE_CHECKING:
    from scpn_phase_orchestrator.binding.types import BindingSpec
    from scpn_phase_orchestrator.runtime.simulation import (
        ScenarioCallback,
        SimulationScenarioContext,
    )

__all__ = [
    "CHAOS_FAULT_KINDS",
    "ChaosExperimentResult",
    "ChaosFault",
    "ChaosSchedule",
    "ResilienceMetrics",
    "compute_resilience",
    "make_chaos_hook",
    "run_resilience_experiment",
]

FloatArray = NDArray[np.float64]

CHAOS_FAULT_KINDS = (
    "coupling_drop",
    "frequency_drift",
    "sensor_noise",
    "drive_dropout",
)


def _positive_int(value: object, *, name: str, minimum: int = 1) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return number


def _finite_real(value: object, *, name: str, minimum: float | None = None) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


@dataclass(frozen=True)
class ChaosFault:
    """One injected fault active over a bounded step window.

    Attributes
    ----------
    kind : str
        Fault type: ``"coupling_drop"`` (scale the coupling matrix down),
        ``"frequency_drift"`` (offset the natural frequencies by a constant),
        ``"sensor_noise"`` (Gaussian phase perturbation), or ``"drive_dropout"``
        (attenuate the external drive ``zeta``).
    start_step : int
        First step at which the fault is active (``>= 1``, so the run starts
        from the spec's own initial state).
    duration_steps : int
        Number of consecutive steps the fault stays active.
    magnitude : float
        Fault strength. For ``coupling_drop`` and ``drive_dropout`` it is a
        fraction in ``[0, 1]``; for ``frequency_drift`` it is an additive offset
        in rad/s; for ``sensor_noise`` it is the noise standard deviation in rad.
    """

    kind: str
    start_step: int
    duration_steps: int
    magnitude: float

    def __post_init__(self) -> None:
        if self.kind not in CHAOS_FAULT_KINDS:
            raise ValueError(f"kind must be one of {list(CHAOS_FAULT_KINDS)}")
        _positive_int(self.start_step, name="start_step", minimum=1)
        _positive_int(self.duration_steps, name="duration_steps", minimum=1)
        if self.kind in ("coupling_drop", "drive_dropout"):
            magnitude = _finite_real(self.magnitude, name="magnitude", minimum=0.0)
            if magnitude > 1.0:
                raise ValueError(f"{self.kind} magnitude must be in [0, 1]")
        else:
            _finite_real(self.magnitude, name="magnitude", minimum=0.0)

    @property
    def end_step(self) -> int:
        """Return the first step after the fault window (exclusive).

        Returns
        -------
        int
            ``start_step + duration_steps``.
        """
        return self.start_step + self.duration_steps

    def active_at(self, step: int) -> bool:
        """Return whether the fault is active at ``step``.

        Parameters
        ----------
        step : int
            The simulation step index.

        Returns
        -------
        bool
            ``True`` when ``start_step <= step < end_step``.
        """
        return self.start_step <= step < self.end_step

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the fault.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the fault fields.
        """
        return {
            "kind": self.kind,
            "start_step": self.start_step,
            "duration_steps": self.duration_steps,
            "magnitude": self.magnitude,
        }


@dataclass(frozen=True)
class ChaosSchedule:
    """An ordered set of faults applied during a resilience experiment.

    Attributes
    ----------
    faults : tuple[ChaosFault, ...]
        The faults to inject; at least one is required.
    """

    faults: tuple[ChaosFault, ...]

    def __post_init__(self) -> None:
        if not self.faults:
            raise ValueError("a chaos schedule requires at least one fault")

    @property
    def last_fault_end(self) -> int:
        """Return the last step at which any fault is still active, plus one.

        Returns
        -------
        int
            The maximum ``end_step`` across the scheduled faults.
        """
        return max(fault.end_step for fault in self.faults)

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the schedule.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping with the fault list and window end.
        """
        return {
            "faults": [fault.to_audit_record() for fault in self.faults],
            "last_fault_end": self.last_fault_end,
        }


@dataclass(frozen=True)
class ResilienceMetrics:
    """Resilience evidence derived from nominal vs perturbed R trajectories.

    Attributes
    ----------
    recovered : bool
        Whether the perturbed run returned within ``recovery_tolerance`` of the
        nominal run after the last fault ended.
    recovery_steps : int | None
        Steps after the last fault end until recovery, or ``None`` if the run
        never recovered within the trajectory.
    max_coherence_drop : float
        Largest positive ``nominal_R - perturbed_R`` across the trajectory.
    stability_margin_erosion : float
        Mean absolute ``nominal_R - perturbed_R`` over the post-fault-onset window.
    final_deviation : float
        Absolute difference of the final order parameters.
    metrics_hash : str
        Deterministic SHA-256 over the audit record (excluding the hash).
    """

    recovered: bool
    recovery_steps: int | None
    max_coherence_drop: float
    stability_margin_erosion: float
    final_deviation: float
    metrics_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the metrics.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of every metric field.
        """
        return {
            "recovered": self.recovered,
            "recovery_steps": self.recovery_steps,
            "max_coherence_drop": self.max_coherence_drop,
            "stability_margin_erosion": self.stability_margin_erosion,
            "final_deviation": self.final_deviation,
            "metrics_hash": self.metrics_hash,
        }


@dataclass(frozen=True)
class ChaosExperimentResult:
    """Full result of one resilience experiment.

    Attributes
    ----------
    spec_name : str
        Name of the binding spec exercised.
    steps : int
        Number of simulation steps.
    seed : int
        Shared RNG seed for the nominal and perturbed runs.
    schedule : ChaosSchedule
        The injected fault schedule.
    metrics : ResilienceMetrics
        The derived resilience evidence.
    nominal_final_r : float
        Final objective order parameter of the nominal run.
    perturbed_final_r : float
        Final objective order parameter of the perturbed run.
    result_hash : str
        Deterministic SHA-256 over the audit record (excluding the hash).
    """

    spec_name: str
    steps: int
    seed: int
    schedule: ChaosSchedule
    metrics: ResilienceMetrics
    nominal_final_r: float
    perturbed_final_r: float
    result_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the experiment.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping including schedule and metrics.
        """
        return {
            "spec_name": self.spec_name,
            "steps": self.steps,
            "seed": self.seed,
            "schedule": self.schedule.to_audit_record(),
            "metrics": self.metrics.to_audit_record(),
            "nominal_final_r": self.nominal_final_r,
            "perturbed_final_r": self.perturbed_final_r,
            "non_actuating": True,
            "result_hash": self.result_hash,
        }


def make_chaos_hook(schedule: ChaosSchedule) -> ScenarioCallback:
    """Build a non-actuating scenario hook that injects the fault schedule.

    The simulation keeps whatever a scenario hook writes into its context, so
    the hook treats each fault as an overlay: at every step it first removes the
    perturbation it applied on the previous step, then applies the faults
    active on this step to the unperturbed values. A fault therefore holds a
    constant strength across its window, never compounds step over step, and
    ends at ``end_step``. Overlapping faults of the same kind combine: drift
    offsets add, coupling and drive factors multiply.

    Removal is exact for the natural frequencies, which only the hook changes.
    When the supervisor rescales the coupling during a ``coupling_drop``
    window, its multiplicative change is carried into the restored coupling;
    under a total drop (``magnitude == 1``) the dropped matrix carries no such
    change and the coupling from before the drop is restored. When the
    supervisor changes ``zeta`` during a ``drive_dropout`` window, its change is
    carried as an additive offset, and a reset to zero is kept as zero.

    Parameters
    ----------
    schedule : ChaosSchedule
        The fault schedule to apply.

    Returns
    -------
    ScenarioCallback
        A callable suitable for ``simulate(..., scenario_hook=...)``.
    """
    overlay = _ChaosOverlay()

    def hook(context: SimulationScenarioContext) -> None:
        """Remove the previous step's overlay, then apply the active faults."""
        overlay.remove(context)
        active = [fault for fault in schedule.faults if fault.active_at(context.step)]
        overlay.apply(active, context)

    return hook


@dataclass
class _ChaosOverlay:
    """Perturbation the chaos hook wrote on the previous step, for removal."""

    omega_offset: float = 0.0
    coupling_factor: float = 1.0
    clean_coupling: CouplingState | None = None
    written_coupling: CouplingState | None = None
    clean_zeta: float | None = None
    written_zeta: float | None = None

    def remove(self, context: SimulationScenarioContext) -> None:
        """Restore the unperturbed values the previous step's overlay replaced."""
        if self.omega_offset != 0.0:
            context.omegas = context.omegas - self.omega_offset
            self.omega_offset = 0.0
        if self.clean_coupling is not None:
            current = context.coupling
            if current is self.written_coupling or self.coupling_factor == 0.0:
                context.coupling = self.clean_coupling
            else:
                context.coupling = CouplingState(
                    knm=current.knm / self.coupling_factor,
                    alpha=current.alpha,
                    active_template=current.active_template,
                    knm_r=current.knm_r,
                )
            self.clean_coupling = None
            self.written_coupling = None
            self.coupling_factor = 1.0
        if self.clean_zeta is not None and self.written_zeta is not None:
            current_zeta = float(context.zeta)
            if current_zeta == self.written_zeta:
                context.zeta = self.clean_zeta
            elif current_zeta != 0.0:
                context.zeta = max(
                    0.0, self.clean_zeta + (current_zeta - self.written_zeta)
                )
            self.clean_zeta = None
            self.written_zeta = None

    def apply(
        self, faults: list[ChaosFault], context: SimulationScenarioContext
    ) -> None:
        """Apply the active faults to the unperturbed context values."""
        omega_offset = 0.0
        coupling_factor = 1.0
        zeta_factor = 1.0
        for fault in faults:
            if fault.kind == "coupling_drop":
                coupling_factor *= 1.0 - fault.magnitude
            elif fault.kind == "frequency_drift":
                omega_offset += fault.magnitude
            elif fault.kind == "sensor_noise":
                noise = context.rng.normal(
                    0.0, fault.magnitude, size=context.phases.shape
                )
                context.phases = context.phases + noise
            else:  # drive_dropout
                zeta_factor *= 1.0 - fault.magnitude
        if omega_offset != 0.0:
            context.omegas = context.omegas + omega_offset
            self.omega_offset = omega_offset
        if coupling_factor != 1.0:
            clean = context.coupling
            written = CouplingState(
                knm=clean.knm * coupling_factor,
                alpha=clean.alpha,
                active_template=clean.active_template,
                knm_r=clean.knm_r,
            )
            context.coupling = written
            self.clean_coupling = clean
            self.written_coupling = written
            self.coupling_factor = coupling_factor
        if zeta_factor != 1.0:
            clean_zeta = float(context.zeta)
            written_zeta = clean_zeta * zeta_factor
            context.zeta = written_zeta
            self.clean_zeta = clean_zeta
            self.written_zeta = written_zeta


def compute_resilience(
    nominal_history: tuple[float, ...],
    perturbed_history: tuple[float, ...],
    *,
    fault_onset_step: int,
    last_fault_end: int,
    recovery_tolerance: float,
) -> ResilienceMetrics:
    """Score resilience from nominal vs perturbed order-parameter trajectories.

    Parameters
    ----------
    nominal_history, perturbed_history : tuple[float, ...]
        Per-step objective order parameters of the nominal and perturbed runs;
        they must share length.
    fault_onset_step : int
        First step at which any fault becomes active; the erosion window starts
        here.
    last_fault_end : int
        First step after the last fault; recovery is measured from here.
    recovery_tolerance : float
        Absolute ``|nominal_R - perturbed_R|`` at or below which the perturbed
        run counts as recovered.

    Returns
    -------
    ResilienceMetrics
        The derived resilience evidence with a deterministic hash.

    Raises
    ------
    ValueError
        If the histories are empty, length-mismatched, or non-finite, or the
        parameters are out of range.
    """
    nominal = np.asarray(nominal_history, dtype=np.float64)
    perturbed = np.asarray(perturbed_history, dtype=np.float64)
    if nominal.size == 0:
        raise ValueError("nominal_history must not be empty")
    if nominal.shape != perturbed.shape:
        raise ValueError("nominal_history and perturbed_history must share length")
    if not (np.all(np.isfinite(nominal)) and np.all(np.isfinite(perturbed))):
        # NaN compares false: np.max(drop) turns NaN and max(0.0, nan) reports
        # "no drop", while |NaN| <= tol fails and hides the non-recovery.
        raise ValueError(
            "nominal_history and perturbed_history must be finite; a non-finite "
            "order parameter means the run diverged"
        )
    steps = int(nominal.size)
    onset = _positive_int(fault_onset_step, name="fault_onset_step", minimum=0)
    end = _positive_int(last_fault_end, name="last_fault_end", minimum=0)
    tolerance = _finite_real(recovery_tolerance, name="recovery_tolerance", minimum=0.0)

    signed_drop = nominal - perturbed
    max_coherence_drop = float(max(0.0, float(np.max(signed_drop))))
    erosion_window = signed_drop[min(onset, steps) :]
    stability_margin_erosion = (
        float(np.mean(np.abs(erosion_window))) if erosion_window.size else 0.0
    )
    final_deviation = float(abs(nominal[-1] - perturbed[-1]))

    recovery_steps: int | None = None
    recovered = False
    for step in range(min(end, steps), steps):
        if abs(nominal[step] - perturbed[step]) <= tolerance:
            recovery_steps = step - min(end, steps)
            recovered = True
            break

    metrics = ResilienceMetrics(
        recovered=recovered,
        recovery_steps=recovery_steps,
        max_coherence_drop=max_coherence_drop,
        stability_margin_erosion=stability_margin_erosion,
        final_deviation=final_deviation,
        metrics_hash="",
    )
    return _with_metrics_hash(metrics)


def _with_metrics_hash(metrics: ResilienceMetrics) -> ResilienceMetrics:
    """Return the metrics augmented with their canonical hash."""
    record = metrics.to_audit_record()
    record.pop("metrics_hash", None)
    serialised = json.dumps(record, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return ResilienceMetrics(
        recovered=metrics.recovered,
        recovery_steps=metrics.recovery_steps,
        max_coherence_drop=metrics.max_coherence_drop,
        stability_margin_erosion=metrics.stability_margin_erosion,
        final_deviation=metrics.final_deviation,
        metrics_hash=digest,
    )


def run_resilience_experiment(
    spec: BindingSpec,
    schedule: ChaosSchedule,
    *,
    steps: int = 200,
    seed: int = 42,
    recovery_tolerance: float = 0.05,
    binding_spec_path: Path | None = None,
) -> ChaosExperimentResult:
    """Run a nominal and a fault-injected simulation and score resilience.

    Both runs share the spec, step count, and seed, so the only difference is the
    injected fault schedule. The simulations are closed-loop (the supervisor
    reacts to the faults); the result is review-only evidence.

    Parameters
    ----------
    spec : BindingSpec
        A validated binding spec.
    schedule : ChaosSchedule
        The fault schedule to inject in the perturbed run.
    steps : int, optional
        Number of simulation steps (default ``200``); must exceed the last fault
        end so recovery can be observed.
    seed : int, optional
        Shared RNG seed (default ``42``).
    recovery_tolerance : float, optional
        Recovery tolerance passed to :func:`compute_resilience` (default ``0.05``).
    binding_spec_path : Path | None, optional
        Path of the spec file. When given, both runs load the domainpack's
        adjacent ``policy.yaml`` exactly as ``spo run`` does, so resilience is
        scored under the same closed loop the pack actually runs; without it
        only the built-in supervisor reacts.

    Returns
    -------
    ChaosExperimentResult
        The schedule, derived metrics, and final order parameters.

    Raises
    ------
    ValueError
        If ``steps`` does not exceed the last fault end, or either run produced
        a non-finite order parameter.
    """
    steps = _positive_int(steps, name="steps", minimum=1)
    if steps <= schedule.last_fault_end:
        raise ValueError("steps must exceed the schedule's last fault end")

    nominal = simulate(
        spec,
        steps=steps,
        seed=seed,
        policy_enabled=True,
        binding_spec_path=binding_spec_path,
    )
    perturbed = simulate(
        spec,
        steps=steps,
        seed=seed,
        policy_enabled=True,
        binding_spec_path=binding_spec_path,
        scenario_hook=make_chaos_hook(schedule),
    )
    fault_onset = min(fault.start_step for fault in schedule.faults)
    metrics = compute_resilience(
        nominal.r_good_history,
        perturbed.r_good_history,
        fault_onset_step=fault_onset,
        last_fault_end=schedule.last_fault_end,
        recovery_tolerance=recovery_tolerance,
    )
    result = ChaosExperimentResult(
        spec_name=nominal.spec_name,
        steps=steps,
        seed=seed,
        schedule=schedule,
        metrics=metrics,
        nominal_final_r=nominal.r_good,
        perturbed_final_r=perturbed.r_good,
        result_hash="",
    )
    return _with_result_hash(result)


def _with_result_hash(result: ChaosExperimentResult) -> ChaosExperimentResult:
    """Return the result augmented with its canonical hash."""
    record = result.to_audit_record()
    record.pop("result_hash", None)
    serialised = json.dumps(record, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return ChaosExperimentResult(
        spec_name=result.spec_name,
        steps=result.steps,
        seed=result.seed,
        schedule=result.schedule,
        metrics=result.metrics,
        nominal_final_r=result.nominal_final_r,
        perturbed_final_r=result.perturbed_final_r,
        result_hash=digest,
    )
