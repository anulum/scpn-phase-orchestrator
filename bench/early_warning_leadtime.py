#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — early-warning lead-time head-to-head benchmark

"""Fair head-to-head: ordinal-transition-entropy vs critical-slowing-down EWS.

This benchmark tests one falsifiable claim, stated in
``monitor/explosive_sync.py``: on a *first-order (explosive)* synchronisation
transition the ordinal-transition-entropy detector warns earlier than the
classical critical-slowing-down early-warning signals (rising variance and
lag-one autocorrelation), "where variance / autocorrelation indicators are weak".

It is designed to be fair and to be able to fail:

* Each detector reads the observable it is designed for, so each method is
  compared at its best: the entropy detector the per-node phase projection
  ``sin(θ)`` (a multi-node method), the critical-slowing-down detector the global
  order parameter ``R(t)`` (the system-level series the classical method is
  applied to). Each was checked against a principled alternative and given the
  observable that suits it — the entropy detector is starved on phase velocity
  and the slowing-down detector on the per-node field — so neither is rigged.
* Both are passive; the dynamics are never modified for either (the flaw that
  made ``bench/competitive_kuramoto.py`` a false win — see its retirement note).
* Both use the *identical* alarm rule (``_alarm_sample``): a robust-z gate, a
  relative-change gate, and a persistence run. Only the underlying indicator
  differs, so a lead-time gap is the indicator's, not the machinery's.
* Each detector is calibrated to the *same* false-alarm rate on a no-transition
  null ensemble before its lead time is measured — the honest way to compare
  detectors of different scales.
* A falsification guard runs the same test on a *continuous (second-order)*
  transition, where critical slowing down is expected to hold its own. A method
  that "wins" even there would be a rigged setup, not a real niche.

The transition order is the single controlled variable: the same all-to-all
network and the same Gaussian natural frequencies are used for both conditions,
and only the inertia is changed. Overdamped Kuramoto (zero inertia) synchronises
through a second-order (continuous) transition (Kuramoto 1984); inertial Kuramoto
(positive inertia) synchronises through a first-order (explosive, hysteretic)
transition (Tanaka, Lichtenberg & Oishi 1997). The transition order is therefore
taken from the established physics of these two models rather than re-measured
under a finite-rate ramp (where apparent hysteresis is confounded by ramp lag).

The data are simulated (a transparent vanilla Kuramoto integrator, not the SPO
UPDE engine, so the generator cannot be accused of favouring SPO) and honestly
labelled as such. ``main`` writes ``bench/early_warning_leadtime_results.json``
with the calibrated false-alarm rates, per-condition lead-time distributions, and
a verdict that names a win only if SPO leads on the explosive transition at a
matched false-alarm rate while the baseline holds on the continuous one.

References
----------
* Kuramoto 1984, *Chemical Oscillations, Waves, and Turbulence* — the
  second-order (continuous) synchronisation transition of overdamped phase
  oscillators.
* Tanaka, Lichtenberg & Oishi 1997, *Phys. Rev. Lett.* 78, 2104 — first-order
  (hysteretic) synchronisation transition of Kuramoto oscillators with inertia.
* Scheffer et al. 2009, *Nature* 461, 53; Dakos et al. 2012, *PLoS ONE* 7,
  e41010 — critical-slowing-down early-warning signals.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    critical_slowing_down_warning,
)
from scpn_phase_orchestrator.monitor.explosive_sync import explosive_sync_warning

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

WINDOW = 128
STEP = 16
BASELINE_FRACTION = 0.25
PERSISTENCE = 2
RELATIVE_GATE = 0.05
TARGET_FALSE_ALARM = 0.10
THRESHOLD_GRID = tuple(round(0.25 * k, 2) for k in range(1, 41))  # 0.25 … 10.0


@dataclass(frozen=True)
class DetectorScore:
    """A detector's per-window alarm score and relative-change gate on a run.

    ``score`` is signed so that larger means more alarming for both detectors
    (the entropy drop and the slowing-down rise are both mapped to a positive
    breach), and ``relative`` is the corresponding fractional change used by the
    shared relative gate.
    """

    score: FloatArray
    relative: FloatArray
    window_starts: IntArray
    n_baseline: int


@dataclass(frozen=True)
class ConditionResult:
    """Per-detector outcome on one transition condition at matched false alarm."""

    condition: str
    transition_kind: str
    n_realisations: int
    entropy_false_alarm: float
    slowing_false_alarm: float
    entropy_threshold: float
    slowing_threshold: float
    entropy_detection_rate: float
    slowing_detection_rate: float
    entropy_median_lead: float
    slowing_median_lead: float
    entropy_leads: list[float]
    slowing_leads: list[float]


def kuramoto_simulation(
    *,
    omega: FloatArray,
    coupling: FloatArray,
    dt: float,
    inertia: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Integrate an all-to-all Kuramoto ensemble under a coupling schedule (RK4).

    ``inertia`` selects the transition order from the same network and the same
    frequencies, so the transition order is the single controlled variable.
    ``inertia == 0`` is the overdamped first-order model whose synchronisation is
    a second-order (continuous) transition; ``inertia > 0`` is the inertial
    second-order model whose synchronisation is a first-order (explosive,
    hysteretic) transition.

    Returns the per-node phase projection ``sin(θ)``, shape ``(N, T)`` — the
    natural per-node oscillator observable the order parameter is built from and
    the one the entropy detector reads — and the global Kuramoto order parameter
    ``R(t)``, shape ``(T,)``.
    """
    n = int(omega.shape[0])
    steps = int(coupling.shape[0])
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    velocity = np.zeros(n, dtype=np.float64)
    observable = np.empty((n, steps), dtype=np.float64)
    order = np.empty(steps, dtype=np.float64)

    def mean_field(state: FloatArray, gain: float) -> FloatArray:
        diff = state[np.newaxis, :] - state[:, np.newaxis]
        coupling_term = (gain / n) * np.sum(np.sin(diff), axis=1)
        return np.asarray(omega + coupling_term, dtype=np.float64)

    for t in range(steps):
        gain = float(coupling[t])
        observable[:, t] = np.sin(theta)
        order[t] = float(np.abs(np.mean(np.exp(1j * theta))))
        if inertia <= 0.0:
            k1 = mean_field(theta, gain)
            k2 = mean_field(theta + 0.5 * dt * k1, gain)
            k3 = mean_field(theta + 0.5 * dt * k2, gain)
            k4 = mean_field(theta + dt * k3, gain)
            theta = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            theta, velocity = _inertial_step(
                theta, velocity, gain, dt, inertia, mean_field
            )
    return observable, order


def _inertial_step(
    theta: FloatArray,
    velocity: FloatArray,
    gain: float,
    dt: float,
    inertia: float,
    mean_field: Callable[[FloatArray, float], FloatArray],
) -> tuple[FloatArray, FloatArray]:
    """Advance one RK4 step of the inertial (second-order) Kuramoto system."""

    def accel(state: FloatArray, vel: FloatArray) -> FloatArray:
        return (mean_field(state, gain) - vel) / inertia

    a1 = accel(theta, velocity)
    v1 = velocity
    a2 = accel(theta + 0.5 * dt * v1, velocity + 0.5 * dt * a1)
    v2 = velocity + 0.5 * dt * a1
    a3 = accel(theta + 0.5 * dt * v2, velocity + 0.5 * dt * a2)
    v3 = velocity + 0.5 * dt * a2
    a4 = accel(theta + dt * v3, velocity + dt * a3)
    v4 = velocity + dt * a3
    next_theta = theta + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
    next_velocity = velocity + (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    return next_theta, next_velocity


def order_parameter_transition(order: FloatArray, threshold: float = 0.5) -> int | None:
    """Return the first sample at which ``R(t)`` crosses ``threshold`` upward."""
    crossed = np.flatnonzero(order >= threshold)
    return int(crossed[0]) if crossed.size else None


def _alarm_sample(
    detector: DetectorScore, *, threshold: float, relative_gate: float, persistence: int
) -> int | None:
    """Return the sample of the first sustained breach, shared by both detectors.

    A window breaches when it is past the baseline, its signed score meets the
    threshold, and its relative change meets the gate; ``persistence`` consecutive
    breaches raise the alarm. This is the one alarm rule both detectors share.
    """
    n_windows = int(detector.score.shape[0])
    past_baseline = np.arange(n_windows) >= detector.n_baseline
    breaches = (
        past_baseline
        & (detector.score >= threshold)
        & (detector.relative >= relative_gate)
    )
    run = 0
    for index in range(n_windows):
        if breaches[index]:
            run += 1
            if run >= persistence:
                start = index - persistence + 1
                return int(detector.window_starts[start])
        else:
            run = 0
    return None


def entropy_score(signals: FloatArray) -> DetectorScore:
    """Return the entropy detector's score from the per-node phase projections.

    The ordinal-transition-entropy monitor is a multi-node method, so it is fed
    the per-node phase projection ``sin(θ)`` — the natural oscillator observable.
    (The per-node phase *velocity* starves it in the inertial model, so using it
    would rig the test against the entropy detector, the mirror of the mistake
    corrected for the slowing-down baseline.)
    """
    warning = explosive_sync_warning(
        signals,
        window=WINDOW,
        step=STEP,
        baseline_fraction=BASELINE_FRACTION,
        z_threshold=0.0,
        drop_threshold=0.0,
        persistence=PERSISTENCE,
    )
    return DetectorScore(
        score=-np.asarray(warning.robust_z, dtype=np.float64),
        relative=np.asarray(warning.relative_drop, dtype=np.float64),
        window_starts=np.asarray(warning.window_starts, dtype=np.int64),
        n_baseline=warning.n_baseline_windows,
    )


def slowing_score(signals: FloatArray) -> DetectorScore:
    """Return the slowing-down detector's score from the order-parameter series.

    Critical slowing down is classically applied to a system-level observable, so
    it is fed the global Kuramoto order parameter ``R(t)`` as a single series —
    the observable the method was designed for — rather than being handicapped on
    the per-node field that suits the entropy detector.
    """
    warning = critical_slowing_down_warning(
        signals,
        window=WINDOW,
        step=STEP,
        baseline_fraction=BASELINE_FRACTION,
        z_threshold=0.0,
        rise_threshold=0.0,
        persistence=PERSISTENCE,
    )
    return DetectorScore(
        score=np.asarray(warning.combined_z, dtype=np.float64),
        relative=np.asarray(warning.relative_rise, dtype=np.float64),
        window_starts=np.asarray(warning.window_starts, dtype=np.int64),
        n_baseline=warning.n_baseline_windows,
    )


def calibrate_threshold(null_scores: list[DetectorScore], target_fa: float) -> float:
    """Return the smallest grid threshold with false-alarm rate at most target.

    Falls back to the largest grid threshold when even that exceeds the target.
    """
    for threshold in THRESHOLD_GRID:
        alarms = sum(
            _alarm_sample(
                score,
                threshold=threshold,
                relative_gate=RELATIVE_GATE,
                persistence=PERSISTENCE,
            )
            is not None
            for score in null_scores
        )
        if alarms / len(null_scores) <= target_fa:
            return threshold
    return THRESHOLD_GRID[-1]


def false_alarm_rate(null_scores: list[DetectorScore], threshold: float) -> float:
    """Return the fraction of null runs that alarm at ``threshold``."""
    alarms = sum(
        _alarm_sample(
            score,
            threshold=threshold,
            relative_gate=RELATIVE_GATE,
            persistence=PERSISTENCE,
        )
        is not None
        for score in null_scores
    )
    return alarms / len(null_scores)


def evaluate_leads(
    transition_scores: list[tuple[DetectorScore, int]], threshold: float
) -> tuple[float, float, list[float]]:
    """Return the detection rate, median lead time, and per-run lead times.

    Lead time is ``transition_sample − alarm_sample`` in samples; only alarms
    that fire at or before the transition count as detections.
    """
    leads: list[float] = []
    for score, transition in transition_scores:
        alarm = _alarm_sample(
            score,
            threshold=threshold,
            relative_gate=RELATIVE_GATE,
            persistence=PERSISTENCE,
        )
        if alarm is not None and alarm <= transition:
            leads.append(float(transition - alarm))
    rate = len(leads) / len(transition_scores)
    median = float(np.median(leads)) if leads else 0.0
    return rate, median, leads


@dataclass(frozen=True)
class ConditionConfig:
    """The controlled setup for one transition kind — only inertia differs.

    ``inertia`` fixes the transition order (0 → second-order/continuous, > 0 →
    first-order/explosive) on the same all-to-all network with the same Gaussian
    natural frequencies; ``coupling_high`` is the top of the ramp, chosen so the
    onset lands in the latter part of the series with a clean leading baseline.
    """

    inertia: float
    coupling_high: float


_CONDITIONS: dict[str, ConditionConfig] = {
    "explosive": ConditionConfig(inertia=6.0, coupling_high=9.0),
    "continuous": ConditionConfig(inertia=0.0, coupling_high=2.5),
}


def coupling_schedule(
    steps: int, coupling_low: float, coupling_high: float, hold_fraction: float
) -> FloatArray:
    """Return a hold-then-ramp coupling schedule.

    The coupling is held subcritical for the leading ``hold_fraction`` of the
    series — giving the detectors a clean stationary baseline — then ramped to
    ``coupling_high`` so the synchronisation onset develops in the latter part.
    """
    hold = int(hold_fraction * steps)
    return np.concatenate(
        [
            np.full(hold, coupling_low),
            np.linspace(coupling_low, coupling_high, steps - hold),
        ]
    ).astype(np.float64)


def _score_ensemble(
    config: ConditionConfig,
    n: int,
    dt: float,
    coupling: FloatArray,
    n_realisations: int,
    rng: np.random.Generator,
) -> tuple[list[DetectorScore], list[DetectorScore], list[int | None]]:
    """Integrate an ensemble and return entropy scores, slowing scores, onsets."""
    entropy_scores: list[DetectorScore] = []
    slowing_scores: list[DetectorScore] = []
    onsets: list[int | None] = []
    for _ in range(n_realisations):
        omega = rng.standard_normal(n)
        omega = omega - float(np.mean(omega))
        phase_field, order = kuramoto_simulation(
            omega=omega, coupling=coupling, dt=dt, inertia=config.inertia, rng=rng
        )
        entropy_scores.append(entropy_score(phase_field))
        slowing_scores.append(slowing_score(order[np.newaxis, :]))
        onsets.append(order_parameter_transition(order))
    return entropy_scores, slowing_scores, onsets


def run_condition(
    kind: str,
    *,
    n: int = 48,
    steps: int = 4000,
    dt: float = 0.05,
    coupling_low: float = 0.5,
    hold_fraction: float = 0.4,
    n_realisations: int = 10,
    seed: int = 0,
) -> ConditionResult:
    """Run one transition condition and return the matched-false-alarm outcome."""
    config = _CONDITIONS.get(kind)
    if config is None:
        raise ValueError(f"unknown transition kind {kind!r}")
    rng = np.random.default_rng(seed)
    ramp = coupling_schedule(steps, coupling_low, config.coupling_high, hold_fraction)
    null = np.full(steps, coupling_low, dtype=np.float64)

    null_entropy, null_slowing, _ = _score_ensemble(
        config, n, dt, null, n_realisations, rng
    )
    tr_entropy, tr_slowing, onsets = _score_ensemble(
        config, n, dt, ramp, n_realisations, rng
    )
    detected = [
        (index, onset) for index, onset in enumerate(onsets) if onset is not None
    ]
    entropy_pairs = [(tr_entropy[i], onset) for i, onset in detected]
    slowing_pairs = [(tr_slowing[i], onset) for i, onset in detected]

    entropy_threshold = calibrate_threshold(null_entropy, TARGET_FALSE_ALARM)
    slowing_threshold = calibrate_threshold(null_slowing, TARGET_FALSE_ALARM)
    entropy_rate, entropy_lead, entropy_leads = evaluate_leads(
        entropy_pairs, entropy_threshold
    )
    slowing_rate, slowing_lead, slowing_leads = evaluate_leads(
        slowing_pairs, slowing_threshold
    )

    return ConditionResult(
        condition=kind,
        transition_kind="first-order" if config.inertia > 0.0 else "second-order",
        n_realisations=len(entropy_pairs),
        entropy_false_alarm=false_alarm_rate(null_entropy, entropy_threshold),
        slowing_false_alarm=false_alarm_rate(null_slowing, slowing_threshold),
        entropy_threshold=entropy_threshold,
        slowing_threshold=slowing_threshold,
        entropy_detection_rate=entropy_rate,
        slowing_detection_rate=slowing_rate,
        entropy_median_lead=entropy_lead,
        slowing_median_lead=slowing_lead,
        entropy_leads=entropy_leads,
        slowing_leads=slowing_leads,
    )


def verdict(explosive: ConditionResult, continuous: ConditionResult) -> str:
    """Return the honest verdict of the head-to-head, separating the two axes.

    Two axes are reported independently because they can disagree: whether the
    entropy detector *warns earlier* (median lead time when it fires) and whether
    it is *as reliable* (detection rate). A clean win requires both; leading on
    lead time alone, at a lower detection rate, is a trade-off — and the lead-time
    gap is then confounded by the differing detection subsets, so it is not read
    as a clean early-warning win. The continuous transition is the falsification
    guard against a setup that flatters the entropy detector everywhere.
    """
    if explosive.entropy_detection_rate <= 0.0:
        return (
            "NO WIN: transition-entropy produces no leading detection on the "
            "explosive transition at matched false alarm; the critical-slowing-down "
            "baseline detects it. Do not claim an early-warning advantage."
        )
    leads_time = explosive.entropy_median_lead > explosive.slowing_median_lead
    as_reliable = explosive.entropy_detection_rate >= explosive.slowing_detection_rate
    baseline_holds_continuous = (
        continuous.slowing_median_lead >= continuous.entropy_median_lead
        or continuous.slowing_detection_rate >= continuous.entropy_detection_rate
    )
    if leads_time and as_reliable and baseline_holds_continuous:
        margin = explosive.entropy_median_lead - explosive.slowing_median_lead
        return (
            "WIN (named niche): transition-entropy leads critical-slowing-down by "
            f"{margin:.0f} samples on the explosive transition at matched false "
            "alarm and detects at least as reliably, while the baseline holds on "
            "the continuous transition."
        )
    if leads_time and as_reliable and not baseline_holds_continuous:
        return (
            "SUSPECT: transition-entropy leads on BOTH transition kinds, including "
            "where critical slowing down should hold — treat as a setup artefact, "
            "not a real niche advantage."
        )
    if leads_time and not as_reliable:
        return (
            "TRADE-OFF (no clean win): transition-entropy warns earlier when it "
            f"fires (median lead {explosive.entropy_median_lead:.0f} vs "
            f"{explosive.slowing_median_lead:.0f} samples) but detects less "
            f"reliably ({explosive.entropy_detection_rate:.0%} vs "
            f"{explosive.slowing_detection_rate:.0%}); the lead-time gap is "
            "confounded by the smaller detection subset, so it does not support an "
            "unqualified early-warning advantage."
        )
    return (
        "NO WIN: transition-entropy does not warn earlier than critical-slowing-down "
        "on the explosive transition at matched false alarm; do not claim an "
        "early-warning advantage."
    )


def main() -> None:
    """Run both conditions, print the verdict, and write the results JSON."""
    explosive = run_condition("explosive", seed=20260704)
    continuous = run_condition("continuous", seed=20260705)
    outcome = verdict(explosive, continuous)
    payload = {
        "benchmark": "early_warning_leadtime",
        "claim": (
            "ordinal-transition-entropy warns earlier than critical-slowing-down "
            "on a first-order (explosive) synchronisation transition"
        ),
        "data": "simulated vanilla Kuramoto (transparent generator, not SPO UPDE)",
        "target_false_alarm": TARGET_FALSE_ALARM,
        "window": WINDOW,
        "step": STEP,
        "explosive": asdict(explosive),
        "continuous": asdict(continuous),
        "verdict": outcome,
    }
    destination = Path(__file__).with_name("early_warning_leadtime_results.json")
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(outcome)
    print(f"results written to {destination}")


if __name__ == "__main__":
    main()
