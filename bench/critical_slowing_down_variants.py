#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — critical-slowing-down variant benchmark

"""Benchmark critical-slowing-down variants on controlled Kuramoto transitions.

The classical CSD detector is the multi-domain winner from the cross-domain
meta-analysis, but its wins are sparse. This benchmark evaluates two refinement
ideas on a transparent synthetic corpus:

* **multi-scale** CSD — combine variance/autocorrelation indices across a bank
  of window lengths;
* **surrogate-aware** threshold — set the alarm gate from a block-bootstrap
  null distribution of the combined score rather than from a Gaussian-MAD
  robust z-score.

Each variant is calibrated to the same target false-alarm rate on an independent
null ensemble, then scored on first-order (explosive) and second-order
(continuous) synchronisation transitions. The results are written as an
honest-audit-style aggregate so the meta-analysis tool can rank the variants
alongside the real-data detectors.

This driver reuses the vanilla Kuramoto simulator and transition definitions
from ``bench/early_warning_leadtime.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    CriticalSlowingDownWarning,
    critical_slowing_down_multiscale_warning,
    critical_slowing_down_warning,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

WINDOW = 128
STEP = 16
BASELINE_FRACTION = 0.25
PERSISTENCE = 2
TARGET_FALSE_ALARM = 0.10


def _load_early_warning_leadtime() -> Any:
    """Load ``bench/early_warning_leadtime.py`` as a module for reuse."""
    here = Path(__file__).resolve().parent
    target = here / "early_warning_leadtime.py"
    spec = importlib.util.spec_from_file_location("early_warning_leadtime", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["early_warning_leadtime"] = module
    spec.loader.exec_module(module)
    return module


_EWL = _load_early_warning_leadtime()


@dataclass(frozen=True)
class DetectorScore:
    """Per-recording score trajectory consumed by the calibration/evaluation."""

    score: FloatArray
    relative: FloatArray
    window_starts: IntArray
    n_baseline: int


def _score_from_warning(warning: CriticalSlowingDownWarning) -> DetectorScore:
    """Turn a CSD warning record into the neutral score shape."""
    return DetectorScore(
        score=np.asarray(warning.combined_z, dtype=np.float64),
        relative=np.asarray(warning.relative_rise, dtype=np.float64),
        window_starts=np.asarray(warning.window_starts, dtype=np.int64),
        n_baseline=warning.n_baseline_windows,
    )


def _baseline_score(order: FloatArray) -> DetectorScore:
    """Classical CSD score on the global order-parameter series."""
    warning = critical_slowing_down_warning(
        order[np.newaxis, :],
        window=WINDOW,
        step=STEP,
        baseline_fraction=BASELINE_FRACTION,
        z_threshold=0.0,
        rise_threshold=0.0,
        persistence=PERSISTENCE,
    )
    return _score_from_warning(warning)


def _multiscale_score(order: FloatArray) -> DetectorScore:
    """Multi-scale CSD score on the global order-parameter series."""
    warning = critical_slowing_down_multiscale_warning(
        order[np.newaxis, :],
        windows=(64, 128, 256),
        step=STEP,
        baseline_fraction=BASELINE_FRACTION,
        z_threshold=0.0,
        rise_threshold=0.0,
        persistence=PERSISTENCE,
        aggregation="max",
    )
    return _score_from_warning(warning)


def _surrogate_score(order: FloatArray) -> DetectorScore:
    """Surrogate-threshold CSD score (threshold is applied later)."""
    # The score trajectory is identical to the baseline; only the gate differs.
    return _baseline_score(order)


def _alarm_sample(
    detector: DetectorScore, *, threshold: float, relative_gate: float
) -> int | None:
    """Return the sample of the first sustained breach."""
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
            if run >= PERSISTENCE:
                start = index - PERSISTENCE + 1
                return int(detector.window_starts[start])
        else:
            run = 0
    return None


def _empirical_threshold(null_scores: list[DetectorScore], target_fa: float) -> float:
    """Return the (1 - target_fa) percentile of null maximum scores."""
    max_scores = np.asarray(
        [float(s.score.max()) for s in null_scores], dtype=np.float64
    )
    return float(np.percentile(max_scores, 100.0 * (1.0 - target_fa)))


def _surrogate_threshold(
    null_orders: list[FloatArray],
    target_fa: float,
    rng: np.random.Generator,
    n_surrogates: int = 100,
) -> float:
    """Return a threshold calibrated on block-bootstrap surrogates of the nulls."""
    from scpn_phase_orchestrator.monitor.critical_slowing_down import (
        _block_bootstrap,
    )

    pooled = np.concatenate([np.asarray(o, dtype=np.float64) for o in null_orders])
    n_samples = int(pooled.shape[0])
    block_length = max(1, n_samples // 20)
    max_scores = np.empty(n_surrogates, dtype=np.float64)
    for idx in range(n_surrogates):
        surrogate = _block_bootstrap(pooled[np.newaxis, :], block_length, rng).ravel()
        score = _baseline_score(surrogate)
        max_scores[idx] = float(score.score.max())
    return float(np.percentile(max_scores, 100.0 * (1.0 - target_fa)))


def _evaluate(
    transition_scores: list[DetectorScore],
    onsets: list[int | None],
    threshold: float,
) -> tuple[float, float, list[float]]:
    """Return detection rate, median lead, and per-run leads at ``threshold``."""
    leads: list[float] = []
    detected = 0
    for score, onset in zip(transition_scores, onsets, strict=True):
        if onset is None:
            continue
        detected += 1
        alarm = _alarm_sample(
            score,
            threshold=threshold,
            relative_gate=0.05,
        )
        if alarm is not None and alarm <= onset:
            leads.append(float(onset - alarm))
    rate = len(leads) / detected if detected else 0.0
    median = float(np.median(leads)) if leads else 0.0
    return rate, median, leads


def _score_ensemble(
    config: Any,
    n: int,
    dt: float,
    coupling: FloatArray,
    n_realisations: int,
    rng: np.random.Generator,
    scorer: Any,
) -> tuple[list[DetectorScore], list[FloatArray], list[int | None]]:
    """Integrate an ensemble and score each realisation with ``scorer``."""
    scores: list[DetectorScore] = []
    orders: list[FloatArray] = []
    onsets: list[int | None] = []
    for _ in range(n_realisations):
        omega = rng.standard_normal(n)
        omega = omega - float(np.mean(omega))
        phase_field, order = _EWL.kuramoto_simulation(
            omega=omega, coupling=coupling, dt=dt, inertia=config.inertia, rng=rng
        )
        orders.append(order)
        scores.append(scorer(order))
        onsets.append(_EWL.order_parameter_transition(order))
    return scores, orders, onsets


def _run_condition(
    kind: str,
    *,
    n: int,
    steps: int,
    dt: float,
    coupling_low: float,
    hold_fraction: float,
    n_realisations: int,
    seed: int,
) -> dict[str, Any]:
    """Run one transition condition for all CSD variants."""
    config = _EWL._CONDITIONS.get(kind)
    if config is None:
        raise ValueError(f"unknown transition kind {kind!r}")
    rng = np.random.default_rng(seed)
    ramp = _EWL.coupling_schedule(
        steps, coupling_low, config.coupling_high, hold_fraction
    )
    null_coupling = np.full(steps, coupling_low, dtype=np.float64)

    variants = {
        "critical_slowing_down_baseline": _baseline_score,
        "critical_slowing_down_multiscale": _multiscale_score,
        "critical_slowing_down_surrogate": _surrogate_score,
    }

    results: dict[str, Any] = {
        "condition": kind,
        "transition_kind": "first-order" if config.inertia > 0.0 else "second-order",
        "n_realisations": n_realisations,
        "target_false_alarm": TARGET_FALSE_ALARM,
        "window": WINDOW,
        "step": STEP,
    }

    for name, scorer in variants.items():
        null_scores, null_orders, _ = _score_ensemble(
            config, n, dt, null_coupling, n_realisations, rng, scorer
        )
        tr_scores, _, onsets = _score_ensemble(
            config, n, dt, ramp, n_realisations, rng, scorer
        )

        if name == "critical_slowing_down_surrogate":
            threshold = _surrogate_threshold(null_orders, TARGET_FALSE_ALARM, rng)
        else:
            threshold = _empirical_threshold(null_scores, TARGET_FALSE_ALARM)

        fa = (
            sum(
                _alarm_sample(s, threshold=threshold, relative_gate=0.05) is not None
                for s in null_scores
            )
            / len(null_scores)
            if null_scores
            else 0.0
        )
        rate, median, leads = _evaluate(tr_scores, onsets, threshold)
        results[name] = {
            "threshold": threshold,
            "achieved_false_alarm": fa,
            "detection_rate": rate,
            "median_lead": median,
            "leads": leads,
        }

    return results


def _binomial_p_value(k: int, n: int, p: float) -> float:
    """Two-sided binomial p-value for H0: success probability = p."""
    from scipy import stats

    return float(stats.binomtest(k, n, p, alternative="two-sided").pvalue)


def _build_aggregate(
    explosive: dict[str, Any], continuous: dict[str, Any], out_dir: Path
) -> Path:
    """Write an honest-audit-style aggregate JSON for the meta-analysis tool."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_fa = explosive["target_false_alarm"]
    variants = [
        "critical_slowing_down_baseline",
        "critical_slowing_down_multiscale",
        "critical_slowing_down_surrogate",
    ]

    per_recording: list[dict[str, Any]] = []
    for condition_name, condition in (
        ("explosive", explosive),
        ("continuous", continuous),
    ):
        detectors: dict[str, Any] = {}
        n_events = condition["n_realisations"]
        for variant in variants:
            entry = condition[variant]
            k = int(round(entry["detection_rate"] * n_events))
            p_value = _binomial_p_value(k, n_events, target_fa)
            detectors[variant] = {
                "detector_name": variant,
                "detection_rate": entry["detection_rate"],
                "p_value": p_value,
                "beats_chance": entry["detection_rate"] > target_fa,
                "median_lead": entry["median_lead"],
                "achieved_false_alarm": entry["achieved_false_alarm"],
            }
        per_recording.append(
            {
                "recording_id": condition_name,
                "n_events": n_events,
                "detectors": detectors,
            }
        )

    summaries: dict[str, Any] = {}
    for variant in variants:
        rates = [
            explosive[variant]["detection_rate"],
            continuous[variant]["detection_rate"],
        ]
        mean_rate = float(np.mean(rates))
        # Geometric mean p-value across the two conditions.
        p_values = [
            _binomial_p_value(
                int(
                    round(
                        explosive[variant]["detection_rate"]
                        * explosive["n_realisations"]
                    )
                ),
                explosive["n_realisations"],
                target_fa,
            ),
            _binomial_p_value(
                int(
                    round(
                        continuous[variant]["detection_rate"]
                        * continuous["n_realisations"]
                    )
                ),
                continuous["n_realisations"],
                target_fa,
            ),
        ]
        geom_p = float(np.exp(np.mean(np.log(np.maximum(p_values, 1e-300)))))
        summaries[variant] = {
            "mean_detection_rate": mean_rate,
            "geometric_mean_p_value": geom_p,
            "fraction_beats_chance": float(np.mean([r > target_fa for r in rates])),
        }

    aggregate = {
        "benchmark": "csd_variant_synthetic",
        "corpus": (
            "vanilla Kuramoto first- and second-order synchronisation transitions"
        ),
        "target_false_alarm": target_fa,
        "n_recordings": 2,
        "recording_ids": ["explosive", "continuous"],
        "per_recording": per_recording,
        **summaries,
        "recommendation": {
            "refine": True,
            "preferred_variant": max(
                variants, key=lambda v: summaries[v]["mean_detection_rate"]
            ),
            "rationale": (
                "Highest mean detection rate across both transition kinds at "
                "matched false alarm."
            ),
        },
    }

    path = out_dir / "csd_variant_synthetic_results.json"
    path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the variant benchmark and write the aggregate JSON."""
    parser = argparse.ArgumentParser(
        description="Benchmark critical-slowing-down variants on Kuramoto transitions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "examples"
        / "real_data"
        / "csd_variant_synthetic",
        help="Directory for the aggregate JSON output.",
    )
    parser.add_argument("--n", type=int, default=32, help="Number of oscillators.")
    parser.add_argument("--steps", type=int, default=3000, help="Simulation steps.")
    parser.add_argument(
        "--n-realisations", type=int, default=10, help="Realisations per condition."
    )
    parser.add_argument("--dt", type=float, default=0.05, help="Integration step.")
    parser.add_argument(
        "--coupling-low", type=float, default=0.5, help="Subcritical coupling."
    )
    parser.add_argument(
        "--hold-fraction",
        type=float,
        default=0.4,
        help="Fraction of the series held at coupling-low.",
    )
    parser.add_argument("--seed", type=int, default=20260709, help="Random seed.")
    args = parser.parse_args(argv)

    common = {
        "n": args.n,
        "steps": args.steps,
        "dt": args.dt,
        "coupling_low": args.coupling_low,
        "hold_fraction": args.hold_fraction,
        "n_realisations": args.n_realisations,
    }
    explosive = _run_condition("explosive", seed=args.seed, **common)
    continuous = _run_condition("continuous", seed=args.seed + 1, **common)

    aggregate_path = _build_aggregate(explosive, continuous, args.output_dir)
    print(f"Aggregate written to {aggregate_path}")
    print("Explosive detection rates:")
    for variant in (
        "critical_slowing_down_baseline",
        "critical_slowing_down_multiscale",
        "critical_slowing_down_surrogate",
    ):
        entry = explosive[variant]
        print(
            f"  {variant}: {entry['detection_rate']:.2%} "
            f"(FA {entry['achieved_false_alarm']:.2%}, "
            f"lead {entry['median_lead']:.0f})"
        )
    print("Continuous detection rates:")
    for variant in (
        "critical_slowing_down_baseline",
        "critical_slowing_down_multiscale",
        "critical_slowing_down_surrogate",
    ):
        entry = continuous[variant]
        print(
            f"  {variant}: {entry['detection_rate']:.2%} "
            f"(FA {entry['achieved_false_alarm']:.2%}, "
            f"lead {entry['median_lead']:.0f})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
