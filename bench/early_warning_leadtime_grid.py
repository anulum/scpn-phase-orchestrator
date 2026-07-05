# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real power-grid early-warning lead-time capstone

"""Real power-grid early-warning capstone: matched-false-alarm instability lead time.

The third domain the early-warning design is proven on: the *same* detector suite
and the *same* matched-false-alarm harness that screened scalp-EEG seizures and
cardiac AF onsets here screen a **growing power-grid oscillation** — a
generator-trip-triggered inter-area mode whose amplitude climbs toward a protective
endpoint — through nothing but a grid adapter. The only grid-specific work is
turning the multi-bus voltage magnitudes into the neutral
:class:`~scpn_phase_orchestrator.monitor.early_warning_suite.SuiteObservables`
bundle; everything downstream (segmentation, calibration, lead measurement,
sealing, verdict) is the shared :mod:`bench.early_warning_domain` harness.

Corpus
------
The PSML dataset (Zheng et al. 2021), a transmission-and-distribution co-simulation
of a 23-bus system: per scenario a millisecond-resolution ``trans.csv`` of the 23
transmission-bus voltage magnitudes and an ``info.csv`` naming the disturbance
(``type``, ``start``, ``end``). It is published under CC BY 4.0 and is
**citation-only here** — obtained from Zenodo and **never redistributed** in this
repository. This module reads it, but its tests exercise every path on **synthetic
arrays and synthetic PSML-format scenarios written in the test**, so the coverage
never depends on the 5.2 GB archive.

Why the grid fits — and its honest framing
-----------------------------------------
Unlike the two-lead ECG, the grid gives a genuine spatial oscillator population
(23 buses), and the transition is a synchronisation *rise* (the buses lock into a
coherent oscillation), the direction the suite was built for. The **transition** is
a ``gen_trip`` scenario whose oscillation *grows* between the trip (``start``) and
the annotated ``end``; the **onset** is that ``end`` — the instability endpoint the
growing oscillation leads to — and the growth before it is the precursor a detector
may lead. This is the textbook critical-slowing-down setting: a declining-damping,
growing oscillation ahead of a protective action. The **false-alarm null** is the
*damped* disturbances (``bus_fault`` / ``branch_trip`` scenarios whose response
does not grow), so the calibration asks the fair question — does the suite lead a
*growing* instability more often than it false-alarms on a *stable* disturbance?

Each transition is scored on a fixed pre-onset segment (a leading baseline plus a
detection horizon ending at ``end``); the observable is the 23-bus voltage field
band-passed to the inter-area / local mode band, per-bus Hilbert analytic phase,
and the cross-bus order parameter. Every alarm or silence is sealed into a
claim-bounded
:class:`~scpn_phase_orchestrator.assurance.early_warning_evidence.EarlyWarningEvidence`;
only the derived, sealed artefacts are committed, never the raw PSML data.

References
----------
* X. Zheng et al. 2021, *PSML: A Multi-scale Time-series Dataset for Machine
  Learning in Decarbonized Energy Grids*, NeurIPS Datasets and Benchmarks — the
  power-system co-simulation corpus (CC BY 4.0).
* Scheffer et al. 2009, *Nature* 461, 53 — generic early-warning signals.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bench.analytic_phase_pipeline import analytic_phase, bandpass, validate_signals
from bench.early_warning_domain import (
    DEFAULT_TARGET_FALSE_ALARM,
    DETECTORS,
    calibrate_detectors,
    domain_verdict,
    evaluate_seizure,
    null_trials,
    slice_observables,
)
from scpn_phase_orchestrator.monitor.early_warning_suite import (
    CRITICAL_SLOWING_DOWN,
    ENSEMBLE_WEIGHTED,
    SYNCHRONISATION,
    TRANSITION_ENTROPY,
    SuiteObservables,
    observables_from_phases,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

FloatArray = NDArray[np.float64]

#: Band-pass passband in hertz — the inter-area and local electromechanical mode
#: band the growing oscillation lives in.
BAND_HZ = (0.2, 5.0)
#: Butterworth band-pass order.
FILTER_ORDER = 4

#: Analysis window length, in seconds (a few electromechanical cycles).
WINDOW_SECONDS = 0.5
#: Hop between consecutive windows, in seconds.
STEP_SECONDS = 0.1
#: Target false-alarm rate the detectors are calibrated to on the damped null.
TARGET_FALSE_ALARM = DEFAULT_TARGET_FALSE_ALARM

#: Full analysis-segment length (baseline + horizon), in seconds.
SEGMENT_SECONDS = 2.0
#: Baseline fraction of an analysis segment (its leading third of sinus growth).
SEGMENT_BASELINE_FRACTION = 1.0 / 3.0
#: Late-third / early-third amplitude ratio above which a ``gen_trip`` oscillation
#: counts as *growing* (an instability transition) rather than damped.
GROWTH_RATIO_THRESHOLD = 1.3
#: Cap on evaluated transitions and null scenarios, so a run over the full corpus
#: stays bounded; the count actually used is reported in the aggregate.
MAX_TRANSITIONS = 12
MAX_NULL_SCENARIOS = 12

#: Disturbance type whose growing oscillation is an instability transition.
GEN_TRIP = "gen_trip"
#: Disturbance types whose damped response forms the false-alarm null.
DAMPED_TYPES = ("bus_fault", "branch_trip")

_OBSERVABLE_CSD = (
    "cross-bus Kuramoto order parameter R(t) of 23-bus voltage analytic phase "
    "(0.2-5 Hz)"
)
_OBSERVABLE_SYNC = "per-bus 23-bus voltage analytic phase (0.2-5 Hz)"
_OBSERVABLE_ENTROPY = "per-bus phase projection sin(phase) of 23-bus voltage (0.2-5 Hz)"
_OBSERVABLE_ENSEMBLE = (
    "fused early-warning suite over 23-bus voltage analytic phase (0.2-5 Hz)"
)
#: The grid observable description sealed into each detector's record.
_OBSERVABLE_DESCRIPTIONS = {
    CRITICAL_SLOWING_DOWN: _OBSERVABLE_CSD,
    SYNCHRONISATION: _OBSERVABLE_SYNC,
    TRANSITION_ENTROPY: _OBSERVABLE_ENTROPY,
    ENSEMBLE_WEIGHTED: _OBSERVABLE_ENSEMBLE,
}

__all__ = [
    "BAND_HZ",
    "DETECTORS",
    "FILTER_ORDER",
    "GROWTH_RATIO_THRESHOLD",
    "SEGMENT_SECONDS",
    "STEP_SECONDS",
    "TARGET_FALSE_ALARM",
    "WINDOW_SECONDS",
    "GridPhaseAdapter",
    "bus_voltages",
    "classify_scenario",
    "discover_scenarios",
    "grid_observables",
    "main",
    "oscillation_growth_ratio",
    "oscillation_info",
]


# --------------------------------------------------------------------------- #
# Observable pipeline (pure — fully exercised on synthetic arrays)             #
# --------------------------------------------------------------------------- #


def grid_observables(
    raw: FloatArray,
    *,
    sampling_rate_hz: float,
    band_hz: tuple[float, float] = BAND_HZ,
    filter_order: int = FILTER_ORDER,
) -> SuiteObservables:
    """Turn a raw multi-bus voltage block into the suite's neutral observables.

    The grid-specific half of the capstone: band-pass each bus voltage to the
    electromechanical mode band (which also removes the DC operating point and slow
    drift), take the per-bus Hilbert analytic phase, and hand the phase field to
    :func:`~scpn_phase_orchestrator.monitor.early_warning_suite.observables_from_phases`.
    The cross-bus order parameter then measures how tightly the buses lock into a
    common oscillation. No decimation — the PMU rate is already low.

    Parameters
    ----------
    raw : FloatArray
        Raw per-bus voltage magnitudes, shape ``(N, T)`` with at least two buses.
    sampling_rate_hz : float
        Sampling rate of ``raw`` in hertz.
    band_hz : tuple[float, float]
        Band-pass passband in hertz.
    filter_order : int
        Butterworth order.

    Returns
    -------
    SuiteObservables
        The per-bus phases, their ``sin`` projection, and the cross-bus order
        parameter, all at ``sampling_rate_hz``.

    Raises
    ------
    ValueError
        If the block has fewer than two buses or is otherwise malformed.
    """
    array = validate_signals(raw, "raw")
    if array.shape[0] < 2:
        raise ValueError("raw must have at least two buses for synchrony")
    filtered = bandpass(
        array,
        sampling_rate_hz=sampling_rate_hz,
        band_hz=band_hz,
        order=filter_order,
    )
    phases = analytic_phase(filtered)
    return observables_from_phases(phases, sampling_rate_hz=sampling_rate_hz)


@dataclass(frozen=True)
class GridPhaseAdapter:
    """The power-grid bridge from raw bus voltages to :class:`SuiteObservables`.

    A ``DomainObservableAdapter`` packaging the common-mode removal, band-pass, and
    Hilbert-phase pipeline, so the neutral suite can screen the grid exactly as it
    screens scalp EEG or cardiac ECG through their own adapters.

    Attributes
    ----------
    sampling_rate_hz : float
        Sampling rate of the raw bus voltages, in hertz.
    band_hz : tuple[float, float]
        Band-pass passband in hertz.
    filter_order : int
        Butterworth order.
    """

    sampling_rate_hz: float
    band_hz: tuple[float, float] = BAND_HZ
    filter_order: int = FILTER_ORDER

    @property
    def domain(self) -> str:
        """Return the domain label ``power_grid``."""
        return "power_grid"

    def observables(self, raw: FloatArray) -> SuiteObservables:
        """Return the neutral observable bundle for one raw multi-bus block.

        Parameters
        ----------
        raw : FloatArray
            Raw per-bus voltage magnitudes, shape ``(N, T)`` with at least two
            buses, at :attr:`sampling_rate_hz`.

        Returns
        -------
        SuiteObservables
            The band-passed analytic-phase bundle the suite reads.
        """
        return grid_observables(
            raw,
            sampling_rate_hz=self.sampling_rate_hz,
            band_hz=self.band_hz,
            filter_order=self.filter_order,
        )


# --------------------------------------------------------------------------- #
# PSML ingestion (thin — touches the citation-only corpus, tested on synthetic  #
# PSML-format scenarios, never on the redistributed archive)                    #
# --------------------------------------------------------------------------- #


def bus_voltages(scenario_dir: str | Path) -> tuple[float, FloatArray]:
    """Read a PSML scenario's bus voltage field and its sampling rate.

    Parameters
    ----------
    scenario_dir : str or Path
        Directory holding the scenario's ``trans.csv``.

    Returns
    -------
    tuple[float, FloatArray]
        The sampling rate in hertz (from the ``Time(s)`` column) and the per-bus
        voltage magnitudes, shape ``(N, T)`` with ``N >= 2``.

    Raises
    ------
    ValueError
        If the scenario carries fewer than two voltage buses.
    """
    path = Path(scenario_dir) / "trans.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    volt_columns = [
        index for index, name in enumerate(header) if name.strip().startswith("VOLT")
    ]
    if len(volt_columns) < 2:
        raise ValueError(f"scenario {scenario_dir} carries fewer than two buses")
    table = np.asarray(rows[1:], dtype=np.float64)
    times = table[:, 0]
    sampling_rate_hz = 1.0 / float(np.median(np.diff(times)))
    voltages = np.ascontiguousarray(table[:, volt_columns].T, dtype=np.float64)
    return sampling_rate_hz, voltages


def oscillation_info(scenario_dir: str | Path) -> dict[str, str]:
    """Return a PSML scenario's ``info.csv`` disturbance metadata as a mapping.

    Parameters
    ----------
    scenario_dir : str or Path
        Directory holding the scenario's ``info.csv``.

    Returns
    -------
    dict[str, str]
        The stripped key/value pairs (``type``, ``start``, ``end``, ``bus1`` …).
    """
    path = Path(scenario_dir) / "info.csv"
    info: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if "," not in line:
                continue
            key, value = line.split(",", 1)
            info[key.strip()] = value.strip()
    return info


def oscillation_growth_ratio(
    voltages: FloatArray,
    *,
    sampling_rate_hz: float,
    start_s: float,
    end_s: float,
) -> float:
    """Return the late-third / early-third mean-deviation ratio over ``[start, end]``.

    A ratio above one means the cross-bus voltage deviation *grows* across the
    event — the signature of a declining-damping, growing oscillation.

    Parameters
    ----------
    voltages : FloatArray
        Per-bus voltage magnitudes, shape ``(N, T)``.
    sampling_rate_hz : float
        Sampling rate in hertz.
    start_s, end_s : float
        Event start and end, in seconds.

    Returns
    -------
    float
        The growth ratio; ``0.0`` when the window is degenerate or the early third
        has no deviation.
    """
    start = int(round(start_s * sampling_rate_hz))
    end = int(round(end_s * sampling_rate_hz))
    if end - start < 3:
        return 0.0
    deviation = np.abs(voltages - voltages.mean(axis=0, keepdims=True)).mean(axis=0)
    third = (end - start) // 3
    early = float(deviation[start : start + third].mean())
    late = float(deviation[end - third : end].mean())
    if early <= 0.0:
        return 0.0
    return late / early


def classify_scenario(
    scenario_dir: str | Path, *, segment_samples: int
) -> tuple[str, int | None]:
    """Classify a scenario as an instability transition, a null, or a skip.

    A ``gen_trip`` scenario whose oscillation grows (ratio ≥
    :data:`GROWTH_RATIO_THRESHOLD`) and leaves a full pre-onset segment inside the
    growth is a ``"transition"`` with onset at the annotated ``end`` sample. A
    damped ``bus_fault`` / ``branch_trip`` scenario long enough for one trial is a
    ``"null"``. Anything else is a ``"skip"``.

    Parameters
    ----------
    scenario_dir : str or Path
        The scenario directory.
    segment_samples : int
        Pre-onset segment length in samples.

    Returns
    -------
    tuple[str, int | None]
        ``("transition", onset_sample)``, ``("null", None)``, or ``("skip", None)``.
    """
    info = oscillation_info(scenario_dir)
    try:
        start_s = float(info["start"])
        end_s = float(info["end"])
    except (KeyError, ValueError):
        return ("skip", None)
    kind = info.get("type", "")
    sampling_rate_hz, voltages = bus_voltages(scenario_dir)
    onset = int(round(end_s * sampling_rate_hz))
    start = int(round(start_s * sampling_rate_hz))
    if kind == GEN_TRIP and end_s > start_s and onset - start > segment_samples:
        ratio = oscillation_growth_ratio(
            voltages, sampling_rate_hz=sampling_rate_hz, start_s=start_s, end_s=end_s
        )
        if ratio >= GROWTH_RATIO_THRESHOLD:
            return ("transition", onset)
    if kind in DAMPED_TYPES and voltages.shape[1] > segment_samples:
        return ("null", None)
    return ("skip", None)


def discover_scenarios(data_dir: str | Path) -> list[Path]:
    """Return the PSML scenario directories under ``data_dir`` in sorted order.

    Parameters
    ----------
    data_dir : str or Path
        A directory whose descendants include scenario folders (each with a
        ``trans.csv`` and an ``info.csv``).

    Returns
    -------
    list[Path]
        The scenario directories, sorted by path.
    """
    root = Path(data_dir)
    scenarios = {
        candidate.parent
        for candidate in root.rglob("trans.csv")
        if (candidate.parent / "info.csv").exists()
    }
    return sorted(scenarios)


# --------------------------------------------------------------------------- #
# Orchestration (I/O shell over the tested logic)                              #
# --------------------------------------------------------------------------- #


def _partition_scenarios(
    scenarios: Sequence[Path], *, segment_samples: int
) -> tuple[list[tuple[Path, int]], list[Path]]:
    """Split discovered scenarios into (transition, onset) pairs and null dirs."""
    transitions: list[tuple[Path, int]] = []
    nulls: list[Path] = []
    for scenario in scenarios:
        kind, onset = classify_scenario(scenario, segment_samples=segment_samples)
        if kind == "transition" and onset is not None:
            transitions.append((scenario, onset))
        elif kind == "null":
            nulls.append(scenario)
    return transitions, nulls


def main(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    segment_seconds: float = SEGMENT_SECONDS,
    baseline_fraction: float = SEGMENT_BASELINE_FRACTION,
    max_transitions: int = MAX_TRANSITIONS,
    max_null_scenarios: int = MAX_NULL_SCENARIOS,
) -> None:
    """Run the capstone over PSML scenarios and write the sealed derived artefacts.

    Discovers the scenarios under ``data_dir`` (raw PSML, citation-only), classifies
    them into growing-instability transitions and damped nulls, calibrates every
    detector to a matched false-alarm rate on the null, seals each detector's alarm
    for each transition, and writes the sealed records plus an aggregate results
    JSON to ``output_dir``. Only the derived, sealed artefacts are written; the raw
    PSML data is never copied.

    Parameters
    ----------
    data_dir : str or Path
        Directory whose descendants include PSML scenario folders.
    output_dir : str or Path
        Directory the sealed derived artefacts are written to.
    segment_seconds : float
        Pre-onset analysis-segment length in seconds; also the null trial length.
    baseline_fraction : float
        Leading baseline fraction of each segment.
    max_transitions, max_null_scenarios : int
        Caps on the evaluated transitions and null scenarios, so a run over the full
        corpus stays bounded; the counts used and dropped are reported.
    """
    data = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenarios = discover_scenarios(data)
    if not scenarios:
        raise ValueError(f"no PSML scenarios found under {data}")
    sampling_rate_hz, _ = bus_voltages(scenarios[0])
    segment_samples = int(round(segment_seconds * sampling_rate_hz))
    window = max(1, int(round(WINDOW_SECONDS * sampling_rate_hz)))
    step = max(1, int(round(STEP_SECONDS * sampling_rate_hz)))
    adapter = GridPhaseAdapter(sampling_rate_hz=sampling_rate_hz)

    transitions, nulls = _partition_scenarios(
        scenarios, segment_samples=segment_samples
    )
    used_transitions = transitions[:max_transitions]
    used_nulls = nulls[:max_null_scenarios]

    null_observables = [adapter.observables(bus_voltages(n)[1]) for n in used_nulls]
    trials = null_trials(null_observables, segment_samples=segment_samples)
    thresholds = calibrate_detectors(
        trials,
        target_fa=TARGET_FALSE_ALARM,
        window=window,
        step=step,
        baseline_fraction=baseline_fraction,
    )

    leads_by_detector: dict[str, list[float]] = {name: [] for name in DETECTORS}
    transition_records: list[dict[str, object]] = []
    for scenario, onset in used_transitions:
        record_id = scenario.name
        segment = slice_observables(
            adapter.observables(bus_voltages(scenario)[1]),
            start=onset - segment_samples,
            stop=onset,
        )
        result = evaluate_seizure(
            segment,
            record_id=record_id,
            onset_sample=segment_samples,
            signal_source=(
                f"PSML {scenario.name} (Zheng et al. 2021) / growing gen_trip "
                f"oscillation / {segment_seconds:g} s pre-onset segment ending at "
                "the annotated instability endpoint"
            ),
            captured_at=f"PSML/{scenario.name}",
            thresholds=thresholds,
            observable_descriptions=_OBSERVABLE_DESCRIPTIONS,
            window=window,
            step=step,
            baseline_fraction=baseline_fraction,
        )
        (out / f"{record_id}_early_warning_evidence.json").write_text(
            json.dumps(result.to_audit_record(), indent=2) + "\n", encoding="utf-8"
        )
        for name, lead in result.lead_seconds().items():
            if lead is not None:
                leads_by_detector[name].append(lead)
        transition_records.append(
            {"record_id": record_id, "lead_seconds": result.lead_seconds()}
        )

    payload = {
        "benchmark": "early_warning_leadtime_grid",
        "corpus": "PSML 23-bus power-system co-simulation (Zheng et al. 2021)",
        "sampling_rate_hz": sampling_rate_hz,
        "band_hz": list(BAND_HZ),
        "window": window,
        "step": step,
        "segment_seconds": segment_seconds,
        "baseline_seconds": baseline_fraction * segment_seconds,
        "horizon_seconds": (1.0 - baseline_fraction) * segment_seconds,
        "target_false_alarm": TARGET_FALSE_ALARM,
        "n_transitions_found": len(transitions),
        "n_null_scenarios_found": len(nulls),
        "n_transitions_evaluated": len(used_transitions),
        "n_null_trials": len(trials),
        "matched_false_alarm_thresholds": thresholds,
        "transitions": transition_records,
        "verdict": domain_verdict(
            leads_by_detector,
            len(used_transitions),
            noun="growing-oscillation instabilities",
            singular="instability",
        ),
    }
    (out / "early_warning_leadtime_grid_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(payload["verdict"])
    print(
        f"{len(transitions)} transitions / {len(nulls)} nulls found; "
        f"evaluated {len(used_transitions)} on {len(trials)} null trials"
    )
    print(f"results written to {out}")


if __name__ == "__main__":  # pragma: no cover - CLI shell over the tested logic
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="directory holding the PSML scenarios")
    parser.add_argument("output_dir", help="directory for the sealed derived output")
    arguments = parser.parse_args()
    main(arguments.data_dir, arguments.output_dir)
