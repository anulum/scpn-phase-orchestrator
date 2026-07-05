# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SCPN-vs-Dakos matched-false-alarm head-to-head runner

"""Head-to-head: SCPN critical slowing down vs Dakos AR(1)-Kendall-τ, same protocol.

Each domain's sealed aggregate already records the SCPN suite's permutation
significance at a matched false-alarm rate. This runner rebuilds the *same* transition
and null segments each capstone uses, scores them with the competitor AR(1)-Kendall-τ
detector (:mod:`bench.competitor_ar1_kendall`) through the *same* matched-false-alarm
calibration and label-permutation test, and writes a side-by-side comparison. The two
detectors read the same one-dimensional signal — the detrended proxy residual for
palaeoclimate, the cross-node Kuramoto order parameter for the multi-node domains — so
the only difference is the detector, not the data or the operating point.

The runner is an I/O shell over already-tested logic: the competitor detector and the
permutation core are unit-tested, and the per-domain segment builders reuse each
capstone's own ingestion, so a comparison run reproduces the capstone segments exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from bench.competitor_ar1_kendall import ar1_kendall_significance
from bench.early_warning_domain import null_trials, slice_observables

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    import numpy as np
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]

__all__ = ["main", "scpn_best_detector"]


def scpn_best_detector(aggregate: dict[str, object]) -> dict[str, object]:
    """Return the SCPN detector with the most leads from a committed aggregate.

    Parameters
    ----------
    aggregate : dict
        A capstone results record carrying a ``permutation_significance`` mapping of
        detector label to its significance record.

    Returns
    -------
    dict
        ``{"detector", "observed_led", "n_transitions", "p_value"}`` of the member with
        the most leads (ties broken by the smaller p-value).
    """
    significance = aggregate["permutation_significance"]
    assert isinstance(significance, dict)
    if "observed_led" in significance:
        # A single-series capstone records one detector's significance directly.
        best_name = "critical_slowing_down"
        record = significance
    else:
        # A multi-node capstone records one significance per detector; pick the member
        # with the most leads (ties broken by the smaller p-value).
        best_name = max(
            significance,
            key=lambda name: (
                significance[name]["observed_led"],
                -significance[name]["p_value"],
            ),
        )
        record = significance[best_name]
    return {
        "detector": best_name,
        "observed_led": record["observed_led"],
        "n_transitions": record["n_transitions"],
        "p_value": record["p_value"],
    }


def _climate_segments(  # pragma: no cover - I/O shell over tested capstone ingestion
    data_dir: Path,
) -> tuple[list[FloatArray], list[FloatArray], int, int]:
    """Rebuild the palaeoclimate transition and null series (detrended residuals)."""
    from bench.early_warning_leadtime_climate import (
        CLIMATE_RECORDS,
        SEGMENT_BASELINE_FRACTION,
        SEGMENT_SAMPLES,
        STEP,
        WINDOW,
        ClimateProxyAdapter,
        load_climate_series,
    )
    from bench.early_warning_single_series import null_series_trials, slice_series

    del SEGMENT_BASELINE_FRACTION  # segmentation uses the sample counts only
    adapter = ClimateProxyAdapter()
    transitions: list[FloatArray] = []
    null_regions = []
    for record in CLIMATE_RECORDS:
        proxy, span = load_climate_series(
            data_dir / record.proxy_file,
            data_dir / record.time_file,
            age_unit_years=record.age_unit_years,
        )
        observable = adapter.observable(proxy, span_years=span)
        start = observable.n_samples - SEGMENT_SAMPLES
        if start < SEGMENT_SAMPLES:
            continue
        transitions.append(
            slice_series(observable, start=start, stop=observable.n_samples).series
        )
        null_regions.append(slice_series(observable, start=0, stop=start))
    trials = null_series_trials(null_regions, segment_samples=SEGMENT_SAMPLES)
    return transitions, [trial.series for trial in trials], WINDOW, STEP


def _grid_segments(  # pragma: no cover - I/O shell over tested capstone ingestion
    data_dir: Path,
) -> tuple[list[FloatArray], list[FloatArray], int, int]:
    """Rebuild the grid transition and null order-parameter series."""
    from bench.early_warning_leadtime_grid import (
        MAX_NULL_SCENARIOS,
        MAX_TRANSITIONS,
        SEGMENT_SECONDS,
        STEP_SECONDS,
        WINDOW_SECONDS,
        GridPhaseAdapter,
        _partition_scenarios,
        bus_voltages,
        discover_scenarios,
    )

    scenarios = discover_scenarios(data_dir)
    rate, _ = bus_voltages(scenarios[0])
    segment_samples = int(round(SEGMENT_SECONDS * rate))
    window = max(1, int(round(WINDOW_SECONDS * rate)))
    step = max(1, int(round(STEP_SECONDS * rate)))
    transitions, nulls = _partition_scenarios(
        scenarios, segment_samples=segment_samples
    )
    adapter = GridPhaseAdapter(sampling_rate_hz=rate)
    null_observables = [
        adapter.observables(bus_voltages(scenario)[1])
        for scenario in nulls[:MAX_NULL_SCENARIOS]
    ]
    trials = null_trials(null_observables, segment_samples=segment_samples)
    transition_series = [
        slice_observables(
            adapter.observables(bus_voltages(scenario)[1]),
            start=onset - segment_samples,
            stop=onset,
        ).order_parameter
        for scenario, onset in transitions[:MAX_TRANSITIONS]
    ]
    return transition_series, [t.order_parameter for t in trials], window, step


def _cardiac_segments(  # pragma: no cover - I/O shell over tested capstone ingestion
    data_dir: Path,
) -> tuple[list[FloatArray], list[FloatArray], int, int]:
    """Rebuild the cardiac transition and null order-parameter series."""
    from bench.early_warning_leadtime_cardiac import (
        AF_RECORDS,
        NULL_RECORDS,
        SEGMENT_SAMPLES,
        SEGMENT_SAMPLES_NATIVE,
        STEP,
        WINDOW,
        CardiacPhaseAdapter,
        _null_observables,
        afib_onsets,
        load_wfdb_leads,
    )

    adapter = CardiacPhaseAdapter()
    trials = null_trials(
        _null_observables(data_dir, NULL_RECORDS, adapter),
        segment_samples=SEGMENT_SAMPLES,
    )
    transition_series: list[FloatArray] = []
    for record_id in AF_RECORDS:
        stem = data_dir / record_id
        onsets = afib_onsets(stem, min_baseline_native=SEGMENT_SAMPLES_NATIVE)
        if not onsets:
            continue
        onset_native = onsets[0]
        leads = load_wfdb_leads(
            stem, sampfrom=onset_native - SEGMENT_SAMPLES_NATIVE, sampto=onset_native
        )
        segment = slice_observables(
            adapter.observables(leads), start=0, stop=SEGMENT_SAMPLES
        )
        transition_series.append(segment.order_parameter)
    return transition_series, [t.order_parameter for t in trials], WINDOW, STEP


def _eeg_segments(  # pragma: no cover - I/O shell over tested capstone ingestion
    data_dir: Path,
) -> tuple[list[FloatArray], list[FloatArray], int, int]:
    """Rebuild the scalp-EEG transition and null order-parameter series."""
    from bench.early_warning_leadtime_eeg import (
        INTERICTAL_RECORDS,
        SEGMENT_SAMPLES,
        SEIZURE_ONSETS_S,
        STEP,
        WINDOW,
        EEGPhaseAdapter,
        load_edf_channels,
    )

    adapter = EEGPhaseAdapter()
    null_observables = [
        adapter.observables(load_edf_channels(data_dir / f"{record}.edf"))
        for record in INTERICTAL_RECORDS
    ]
    trials = null_trials(null_observables, segment_samples=SEGMENT_SAMPLES)
    transition_series: list[FloatArray] = []
    for record_id, onset_s in SEIZURE_ONSETS_S.items():
        observables = adapter.observables(
            load_edf_channels(data_dir / f"{record_id}.edf")
        )
        onset_sample = int(round(onset_s * observables.sampling_rate_hz))
        if onset_sample < SEGMENT_SAMPLES:
            continue
        segment = slice_observables(
            observables, start=onset_sample - SEGMENT_SAMPLES, stop=onset_sample
        )
        transition_series.append(segment.order_parameter)
    return transition_series, [t.order_parameter for t in trials], WINDOW, STEP


#: Each domain: its committed SCPN aggregate and the segment builder that rebuilds the
#: transition and null one-dimensional signals the competitor scores.
_DOMAINS = {
    "palaeoclimate": (
        "examples/real_data/dakos_climate_transitions/"
        "early_warning_leadtime_climate_results.json",
        _climate_segments,
        "detrended proxy residual",
    ),
    "grid": (
        "examples/real_data/psml_grid_oscillation/early_warning_leadtime_grid_results.json",
        _grid_segments,
        "cross-bus Kuramoto order parameter",
    ),
    "cardiac": (
        "examples/real_data/afdb_atrial_fibrillation/"
        "early_warning_leadtime_cardiac_results.json",
        _cardiac_segments,
        "cross-lead Kuramoto order parameter",
    ),
    "eeg": (
        "examples/real_data/chb01_seizures/early_warning_leadtime_eeg_results.json",
        _eeg_segments,
        "cross-channel Kuramoto order parameter",
    ),
}


def main(  # pragma: no cover - I/O shell over the tested detector and permutation core
    output_dir: str | Path,
    *,
    climate_dir: str | Path | None = None,
    grid_dir: str | Path | None = None,
    cardiac_dir: str | Path | None = None,
    eeg_dir: str | Path | None = None,
) -> None:
    """Run the SCPN-vs-Dakos head-to-head over every supplied domain corpus.

    For each domain with a supplied raw-corpus directory, the runner rebuilds the
    capstone's transition and null segments, scores them with the AR(1)-Kendall-τ
    competitor at a matched false alarm, reads the SCPN suite's best-member
    significance from the committed aggregate, and writes a
    ``head_to_head_ar1_kendall.json`` with the two detectors side by side. Raw corpora
    are read but never copied.

    Parameters
    ----------
    output_dir : str or Path
        Directory the comparison JSON is written to.
    climate_dir, grid_dir, cardiac_dir, eeg_dir : str or Path or None
        Raw-corpus directories; a domain is skipped when its directory is ``None``.
    """
    repo = Path(__file__).resolve().parents[1]
    dirs = {
        "palaeoclimate": climate_dir,
        "grid": grid_dir,
        "cardiac": cardiac_dir,
        "eeg": eeg_dir,
    }
    comparisons: list[dict[str, object]] = []
    for domain, (aggregate_path, builder, signal) in _DOMAINS.items():
        data_dir = dirs[domain]
        if data_dir is None:
            continue
        aggregate = json.loads((repo / aggregate_path).read_text(encoding="utf-8"))
        transitions, nulls, window, step = builder(Path(data_dir))
        dakos = ar1_kendall_significance(transitions, nulls, window=window, step=step)
        scpn = scpn_best_detector(aggregate)
        comparisons.append(
            {
                "domain": domain,
                "signal": signal,
                "n_transitions": scpn["n_transitions"],
                "scpn": scpn,
                "dakos_ar1_kendall": dakos.to_audit_record(),
            }
        )
        print(
            f"{domain}: SCPN {scpn['detector']} {scpn['observed_led']}/"
            f"{scpn['n_transitions']} p={scpn['p_value']:.3f}  |  Dakos "
            f"{dakos.significance.observed_led}/{dakos.significance.n_transitions} "
            f"p={dakos.significance.p_value:.3f}"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "head_to_head_ar1_kendall",
        "question": (
            "At a matched false-alarm rate, does the canonical Dakos et al. 2008 "
            "AR(1)-Kendall-tau detector beat the SCPN suite, or chance?"
        ),
        "comparisons": comparisons,
    }
    (out / "head_to_head_ar1_kendall.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"comparison written to {out}")


if __name__ == "__main__":  # pragma: no cover - CLI shell
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--climate-dir")
    parser.add_argument("--grid-dir")
    parser.add_argument("--cardiac-dir")
    parser.add_argument("--eeg-dir")
    arguments = parser.parse_args()
    main(
        arguments.output_dir,
        climate_dir=arguments.climate_dir,
        grid_dir=arguments.grid_dir,
        cardiac_dir=arguments.cardiac_dir,
        eeg_dir=arguments.eeg_dir,
    )
