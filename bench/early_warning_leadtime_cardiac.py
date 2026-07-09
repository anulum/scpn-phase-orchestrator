# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real cardiac-ECG early-warning lead-time capstone

"""Real cardiac-ECG early-warning capstone: matched-false-alarm AF-onset lead time.

The second domain proof that the early-warning design is domain-adaptable: the
*same* detector suite and the *same* matched-false-alarm harness that screened
scalp-EEG seizures (:mod:`bench.early_warning_leadtime_eeg`) here screen the onset
of atrial fibrillation (AF) in the surface ECG, through nothing but a different
adapter. The only cardiac-specific work is turning the ECG leads into the neutral
:class:`~scpn_phase_orchestrator.monitor.early_warning_suite.SuiteObservables`
bundle; everything downstream (segmentation, calibration, lead measurement,
sealing, verdict) is the shared :mod:`bench.early_warning_domain` harness.

Corpus
------
The MIT-BIH Atrial Fibrillation Database (Moody & Mark 1983; Goldberger et al.
2000, PhysioNet): two-lead ambulatory ECG sampled at 250 Hz, ~10 h per record,
with clinician-reviewed rhythm annotations that mark each ``(AFIB`` onset. The raw
recordings are **citation-only** — downloaded from PhysioNet and **never
redistributed here**. This module reads them, but its tests exercise every
signal-processing and evaluation path on **synthetic arrays and a synthetic WFDB
record** so the coverage never depends on the protected corpus.

Observable pipeline and its honest limits
-----------------------------------------
Each recording is turned into one decimated analytic-phase field via the shared
:mod:`bench.analytic_phase_pipeline`: band-pass to the QRS-dominant band, per-lead
Hilbert analytic phase, phase-consistent decimation 250 → 50 Hz. The neutral
bundle's ``sin(phase)`` projection and cross-lead order parameter follow. Two
honest caveats belong on this proof, not buried:

* **Thin population.** The database carries only two ECG leads, so the
  cross-``node`` order parameter is a two-lead inter-lead phase coherence — a
  genuine but thin population next to the 23-channel scalp-EEG field. The suite's
  synchrony detector has correspondingly limited power here.
* **Opposite direction.** A seizure onset is a synchronisation *rise*; AF onset is
  a *loss* of organised atrial activity — a desynchronisation. The suite's
  rising-synchronisation member is therefore not expected to fire on AF; critical
  slowing down (a variance rise, direction-agnostic to the transition) and the
  ordinal-transition entropy are the members with a chance. Whatever the members
  do, the result is sealed and reported honestly.

Each AF onset is scored on a fixed pre-onset segment — a leading
``BASELINE_SECONDS`` of sinus rhythm followed by a ``HORIZON_SECONDS`` detection
window ending at onset (:data:`SEGMENT_SECONDS` total). Only onsets with a full
segment of preceding non-AF rhythm are evaluated; the rest are excluded and
reported, never counted as silent nulls. The matched false-alarm threshold is
calibrated on non-overlapping trials cut from long AF-free (sinus) stretches, and
every alarm or silence is sealed into a claim-bounded
:class:`~scpn_phase_orchestrator.assurance.early_warning_evidence.EarlyWarningEvidence`.
Only the derived, sealed artefacts are committed; the raw ECG never is.

References
----------
* G. B. Moody and R. G. Mark 1983, *Computers in Cardiology* 10:227 — the MIT-BIH
  Atrial Fibrillation Database.
* A. L. Goldberger et al. 2000, *Circulation* 101(23):e215 — PhysioNet /
  PhysioBank.
* Scheffer et al. 2009, *Nature* 461, 53 — generic early-warning signals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bench.analytic_phase_pipeline import (
    analytic_phase,
    bandpass,
    decimate_analytic_phase,
    validate_signals,
)
from bench.early_warning_domain import (
    DEFAULT_TARGET_FALSE_ALARM,
    DETECTORS,
    DETECTORS_MULTISCALE,
    calibrate_detectors,
    domain_verdict,
    evaluate_seizure,
    null_trials,
    permutation_significance_by_detector,
    slice_observables,
)
from scpn_phase_orchestrator.monitor.early_warning_suite import (
    CRITICAL_SLOWING_DOWN,
    CRITICAL_SLOWING_DOWN_MULTISCALE,
    ENSEMBLE_WEIGHTED,
    SYNCHRONISATION,
    TRANSITION_ENTROPY,
    SuiteObservables,
    observables_from_phases,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

FloatArray = NDArray[np.float64]

#: Native sampling rate of the MIT-BIH AFDB recordings, in hertz.
SAMPLING_RATE_HZ = 250.0
#: Decimation factor applied to the analytic-phase components (250 → 50 Hz).
DECIMATION = 5
#: Analysis sampling rate after decimation, in hertz.
DECIMATED_RATE_HZ = SAMPLING_RATE_HZ / DECIMATION
#: Band-pass passband in hertz — the QRS-dominant band carrying the rhythm's
#: inter-lead timing, below the decimated Nyquist.
BAND_HZ = (5.0, 20.0)
#: Butterworth band-pass order.
FILTER_ORDER = 4

#: Analysis window length in samples (4 s at 50 Hz — several cardiac cycles).
WINDOW = 200
#: Hop between consecutive windows in samples (0.5 s at 50 Hz).
STEP = 25
#: Target false-alarm rate the detectors are calibrated to on the sinus null.
TARGET_FALSE_ALARM = DEFAULT_TARGET_FALSE_ALARM

#: Leading sinus-rhythm baseline length, in seconds, of an analysis segment.
BASELINE_SECONDS = 300.0
#: Detection horizon, in seconds, before onset — the pre-onset window the suite
#: is allowed to warn within, and the maximum measurable lead.
HORIZON_SECONDS = 600.0
#: Full analysis-segment length (baseline + horizon), in seconds.
SEGMENT_SECONDS = BASELINE_SECONDS + HORIZON_SECONDS
#: Analysis-segment length in decimated samples.
SEGMENT_SAMPLES = int(SEGMENT_SECONDS * DECIMATED_RATE_HZ)
#: Analysis-segment length in native samples.
SEGMENT_SAMPLES_NATIVE = int(SEGMENT_SECONDS * SAMPLING_RATE_HZ)
#: Baseline fraction of an analysis segment (its leading ``BASELINE_SECONDS``).
SEGMENT_BASELINE_FRACTION = BASELINE_SECONDS / SEGMENT_SECONDS
#: Trials cut from a single sinus stretch for the matched-false-alarm null, so the
#: calibration is fine without loading whole ten-hour records.
NULL_TRIALS_PER_RECORD = 10

#: Rhythm annotation for atrial fibrillation and for normal (sinus) rhythm.
AFIB_LABEL = "(AFIB"
NORMAL_LABEL = "(N"

_OBSERVABLE_CSD = (
    "cross-lead Kuramoto order parameter R(t) of 2-lead ECG analytic phase "
    "(5-20 Hz, decimated to 50 Hz)"
)
_OBSERVABLE_CSD_MULTISCALE = (
    "multi-scale cross-lead Kuramoto order parameter R(t) of 2-lead ECG "
    "analytic phase (5-20 Hz, decimated to 50 Hz)"
)
_OBSERVABLE_SYNC = "per-lead 2-lead ECG analytic phase (5-20 Hz, decimated to 50 Hz)"
_OBSERVABLE_ENTROPY = (
    "per-lead phase projection sin(phase) of 2-lead ECG (5-20 Hz, decimated to 50 Hz)"
)
_OBSERVABLE_ENSEMBLE = (
    "fused early-warning suite over 2-lead ECG analytic phase "
    "(5-20 Hz, decimated to 50 Hz)"
)
#: The cardiac observable description sealed into each detector's record.
_OBSERVABLE_DESCRIPTIONS = {
    CRITICAL_SLOWING_DOWN: _OBSERVABLE_CSD,
    SYNCHRONISATION: _OBSERVABLE_SYNC,
    TRANSITION_ENTROPY: _OBSERVABLE_ENTROPY,
    ENSEMBLE_WEIGHTED: _OBSERVABLE_ENSEMBLE,
}
_OBSERVABLE_DESCRIPTIONS_MULTISCALE = {
    **_OBSERVABLE_DESCRIPTIONS,
    CRITICAL_SLOWING_DOWN_MULTISCALE: _OBSERVABLE_CSD_MULTISCALE,
}

#: AFDB records whose first clean AF onset is evaluated (each has ≥ one onset with
#: a full pre-onset sinus segment).
AF_RECORDS: tuple[str, ...] = (
    "04043",
    "04048",
    "04746",
    "04908",
    "05091",
    "07879",
)
#: AFDB records whose longest AF-free (sinus) stretch forms the false-alarm null.
NULL_RECORDS: tuple[str, ...] = (
    "04015",
    "04126",
)

__all__ = [
    "BAND_HZ",
    "DECIMATED_RATE_HZ",
    "DECIMATION",
    "DETECTORS",
    "FILTER_ORDER",
    "SAMPLING_RATE_HZ",
    "STEP",
    "TARGET_FALSE_ALARM",
    "WINDOW",
    "CardiacPhaseAdapter",
    "afib_onsets",
    "cardiac_observables",
    "load_wfdb_leads",
    "longest_sinus_span",
    "main",
    "rhythm_transitions",
]


# --------------------------------------------------------------------------- #
# Observable pipeline (pure — fully exercised on synthetic arrays)             #
# --------------------------------------------------------------------------- #


def cardiac_observables(
    raw: FloatArray,
    *,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    decimation: int = DECIMATION,
    band_hz: tuple[float, float] = BAND_HZ,
    filter_order: int = FILTER_ORDER,
) -> SuiteObservables:
    """Turn a raw multi-lead ECG block into the suite's neutral observables.

    The cardiac-specific half of the capstone: band-pass, per-lead Hilbert
    analytic phase, and phase-consistent decimation to a per-lead phase field,
    from which
    :func:`~scpn_phase_orchestrator.monitor.early_warning_suite.observables_from_phases`
    derives the neutral bundle the domain-neutral suite reads.

    Parameters
    ----------
    raw : FloatArray
        Raw per-lead ECG, shape ``(N, T)`` with at least two leads (the order
        parameter and synchrony are undefined for a single lead).
    sampling_rate_hz : float
        Native sampling rate of ``raw`` in hertz.
    decimation : int
        Decimation factor applied to the analytic-phase components.
    band_hz : tuple[float, float]
        Band-pass passband in hertz.
    filter_order : int
        Butterworth order.

    Returns
    -------
    SuiteObservables
        The reconstructed phases, the ``sin(phase)`` field, and the cross-lead
        order parameter, all at ``sampling_rate_hz / decimation``.

    Raises
    ------
    ValueError
        If the recording has fewer than two leads or is otherwise malformed.
    """
    array = validate_signals(raw, "raw")
    if array.shape[0] < 2:
        raise ValueError("raw must have at least two leads for synchrony")
    filtered = bandpass(
        array, sampling_rate_hz=sampling_rate_hz, band_hz=band_hz, order=filter_order
    )
    phases = decimate_analytic_phase(analytic_phase(filtered), factor=decimation)
    return observables_from_phases(
        phases, sampling_rate_hz=sampling_rate_hz / decimation
    )


@dataclass(frozen=True)
class CardiacPhaseAdapter:
    """The cardiac-ECG bridge from raw leads to :class:`SuiteObservables`.

    A ``DomainObservableAdapter`` packaging the band-pass / Hilbert / decimation
    pipeline, so the neutral suite can screen the ECG exactly as it screens scalp
    EEG or grid signals through their own adapters.

    Attributes
    ----------
    sampling_rate_hz : float
        Native sampling rate of the raw leads, in hertz.
    decimation : int
        Decimation factor applied to the analytic-phase components.
    band_hz : tuple[float, float]
        Band-pass passband in hertz.
    filter_order : int
        Butterworth order.
    """

    sampling_rate_hz: float = SAMPLING_RATE_HZ
    decimation: int = DECIMATION
    band_hz: tuple[float, float] = BAND_HZ
    filter_order: int = FILTER_ORDER

    @property
    def domain(self) -> str:
        """Return the domain label ``cardiac_ecg``."""
        return "cardiac_ecg"

    def observables(self, raw: FloatArray) -> SuiteObservables:
        """Return the neutral observable bundle for one raw ECG recording.

        Parameters
        ----------
        raw : FloatArray
            Raw per-lead ECG, shape ``(N, T)`` with at least two leads, at
            :attr:`sampling_rate_hz`.

        Returns
        -------
        SuiteObservables
            The decimated analytic-phase bundle the suite reads.
        """
        return cardiac_observables(
            raw,
            sampling_rate_hz=self.sampling_rate_hz,
            decimation=self.decimation,
            band_hz=self.band_hz,
            filter_order=self.filter_order,
        )


# --------------------------------------------------------------------------- #
# WFDB ingestion (thin — touches the citation-only corpus, tested on a          #
# synthetic WFDB record, never on the redistributed raw recordings)             #
# --------------------------------------------------------------------------- #


def load_wfdb_leads(
    record_path: str | Path,
    *,
    sampfrom: int = 0,
    sampto: int | None = None,
    expected_rate_hz: float = SAMPLING_RATE_HZ,
) -> FloatArray:
    """Read the ECG leads of a WFDB record segment into a ``(N, T)`` array.

    Non-finite gap samples (WFDB marks dropouts as ``NaN``) are zeroed so the
    band-pass never sees a non-finite value; the returned block is contiguous and
    ready for :func:`cardiac_observables`.

    Parameters
    ----------
    record_path : str or Path
        Path to the WFDB record *stem* (no extension).
    sampfrom, sampto : int
        Half-open native-sample range to read; ``sampto`` of ``None`` reads to the
        end.
    expected_rate_hz : float
        Sampling rate the record must carry.

    Returns
    -------
    FloatArray
        The leads, shape ``(N, T)`` with ``N >= 2``.

    Raises
    ------
    ValueError
        If the record rate differs from ``expected_rate_hz`` or it carries fewer
        than two leads.
    """
    import wfdb

    record = wfdb.rdrecord(str(record_path), sampfrom=sampfrom, sampto=sampto)
    if abs(float(record.fs) - expected_rate_hz) > 1.0e-6:
        raise ValueError(
            f"record {record_path} is sampled at {record.fs} Hz, not {expected_rate_hz}"
        )
    signal = np.asarray(record.p_signal, dtype=np.float64).T
    if signal.shape[0] < 2:
        raise ValueError(f"record {record_path} carries fewer than two leads")
    return np.ascontiguousarray(np.nan_to_num(signal), dtype=np.float64)


def rhythm_transitions(record_path: str | Path) -> list[tuple[int, str]]:
    """Return the record's ``(sample, rhythm-label)`` transitions in time order.

    Parameters
    ----------
    record_path : str or Path
        Path to the WFDB record *stem* (no extension); its ``atr`` annotation
        carries the rhythm-change stream.

    Returns
    -------
    list[tuple[int, str]]
        Each annotated rhythm change as ``(native_sample, label)`` (labels such
        as ``(N`` for sinus and ``(AFIB`` for atrial fibrillation).
    """
    import wfdb

    annotation = wfdb.rdann(str(record_path), "atr")
    return [
        (int(sample), str(note).strip("\x00"))
        for sample, note in zip(annotation.sample, annotation.aux_note, strict=True)
    ]


def afib_onsets(
    record_path: str | Path, *, min_baseline_native: int = SEGMENT_SAMPLES_NATIVE
) -> list[int]:
    """Return the native-sample AF onsets with a full clean pre-onset baseline.

    An onset qualifies when the rhythm changes *into* ``(AFIB`` from a non-AF
    rhythm and at least ``min_baseline_native`` samples of that non-AF rhythm
    precede it, so the pre-onset segment is guaranteed sinus and any alarm is a
    genuine lead.

    Parameters
    ----------
    record_path : str or Path
        Path to the WFDB record stem.
    min_baseline_native : int
        Required preceding non-AF span, in native samples.

    Returns
    -------
    list[int]
        Qualifying AF onset sample indices, in time order.
    """
    transitions = rhythm_transitions(record_path)
    onsets: list[int] = []
    for index, (sample, label) in enumerate(transitions):
        if label != AFIB_LABEL:
            continue
        previous_sample = transitions[index - 1][0] if index > 0 else 0
        previous_label = transitions[index - 1][1] if index > 0 else NORMAL_LABEL
        clean = previous_label != AFIB_LABEL
        if clean and sample - previous_sample >= min_baseline_native:
            onsets.append(sample)
    return onsets


def longest_sinus_span(record_path: str | Path) -> tuple[int, int]:
    """Return the ``(start, stop)`` native-sample range of the longest sinus span.

    The record's longest continuous ``(N`` stretch carries no AF, so it is a clean
    no-transition null. The stop is the next rhythm change, or the record end.

    Parameters
    ----------
    record_path : str or Path
        Path to the WFDB record stem.

    Returns
    -------
    tuple[int, int]
        The half-open native-sample range of the longest sinus stretch.

    Raises
    ------
    ValueError
        If the record annotates no sinus rhythm.
    """
    import wfdb

    header = wfdb.rdheader(str(record_path))
    end = int(header.sig_len)
    transitions = rhythm_transitions(record_path)
    best: tuple[int, int] | None = None
    for index, (sample, label) in enumerate(transitions):
        if label != NORMAL_LABEL:
            continue
        stop = transitions[index + 1][0] if index + 1 < len(transitions) else end
        if best is None or stop - sample > best[1] - best[0]:
            best = (sample, stop)
    if best is None:
        raise ValueError(f"record {record_path} annotates no sinus rhythm")
    return best


# --------------------------------------------------------------------------- #
# Orchestration (I/O shell over the tested logic)                              #
# --------------------------------------------------------------------------- #


def _null_observables(
    data: Path, null_records: Sequence[str], adapter: CardiacPhaseAdapter
) -> list[SuiteObservables]:
    """Build the sinus-null observables, one per null record's longest span.

    Each null span is capped at :data:`NULL_TRIALS_PER_RECORD` segments so the
    calibration is fine without decimating whole ten-hour records.
    """
    cap = NULL_TRIALS_PER_RECORD * SEGMENT_SAMPLES_NATIVE
    null: list[SuiteObservables] = []
    for record in null_records:
        stem = data / record
        start, stop = longest_sinus_span(stem)
        stop = min(stop, start + cap)
        leads = load_wfdb_leads(stem, sampfrom=start, sampto=stop)
        null.append(adapter.observables(leads))
    return null


def main(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    af_records: Sequence[str] = AF_RECORDS,
    null_records: Sequence[str] = NULL_RECORDS,
    segment_samples: int = SEGMENT_SAMPLES,
    baseline_fraction: float = SEGMENT_BASELINE_FRACTION,
    multiscale: bool = False,
) -> None:
    """Run the capstone over MIT-BIH AFDB and write the sealed derived artefacts.

    Reads the sinus null and the annotated AF onsets from ``data_dir`` (raw WFDB,
    citation-only), calibrates every detector to a matched false-alarm rate on the
    sinus null, seals each detector's alarm for the first clean onset of each AF
    record, and writes the sealed records plus an aggregate results JSON to
    ``output_dir``. Only the derived, sealed artefacts are written; the raw ECG is
    never copied.

    Parameters
    ----------
    data_dir : str or Path
        Directory holding the WFDB record files (``<record>.hea``/``.dat``/``.atr``).
    output_dir : str or Path
        Directory the sealed derived artefacts are written to.
    af_records : sequence of str
        Record stems whose first clean AF onset is evaluated.
    null_records : sequence of str
        Record stems whose longest sinus span forms the false-alarm null.
    segment_samples : int
        Pre-onset analysis-segment length in decimated samples; also the null
        trial length. Defaults to :data:`SEGMENT_SAMPLES`.
    baseline_fraction : float
        Leading baseline fraction of each segment. Defaults to
        :data:`SEGMENT_BASELINE_FRACTION`.
    multiscale : bool
        If True, also evaluate and seal the multi-scale CSD detector.
    """
    data = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    segment_seconds = segment_samples / DECIMATED_RATE_HZ
    segment_native = int(round(segment_seconds * SAMPLING_RATE_HZ))
    adapter = CardiacPhaseAdapter()

    null_observables = _null_observables(data, null_records, adapter)
    trials = null_trials(null_observables, segment_samples=segment_samples)
    calibration = calibrate_detectors(
        trials,
        target_fa=TARGET_FALSE_ALARM,
        window=WINDOW,
        step=STEP,
        baseline_fraction=baseline_fraction,
        multiscale=multiscale,
    )
    thresholds = calibration.thresholds

    detector_set = DETECTORS_MULTISCALE if multiscale else DETECTORS
    observable_descriptions = (
        _OBSERVABLE_DESCRIPTIONS_MULTISCALE if multiscale else _OBSERVABLE_DESCRIPTIONS
    )
    leads_by_detector: dict[str, list[float]] = {name: [] for name in detector_set}
    onset_records: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    transition_segments: list[SuiteObservables] = []
    for record_id in af_records:
        stem = data / record_id
        onsets = afib_onsets(stem, min_baseline_native=segment_native)
        if not onsets:
            # No onset has a full clean pre-onset segment; reported, not a null.
            excluded.append({"record_id": record_id, "reason": "no clean onset"})
            continue
        onset_native = onsets[0]
        leads = load_wfdb_leads(
            stem, sampfrom=onset_native - segment_native, sampto=onset_native
        )
        segment = slice_observables(
            adapter.observables(leads), start=0, stop=segment_samples
        )
        transition_segments.append(segment)
        result = evaluate_seizure(
            segment,
            record_id=record_id,
            onset_sample=segment_samples,
            signal_source=(
                f"MIT-BIH AFDB {record_id} (Moody & Mark 1983) / AF onset "
                f"{onset_native / SAMPLING_RATE_HZ:g} s / {segment_seconds:g} s "
                "pre-onset segment"
            ),
            captured_at=f"AFDB/{record_id}",
            thresholds=thresholds,
            observable_descriptions=observable_descriptions,
            window=WINDOW,
            step=STEP,
            baseline_fraction=baseline_fraction,
            multiscale=multiscale,
        )
        (out / f"{record_id}_early_warning_evidence.json").write_text(
            json.dumps(result.to_audit_record(), indent=2) + "\n", encoding="utf-8"
        )
        for name, lead in result.lead_seconds().items():
            if lead is not None:
                leads_by_detector[name].append(lead)
        onset_records.append(
            {
                "record_id": record_id,
                "onset_s": round(onset_native / SAMPLING_RATE_HZ, 1),
                "lead_seconds": result.lead_seconds(),
            }
        )

    significance = permutation_significance_by_detector(
        transition_segments,
        trials,
        thresholds=thresholds,
        window=WINDOW,
        step=STEP,
        baseline_fraction=baseline_fraction,
        multiscale=multiscale,
    )

    result_filename = (
        "early_warning_leadtime_cardiac_multiscale_results.json"
        if multiscale
        else "early_warning_leadtime_cardiac_results.json"
    )
    payload = {
        "benchmark": "early_warning_leadtime_cardiac",
        "corpus": "MIT-BIH Atrial Fibrillation Database (Moody & Mark 1983)",
        "multiscale": multiscale,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "decimated_rate_hz": DECIMATED_RATE_HZ,
        "band_hz": list(BAND_HZ),
        "window": WINDOW,
        "step": STEP,
        "baseline_seconds": baseline_fraction * segment_seconds,
        "horizon_seconds": (1.0 - baseline_fraction) * segment_seconds,
        "segment_seconds": segment_seconds,
        "target_false_alarm": TARGET_FALSE_ALARM,
        "sinus_null_records": list(null_records),
        "n_null_trials": len(trials),
        "matched_false_alarm_thresholds": thresholds,
        "achieved_false_alarm": calibration.achieved_false_alarm,
        "permutation_significance": {
            name: result.to_audit_record() for name, result in significance.items()
        },
        "af_onsets": onset_records,
        "excluded_records": excluded,
        "verdict": domain_verdict(
            leads_by_detector,
            len(onset_records),
            noun="AF onsets",
            singular="AF onset",
            multiscale=multiscale,
        ),
    }
    (out / result_filename).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(payload["verdict"])
    print(
        f"{len(trials)} null trials; {len(excluded)} records excluded (no clean onset)"
    )
    print(f"results written to {out}")


if __name__ == "__main__":  # pragma: no cover - CLI shell over the tested logic
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="directory holding the MIT-BIH AFDB records")
    parser.add_argument("output_dir", help="directory for the sealed derived output")
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="also evaluate the multi-scale CSD detector",
    )
    arguments = parser.parse_args()
    main(arguments.data_dir, arguments.output_dir, multiscale=arguments.multiscale)
