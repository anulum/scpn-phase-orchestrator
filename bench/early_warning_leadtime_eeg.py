# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real scalp-EEG early-warning lead-time capstone

"""Real scalp-EEG early-warning capstone: matched-false-alarm seizure lead time.

A fair synthetic head-to-head (``bench/early_warning_leadtime.py``) and the real
EEG check behind it established that *detection* is a commodity: no single
early-warning indicator beats the others by a decisive margin, so "our detector
warns earlier" is not a claim this platform can honestly make. What it *can*
claim — and what this capstone demonstrates on real data — is the auditable
envelope: the same three-member detector suite and its fusion, run on a public
scalp-EEG seizure corpus, calibrated to a matched false-alarm rate on genuine
interictal recordings, with every alarm (or silence) sealed into a
content-addressed, claim-bounded :class:`EarlyWarningEvidence` record carrying an
honest lead time.

This capstone is the scalp-EEG *adapter* onto the domain-neutral machinery. The
signal processing here — band-pass, Hilbert analytic phase, phase-consistent
decimation — is the only EEG-specific work; it produces the neutral
:class:`~scpn_phase_orchestrator.monitor.early_warning_suite.SuiteObservables`
bundle, and everything downstream (segmentation, matched-false-alarm calibration,
lead measurement, sealing, verdict) is the shared harness in
:mod:`bench.early_warning_domain`, reused unchanged by the cardiac and grid
capstones. :class:`EEGPhaseAdapter` packages the pipeline as a
:class:`~scpn_phase_orchestrator.monitor.early_warning_suite.DomainObservableAdapter`.

Corpus
------
The CHB-MIT Scalp EEG Database (Shoeb 2009; Goldberger et al. 2000, PhysioNet):
23-channel bipolar scalp EEG sampled at 256 Hz, with clinician-annotated seizure
onset times. The raw EDF recordings are **citation-only**: they are downloaded
from PhysioNet and are **never redistributed here**. This module reads them, but
its tests exercise every signal-processing and evaluation path on **synthetic
arrays** so the coverage of the logic never depends on the protected corpus.

Observable pipeline
-------------------
Each recording is turned into one coherent decimated analytic-phase field, from
which all three suite members read complementary moments:

1. Zero-phase Butterworth band-pass, 4–30 Hz — the θ–β range that carries the
   pre-ictal synchronisation build-up, away from mains and drift.
2. Per-channel Hilbert analytic phase ``φ(t)``.
3. Decimation 256 → 32 Hz applied to the *continuous* components ``sin φ`` and
   ``cos φ`` (a wrapped phase must never be low-pass filtered — its ±π jumps are
   not band-limited), with the phase reconstructed by ``atan2``. The neutral
   bundle's ``sin(phase)`` projection and cross-channel order parameter are then
   derived from that single decimated field, so the members read one consistent
   representation rather than three independently filtered ones:

   * critical slowing down reads the cross-channel order parameter
     ``R(t) = |⟨e^{iφ}⟩|`` (second-moment variance / autocorrelation rise);
   * rising synchronisation reads the per-channel phases (first-moment coherence
     rise, its own internal ``R``);
   * ordinal-transition entropy reads the per-channel projection ``sin φ`` (a
     regularisation drop).

Window 128, step 16 at 32 Hz — a 4-second window hopped every 0.5 s.

Fixed pre-onset segment — a clean baseline
------------------------------------------
Each seizure is evaluated on a fixed segment ending at onset: a leading
``BASELINE_SECONDS`` baseline followed by a ``HORIZON_SECONDS`` detection window
(:data:`SEGMENT_SECONDS` total). The baseline is a fixed pre-onset interval, not
a fraction of the whole recording, so an early-onset seizure cannot contaminate
its own baseline with ictal samples; the onset sits at the segment end, so every
window is pre-ictal and any alarm is a genuine lead. A seizure whose onset is
earlier than :data:`SEGMENT_SECONDS` cannot form a clean baseline and is
**excluded and reported as such**, never counted as a silent null.

Matched false alarm — the honest comparison
--------------------------------------------
A lead time is only meaningful at a fixed false-alarm rate, and one ictal file
cannot self-calibrate its own. Each interictal recording is cut into many
non-overlapping null trials of the *same* length and baseline/horizon structure
as a seizure's pre-onset segment (:func:`~bench.early_warning_domain.null_trials`),
giving a fine false-alarm estimate rather than a handful of whole-recording
trials. Each detector's threshold is the smallest that holds the trial
false-alarm rate at or below :data:`TARGET_FALSE_ALARM`; only then is its lead —
``onset − alarm`` in samples, converted to seconds — measured on the seizures.
**The gain from fusion is reported as improved matched-false-alarm lead, never as
a raw detection rate**: an OR of the members trivially raises the rate by spending
the false-alarm budget. If the fusion does not beat the best single member at
matched false alarm, this capstone says so — a valid result, and the auditable
moat holds regardless.

Every evaluated seizure yields four sealed :class:`EarlyWarningEvidence` records
(three members + fusion), including a sealed *silence* when a detector does not
fire — the honest half of the moat. Only these derived, sealed artefacts are
committed, mirroring ``examples/real_data/iso_ne_case1/``; the raw EDF never is.

References
----------
* A. H. Shoeb, *Application of Machine Learning to Epileptic Seizure Onset
  Detection and Treatment*, PhD thesis, MIT, 2009 — the CHB-MIT corpus.
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
    from collections.abc import Mapping, Sequence

FloatArray = NDArray[np.float64]

#: Native sampling rate of the CHB-MIT scalp-EEG recordings, in hertz.
SAMPLING_RATE_HZ = 256.0
#: Decimation factor applied to the analytic-phase components (256 → 32 Hz).
DECIMATION = 8
#: Analysis sampling rate after decimation, in hertz.
DECIMATED_RATE_HZ = SAMPLING_RATE_HZ / DECIMATION
#: Band-pass passband in hertz — the θ–β range carrying pre-ictal synchronisation.
BAND_HZ = (4.0, 30.0)
#: Butterworth band-pass order.
FILTER_ORDER = 4

#: Analysis window length in samples (4 s at 32 Hz).
WINDOW = 128
#: Hop between consecutive windows in samples (0.5 s at 32 Hz).
STEP = 16
#: Target false-alarm rate the detectors are calibrated to on the interictal null.
TARGET_FALSE_ALARM = DEFAULT_TARGET_FALSE_ALARM

#: Leading baseline length, in seconds, of an analysis segment. Fixed (not a
#: fraction of the whole recording) so an early-onset seizure cannot contaminate
#: the baseline with ictal samples.
BASELINE_SECONDS = 300.0
#: Detection horizon, in seconds, before onset — the pre-onset window the suite
#: is allowed to warn within, and the maximum measurable lead.
HORIZON_SECONDS = 600.0
#: Full analysis-segment length (baseline + horizon), in seconds.
SEGMENT_SECONDS = BASELINE_SECONDS + HORIZON_SECONDS
#: Analysis-segment length in decimated samples.
SEGMENT_SAMPLES = int(SEGMENT_SECONDS * DECIMATED_RATE_HZ)
#: Baseline fraction of an analysis segment (its leading ``BASELINE_SECONDS``).
SEGMENT_BASELINE_FRACTION = BASELINE_SECONDS / SEGMENT_SECONDS

_OBSERVABLE_CSD = (
    "cross-channel Kuramoto order parameter R(t) of scalp-EEG analytic phase "
    "(4-30 Hz, decimated to 32 Hz)"
)
_OBSERVABLE_SYNC = "per-channel scalp-EEG analytic phase (4-30 Hz, decimated to 32 Hz)"
_OBSERVABLE_ENTROPY = (
    "per-channel phase projection sin(phase) of scalp EEG (4-30 Hz, decimated to 32 Hz)"
)
_OBSERVABLE_ENSEMBLE = (
    "fused early-warning suite over scalp-EEG analytic phase "
    "(4-30 Hz, decimated to 32 Hz)"
)
#: The scalp-EEG observable description sealed into each detector's record.
_OBSERVABLE_DESCRIPTIONS = {
    CRITICAL_SLOWING_DOWN: _OBSERVABLE_CSD,
    SYNCHRONISATION: _OBSERVABLE_SYNC,
    TRANSITION_ENTROPY: _OBSERVABLE_ENTROPY,
    ENSEMBLE_WEIGHTED: _OBSERVABLE_ENSEMBLE,
}

#: Annotated seizure onset times (seconds into each CHB-MIT chb01 record).
SEIZURE_ONSETS_S: dict[str, int] = {
    "chb01_03": 2996,
    "chb01_04": 1467,
    "chb01_15": 1732,
    "chb01_16": 1015,
    "chb01_18": 1720,
    "chb01_21": 327,
    "chb01_26": 1862,
}
#: A few seizure-free chb01 records used as the interictal false-alarm null.
INTERICTAL_RECORDS: tuple[str, ...] = (
    "chb01_01",
    "chb01_02",
    "chb01_05",
    "chb01_06",
    "chb01_07",
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
    "EEGPhaseAdapter",
    "edf_start_datetime",
    "eeg_observables",
    "load_edf_channels",
    "main",
]


# --------------------------------------------------------------------------- #
# Observable pipeline (pure — fully exercised on synthetic arrays)             #
# --------------------------------------------------------------------------- #


def eeg_observables(
    raw: FloatArray,
    *,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    decimation: int = DECIMATION,
    band_hz: tuple[float, float] = BAND_HZ,
    filter_order: int = FILTER_ORDER,
) -> SuiteObservables:
    """Turn a raw multichannel recording into the suite's neutral observables.

    This is the scalp-EEG-specific half of the capstone: band-pass, Hilbert
    analytic phase, and phase-consistent decimation to a per-channel phase field,
    from which
    :func:`~scpn_phase_orchestrator.monitor.early_warning_suite.observables_from_phases`
    derives the neutral bundle the domain-neutral suite reads.

    Parameters
    ----------
    raw : FloatArray
        Raw per-channel EEG, shape ``(N, T)`` with at least two channels (the
        order parameter and synchrony are undefined for a single channel).
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
        The reconstructed phases, the ``sin(phase)`` field, and the cross-channel
        order parameter, all at ``sampling_rate_hz / decimation``.

    Raises
    ------
    ValueError
        If the recording has fewer than two channels or is otherwise malformed.
    """
    array = validate_signals(raw, "raw")
    if array.shape[0] < 2:
        raise ValueError("raw must have at least two channels for synchrony")
    filtered = bandpass(
        array, sampling_rate_hz=sampling_rate_hz, band_hz=band_hz, order=filter_order
    )
    phases = decimate_analytic_phase(analytic_phase(filtered), factor=decimation)
    return observables_from_phases(
        phases, sampling_rate_hz=sampling_rate_hz / decimation
    )


@dataclass(frozen=True)
class EEGPhaseAdapter:
    """The scalp-EEG bridge from raw EDF channels to :class:`SuiteObservables`.

    A ``DomainObservableAdapter`` packaging the band-pass / Hilbert / decimation
    pipeline, so the neutral suite can screen scalp EEG exactly as it screens
    cardiac or grid signals through their own adapters.

    Attributes
    ----------
    sampling_rate_hz : float
        Native sampling rate of the raw channels, in hertz.
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
        """Return the domain label ``scalp_eeg``."""
        return "scalp_eeg"

    def observables(self, raw: FloatArray) -> SuiteObservables:
        """Return the neutral observable bundle for one raw EEG recording.

        Parameters
        ----------
        raw : FloatArray
            Raw per-channel scalp EEG, shape ``(N, T)`` with at least two
            channels, at :attr:`sampling_rate_hz`.

        Returns
        -------
        SuiteObservables
            The decimated analytic-phase bundle the suite reads.
        """
        return eeg_observables(
            raw,
            sampling_rate_hz=self.sampling_rate_hz,
            decimation=self.decimation,
            band_hz=self.band_hz,
            filter_order=self.filter_order,
        )


# --------------------------------------------------------------------------- #
# EDF ingestion (thin — touches the citation-only corpus, tested on synthetic  #
# EDF fixtures, never on the redistributed raw recordings)                      #
# --------------------------------------------------------------------------- #


def load_edf_channels(
    path: str | Path, *, expected_rate_hz: float = SAMPLING_RATE_HZ
) -> FloatArray:
    """Read the same-rate signal channels of an EDF recording into an array.

    Only channels sampled at ``expected_rate_hz`` are kept (CHB-MIT files
    occasionally carry an off-rate annotation or dummy channel), so the returned
    array is a clean ``(N, T)`` block ready for :func:`eeg_observables`.

    Parameters
    ----------
    path : str or Path
        Path to the EDF recording.
    expected_rate_hz : float
        Sampling rate a channel must have to be included.

    Returns
    -------
    FloatArray
        The selected channels, shape ``(N, T)``.

    Raises
    ------
    ValueError
        If no channel matches ``expected_rate_hz``.
    """
    import pyedflib

    reader = pyedflib.EdfReader(str(path))
    try:
        selected = [
            index
            for index in range(reader.signals_in_file)
            if abs(float(reader.getSampleFrequency(index)) - expected_rate_hz) < 1.0e-6
        ]
        if not selected:
            raise ValueError(
                f"no channel in {path} is sampled at {expected_rate_hz} Hz"
            )
        channels = [
            np.asarray(reader.readSignal(index), dtype=np.float64) for index in selected
        ]
    finally:
        reader.close()
    return np.ascontiguousarray(np.vstack(channels), dtype=np.float64)


def edf_start_datetime(path: str | Path) -> str:
    """Return the EDF recording's start datetime as an ISO-8601 string.

    CHB-MIT anonymises the physical capture instant, so this is provenance from
    the file rather than a wall-clock reading, and is carried into the seals
    verbatim for reproducibility.

    Parameters
    ----------
    path : str or Path
        Path to the EDF recording.

    Returns
    -------
    str
        The recording's start datetime in ISO-8601 form.
    """
    import pyedflib

    reader = pyedflib.EdfReader(str(path))
    try:
        start = reader.getStartdatetime()
    finally:
        reader.close()
    return start.isoformat()


# --------------------------------------------------------------------------- #
# Orchestration (I/O shell over the tested logic)                              #
# --------------------------------------------------------------------------- #


def main(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    interictal_records: Sequence[str] = INTERICTAL_RECORDS,
    seizures: Mapping[str, int] = SEIZURE_ONSETS_S,
    segment_samples: int = SEGMENT_SAMPLES,
    baseline_fraction: float = SEGMENT_BASELINE_FRACTION,
) -> None:
    """Run the capstone over CHB-MIT chb01 and write the sealed derived artefacts.

    Reads the interictal null and the annotated seizures from ``data_dir`` (raw
    EDF, citation-only), calibrates every detector to a matched false-alarm rate,
    seals each detector's alarm per seizure, and writes the sealed records plus an
    aggregate results JSON to ``output_dir``. Only the derived, sealed artefacts
    are written; the raw EDF is never copied.

    Parameters
    ----------
    data_dir : str or Path
        Directory holding the ``<record>.edf`` recordings.
    output_dir : str or Path
        Directory the sealed derived artefacts are written to.
    interictal_records : sequence of str
        Seizure-free record stems forming the false-alarm null.
    seizures : Mapping[str, int]
        Record stem to annotated onset (seconds) for each seizure to evaluate.
    segment_samples : int
        Pre-onset analysis-segment length in decimated samples; also the null
        trial length. Defaults to :data:`SEGMENT_SAMPLES`.
    baseline_fraction : float
        Leading baseline fraction of each segment. Defaults to
        :data:`SEGMENT_BASELINE_FRACTION`.
    """
    data = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    segment_seconds = segment_samples / DECIMATED_RATE_HZ
    adapter = EEGPhaseAdapter()

    # Finer matched-false-alarm null: cut each interictal hour into many trials of
    # the same length and structure as a seizure's pre-onset analysis segment.
    null_observables = [
        adapter.observables(load_edf_channels(data / f"{record}.edf"))
        for record in interictal_records
    ]
    trials = null_trials(null_observables, segment_samples=segment_samples)
    calibration = calibrate_detectors(
        trials,
        target_fa=TARGET_FALSE_ALARM,
        window=WINDOW,
        step=STEP,
        baseline_fraction=baseline_fraction,
    )
    thresholds = calibration.thresholds

    leads_by_detector: dict[str, list[float]] = {name: [] for name in DETECTORS}
    seizure_records: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for record_id, onset_s in seizures.items():
        path = data / f"{record_id}.edf"
        observables = adapter.observables(load_edf_channels(path))
        onset_sample = int(round(onset_s * observables.sampling_rate_hz))
        if onset_sample < segment_samples:
            # Too early for a clean pre-onset baseline; excluded, not counted null.
            excluded.append({"record_id": record_id, "onset_s": onset_s})
            continue
        # Fixed pre-onset segment with a guaranteed-clean leading baseline; the
        # onset sits at the segment end, so any alarm is a genuine lead.
        segment = slice_observables(
            observables, start=onset_sample - segment_samples, stop=onset_sample
        )
        result = evaluate_seizure(
            segment,
            record_id=record_id,
            onset_sample=segment_samples,
            signal_source=(
                f"CHB-MIT scalp EEG {record_id} (Shoeb 2009) / seizure onset "
                f"{onset_s} s / {segment_seconds:g} s pre-onset segment"
            ),
            captured_at=edf_start_datetime(path),
            thresholds=thresholds,
            observable_descriptions=_OBSERVABLE_DESCRIPTIONS,
            window=WINDOW,
            step=STEP,
            baseline_fraction=baseline_fraction,
        )
        (out / f"{record_id}_early_warning_evidence.json").write_text(
            json.dumps(result.to_audit_record(), indent=2) + "\n", encoding="utf-8"
        )
        for name, lead in result.lead_seconds().items():
            if lead is not None:
                leads_by_detector[name].append(lead)
        seizure_records.append(
            {
                "record_id": record_id,
                "onset_s": onset_s,
                "lead_seconds": result.lead_seconds(),
            }
        )

    payload = {
        "benchmark": "early_warning_leadtime_eeg",
        "corpus": "CHB-MIT Scalp EEG Database, subject chb01 (Shoeb 2009)",
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "decimated_rate_hz": DECIMATED_RATE_HZ,
        "band_hz": list(BAND_HZ),
        "window": WINDOW,
        "step": STEP,
        "baseline_seconds": baseline_fraction * segment_seconds,
        "horizon_seconds": (1.0 - baseline_fraction) * segment_seconds,
        "segment_seconds": segment_seconds,
        "target_false_alarm": TARGET_FALSE_ALARM,
        "interictal_null_records": list(interictal_records),
        "n_null_trials": len(trials),
        "matched_false_alarm_thresholds": thresholds,
        "achieved_false_alarm": calibration.achieved_false_alarm,
        "seizures": seizure_records,
        "excluded_seizures": excluded,
        "verdict": domain_verdict(
            leads_by_detector,
            len(seizure_records),
            noun="seizures",
            singular="seizure",
        ),
    }
    (out / "early_warning_leadtime_eeg_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(payload["verdict"])
    print(f"{len(trials)} null trials; {len(excluded)} seizures excluded (early onset)")
    print(f"results written to {out}")


if __name__ == "__main__":  # pragma: no cover - CLI shell over the tested logic
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="directory holding the CHB-MIT chb01 EDFs")
    parser.add_argument("output_dir", help="directory for the sealed derived output")
    arguments = parser.parse_args()
    main(arguments.data_dir, arguments.output_dir)
