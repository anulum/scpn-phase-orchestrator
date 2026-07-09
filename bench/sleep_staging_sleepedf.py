# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sleep-EDF N3 vs Wake honest detector audit

"""Honest audit of a slow-wave detector for N3 sleep on PhysioNet Sleep-EDF.

Loads one PhysioNet Sleep-EDF Expanded recording, extracts the Fpz-Cz EEG
channel, computes a per-epoch normalized delta-band analytic-amplitude score,
and audits its ability to separate expert-scored N3 epochs from Wake epochs at
a matched false-alarm rate. The resulting verdict is sealed into a
content-addressed JSON record.

The detector is intentionally simple: N3 (slow-wave sleep) is defined by
high-amplitude delta oscillations, so the score is the mean delta-band Hilbert
envelope divided by the mean broadband Hilbert envelope. This measures the
fraction of the EEG's total instantaneous power that lives in the delta band.
A cross-band Kuramoto-R phase-coherence score was evaluated during development
and did not separate N3 from Wake at the matched-false-alarm bar; the amplitude
envelope is the oscillator observable that carries the discriminative signal.

Usage:
    python bench/sleep_staging_sleepedf.py \
        SC4001E0-PSG.edf SC4001EC-Hypnogram.edf \
        examples/real_data/sleepedf_staging

Raw EDF files are citation-only and are never redistributed by this script.
Only derived sealed audit records and aggregate summaries are written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, hilbert

from scpn_phase_orchestrator.evaluation.auditor import audit_detector
from scpn_phase_orchestrator.evaluation.record import seal_detector_audit

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

FloatArray = NDArray[np.float64]

#: EEG channel used by the detector.
CHANNEL_LABEL = "EEG Fpz-Cz"
#: Native sampling rate of Sleep-EDF cassette recordings, in hertz.
SAMPLING_RATE_HZ = 100.0
#: Epoch length for sleep staging, in seconds.
EPOCH_SECONDS = 30.0
#: Delta band passband, in hertz.
DELTA_BAND_HZ = (0.5, 4.0)
#: Butterworth band-pass order.
FILTER_ORDER = 3
#: Target false-alarm rate for the matched-FA calibration.
TARGET_FALSE_ALARM = 0.10
#: Permutation seed for the significance test.
PERMUTATION_SEED = 42
#: Number of permutations for the significance test.
PERMUTATIONS = 10_000
#: Significance level for the ``beats_chance`` convenience flag.
ALPHA = 0.05
#: Decimal places to which per-epoch scores are rounded before auditing. This
#: insulates the sealed record from cross-environment floating-point noise
#: (different BLAS/SIMD paths) while preserving detection sensitivity.
SCORE_PRECISION = 6

#: Mapping from Sleep-EDF annotation strings to canonical stage labels.
STAGE_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": "Unknown",
}


# --------------------------------------------------------------------------- #
# Signal processing (pure, deterministic, exercised on synthetic arrays)       #
# --------------------------------------------------------------------------- #


def _bandpass(sig: FloatArray, fs: float, lo: float, hi: float) -> FloatArray:
    """Return a zero-phase Butterworth band-pass filtered signal."""
    nyq = fs / 2.0
    b, a = butter(FILTER_ORDER, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def _load_channel(path: str | Path, label: str) -> FloatArray:
    """Read the first signal matching ``label`` from an EDF file.

    Parameters
    ----------
    path : str or Path
        Path to the EDF recording.
    label : str
        Signal label to match (substring match).

    Returns
    -------
    FloatArray
        The matched channel, shape ``(T,)``.

    Raises
    ------
    ValueError
        If no channel matches ``label``.
    """
    import pyedflib

    reader = pyedflib.EdfReader(str(path))
    try:
        labels = reader.getSignalLabels()
        selected = next(
            (i for i, lab in enumerate(labels) if label in lab),
            None,
        )
        if selected is None:
            raise ValueError(f"no channel matching {label!r} in {path}")
        data = reader.readSignal(selected)
    finally:
        reader.close()
    return np.asarray(data, dtype=np.float64)


def _load_annotations(path: str | Path) -> tuple[FloatArray, FloatArray, list[str]]:
    """Return onset, duration, description arrays from a Sleep-EDF hypnogram."""
    import pyedflib

    reader = pyedflib.EdfReader(str(path))
    try:
        onsets, durations, descriptions = reader.readAnnotations()
    finally:
        reader.close()
    return (
        np.asarray(onsets, dtype=np.float64),
        np.asarray(durations, dtype=np.float64),
        list(descriptions),
    )


def _epoch_scores(eeg: FloatArray, fs: float) -> FloatArray:
    """Return the normalized delta-band analytic-amplitude score per epoch.

    For each 30-second epoch the score is

        mean(|H(delta_filter(eeg))|) / mean(|H(eeg)|)

    where ``H`` is the Hilbert transform. This is the fraction of the EEG's
    instantaneous broadband power that is concentrated in the delta band.

    Parameters
    ----------
    eeg : FloatArray
        Raw EEG channel, shape ``(T,)``.
    fs : float
        Sampling rate in hertz.

    Returns
    -------
    FloatArray
        Score per epoch, shape ``(n_epochs,)``.
    """
    delta = _bandpass(eeg, fs, DELTA_BAND_HZ[0], DELTA_BAND_HZ[1])
    delta_env = np.abs(hilbert(delta))
    broad_env = np.abs(hilbert(eeg))

    epoch_len = int(EPOCH_SECONDS * fs)
    n_epochs = len(eeg) // epoch_len
    scores = np.empty(n_epochs, dtype=np.float64)
    for e in range(n_epochs):
        start = e * epoch_len
        stop = start + epoch_len
        scores[e] = round(
            float(delta_env[start:stop].mean() / broad_env[start:stop].mean()),
            SCORE_PRECISION,
        )
    return scores


def _epoch_stages(
    n_epochs: int,
    onsets: FloatArray,
    durations: FloatArray,
    descriptions: Sequence[str],
) -> list[str]:
    """Map hypnogram annotations to one stage label per epoch.

    Parameters
    ----------
    n_epochs : int
        Number of epochs.
    onsets, durations : FloatArray
        Annotation onsets and durations in seconds.
    descriptions : sequence of str
        Annotation descriptions.

    Returns
    -------
    list[str]
        One canonical stage label per epoch.
    """
    stages = ["Unknown"] * n_epochs
    for onset, duration, desc in zip(onsets, durations, descriptions, strict=True):
        stage = STAGE_MAP.get(desc, "Unknown")
        if stage == "Unknown":
            continue
        start_epoch = int(onset / EPOCH_SECONDS)
        duration_epochs = max(1, int(duration / EPOCH_SECONDS))
        for e in range(start_epoch, min(start_epoch + duration_epochs, n_epochs)):
            stages[e] = stage
    return stages


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #


def main(
    psg_path: str | Path,
    hypnogram_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Run the N3-vs-Wake audit and write sealed derived artefacts.

    Parameters
    ----------
    psg_path : str or Path
        Path to the Sleep-EDF PSG recording.
    hypnogram_path : str or Path
        Path to the expert hypnogram annotation EDF.
    output_dir : str or Path
        Directory for the sealed audit record and summary JSON.
    """
    psg_path = Path(psg_path)
    hypnogram_path = Path(hypnogram_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eeg = _load_channel(psg_path, CHANNEL_LABEL)
    onsets, durations, descriptions = _load_annotations(hypnogram_path)

    scores = _epoch_scores(eeg, SAMPLING_RATE_HZ)
    stages = _epoch_stages(len(scores), onsets, durations, descriptions)

    event_scores = [float(scores[i]) for i, s in enumerate(stages) if s == "N3"]
    null_scores = [float(scores[i]) for i, s in enumerate(stages) if s == "Wake"]

    if not event_scores:
        raise ValueError("no N3 epochs found in hypnogram")
    if not null_scores:
        raise ValueError("no Wake epochs found in hypnogram")

    audit = audit_detector(
        event_scores=event_scores,
        null_scores=null_scores,
        detector_name="normalized_delta_envelope",
        target_false_alarm=TARGET_FALSE_ALARM,
        n_permutations=PERMUTATIONS,
        seed=PERMUTATION_SEED,
        alpha=ALPHA,
    )

    corpus_id = f"sleepedf-n3-vs-wake-{psg_path.stem.lower()}"
    captured_at = "2026-07-08T21:00:00Z"
    sealed = seal_detector_audit(
        audit,
        corpus_id=corpus_id,
        captured_at=captured_at,
    )

    audit_record = sealed.to_record()
    audit_path = out / "sleepedf_n3_vs_wake_audit.json"
    audit_path.write_text(json.dumps(audit_record, indent=2) + "\n", encoding="utf-8")

    summary = {
        "benchmark": "sleep_staging_sleepedf",
        "corpus": "PhysioNet Sleep-EDF Expanded",
        "subject_recording": psg_path.stem,
        "hypnogram": hypnogram_path.stem,
        "channel": CHANNEL_LABEL,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "epoch_seconds": EPOCH_SECONDS,
        "delta_band_hz": list(DELTA_BAND_HZ),
        "n_epochs": len(scores),
        "n_n3": len(event_scores),
        "n_wake": len(null_scores),
        "score_mean_n3": round(float(np.mean(event_scores)), 6),
        "score_mean_wake": round(float(np.mean(null_scores)), 6),
        "target_false_alarm": TARGET_FALSE_ALARM,
        "detector_name": audit.detector_name,
        "matched_threshold": audit.matched_threshold,
        "achieved_false_alarm": audit.achieved_false_alarm,
        "detection_rate": audit.detection_rate,
        "n_events_alarmed": audit.n_events_alarmed,
        "p_value": audit.p_value,
        "beats_chance": audit.beats_chance,
        "permutation_seed": PERMUTATION_SEED,
        "n_permutations": PERMUTATIONS,
        "corpus_id": corpus_id,
        "captured_at": captured_at,
        "audit_content_hash": audit_record["content_hash"],
        "source_files": {
            "psg_sha256": _file_sha256(psg_path),
            "hypnogram_sha256": _file_sha256(hypnogram_path),
        },
    }
    summary_path = out / "sleepedf_n3_vs_wake_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"Sleep-EDF N3 vs Wake audit: "
        f"n_n3={len(event_scores)}, n_wake={len(null_scores)}, "
        f"detection_rate={audit.detection_rate:.3f}, "
        f"achieved_fa={audit.achieved_false_alarm:.3f}, "
        f"p_value={audit.p_value:.4f}, beats_chance={audit.beats_chance}"
    )
    print(f"Sealed audit written to {audit_path}")
    print(f"Summary written to {summary_path}")


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``.

    This is recorded for provenance only; the integrity test guards the sealed
    audit record, not the raw EDF files.
    """
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":  # pragma: no cover - CLI shell over tested logic
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "psg_edf",
        help="Sleep-EDF PSG recording (e.g. SC4001E0-PSG.edf)",
    )
    parser.add_argument(
        "hypnogram_edf",
        help="Sleep-EDF hypnogram annotation EDF (e.g. SC4001EC-Hypnogram.edf)",
    )
    parser.add_argument("output_dir", help="Directory for sealed audit output")
    args = parser.parse_args()
    main(args.psg_edf, args.hypnogram_edf, args.output_dir)
