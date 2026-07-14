# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CAP multi-channel N3 vs Wake honest detector audit

"""Honest audit of three slow-wave detectors on PhysioNet CAP Sleep Database.

Loads one or more PhysioNet CAP Sleep Database recordings, selects the
available EEG channels, parses the REMlogic text hypnogram into 30-second
epochs, and audits three detectors' ability to separate expert-scored N3 epochs
from Wake epochs at a matched false-alarm rate.

Detector 1 is the normalized delta-band Hilbert envelope averaged across
channels. Detector 2 is the multi-channel delta-phase Kuramoto order parameter
(R), computed across channels at every time sample and then averaged over the
epoch. Detector 3 is the SNR-weighted variant of detector 2, where each
channel's contribution is weighted by the square root of its local delta-band
SNR. All three are sealed into content-addressed JSON records.

Single-recording usage:
    python bench/cap_multichannel_n3_vs_wake.py \
        scratchpad/cap_data/n1.edf \
        scratchpad/cap_data/n1.txt \
        examples/real_data/cap_multichannel_staging

Batch usage:
    python bench/cap_multichannel_n3_vs_wake.py \
        --manifest cap_multichannel_manifest.csv \
        examples/real_data/cap_multichannel_staging

A manifest CSV has one header row and rows of the form
``recording_id,edf_path,txt_path``. In batch mode each recording's artefacts
are written to ``OUTPUT_DIR/RECORDING_ID/`` and an aggregate comparison JSON is
written to ``OUTPUT_DIR/cap_multichannel_aggregate.json``.

Raw EDF files are citation-only and are never redistributed by this script.
Only derived sealed audit records and aggregate summaries are written.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, hilbert, resample

from bench.honest_dataset_audit import (
    AuditConfig,
    compute_aggregate,
    file_sha256,
    run_audit,
    write_aggregate,
)
from scpn_phase_orchestrator.monitor.adaptive_kuramoto import (
    compute_adaptive_kuramoto_scores,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

FloatArray = NDArray[np.float64]

#: Native/common sampling rate used for analysis, in hertz.
#: CAP recordings are typically sampled at 100 Hz or 512 Hz. To keep the
#: analysis reproducible across machines, all EEG channels are resampled to
#: this rate before feature extraction.
TARGET_SAMPLING_RATE_HZ = 100.0
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
#: Decimal places to which per-epoch scores are rounded before auditing.
SCORE_PRECISION = 6

#: Mapping from REMlogic stage event strings to canonical stage labels.
STAGE_MAP = {
    "SLEEP-S0": "Wake",
    "SLEEP-S1": "N1",
    "SLEEP-S2": "N2",
    "SLEEP-S3": "N3",
    "SLEEP-S4": "N3",
    "SLEEP-REM": "REM",
}

#: Preferred monopolar EEG channel labels (substring match).
MONOPOLAR_LABELS = [
    "EEG F3",
    "EEG F4",
    "EEG C3",
    "EEG C4",
    "EEG O1",
    "EEG O2",
]

#: Fallback bipolar EEG derivations (substring match).
#: CAP Sleep Database uses F1/F2 instead of Fp1/Fp2 for the frontal leads.
BIPOLAR_LABELS = [
    "F1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "F2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]


# --------------------------------------------------------------------------- #
# Signal processing                                                            #
# --------------------------------------------------------------------------- #


def _bandpass(sig: FloatArray, fs: float, lo: float, hi: float) -> FloatArray:
    """Return a zero-phase Butterworth band-pass filtered signal."""
    nyq = fs / 2.0
    b, a = butter(FILTER_ORDER, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def _resample_to_rate(
    sig: FloatArray,
    source_fs: float,
    target_fs: float,
) -> FloatArray:
    """Resample ``sig`` from ``source_fs`` to ``target_fs`` deterministically."""
    if source_fs == target_fs:
        return sig
    ratio = target_fs / source_fs
    n_target = int(round(len(sig) * ratio))
    return resample(sig, n_target)


def _load_eeg_channels(
    path: str | Path,
    target_fs: float = TARGET_SAMPLING_RATE_HZ,
) -> tuple[FloatArray, list[str], float]:
    """Load EEG channels from a CAP EDF, resample to ``target_fs``.

    Parameters
    ----------
    path : str or Path
        Path to the EDF recording.
    target_fs : float
        Target sampling rate in hertz.

    Returns
    -------
    data : FloatArray
        Array of shape ``(n_channels, n_samples)``.
    labels : list[str]
        The selected channel labels.
    fs : float
        The common sampling rate ``target_fs``.

    Raises
    ------
    ValueError
        If fewer than 3 EEG channels are found.
    """
    import pyedflib

    reader = pyedflib.EdfReader(str(path))
    try:
        labels = reader.getSignalLabels()
        selected_indices: list[int] = []
        selected_labels: list[str] = []
        for candidate in MONOPOLAR_LABELS + BIPOLAR_LABELS:
            for i, lab in enumerate(labels):
                if candidate in lab and i not in selected_indices:
                    selected_indices.append(i)
                    selected_labels.append(lab)
                    break
        if len(selected_indices) < 3:
            available = "\n".join(labels)
            raise ValueError(
                f"found only {len(selected_indices)} EEG channels "
                f"(need >= 3) in {path}\nAvailable labels:\n{available}"
            )

        channel_signals: list[FloatArray] = []
        min_duration_samples: int | None = None
        source_fs_list: list[float] = []
        for idx in selected_indices:
            fs_i = reader.getSampleFrequency(idx)
            source_fs_list.append(fs_i)
            sig = reader.readSignal(idx)
            resampled = _resample_to_rate(
                np.asarray(sig, dtype=np.float64), fs_i, target_fs
            )
            channel_signals.append(resampled)
            if min_duration_samples is None or len(resampled) < min_duration_samples:
                min_duration_samples = len(resampled)
    finally:
        reader.close()

    # Trim all channels to the common length (recordings may differ by a few
    # samples due to rounding during resampling).
    if min_duration_samples is None:
        raise ValueError("no EEG channels loaded")
    data = np.stack([sig[:min_duration_samples] for sig in channel_signals])
    return data, selected_labels, target_fs


# --------------------------------------------------------------------------- #
# Annotation parsing                                                           #
# --------------------------------------------------------------------------- #


def _find_header_index(lines: list[str]) -> int:
    """Return the line index just after the REMlogic stage header.

    The header line starts with ``Sleep Stage`` and may or may not include a
    ``Position`` column. If no header is found, raise ``ValueError``.
    """
    for i, line in enumerate(lines):
        if line.startswith("Sleep Stage") and "Event" in line and "Duration" in line:
            return i + 1
    raise ValueError("could not find REMlogic header")


def _parse_remlogic_stages(path: str | Path) -> list[tuple[float, float, str]]:
    """Parse REMlogic text export into stage intervals.

    Returns
    -------
    list[tuple[float, float, str]]
        List of ``(onset_seconds, duration_seconds, canonical_stage)`` for stage
        rows. Non-stage events are skipped.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()

    start_idx = _find_header_index(lines)
    header_parts = lines[start_idx - 1].split("\t")
    try:
        event_idx = header_parts.index("Event")
    except ValueError as exc:
        raise ValueError("could not find 'Event' column in REMlogic header") from exc
    try:
        duration_idx = header_parts.index("Duration[s]")
    except ValueError as exc:
        raise ValueError(
            "could not find 'Duration[s]' column in REMlogic header"
        ) from exc

    intervals: list[tuple[float, float, str]] = []
    reader = csv.reader(lines[start_idx:], delimiter="\t")
    for row in reader:
        if max(event_idx, duration_idx) >= len(row):
            continue
        event = row[event_idx].strip()
        stage = STAGE_MAP.get(event)
        if stage is None:
            continue
        try:
            duration = float(row[duration_idx].strip())
        except ValueError as exc:
            raise ValueError(f"invalid duration in row {row!r}") from exc
        # Onset in seconds since recording start. We derive it from the ordinal
        # position of intervals assuming contiguous stage rows, which matches the
        # REMlogic export for these recordings.
        onset = sum(d for _, d, _ in intervals)
        intervals.append((onset, duration, stage))
    return intervals


def _epoch_stages(
    n_epochs: int,
    intervals: Sequence[tuple[float, float, str]],
) -> list[str]:
    """Map stage intervals to one canonical stage label per epoch.

    Parameters
    ----------
    n_epochs : int
        Number of epochs.
    intervals : sequence of tuple
        ``(onset_seconds, duration_seconds, stage)``.

    Returns
    -------
    list[str]
        One canonical stage label per epoch.
    """
    stages = ["Unknown"] * n_epochs
    for onset, duration, stage in intervals:
        start_epoch = int(onset / EPOCH_SECONDS)
        duration_epochs = max(1, int(round(duration / EPOCH_SECONDS)))
        for e in range(start_epoch, min(start_epoch + duration_epochs, n_epochs)):
            stages[e] = stage
    return stages


# --------------------------------------------------------------------------- #
# Detectors                                                                    #
# --------------------------------------------------------------------------- #


def _envelope_scores(data: FloatArray, fs: float) -> FloatArray:
    """Normalized delta-envelope detector scores per epoch.

    Parameters
    ----------
    data : FloatArray
        EEG channels, shape ``(n_channels, n_samples)``.
    fs : float
        Sampling rate in hertz.

    Returns
    -------
    FloatArray
        Score per epoch, shape ``(n_epochs,)``.
    """
    n_channels, n_samples = data.shape
    epoch_len = int(EPOCH_SECONDS * fs)
    n_epochs = n_samples // epoch_len
    scores = np.zeros(n_epochs, dtype=np.float64)

    for c in range(n_channels):
        delta = _bandpass(data[c], fs, DELTA_BAND_HZ[0], DELTA_BAND_HZ[1])
        delta_env = np.abs(hilbert(delta))
        broad_env = np.abs(hilbert(data[c]))
        for e in range(n_epochs):
            start = e * epoch_len
            stop = start + epoch_len
            ratio = delta_env[start:stop].mean() / broad_env[start:stop].mean()
            scores[e] += ratio / n_channels

    return np.round(scores, SCORE_PRECISION)


def _kuramoto_scores(data: FloatArray, fs: float) -> FloatArray:
    """Multi-channel delta-phase Kuramoto-R detector scores per epoch.

    For each time sample the Kuramoto order parameter across channels is

        R(t) = | (1/C) sum_c exp(i phi_c(t)) |

    where ``phi_c`` is the Hilbert phase of the delta-filtered channel. The
    epoch score is the mean of ``R(t)`` over the epoch.

    Parameters
    ----------
    data : FloatArray
        EEG channels, shape ``(n_channels, n_samples)``.
    fs : float
        Sampling rate in hertz.

    Returns
    -------
    FloatArray
        Score per epoch, shape ``(n_epochs,)``.
    """
    n_channels, n_samples = data.shape
    epoch_len = int(EPOCH_SECONDS * fs)
    n_epochs = n_samples // epoch_len

    phases = np.empty((n_channels, n_samples), dtype=np.float64)
    for c in range(n_channels):
        delta = _bandpass(data[c], fs, DELTA_BAND_HZ[0], DELTA_BAND_HZ[1])
        analytic = hilbert(delta)
        phases[c, :] = np.angle(analytic)

    # R(t) per sample, then average per epoch.
    r_t = np.abs(np.exp(1j * phases).mean(axis=0))
    scores = r_t[: n_epochs * epoch_len].reshape(n_epochs, epoch_len).mean(axis=1)

    return np.round(scores, SCORE_PRECISION)


def _snr_weighted_kuramoto_scores(data: FloatArray, fs: float) -> FloatArray:
    """SNR-weighted multi-channel delta-phase Kuramoto-R detector scores per epoch.

    For each channel the delta-band Hilbert phase is extracted. At every time
    sample the channel's weight is derived from its local delta-band SNR
    (delta power / total power) within the current epoch, soft-scaled by sqrt
    to prevent a single channel from dominating. The weighted Kuramoto order
    parameter is

        R(t) = | sum_c w_c(t) exp(i phi_c(t)) | / sum_c w_c(t)

    and the epoch score is the mean of R(t) over the epoch.

    Parameters
    ----------
    data : FloatArray
        EEG channels, shape ``(n_channels, n_samples)``.
    fs : float
        Sampling rate in hertz.

    Returns
    -------
    FloatArray
        Score per epoch, shape ``(n_epochs,)``.
    """
    n_channels, n_samples = data.shape
    epoch_len = int(EPOCH_SECONDS * fs)
    n_epochs = n_samples // epoch_len

    phases = np.empty((n_channels, n_samples), dtype=np.float64)
    for c in range(n_channels):
        delta = _bandpass(data[c], fs, DELTA_BAND_HZ[0], DELTA_BAND_HZ[1])
        analytic = hilbert(delta)
        phases[c, :] = np.angle(analytic)

    # Compute per-channel, per-epoch delta and total power for SNR weights.
    # We do this without an explicit inner loop by reshaping to
    # (n_channels, n_epochs, epoch_len).
    epoch_data = data[:, : n_epochs * epoch_len].reshape(
        n_channels, n_epochs, epoch_len
    )
    delta_data = np.empty_like(epoch_data)
    for c in range(n_channels):
        delta = _bandpass(data[c], fs, DELTA_BAND_HZ[0], DELTA_BAND_HZ[1])
        delta_data[c] = delta[: n_epochs * epoch_len].reshape(n_epochs, epoch_len)

    delta_power = (delta_data**2).mean(axis=2)
    total_power = (epoch_data**2).mean(axis=2) + 1e-12
    snr = delta_power / total_power
    weights = np.sqrt(np.maximum(snr, 0.0))  # soft, non-negative weights

    # Expand weights to per-sample shape (n_channels, n_epochs, epoch_len).
    weights_per_sample = np.repeat(weights[:, :, np.newaxis], epoch_len, axis=2)

    # phases reshaped to (n_channels, n_epochs, epoch_len).
    phases_reshaped = phases[:, : n_epochs * epoch_len].reshape(
        n_channels, n_epochs, epoch_len
    )
    r_t_per_epoch = np.abs(
        (weights_per_sample * np.exp(1j * phases_reshaped)).sum(axis=0)
        / weights.sum(axis=0)[:, np.newaxis]
    )
    scores = r_t_per_epoch.mean(axis=1)

    return np.round(scores, SCORE_PRECISION)


def _adaptive_kuramoto_scores(data: FloatArray, fs: float) -> FloatArray:
    """Adaptive quality-weighted multi-channel delta-phase Kuramoto scores.

    Uses the reusable ``monitor.adaptive_kuramoto`` implementation: channels
    are weighted by delta-band SNR penalised by excess kurtosis, and each
    epoch is pooled with the median of the weighted Kuramoto order parameter.
    """
    scores, _ = compute_adaptive_kuramoto_scores(
        data,
        fs,
        band_hz=DELTA_BAND_HZ,
        epoch_seconds=EPOCH_SECONDS,
        score_precision=SCORE_PRECISION,
    )
    return scores


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #


def _run_audit(
    scores: FloatArray,
    stages: list[str],
    detector_name: str,
    corpus_id: str,
    captured_at: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Audit one detector and return sealed record + summary fragment."""
    config = AuditConfig(
        target_false_alarm=TARGET_FALSE_ALARM,
        n_permutations=PERMUTATIONS,
        seed=PERMUTATION_SEED,
        alpha=ALPHA,
        captured_at=captured_at,
    )
    return run_audit(
        scores=scores,
        labels=stages,
        detector_name=detector_name,
        corpus_id=corpus_id,
        config=config,
        event_label="N3",
        null_label="Wake",
    )


def _process_recording(
    recording_id: str,
    edf_path: Path,
    txt_path: Path,
    output_dir: Path,
    captured_at: str,
) -> dict[str, object]:
    """Run all detectors on one recording and write sealed artefacts.

    Returns the per-recording comparison fragment used by the batch aggregator.
    """
    data, channel_labels, fs = _load_eeg_channels(edf_path)
    intervals = _parse_remlogic_stages(txt_path)

    epoch_len = int(EPOCH_SECONDS * fs)
    n_epochs = data.shape[1] // epoch_len
    stages = _epoch_stages(n_epochs, intervals)

    n_n3 = sum(1 for s in stages if s == "N3")
    n_wake = sum(1 for s in stages if s == "Wake")
    if n_n3 == 0:
        raise ValueError(f"no N3 epochs found in {txt_path}")
    if n_wake == 0:
        raise ValueError(f"no Wake epochs found in {txt_path}")

    corpus_id = f"cap-{recording_id}-multichannel-n3-vs-wake"

    envelope_scores_arr = _envelope_scores(data, fs)
    kuramoto_scores_arr = _kuramoto_scores(data, fs)
    snr_kuramoto_scores_arr = _snr_weighted_kuramoto_scores(data, fs)
    adaptive_kuramoto_scores_arr = _adaptive_kuramoto_scores(data, fs)

    envelope_audit, envelope_summary = _run_audit(
        envelope_scores_arr,
        stages,
        "normalized_delta_envelope",
        corpus_id,
        captured_at,
    )
    kuramoto_audit, kuramoto_summary = _run_audit(
        kuramoto_scores_arr,
        stages,
        "multi_channel_delta_kuramoto",
        corpus_id,
        captured_at,
    )
    snr_kuramoto_audit, snr_kuramoto_summary = _run_audit(
        snr_kuramoto_scores_arr,
        stages,
        "snr_weighted_delta_kuramoto",
        corpus_id,
        captured_at,
    )
    adaptive_kuramoto_audit, adaptive_kuramoto_summary = _run_audit(
        adaptive_kuramoto_scores_arr,
        stages,
        "adaptive_kuramoto",
        corpus_id,
        captured_at,
    )

    out = output_dir / recording_id
    out.mkdir(parents=True, exist_ok=True)

    prefix = f"cap_{recording_id}"
    (out / f"{prefix}_delta_envelope_audit.json").write_text(
        json.dumps(envelope_audit, indent=2) + "\n", encoding="utf-8"
    )
    (out / f"{prefix}_delta_envelope_summary.json").write_text(
        json.dumps(envelope_summary, indent=2) + "\n", encoding="utf-8"
    )
    (out / f"{prefix}_multichannel_kuramoto_audit.json").write_text(
        json.dumps(kuramoto_audit, indent=2) + "\n", encoding="utf-8"
    )
    (out / f"{prefix}_multichannel_kuramoto_summary.json").write_text(
        json.dumps(kuramoto_summary, indent=2) + "\n", encoding="utf-8"
    )
    (out / f"{prefix}_snr_weighted_kuramoto_audit.json").write_text(
        json.dumps(snr_kuramoto_audit, indent=2) + "\n", encoding="utf-8"
    )
    (out / f"{prefix}_snr_weighted_kuramoto_summary.json").write_text(
        json.dumps(snr_kuramoto_summary, indent=2) + "\n", encoding="utf-8"
    )
    (out / f"{prefix}_adaptive_kuramoto_audit.json").write_text(
        json.dumps(adaptive_kuramoto_audit, indent=2) + "\n", encoding="utf-8"
    )
    (out / f"{prefix}_adaptive_kuramoto_summary.json").write_text(
        json.dumps(adaptive_kuramoto_summary, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "recording_id": recording_id,
        "channels": channel_labels,
        "sampling_rate_hz": fs,
        "n_epochs": n_epochs,
        "n_n3": n_n3,
        "n_wake": n_wake,
        "source_files": {
            "edf_sha256": file_sha256(edf_path),
            "txt_sha256": file_sha256(txt_path),
        },
        "detectors": {
            "normalized_delta_envelope": envelope_summary,
            "multi_channel_delta_kuramoto": kuramoto_summary,
            "snr_weighted_delta_kuramoto": snr_kuramoto_summary,
            "adaptive_kuramoto": adaptive_kuramoto_summary,
        },
    }


def _write_single_recording_comparison(
    output_dir: Path,
    recording_id: str,
    fragment: dict[str, object],
) -> None:
    """Write the flat comparison JSON for single-recording mode."""
    comparison = {
        "benchmark": "cap_multichannel_n3_vs_wake",
        "corpus": "PhysioNet CAP Sleep Database",
        "subject_recording": recording_id,
        "channels": fragment["channels"],
        "sampling_rate_hz": fragment["sampling_rate_hz"],
        "epoch_seconds": EPOCH_SECONDS,
        "delta_band_hz": list(DELTA_BAND_HZ),
        "n_epochs": fragment["n_epochs"],
        "n_n3": fragment["n_n3"],
        "n_wake": fragment["n_wake"],
        "target_false_alarm": TARGET_FALSE_ALARM,
        "detectors": fragment["detectors"],
        "source_files": fragment["source_files"],
    }
    (output_dir / f"cap_{recording_id}_detector_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )


def _cap_recommendation(
    records: list[dict[str, object]],
    stats_by_detector: dict[str, object],
) -> dict[str, object]:
    """CAP-specific recommendation based on the Kuramoto refinement experiment."""
    envelope_stats = stats_by_detector["normalized_delta_envelope"]
    kuramoto_stats = stats_by_detector["multi_channel_delta_kuramoto"]
    snr_kuramoto_stats = stats_by_detector["snr_weighted_delta_kuramoto"]
    adaptive_stats = stats_by_detector["adaptive_kuramoto"]

    # Rank variants by mean detection rate.
    variant_means = {
        "normalized_delta_envelope": envelope_stats["mean_detection_rate"],  # type: ignore[index]  # audit-record dict is dynamically typed (JSON)
        "multi_channel_delta_kuramoto": kuramoto_stats["mean_detection_rate"],  # type: ignore[index]  # audit-record dict is dynamically typed (JSON)
        "snr_weighted_delta_kuramoto": snr_kuramoto_stats["mean_detection_rate"],  # type: ignore[index]  # audit-record dict is dynamically typed (JSON)
        "adaptive_kuramoto": adaptive_stats["mean_detection_rate"],  # type: ignore[index]  # audit-record dict is dynamically typed (JSON)
    }
    best_variant = max(variant_means, key=variant_means.get)  # type: ignore[arg-type]  # audit-record dict is dynamically typed (JSON)

    adaptive_improves_kuramoto = (
        adaptive_stats["mean_detection_rate"]  # type: ignore[index]  # audit-record dict is dynamically typed (JSON)
        >= kuramoto_stats["mean_detection_rate"]  # type: ignore[index]  # audit-record dict is dynamically typed (JSON)
    )
    adaptive_wins_any = any(
        r["detectors"]["adaptive_kuramoto"]["detection_rate"]  # type: ignore[index]  # audit-record dict is dynamically typed (JSON)
        >= r["detectors"]["multi_channel_delta_kuramoto"]["detection_rate"]  # type: ignore[index]  # audit-record dict is dynamically typed (JSON)
        for r in records
    )

    if best_variant == "adaptive_kuramoto":
        return {
            "refine_kuramoto": True,
            "preferred_variant": "adaptive_kuramoto",
            "rationale": (
                "Adaptive quality-weighted Kuramoto has the highest mean detection "
                "rate on the panel, validating the channel-quality and robust-"
                "pooling refinement over the simple mean-R and SNR-weighted variants."
            ),
        }
    if adaptive_improves_kuramoto or adaptive_wins_any:
        return {
            "refine_kuramoto": True,
            "preferred_variant": "adaptive_kuramoto",
            "rationale": (
                "Adaptive Kuramoto improves over the simple mean-R variant, but it "
                "has not yet caught the envelope; further refinement (more channel "
                "configurations or multi-band fusion) is warranted."
            ),
        }
    return {
        "refine_kuramoto": False,
        "preferred_variant": "normalized_delta_envelope",
        "rationale": (
            "Adaptive Kuramoto does not improve over the simple mean-R Kuramoto "
            "detector on this panel; further investment in this exact "
            "spatial-R feature is not supported by the data."
        ),
    }


def _compute_aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    """Compute cross-recording aggregate statistics and recommendation."""
    return compute_aggregate(
        records=records,
        benchmark="cap_multichannel_n3_vs_wake",
        corpus="PhysioNet CAP Sleep Database",
        detector_names=[
            "normalized_delta_envelope",
            "multi_channel_delta_kuramoto",
            "snr_weighted_delta_kuramoto",
            "adaptive_kuramoto",
        ],
        recommendation_fn=_cap_recommendation,
    )


def main(
    edf_path: str | Path | None,
    txt_path: str | Path | None,
    output_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    captured_at: str = "2026-07-08T22:00:00Z",
) -> None:
    """Run the N3-vs-Wake audit in single-recording or batch mode.

    Parameters
    ----------
    edf_path, txt_path : str or Path, optional
        Paths for single-recording mode. Ignored when ``manifest_path`` is given.
    output_dir : str or Path
        Directory for sealed audit output.
    manifest_path : str or Path, optional
        Path to a CSV manifest for batch mode. Each row after the header must be
        ``recording_id,edf_path,txt_path``.
    captured_at : str
        Provenance timestamp for sealed records.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if manifest_path is not None:
        records: list[dict[str, object]] = []
        with Path(manifest_path).open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                recording_id, edf, txt = row[0].strip(), row[1].strip(), row[2].strip()
                fragment = _process_recording(
                    recording_id, Path(edf), Path(txt), out, captured_at
                )
                records.append(fragment)
                env_dr = fragment["detectors"]["normalized_delta_envelope"][  # type: ignore[index]  # audit-record dict is dynamically typed (JSON)
                    "detection_rate"
                ]
                kur_dr = fragment["detectors"]["multi_channel_delta_kuramoto"][  # type: ignore[index]  # audit-record dict is dynamically typed (JSON)
                    "detection_rate"
                ]
                snr_dr = fragment["detectors"]["snr_weighted_delta_kuramoto"][  # type: ignore[index]  # audit-record dict is dynamically typed (JSON)
                    "detection_rate"
                ]
                adapt_dr = fragment["detectors"]["adaptive_kuramoto"]["detection_rate"]
                print(
                    f"{recording_id}: envelope DR={env_dr:.3f}, "
                    f"kuramoto DR={kur_dr:.3f}, "
                    f"snr-weighted DR={snr_dr:.3f}, "
                    f"adaptive DR={adapt_dr:.3f}"
                )
        aggregate = _compute_aggregate(records)
        write_aggregate(out, aggregate)
        aggregate_path = out / "cap_multichannel_aggregate.json"
        print(f"Aggregate comparison written to {aggregate_path}")
        return

    if edf_path is None or txt_path is None:
        raise ValueError("either --manifest or both edf and txt paths are required")

    edf_path_p = Path(edf_path)
    recording_id = edf_path_p.stem
    fragment = _process_recording(
        recording_id, edf_path_p, Path(txt_path), out, captured_at
    )
    _write_single_recording_comparison(out, recording_id, fragment)
    print(
        f"CAP {recording_id} multi-channel N3 vs Wake audit:\n"
        f"  channels={fragment['channels']}\n"
        f"  n_n3={fragment['n_n3']}, n_wake={fragment['n_wake']}\n"
        f"  envelope: detection_rate={fragment['detectors']['normalized_delta_envelope']['detection_rate']:.3f}, "  # noqa: E501
        f"achieved_fa={fragment['detectors']['normalized_delta_envelope']['achieved_false_alarm']:.3f}, "  # noqa: E501
        f"p={fragment['detectors']['normalized_delta_envelope']['p_value']:.4f}\n"  # noqa: E501
        f"  kuramoto: detection_rate={fragment['detectors']['multi_channel_delta_kuramoto']['detection_rate']:.3f}, "  # noqa: E501
        f"achieved_fa={fragment['detectors']['multi_channel_delta_kuramoto']['achieved_false_alarm']:.3f}, "  # noqa: E501
        f"p={fragment['detectors']['multi_channel_delta_kuramoto']['p_value']:.4f}\n"  # noqa: E501
        f"  snr-weighted: detection_rate={fragment['detectors']['snr_weighted_delta_kuramoto']['detection_rate']:.3f}, "  # noqa: E501
        f"achieved_fa={fragment['detectors']['snr_weighted_delta_kuramoto']['achieved_false_alarm']:.3f}, "  # noqa: E501
        f"p={fragment['detectors']['snr_weighted_delta_kuramoto']['p_value']:.4f}\n"  # noqa: E501
        f"  adaptive: detection_rate={fragment['detectors']['adaptive_kuramoto']['detection_rate']:.3f}, "  # noqa: E501
        f"achieved_fa={fragment['detectors']['adaptive_kuramoto']['achieved_false_alarm']:.3f}, "  # noqa: E501
        f"p={fragment['detectors']['adaptive_kuramoto']['p_value']:.4f}"  # noqa: E501
    )
    print(f"Sealed audits and comparison written to {out}")


if __name__ == "__main__":  # pragma: no cover - CLI shell over tested logic
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        help="CSV manifest for batch mode: recording_id,edf_path,txt_path",
    )
    parser.add_argument(
        "edf",
        nargs="?",
        help="CAP recording EDF (e.g. n1.edf); required in single-recording mode",
    )
    parser.add_argument(
        "txt",
        nargs="?",
        help=(
            "REMlogic text annotation file (e.g. n1.txt); "
            "required in single-recording mode"
        ),
    )
    parser.add_argument("output_dir", help="Directory for sealed audit output")
    args = parser.parse_args()
    main(args.edf, args.txt, args.output_dir, manifest_path=args.manifest)
