# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CHB-MIT multi-channel Kuramoto detector audit

"""Honest audit of multi-channel Kuramoto detectors on CHB-MIT chb01 seizures.

Loads the public CHB-MIT Scalp EEG recordings for subject ``chb01`` from a local
``DATA_DIR`` (the raw EDFs are citation-only and are not redistributed),
extracts a 5-minute pre-ictal window before each annotated seizure, and pits it
against seizure-free interictal recordings. Two detectors are compared:

1. **mean_kuramoto** — unweighted mean phase coherence across all 23 bipolar
   channels (the textbook Kuramoto order parameter).
2. **adaptive_kuramoto** — quality-weighted phase coherence where each channel's
   contribution is driven by band-limited SNR penalised by excess kurtosis,
   pooled robustly with the median.

Both detectors are run in two bands: delta (0.5–4 Hz) and the seizure dynamics
band (4–30 Hz). Every detector is audited at a matched 10 % false-alarm rate,
with a label-permutation significance test (10 000 relabellings, seed 42), and
the AUC is reported as a second separation metric.

Usage:
    python bench/chbmit_multichannel_kuramoto.py \
        data/chb01_seizures \
        examples/real_data/chb01_seizures_multichannel_kuramoto
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the repository root and ``src/`` discoverable whether the script is run
# directly, via ``python -m bench...``, or through ``PYTHONPATH``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.signal import butter, filtfilt, hilbert  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from bench.honest_dataset_audit import (  # noqa: E402
    AuditConfig,
    file_sha256,
    run_audit,
    write_audit_files,
)
from scpn_phase_orchestrator.monitor.adaptive_kuramoto import (  # noqa: E402
    compute_adaptive_kuramoto_scores,
)

FloatArray = NDArray[np.float64]

#: CHB-MIT summary file URL (seizure annotations are fetched, not redistributed).
PHYSIONET_BASE = "https://physionet.org/files/chbmit/1.0.0"
SUBJECT = "chb01"
SUMMARY_URL = f"{PHYSIONET_BASE}/{SUBJECT}/{SUBJECT}-summary.txt"

#: Analysis parameters.
EPOCH_SECONDS = 30.0
PREICTAL_SECONDS = 300.0
FILTER_ORDER = 3
#: Common analysis rate after anti-alias resampling. 64 Hz preserves the delta
#: (0.5–4 Hz) and seizure dynamics (4–30 Hz) bands and keeps Hilbert phase
#: extraction fast; the resampler's anti-alias filter removes energy above 32 Hz.
TARGET_SAMPLING_RATE_HZ = 64.0
TARGET_FALSE_ALARM = 0.10
N_PERMUTATIONS = 10_000
PERMUTATION_SEED = 42
ALPHA = 0.05
SCORE_PRECISION = 6

#: Seizure-free recordings used as the null class.
INTERICTAL_FILES = [
    "chb01_01.edf",
    "chb01_02.edf",
    "chb01_05.edf",
    "chb01_06.edf",
    "chb01_07.edf",
]

#: Frequency bands under comparison.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "seizure": (4.0, 30.0),
}


def _parse_summary() -> dict[str, tuple[int, int]]:
    """Return ``{edf_filename: (seizure_start_s, seizure_end_s)}`` from PhysioNet."""
    text = urllib.request.urlopen(SUMMARY_URL, timeout=30).read().decode()  # noqa: S310
    seizures: dict[str, tuple[int, int]] = {}
    current_file: str | None = None
    start: int | None = None
    for line in text.splitlines():
        m = re.match(r"File Name:\s+(\S+)", line)
        if m:
            current_file = m.group(1)
            continue
        m = re.match(r"Seizure\s+\d*\s*Start Time:\s+(\d+)", line)
        if not m:
            m = re.match(r"Seizure Start Time:\s+(\d+)", line)
        if m and current_file:
            start = int(m.group(1))
            continue
        m = re.match(r"Seizure\s+\d*\s*End Time:\s+(\d+)", line)
        if not m:
            m = re.match(r"Seizure End Time:\s+(\d+)", line)
        if m and current_file and start is not None:
            seizures[current_file] = (start, int(m.group(1)))
            start = None
    return seizures


def _load_edf(path: Path) -> tuple[FloatArray, float, list[str]]:
    """Load all channels from an EDF and return (data, fs, labels).

    All channels are assumed to share a common sampling rate. The returned
    ``data`` array has shape ``(n_channels, n_samples)``.
    """
    import pyedflib

    reader = pyedflib.EdfReader(str(path))
    try:
        labels = reader.getSignalLabels()
        n_channels = reader.signals_in_file
        fs_values = {reader.getSampleFrequency(i) for i in range(n_channels)}
        if len(fs_values) != 1:
            raise ValueError(f"heterogeneous sampling rates in {path}: {fs_values}")
        fs = float(next(iter(fs_values)))
        signals = [
            np.asarray(reader.readSignal(i), dtype=np.float64)
            for i in range(n_channels)
        ]
        min_len = min(len(s) for s in signals)
        data = np.stack([s[:min_len] for s in signals])
    finally:
        reader.close()
    return data, fs, labels


def _bandpass(sig: FloatArray, fs: float, lo: float, hi: float) -> FloatArray:
    """Zero-phase Butterworth band-pass filter."""
    nyq = fs / 2.0
    b, a = butter(FILTER_ORDER, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def _decimate_to_rate(
    data: FloatArray,
    source_fs: float,
    target_fs: float = TARGET_SAMPLING_RATE_HZ,
) -> tuple[FloatArray, float]:
    """Anti-alias resample ``data`` along the time axis to ``target_fs``.

    The CHB-MIT recordings are sampled at 256 Hz; resampling to 64 Hz before
    band-pass filtering and Hilbert phase extraction cuts run time by an order
    of magnitude while preserving both the delta and seizure dynamics bands.
    """
    if source_fs == target_fs:
        return data, float(source_fs)
    q = int(round(source_fs / target_fs))
    if abs(source_fs / q - target_fs) > 1e-6:
        raise ValueError(
            f"cannot resample {source_fs} Hz to {target_fs} Hz by integer factor"
        )
    from scipy.signal import resample_poly

    return resample_poly(data, 1, q, axis=-1), float(target_fs)


def _mean_kuramoto_scores(
    data: FloatArray,
    fs: float,
    band_hz: tuple[float, float],
    epoch_seconds: float = EPOCH_SECONDS,
) -> FloatArray:
    """Return per-epoch unweighted mean-R Kuramoto scores.

    For each channel the band-passed Hilbert phase is extracted. The instantaneous
    Kuramoto order parameter ``R(t)`` is the magnitude of the mean complex
    exponential across channels; the epoch score is the mean of ``R(t)`` over the
    epoch.
    """
    n_channels, n_samples = data.shape
    epoch_len = int(epoch_seconds * fs)
    n_epochs = n_samples // epoch_len
    if n_epochs == 0:
        raise ValueError("signal shorter than one epoch")

    trimmed = data[:, : n_epochs * epoch_len]
    filtered = _bandpass(trimmed, fs, band_hz[0], band_hz[1])
    phases = np.angle(hilbert(filtered, axis=1))

    r_t = np.abs(np.exp(1j * phases).mean(axis=0))
    scores = r_t.reshape(n_epochs, epoch_len).mean(axis=1)
    return np.round(scores, SCORE_PRECISION)


def _adaptive_kuramoto_scores(
    data: FloatArray,
    fs: float,
    band_hz: tuple[float, float],
    epoch_seconds: float = EPOCH_SECONDS,
    weight_mode: str = "snr_kurtosis",
) -> FloatArray:
    """Return per-epoch adaptive quality-weighted Kuramoto scores."""
    scores, _ = compute_adaptive_kuramoto_scores(
        data,
        fs,
        band_hz=band_hz,
        epoch_seconds=epoch_seconds,
        weight_mode=weight_mode,
        score_precision=SCORE_PRECISION,
    )
    return scores


def _plv_kuramoto_scores(
    data: FloatArray,
    fs: float,
    band_hz: tuple[float, float],
    epoch_seconds: float = EPOCH_SECONDS,
) -> FloatArray:
    """Return per-epoch PLV-to-mean-field weighted Kuramoto scores."""
    return _adaptive_kuramoto_scores(
        data, fs, band_hz, epoch_seconds, weight_mode="plv_mean_field"
    )


def _preictal_epoch_indices(
    n_epochs: int,
    onset_s: float,
    epoch_seconds: float = EPOCH_SECONDS,
    preictal_seconds: float = PREICTAL_SECONDS,
) -> list[int]:
    """Return indices of epochs that fully lie in the pre-ictal window."""
    indices = []
    for e in range(n_epochs):
        epoch_start = e * epoch_seconds
        epoch_end = (e + 1) * epoch_seconds
        if epoch_start >= onset_s - preictal_seconds and epoch_end <= onset_s:
            indices.append(e)
    return indices


def _compute_auc(event_scores: FloatArray, null_scores: FloatArray) -> float:
    """Mann-Whitney / ROC AUC for event scores larger than null scores."""
    n1 = len(event_scores)
    n2 = len(null_scores)
    if n1 == 0 or n2 == 0:
        return float("nan")
    combined = np.concatenate([event_scores, null_scores])
    labels = np.concatenate([np.ones(n1, dtype=int), np.zeros(n2, dtype=int)])
    ranks = rankdata(combined)
    u_stat = ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0
    return float(u_stat / (n1 * n2))


def _detector_name(kind: str, band: str) -> str:
    """Canonical detector identifier, e.g. ``mean_kuramoto_delta``."""
    return f"{kind}_kuramoto_{band}"


def _compute_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build cross-seizure aggregate statistics and a data-driven recommendation."""
    detector_names = list(records[0]["detectors"].keys())
    stats: dict[str, Any] = {}
    for det in detector_names:
        drs = np.asarray([r["detectors"][det]["detection_rate"] for r in records])
        fas = np.asarray([r["detectors"][det]["achieved_false_alarm"] for r in records])
        pvals = np.asarray([r["detectors"][det]["p_value"] for r in records])
        aucs = np.asarray([r["detectors"][det]["auc"] for r in records])
        beats = np.asarray([r["detectors"][det]["beats_chance"] for r in records])
        stats[det] = {
            "mean_detection_rate": round(float(drs.mean()), 6),
            "std_detection_rate": round(float(drs.std(ddof=0)), 6),
            "mean_achieved_false_alarm": round(float(fas.mean()), 6),
            "geometric_mean_p_value": round(
                float(np.exp(np.mean(np.log(np.maximum(pvals, 1e-300))))), 6
            ),
            "mean_auc": round(float(aucs.mean()), 6),
            "fraction_beats_chance": round(float(beats.mean()), 6),
        }

    best = max(stats, key=lambda d: stats[d]["mean_detection_rate"])
    best_stats = stats[best]
    any_beats = any(r["detectors"][best]["beats_chance"] for r in records)

    if not any_beats:
        recommendation = {
            "refine_kuramoto": False,
            "preferred_variant": best,
            "rationale": (
                f"{best} has the highest mean detection rate "
                f"({best_stats['mean_detection_rate']:.3f}) on this panel, but it does "
                "not beat chance on any seizure; further refinement of this exact "
                "multi-channel Kuramoto feature is not supported by the data."
            ),
        }
    else:
        recommendation = {
            "refine_kuramoto": True,
            "preferred_variant": best,
            "rationale": (
                f"{best} has the highest mean detection rate "
                f"({best_stats['mean_detection_rate']:.3f}) and mean AUC "
                f"({best_stats['mean_auc']:.3f}) on this panel; it is the preferred "
                "multi-channel Kuramoto variant for further refinement."
            ),
        }

    return {
        "benchmark": "chbmit_multichannel_kuramoto",
        "corpus": "PhysioNet CHB-MIT Scalp EEG Database, subject chb01",
        "target_false_alarm": TARGET_FALSE_ALARM,
        "n_seizures": len(records),
        "recording_ids": [r["recording_id"] for r in records],
        "epoch_seconds": EPOCH_SECONDS,
        "preictal_seconds": PREICTAL_SECONDS,
        "bands": {k: list(v) for k, v in BANDS.items()},
        "interictal_files": INTERICTAL_FILES,
        "per_seizure": records,
        **stats,
        "recommendation": recommendation,
    }


def main(data_dir: Path, output_dir: Path) -> None:
    """Run the full multi-channel Kuramoto audit and write sealed artefacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching seizure annotations from PhysioNet...")
    annotations = _parse_summary()
    available_edfs = {p.name for p in data_dir.glob("*.edf")}
    seizures = {
        fname: (start, end)
        for fname, (start, end) in annotations.items()
        if fname in available_edfs and start > PREICTAL_SECONDS
    }
    print(f"  {len(seizures)} seizures with clean pre-ictal windows")

    config = AuditConfig(
        target_false_alarm=TARGET_FALSE_ALARM,
        n_permutations=N_PERMUTATIONS,
        seed=PERMUTATION_SEED,
        alpha=ALPHA,
        score_precision=SCORE_PRECISION,
        captured_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    # ------------------------------------------------------------------ #
    # Load interictal null recordings and score them with every detector.  #
    # ------------------------------------------------------------------ #
    print("Scoring interictal null recordings...")
    null_score_batches: dict[str, list[FloatArray]] = {
        _detector_name(kind, band): []
        for band in BANDS
        for kind in ("mean", "adaptive")
    }
    null_score_batches["plv_kuramoto_seizure"] = []
    for fname in INTERICTAL_FILES:
        path = data_dir / fname
        if not path.exists():
            print(f"  skipping missing interictal file {fname}")
            continue
        data, fs, labels = _load_edf(path)
        data, fs = _decimate_to_rate(data, fs)
        for band_name, band_hz in BANDS.items():
            null_score_batches[_detector_name("mean", band_name)].append(
                _mean_kuramoto_scores(data, fs, band_hz, EPOCH_SECONDS)
            )
            null_score_batches[_detector_name("adaptive", band_name)].append(
                _adaptive_kuramoto_scores(data, fs, band_hz, EPOCH_SECONDS)
            )
        null_score_batches["plv_kuramoto_seizure"].append(
            _plv_kuramoto_scores(data, fs, BANDS["seizure"], EPOCH_SECONDS)
        )
        print(
            f"  {fname}: {data.shape[0]} channels, {data.shape[1] / fs:.0f}s "
            f"at {fs:.0f} Hz"
        )

    null_scores = {
        name: np.concatenate(batches) if batches else np.array([], dtype=np.float64)
        for name, batches in null_score_batches.items()
    }
    for name, scores in null_scores.items():
        print(f"  {name}: {len(scores)} null epochs")

    # ------------------------------------------------------------------ #
    # Audit each seizure's pre-ictal epochs against the pooled null.       #
    # ------------------------------------------------------------------ #
    records: list[dict[str, Any]] = []
    for fname in sorted(seizures):
        onset_s, end_s = seizures[fname]
        path = data_dir / fname
        print(f"\nProcessing {fname} (seizure {onset_s}-{end_s}s)...")
        data, fs, labels = _load_edf(path)
        data, fs = _decimate_to_rate(data, fs)
        n_epochs = data.shape[1] // int(EPOCH_SECONDS * fs)
        event_indices = _preictal_epoch_indices(n_epochs, onset_s)
        print(
            f"  {data.shape[0]} channels, {data.shape[1] / fs:.0f}s at {fs:.0f} Hz, "
            f"{len(event_indices)} pre-ictal epochs"
        )
        if not event_indices:
            print("  no usable pre-ictal epochs; skipping")
            continue

        rec_out = output_dir / Path(fname).stem
        rec_out.mkdir(parents=True, exist_ok=True)
        per_detector: dict[str, Any] = {}

        for band_name, band_hz in BANDS.items():
            for kind, scorer in (
                ("mean", _mean_kuramoto_scores),
                ("adaptive", _adaptive_kuramoto_scores),
            ):
                det_name = _detector_name(kind, band_name)
                full_scores = scorer(data, fs, band_hz, EPOCH_SECONDS)
                event_scores = full_scores[event_indices]
                combined_scores = np.concatenate(
                    [event_scores, null_scores[det_name]]
                )
                combined_labels = ["preictal"] * len(event_scores) + [
                    "interictal"
                ] * len(null_scores[det_name])
                audit_record, summary = run_audit(
                    scores=combined_scores,
                    labels=combined_labels,
                    detector_name=det_name,
                    corpus_id=f"chbmit-{Path(fname).stem}-preictal-vs-interictal",
                    config=config,
                    event_label="preictal",
                    null_label="interictal",
                )
                summary["auc"] = round(
                    _compute_auc(event_scores, null_scores[det_name]), 6
                )
                summary["preictal_score_mean"] = round(float(event_scores.mean()), 6)
                summary["interictal_score_mean"] = round(
                    float(null_scores[det_name].mean()), 6
                )
                per_detector[det_name] = summary
                write_audit_files(
                    output_dir=rec_out,
                    prefix=Path(fname).stem,
                    detector_name=det_name,
                    audit_record=audit_record,
                    summary=summary,
                )

        # Extra PLV-to-mean-field weighted variant for the seizure band.
        det_name = "plv_kuramoto_seizure"
        full_scores = _plv_kuramoto_scores(
            data, fs, BANDS["seizure"], EPOCH_SECONDS
        )
        event_scores = full_scores[event_indices]
        combined_scores = np.concatenate([event_scores, null_scores[det_name]])
        combined_labels = ["preictal"] * len(event_scores) + [
            "interictal"
        ] * len(null_scores[det_name])
        audit_record, summary = run_audit(
            scores=combined_scores,
            labels=combined_labels,
            detector_name=det_name,
            corpus_id=f"chbmit-{Path(fname).stem}-preictal-vs-interictal",
            config=config,
            event_label="preictal",
            null_label="interictal",
        )
        summary["auc"] = round(
            _compute_auc(event_scores, null_scores[det_name]), 6
        )
        summary["preictal_score_mean"] = round(float(event_scores.mean()), 6)
        summary["interictal_score_mean"] = round(
            float(null_scores[det_name].mean()), 6
        )
        per_detector[det_name] = summary
        write_audit_files(
            output_dir=rec_out,
            prefix=Path(fname).stem,
            detector_name=det_name,
            audit_record=audit_record,
            summary=summary,
        )

        records.append(
            {
                "recording_id": Path(fname).stem,
                "seizure_onset_s": onset_s,
                "seizure_end_s": end_s,
                "n_channels": data.shape[0],
                "sampling_rate_hz": fs,
                "n_preictal_epochs": len(event_indices),
                "edf_sha256": file_sha256(path),
                "detectors": per_detector,
            }
        )

    # ------------------------------------------------------------------ #
    # Aggregate and write.                                                 #
    # ------------------------------------------------------------------ #
    if not records:
        raise RuntimeError("no seizures produced usable pre-ictal epochs")

    aggregate = _compute_aggregate(records)
    aggregate_path = output_dir / "chbmit_multichannel_kuramoto.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("CHB-MIT chb01 multi-channel Kuramoto audit")
    print(f"{'=' * 60}")
    print(f"Seizures audited: {aggregate['n_seizures']}")
    detector_names = list(records[0]["detectors"].keys())
    for det in detector_names:
        stats = aggregate[det]
        print(
            f"{det:32s}  DR={stats['mean_detection_rate']:.3f} "
            f"(±{stats['std_detection_rate']:.3f})  "
            f"AUC={stats['mean_auc']:.3f}  "
            f"p_geo={stats['geometric_mean_p_value']:.4f}  "
            f"beats={stats['fraction_beats_chance']:.2f}"
        )
    print(f"\nAggregate written to {aggregate_path}")


if __name__ == "__main__":  # pragma: no cover - CLI shell over tested logic
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/chb01_seizures",
        help="Directory containing the chb01 EDF files",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="examples/real_data/chb01_seizures_multichannel_kuramoto",
        help="Directory for sealed audit output",
    )
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.output_dir))
