# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CHB-MIT cross-subject Kuramoto generalisation audit

"""Leave-one-subject-out generalisation audit of multi-channel Kuramoto detectors.

The single-subject CHB-MIT chb01 study found that a global top-k PLV Kuramoto
detector beats the unweighted mean-R baseline. That result tuned ``k`` on the
same subject it was scored on. This audit asks the honest question instead:
**does the detector generalise across subjects?**

For each held-out subject S:

1. calibrate ``k`` ONLY on the training subjects (the k that maximises mean
   training AUC of top-k PLV) — S's own data never influences its own ``k``;
2. score top-k PLV (with that calibrated ``k``) and the mean-R baseline on S's
   pre-ictal windows versus S's interictal null epochs;
3. report the out-of-sample AUC and detection rate for both detectors.

Raw EDFs are citation-only (PhysioNet CHB-MIT 1.0.0) and are read from a local
``DATA_DIR``; they are never redistributed. Only the derived sealed comparison
JSON is committed.

Usage:
    python bench/chbmit_crosssubject_validation.py \
        /path/to/chbmit_data \
        examples/real_data/chbmit_crosssubject_kuramoto \
        chb01 chb02 chb03 chb04 chb05
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, hilbert, resample_poly
from scipy.stats import rankdata

FloatArray = NDArray[np.float64]

BAND_HZ = (4.0, 30.0)
TARGET_SAMPLING_RATE_HZ = 64.0
EPOCH_SECONDS = 30.0
PREICTAL_SECONDS = 300.0
FILTER_ORDER = 3
TARGET_FALSE_ALARM = 0.10
N_INTERICTAL = 5
#: Candidate channel counts for the global top-k PLV selection.
K_GRID = (10, 13, 15, 17, 20, 23)
SCORE_PRECISION = 6


def parse_summary(data_dir: Path, subject: str) -> tuple[dict[str, int], list[str]]:
    """Return ``({seizure_stem: onset_s}, [interictal_stems])`` for ``subject``.

    Only seizures whose onset leaves a full pre-ictal window (onset greater than
    ``PREICTAL_SECONDS``) are kept. Seizure-free files become interictal nulls.
    """
    text = (data_dir / f"{subject}-summary.txt").read_text(encoding="utf-8")
    seizures: dict[str, int] = {}
    interictal: list[str] = []
    current: str | None = None
    n_seiz = 0
    for line in text.splitlines():
        m = re.match(r"File Name:\s+(\S+\.edf)", line)
        if m:
            if current and n_seiz == 0:
                interictal.append(current[:-4])
            current, n_seiz = m.group(1), 0
            continue
        m = re.match(r"Number of Seizures in File:\s+(\d+)", line)
        if m:
            n_seiz = int(m.group(1))
            continue
        m = re.match(r"Seizure(?:\s+\d+)?\s+Start Time:\s+(\d+)", line)
        if m and current and int(m.group(1)) > PREICTAL_SECONDS:
            seizures[current[:-4]] = int(m.group(1))
    if current and n_seiz == 0:
        interictal.append(current[:-4])
    return seizures, interictal


def load_edf(path: Path) -> FloatArray:
    """Load all channels from an EDF, resampled to ``TARGET_SAMPLING_RATE_HZ``."""
    import pyedflib

    reader = pyedflib.EdfReader(str(path))
    try:
        n = reader.signals_in_file
        fs = reader.getSampleFrequency(0)
        signals = [np.asarray(reader.readSignal(i), dtype=np.float64) for i in range(n)]
        min_len = min(len(s) for s in signals)
        data = np.stack([s[:min_len] for s in signals])
    finally:
        reader.close()
    q = int(round(fs / TARGET_SAMPLING_RATE_HZ))
    return resample_poly(data, 1, q, axis=-1)


def _epoch_phases(data: FloatArray) -> tuple[FloatArray, int]:
    """Return band-passed Hilbert phases, shape ``(channels, epochs, samples)``."""
    epoch_len = int(EPOCH_SECONDS * TARGET_SAMPLING_RATE_HZ)
    n_epochs = data.shape[1] // epoch_len
    nyq = TARGET_SAMPLING_RATE_HZ / 2.0
    b, a = butter(FILTER_ORDER, [BAND_HZ[0] / nyq, BAND_HZ[1] / nyq], btype="band")
    filtered = filtfilt(b, a, data[:, : n_epochs * epoch_len], axis=-1)
    phases = np.angle(hilbert(filtered, axis=1))
    return phases.reshape(data.shape[0], n_epochs, epoch_len), n_epochs


def mean_r_scores(phase_epochs: FloatArray) -> FloatArray:
    """Per-epoch unweighted mean-R Kuramoto score."""
    return np.abs(np.exp(1j * phase_epochs).mean(axis=0)).mean(axis=1)


def topk_plv_scores(phase_epochs: FloatArray, k: int) -> FloatArray:
    """Per-epoch global top-k PLV: mean-R over the k best mean-field-locked channels."""
    mean_field = np.angle(np.exp(1j * phase_epochs).mean(axis=0))
    plv = np.abs(np.exp(1j * (phase_epochs - mean_field[np.newaxis])).mean(axis=2))
    k = min(k, phase_epochs.shape[0])
    selected = np.argsort(plv.mean(axis=1))[::-1][:k]
    return np.abs(np.exp(1j * phase_epochs[selected]).mean(axis=0)).mean(axis=1)


def auc(event: FloatArray, null: FloatArray) -> float:
    """Mann-Whitney / ROC AUC that ``event`` scores exceed ``null`` scores."""
    if len(event) == 0 or len(null) == 0:
        return float("nan")
    combined = np.concatenate([event, null])
    labels = np.concatenate([np.ones(len(event)), np.zeros(len(null))])
    ranks = rankdata(combined)
    u = ranks[labels == 1].sum() - len(event) * (len(event) + 1) / 2.0
    return float(u / (len(event) * len(null)))


def detection_rate(event: FloatArray, null: FloatArray) -> float:
    """Fraction of event epochs above the null threshold matched to 10% false alarm."""
    if len(event) == 0 or len(null) == 0:
        return float("nan")
    threshold = float(np.quantile(null, 1.0 - TARGET_FALSE_ALARM))
    return float(np.mean(event > threshold))


def subject_scores(
    data_dir: Path, subject: str
) -> dict[str, tuple[FloatArray, FloatArray]]:
    """Return ``{detector: (event_scores, null_scores)}`` for one subject."""
    seizures, interictal = parse_summary(data_dir, subject)
    nulls = interictal[:N_INTERICTAL]
    detectors = ["mean_r", *(f"topk{k}" for k in K_GRID)]
    event: dict[str, list[FloatArray]] = {d: [] for d in detectors}
    null: dict[str, list[FloatArray]] = {d: [] for d in detectors}

    for stem in nulls:
        path = data_dir / f"{stem}.edf"
        if not path.exists():
            continue
        ph, _ = _epoch_phases(load_edf(path))
        null["mean_r"].append(mean_r_scores(ph))
        for k in K_GRID:
            null[f"topk{k}"].append(topk_plv_scores(ph, k))

    for stem, onset in seizures.items():
        path = data_dir / f"{stem}.edf"
        if not path.exists():
            continue
        ph, n_epochs = _epoch_phases(load_edf(path))
        idx = [
            e
            for e in range(n_epochs)
            if e * EPOCH_SECONDS >= onset - PREICTAL_SECONDS
            and (e + 1) * EPOCH_SECONDS <= onset
        ]
        if not idx:
            continue
        event["mean_r"].append(mean_r_scores(ph)[idx])
        for k in K_GRID:
            event[f"topk{k}"].append(topk_plv_scores(ph, k)[idx])

    return {
        d: (
            np.concatenate(event[d]) if event[d] else np.array([], dtype=np.float64),
            np.concatenate(null[d]) if null[d] else np.array([], dtype=np.float64),
        )
        for d in detectors
    }


def run(data_dir: Path, subjects: list[str]) -> dict[str, Any]:
    """Leave-one-subject-out cross-subject validation. Returns the sealed record."""
    scored = {}
    for subject in subjects:
        scored[subject] = subject_scores(data_dir, subject)

    per_subject = []
    for held in subjects:
        train = [s for s in subjects if s != held]
        best_k, best_train_auc = K_GRID[-1], -1.0
        for k in K_GRID:
            train_aucs = [auc(*scored[s][f"topk{k}"]) for s in train]
            mean_auc = float(np.nanmean(train_aucs))
            if mean_auc > best_train_auc:
                best_train_auc, best_k = mean_auc, k
        ev_t, nu_t = scored[held][f"topk{best_k}"]
        ev_m, nu_m = scored[held]["mean_r"]
        per_subject.append(
            {
                "subject": held,
                "calibrated_k": best_k,
                "n_preictal_epochs": int(len(ev_m)),
                "topk_plv": {
                    "auc": round(auc(ev_t, nu_t), SCORE_PRECISION),
                    "detection_rate": round(
                        detection_rate(ev_t, nu_t), SCORE_PRECISION
                    ),
                },
                "mean_r": {
                    "auc": round(auc(ev_m, nu_m), SCORE_PRECISION),
                    "detection_rate": round(
                        detection_rate(ev_m, nu_m), SCORE_PRECISION
                    ),
                },
            }
        )

    topk_aucs = np.array([r["topk_plv"]["auc"] for r in per_subject])
    mean_aucs = np.array([r["mean_r"]["auc"] for r in per_subject])
    wins = int(np.sum(topk_aucs > mean_aucs))
    return {
        "benchmark": "chbmit_crosssubject_kuramoto",
        "corpus": "PhysioNet CHB-MIT Scalp EEG Database",
        "question": "does top-k PLV Kuramoto generalise across subjects (LOSO)?",
        "subjects": subjects,
        "band_hz": list(BAND_HZ),
        "epoch_seconds": EPOCH_SECONDS,
        "preictal_seconds": PREICTAL_SECONDS,
        "target_false_alarm": TARGET_FALSE_ALARM,
        "k_grid": list(K_GRID),
        "per_subject": per_subject,
        "topk_plv_beats_mean_r_subjects": wins,
        "n_subjects": len(per_subject),
        "mean_auc_topk_plv": round(float(np.nanmean(topk_aucs)), SCORE_PRECISION),
        "mean_auc_mean_r": round(float(np.nanmean(mean_aucs)), SCORE_PRECISION),
    }


def main(data_dir: Path, output_dir: Path, subjects: list[str]) -> None:
    """Run the audit and write the sealed cross-subject comparison JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    record = run(data_dir, subjects)
    out = output_dir / "chbmit_crosssubject_kuramoto.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"{'=' * 60}")
    print("CHB-MIT cross-subject Kuramoto generalisation (leave-one-subject-out)")
    print(f"{'=' * 60}")
    for r in record["per_subject"]:
        print(
            f"held={r['subject']:6s} k*={r['calibrated_k']:2d}  "
            f"top-k PLV AUC={r['topk_plv']['auc']:.3f}  "
            f"mean-R AUC={r['mean_r']['auc']:.3f}"
        )
    print(
        f"\nOut-of-sample: top-k PLV beats mean-R on "
        f"{record['topk_plv_beats_mean_r_subjects']}/{record['n_subjects']} subjects; "
        f"mean AUC top-k={record['mean_auc_topk_plv']:.3f} "
        f"vs mean-R={record['mean_auc_mean_r']:.3f}"
    )
    print(f"\nSealed to {out}")


if __name__ == "__main__":  # pragma: no cover - CLI shell over tested logic
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir", type=Path, help="Directory with chbNN-summary.txt + EDFs"
    )
    parser.add_argument("output_dir", type=Path, help="Directory for the sealed JSON")
    parser.add_argument("subjects", nargs="*", default=["chb01"], help="Subject IDs")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.subjects or ["chb01"])
