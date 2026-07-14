# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Top-k PLV channel-selection diagnostic

"""Fast diagnostic: does selecting top-k channels by PLV beat mean-R on chb01?

Loads each EDF once, caches decimated seizure-band phases and PLV weights, then
evaluates mean-R plus top-k variants (per-epoch and global selection) against the
same pooled interictal null recordings used by bench/chbmit_multichannel_kuramoto.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.signal import butter, filtfilt, hilbert  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from bench.chbmit_multichannel_kuramoto import (  # noqa: E402
    BANDS,
    EPOCH_SECONDS,
    INTERICTAL_FILES,
    PREICTAL_SECONDS,
    TARGET_SAMPLING_RATE_HZ,
    _decimate_to_rate,
    _load_edf,
    _parse_summary,
    _preictal_epoch_indices,
)
from scpn_phase_orchestrator.monitor.adaptive_kuramoto import (  # noqa: E402
    compute_phase_locking_weights,
)

FloatArray = NDArray[np.float64]

FILTER_ORDER = 3
BAND = BANDS["seizure"]
K_VALUES = [15, 20, 23]


def _bandpass(sig: FloatArray, fs: float, lo: float, hi: float) -> FloatArray:
    nyq = fs / 2.0
    b, a = butter(FILTER_ORDER, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def _phases_and_weights(
    data: FloatArray, fs: float, epoch_seconds: float
) -> tuple[FloatArray, FloatArray]:
    """Return seizure-band phases and per-epoch PLV-to-mean-field weights."""
    filtered = _bandpass(data, fs, BAND[0], BAND[1])
    phases = np.angle(hilbert(filtered, axis=1))
    weights = compute_phase_locking_weights(phases, fs, epoch_seconds=epoch_seconds)
    return phases, weights


def _mean_r_per_epoch(
    phases: FloatArray, fs: float, epoch_seconds: float
) -> FloatArray:
    epoch_len = int(epoch_seconds * fs)
    n_epochs = phases.shape[1] // epoch_len
    epochs = phases[:, : n_epochs * epoch_len].reshape(
        phases.shape[0], n_epochs, epoch_len
    )
    r_t = np.abs(np.exp(1j * epochs).mean(axis=0))
    return r_t.mean(axis=1)


def _topk_per_epoch_all_k(
    phases: FloatArray,
    weights: FloatArray,
    fs: float,
    epoch_seconds: float,
    k_values: list[int],
) -> dict[int, FloatArray]:
    """Return per-epoch top-k mean-R scores for every k in ``k_values``.

    Channels are selected independently for each epoch by PLV to the mean field.
    All k values are computed in one vectorised pass.
    """
    epoch_len = int(epoch_seconds * fs)
    n_channels, n_epochs_total = phases.shape
    n_epochs = n_epochs_total // epoch_len
    epochs = phases[:, : n_epochs * epoch_len].reshape(
        n_channels, n_epochs, epoch_len
    )
    exp_epochs = np.exp(1j * epochs)  # (C, E, L)

    # Sort channels by weight per epoch (ascending).
    sorted_idx = np.argsort(weights, axis=0)  # (C, E)
    epoch_idx = np.arange(n_epochs)[np.newaxis, :]  # (1, E)
    sorted_exp = exp_epochs[sorted_idx, epoch_idx, :]  # (C, E, L)
    cumsum = np.cumsum(sorted_exp, axis=0)  # (C, E, L)

    scores: dict[int, FloatArray] = {}
    for k in k_values:
        if k >= n_channels:
            top_sum = cumsum[-1]
        else:
            top_sum = cumsum[-1] - cumsum[n_channels - k - 1]
        r_t = np.abs(top_sum / k)
        scores[k] = r_t.mean(axis=1)
    return scores


def _topk_global_all_k(
    phases: FloatArray,
    weights: FloatArray,
    fs: float,
    epoch_seconds: float,
    k_values: list[int],
) -> dict[int, FloatArray]:
    """Return top-k mean-R scores for channels selected globally by mean PLV."""
    mean_weight = weights.mean(axis=1)
    sorted_idx = np.argsort(mean_weight)
    scores: dict[int, FloatArray] = {}
    for k in k_values:
        top_idx = sorted_idx[-k:]
        scores[k] = _mean_r_per_epoch(phases[top_idx], fs, epoch_seconds)
    return scores


def _compute_auc(event_scores: FloatArray, null_scores: FloatArray) -> float:
    n1, n2 = len(event_scores), len(null_scores)
    if n1 == 0 or n2 == 0:
        return float("nan")
    combined = np.concatenate([event_scores, null_scores])
    labels = np.concatenate([np.ones(n1), np.zeros(n2)])
    ranks = rankdata(combined)
    u_stat = ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0
    return float(u_stat / (n1 * n2))


def _dr_at_fa(event_scores: FloatArray, null_scores: FloatArray, fa: float) -> float:
    threshold = np.quantile(null_scores, 1.0 - fa)
    return float(np.mean(event_scores >= threshold))


def main() -> None:
    data_dir = Path("data/chb01_seizures")
    annotations = _parse_summary()
    available = {p.name for p in data_dir.glob("*.edf")}
    seizures = {
        fname: (start, end)
        for fname, (start, end) in annotations.items()
        if fname in available and start > PREICTAL_SECONDS
    }

    print("Loading interictal recordings and computing PLV weights...")
    null_phases: list[FloatArray] = []
    null_weights: list[FloatArray] = []
    for fname in INTERICTAL_FILES:
        path = data_dir / fname
        if not path.exists():
            continue
        t0 = time.time()
        data, fs, _ = _load_edf(path)
        data, fs = _decimate_to_rate(data, fs)
        phases, weights = _phases_and_weights(data, fs, EPOCH_SECONDS)
        null_phases.append(phases)
        null_weights.append(weights)
        print(f"  {fname}: {phases.shape[0]} ch, {time.time()-t0:.1f}s")

    print("Loading seizure recordings and computing PLV weights...")
    seizure_phases: dict[str, FloatArray] = {}
    seizure_weights: dict[str, FloatArray] = {}
    for fname in sorted(seizures):
        path = data_dir / fname
        t0 = time.time()
        data, fs, _ = _load_edf(path)
        data, fs = _decimate_to_rate(data, fs)
        phases, weights = _phases_and_weights(data, fs, EPOCH_SECONDS)
        seizure_phases[fname] = phases
        seizure_weights[fname] = weights
        print(f"  {fname}: {phases.shape[0]} ch, {time.time()-t0:.1f}s")

    print("Computing null scores...")
    null_mean_r = np.concatenate(
        [
            _mean_r_per_epoch(p, TARGET_SAMPLING_RATE_HZ, EPOCH_SECONDS)
            for p in null_phases
        ]
    )
    topk_null: dict[str, dict[int, FloatArray]] = {
        "per_epoch": {},
        "global": {},
    }
    for k in K_VALUES:
        topk_null["per_epoch"][k] = np.concatenate(
            [
                _topk_per_epoch_all_k(
                    p, w, TARGET_SAMPLING_RATE_HZ, EPOCH_SECONDS, [k]
                )[k]
                for p, w in zip(null_phases, null_weights, strict=True)
            ]
        )
        topk_null["global"][k] = np.concatenate(
            [
                _topk_global_all_k(
                    p, w, TARGET_SAMPLING_RATE_HZ, EPOCH_SECONDS, [k]
                )[k]
                for p, w in zip(null_phases, null_weights, strict=True)
            ]
        )
        n_pe = len(topk_null["per_epoch"][k])
        n_gl = len(topk_null["global"][k])
        print(f"  k={k}: per-epoch {n_pe} epochs, global {n_gl} epochs")

    print("Computing per-seizure scores...")
    rows: list[dict[str, Any]] = []
    for fname in sorted(seizures):
        onset_s, _ = seizures[fname]
        phases = seizure_phases[fname]
        weights = seizure_weights[fname]
        n_epochs = phases.shape[1] // int(EPOCH_SECONDS * TARGET_SAMPLING_RATE_HZ)
        event_idx = _preictal_epoch_indices(n_epochs, onset_s)
        if not event_idx:
            continue

        mean_r = _mean_r_per_epoch(phases, TARGET_SAMPLING_RATE_HZ, EPOCH_SECONDS)
        row: dict[str, Any] = {
            "file": fname,
            "mean_r": {
                "auc": _compute_auc(mean_r[event_idx], null_mean_r),
                "dr": _dr_at_fa(mean_r[event_idx], null_mean_r, 0.10),
            },
        }
        for mode, func in (
            ("per_epoch", _topk_per_epoch_all_k),
            ("global", _topk_global_all_k),
        ):
            all_scores = func(
                phases, weights, TARGET_SAMPLING_RATE_HZ, EPOCH_SECONDS, K_VALUES
            )
            row[mode] = {
                k: {
                    "auc": _compute_auc(scores[event_idx], topk_null[mode][k]),
                    "dr": _dr_at_fa(scores[event_idx], topk_null[mode][k], 0.10),
                }
                for k, scores in all_scores.items()
            }
        rows.append(row)
        print(f"  {fname}: done")

    agg: dict[str, Any] = {"baseline": {}, "per_epoch": {}, "global": {}}
    agg["baseline"] = {
        "mean_auc": round(float(np.mean([r["mean_r"]["auc"] for r in rows])), 6),
        "mean_dr": round(float(np.mean([r["mean_r"]["dr"] for r in rows])), 6),
    }
    for mode in ("per_epoch", "global"):
        for k in K_VALUES:
            agg[mode][k] = {
                "mean_auc": round(
                    float(np.mean([r[mode][k]["auc"] for r in rows])), 6
                ),
                "mean_dr": round(
                    float(np.mean([r[mode][k]["dr"] for r in rows])), 6
                ),
            }

    out = Path("examples/real_data/chb01_seizures_multichannel_kuramoto")
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "topk_plv_diagnostic.json"
    out_path.write_text(
        json.dumps({"per_seizure": rows, "aggregate": agg}, indent=2) + "\n",
        encoding="utf-8",
    )

    base_auc = agg["baseline"]["mean_auc"]
    base_dr = agg["baseline"]["mean_dr"]
    print(f"\nBaseline mean-R: AUC={base_auc:.3f} DR={base_dr:.3f}")
    print("Top-k per-epoch selection:")
    for k, v in agg["per_epoch"].items():
        print(f"  k={k:2d}: AUC={v['mean_auc']:.3f} DR={v['mean_dr']:.3f}")
    print("Top-k global selection:")
    for k, v in agg["global"].items():
        print(f"  k={k:2d}: AUC={v['mean_auc']:.3f} DR={v['mean_dr']:.3f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
