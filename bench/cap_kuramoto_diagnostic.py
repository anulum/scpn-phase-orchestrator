# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CAP multi-channel Kuramoto diagnostic

"""Diagnose why the delta-phase Kuramoto detector succeeds or fails per recording.

Loads the four PhysioNet CAP Sleep Database recordings already audited by
``bench/cap_multichannel_n3_vs_wake.py``, computes per-epoch signal properties,
and correlates them with the detector-score gap. The output is a diagnostic
JSON plus a Markdown report that recommends the next Kuramoto variant to build.

Usage:
    python bench/cap_kuramoto_diagnostic.py \
        examples/real_data/cap_multichannel_staging

The script expects the aggregate comparison JSON produced by the batch audit and
writes ``cap_kuramoto_diagnostic.json`` into the same directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from bench.cap_multichannel_n3_vs_wake import (
    DELTA_BAND_HZ,
    EPOCH_SECONDS,
    TARGET_SAMPLING_RATE_HZ,
    _bandpass,
    _envelope_scores,
    _epoch_stages,
    _kuramoto_scores,
    _load_eeg_channels,
    _parse_remlogic_stages,
)

FloatArray = NDArray[np.float64]

#: High-frequency band used as an artifact proxy, in hertz.
#: Upper edge is below the 100 Hz Nyquist frequency.
ARTIFACT_BAND_HZ = (30.0, 45.0)
#: Broadband band used for SNR denominators, in hertz.
BROADBAND_CUTOFF_HZ = 45.0
#: Precision for JSON float values.
ROUND_DIGITS = 6


def _analytic(sig: FloatArray) -> FloatArray:
    """Return the analytic signal (Hilbert transform)."""
    from scipy.signal import hilbert

    return hilbert(sig)


def _kurtosis(sig: FloatArray) -> float:
    """Return excess kurtosis of ``sig``."""
    sig = np.asarray(sig, dtype=np.float64)
    if sig.std() == 0:
        return 0.0
    z = (sig - sig.mean()) / sig.std()
    return float(np.mean(z**4) - 3.0)


def _epoch_features(
    data: FloatArray,
    fs: float,
) -> dict[str, FloatArray]:
    """Compute per-epoch diagnostic features.

    Parameters
    ----------
    data : FloatArray
        EEG channels, shape ``(n_channels, n_samples)``.
    fs : float
        Sampling rate in hertz.

    Returns
    -------
    dict[str, FloatArray]
        Each value has shape ``(n_epochs,)``.
    """
    n_channels, n_samples = data.shape
    epoch_len = int(EPOCH_SECONDS * fs)
    n_epochs = n_samples // epoch_len

    delta_snr = np.zeros(n_epochs, dtype=np.float64)
    delta_env_mean = np.zeros(n_epochs, dtype=np.float64)
    delta_env_std = np.zeros(n_epochs, dtype=np.float64)
    phase_circvar = np.zeros(n_epochs, dtype=np.float64)
    kuramoto_r_mean = np.zeros(n_epochs, dtype=np.float64)
    kuramoto_r_std = np.zeros(n_epochs, dtype=np.float64)
    hf_power_ratio = np.zeros(n_epochs, dtype=np.float64)
    signal_kurtosis = np.zeros(n_epochs, dtype=np.float64)

    # Pre-compute filtered signals, phases, and artifact-band power for all channels.
    delta_signals: list[FloatArray] = []
    delta_envs: list[FloatArray] = []
    artifact_power: list[FloatArray] = []
    raw_power: list[FloatArray] = []
    phases = np.empty((n_channels, n_samples), dtype=np.float64)
    for c in range(n_channels):
        delta = _bandpass(data[c], fs, DELTA_BAND_HZ[0], DELTA_BAND_HZ[1])
        delta_signals.append(delta)
        delta_env = np.abs(_analytic(delta))
        delta_envs.append(delta_env)
        phases[c, :] = np.angle(_analytic(delta))
        artifact = _bandpass(data[c], fs, ARTIFACT_BAND_HZ[0], ARTIFACT_BAND_HZ[1])
        artifact_power.append(artifact**2)
        raw_power.append(data[c] ** 2)

    r_t = np.abs(np.exp(1j * phases).mean(axis=0))

    for e in range(n_epochs):
        start = e * epoch_len
        stop = start + epoch_len

        # Delta SNR: mean delta power / mean broadband power, averaged over channels.
        snr_per_channel: list[float] = []
        env_per_channel: list[float] = []
        for c in range(n_channels):
            epoch_delta = delta_signals[c][start:stop]
            epoch_raw = data[c][start:stop]
            delta_power = float(np.mean(epoch_delta**2))
            broad_power = float(np.mean(epoch_raw**2)) + 1e-12
            snr_per_channel.append(delta_power / broad_power)
            env_per_channel.append(float(delta_envs[c][start:stop].mean()))
        delta_snr[e] = float(np.mean(snr_per_channel))
        delta_env_mean[e] = float(np.mean(env_per_channel))
        delta_env_std[e] = float(np.std(env_per_channel))

        # Circular variance of delta phases across channels.
        epoch_phases = phases[:, start:stop]
        mean_sin = np.sin(epoch_phases).mean(axis=0)
        mean_cos = np.cos(epoch_phases).mean(axis=0)
        r_mean = np.sqrt(mean_sin**2 + mean_cos**2)
        phase_circvar[e] = float(1.0 - r_mean.mean())

        # Kuramoto R statistics.
        epoch_r = r_t[start:stop]
        kuramoto_r_mean[e] = float(epoch_r.mean())
        kuramoto_r_std[e] = float(epoch_r.std())

        # Artifact proxies.
        hf_ratios: list[float] = []
        kurts: list[float] = []
        for c in range(n_channels):
            epoch_raw = data[c][start:stop]
            kurt = float(_kurtosis(epoch_raw))
            kurts.append(kurt)
            hf_power = float(np.mean(artifact_power[c][start:stop]))
            total_power = float(np.mean(raw_power[c][start:stop])) + 1e-12
            hf_ratios.append(hf_power / total_power)
        hf_power_ratio[e] = float(np.mean(hf_ratios))
        signal_kurtosis[e] = float(np.mean(kurts))

    return {
        "delta_snr": np.round(delta_snr, ROUND_DIGITS),
        "delta_env_mean": np.round(delta_env_mean, ROUND_DIGITS),
        "delta_env_std": np.round(delta_env_std, ROUND_DIGITS),
        "phase_circvar": np.round(phase_circvar, ROUND_DIGITS),
        "kuramoto_r_mean": np.round(kuramoto_r_mean, ROUND_DIGITS),
        "kuramoto_r_std": np.round(kuramoto_r_std, ROUND_DIGITS),
        "hf_power_ratio": np.round(hf_power_ratio, ROUND_DIGITS),
        "signal_kurtosis": np.round(signal_kurtosis, ROUND_DIGITS),
    }


def _recording_summary(
    recording_id: str,
    edf_path: Path,
    txt_path: Path,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    """Compute diagnostic features and summary for one recording."""
    data, channel_labels, fs = _load_eeg_channels(
        edf_path, target_fs=TARGET_SAMPLING_RATE_HZ
    )
    intervals = _parse_remlogic_stages(txt_path)

    epoch_len = int(EPOCH_SECONDS * fs)
    n_epochs = data.shape[1] // epoch_len
    stages = _epoch_stages(n_epochs, intervals)

    features = _epoch_features(data, fs)
    envelope_scores = _envelope_scores(data, fs)
    kuramoto_scores = _kuramoto_scores(data, fs)

    # Per-epoch score gap for correlating with features.
    score_gap = np.round(envelope_scores - kuramoto_scores, ROUND_DIGITS)

    # Aggregate detector results for this recording.
    per_rec = {r["recording_id"]: r for r in aggregate["per_recording"]}
    rec_agg = per_rec[recording_id]
    env_summary = rec_agg["detectors"]["normalized_delta_envelope"]
    kur_summary = rec_agg["detectors"]["multi_channel_delta_kuramoto"]

    n3_indices = [i for i, s in enumerate(stages) if s == "N3"]
    wake_indices = [i for i, s in enumerate(stages) if s == "Wake"]

    def _mean_over(indices: list[int], arr: FloatArray) -> float:
        if not indices:
            return 0.0
        return round(float(arr[indices].mean()), ROUND_DIGITS)

    return {
        "recording_id": recording_id,
        "n_channels": len(channel_labels),
        "channels": channel_labels,
        "sampling_rate_hz": fs,
        "n_epochs": n_epochs,
        "n_n3": len(n3_indices),
        "n_wake": len(wake_indices),
        "detector_results": {
            "envelope_detection_rate": env_summary["detection_rate"],
            "kuramoto_detection_rate": kur_summary["detection_rate"],
            "detection_rate_gap": round(
                env_summary["detection_rate"] - kur_summary["detection_rate"],
                ROUND_DIGITS,
            ),
        },
        "n3_feature_means": {
            name: _mean_over(n3_indices, arr) for name, arr in features.items()
        },
        "wake_feature_means": {
            name: _mean_over(wake_indices, arr) for name, arr in features.items()
        },
        "feature_distributions": {
            name: {
                "n3": [float(x) for x in np.round(arr[n3_indices], ROUND_DIGITS)],
                "wake": [float(x) for x in np.round(arr[wake_indices], ROUND_DIGITS)],
            }
            for name, arr in features.items()
        },
        "score_distributions": {
            "envelope": {
                "n3": [float(x) for x in envelope_scores[n3_indices]],
                "wake": [float(x) for x in envelope_scores[wake_indices]],
            },
            "kuramoto": {
                "n3": [float(x) for x in kuramoto_scores[n3_indices]],
                "wake": [float(x) for x in kuramoto_scores[wake_indices]],
            },
            "gap": {
                "n3": [float(x) for x in score_gap[n3_indices]],
                "wake": [float(x) for x in score_gap[wake_indices]],
            },
        },
    }


def _correlate_with_gap(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Correlate per-recording N3 feature means with the detector gap.

    Returns a sorted list of features by absolute Spearman correlation with the
    envelope-minus-Kuramoto detection-rate gap. Positive correlation means the
    feature rises where the envelope beats Kuramoto by more.
    """
    feature_names = list(summaries[0]["n3_feature_means"].keys())
    gaps = np.array([s["detector_results"]["detection_rate_gap"] for s in summaries])
    kuramoto_drs = np.array(
        [s["detector_results"]["kuramoto_detection_rate"] for s in summaries]
    )

    rows: list[dict[str, Any]] = []
    for name in feature_names:
        values = np.array([s["n3_feature_means"][name] for s in summaries])
        if np.std(values) == 0:
            corr_gap = 0.0
            corr_kur = 0.0
            p_gap = 1.0
            p_kur = 1.0
        else:
            corr_gap, p_gap = spearmanr(values, gaps)
            corr_kur, p_kur = spearmanr(values, kuramoto_drs)
        rows.append(
            {
                "feature": name,
                "correlation_with_gap": round(float(corr_gap), ROUND_DIGITS),
                "p_value_gap": round(float(p_gap), ROUND_DIGITS),
                "correlation_with_kuramoto_dr": round(float(corr_kur), ROUND_DIGITS),
                "p_value_kuramoto_dr": round(float(p_kur), ROUND_DIGITS),
            }
        )

    rows.sort(key=lambda r: abs(r["correlation_with_gap"]), reverse=True)
    return {
        "feature_correlations": rows,
        "top_predictor": rows[0]["feature"] if rows else None,
    }


def _recommend(
    correlations: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Produce a data-driven recommendation for the next Kuramoto variant."""
    top = correlations["top_predictor"]
    # Recording where Kuramoto beats the envelope by the largest margin.
    rec_id = min(
        summaries,
        key=lambda s: s["detector_results"]["detection_rate_gap"],
    )["recording_id"]

    if top in ("hf_power_ratio", "signal_kurtosis"):
        change = "artifact_aware_channel_selection"
        rationale = (
            f"{top} is the strongest predictor of the envelope-Kuramoto gap; "
            "channels corrupted by high-frequency artifacts should be dropped or "
            "down-weighted before computing R(t)."
        )
    elif top == "phase_circvar":
        change = "adaptive_channel_selection"
        rationale = (
            "High inter-channel circular variance predicts Kuramoto failure; "
            "dropping the least coherent channels before computing R(t) should "
            "raise mean R on clean recordings."
        )
    elif top == "delta_snr":
        change = "snr_weighted_kuramoto"
        rationale = (
            "Delta-band SNR predicts Kuramoto success; weighting each channel by "
            "its delta SNR before computing R(t) should emphasize coherent slow-wave "
            "activity."
        )
    elif top in ("kuramoto_r_std", "delta_env_std"):
        change = "robust_temporal_aggregation"
        rationale = (
            "Temporal instability of R(t) predicts poor discrimination; using a "
            "robust percentile or median of R(t) instead of the mean should reduce "
            "artifact sensitivity."
        )
    else:
        change = "snr_weighted_kuramoto"
        rationale = (
            "No single feature dominates; the safest first refinement is "
            f"SNR-weighted Kuramoto, validated on the recording where "
            f"Kuramoto already works ({rec_id})."
        )

    return {
        "next_variant": change,
        "rationale": rationale,
        "recording_where_kuramoto_works": rec_id,
    }


def main(output_dir: str | Path) -> None:
    """Run the diagnostic and write the JSON report."""
    out = Path(output_dir)
    aggregate_path = out / "cap_multichannel_aggregate.json"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"aggregate not found: {aggregate_path}")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))

    # Resolve recording paths from the manifest, relative to the project root
    # (the manifest stores project-relative paths).
    manifest_path = out / "cap_multichannel_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest_rows: dict[str, tuple[Path, Path]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            recording_id, edf, txt = row[0].strip(), row[1].strip(), row[2].strip()
            manifest_rows[recording_id] = (Path(edf), Path(txt))

    summaries: list[dict[str, Any]] = []
    for rec in aggregate["per_recording"]:
        recording_id = rec["recording_id"]
        edf_path, txt_path = manifest_rows[recording_id]
        if not edf_path.exists():
            raise FileNotFoundError(
                f"could not locate EDF for {recording_id}: {edf_path}"
            )
        if not txt_path.exists():
            raise FileNotFoundError(
                f"could not locate TXT for {recording_id}: {txt_path}"
            )

        summaries.append(
            _recording_summary(recording_id, edf_path, txt_path, aggregate)
        )
        print(f"{recording_id}: diagnostic features computed")

    correlations = _correlate_with_gap(summaries)
    recommendation = _recommend(correlations, summaries)

    diagnostic = {
        "benchmark": "cap_kuramoto_diagnostic",
        "corpus": "PhysioNet CAP Sleep Database",
        "n_recordings": len(summaries),
        "recording_ids": [s["recording_id"] for s in summaries],
        "per_recording": summaries,
        "correlations": correlations,
        "recommendation": recommendation,
    }

    out_path = out / "cap_kuramoto_diagnostic.json"
    out_path.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    print(f"Diagnostic report written to {out_path}")


if __name__ == "__main__":  # pragma: no cover - CLI shell over tested logic
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        help="Directory containing cap_multichannel_aggregate.json",
    )
    args = parser.parse_args()
    main(args.output_dir)
