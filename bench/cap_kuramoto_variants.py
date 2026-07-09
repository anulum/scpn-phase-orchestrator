# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CAP delta-phase Kuramoto detector variants

"""Honest audit of a family of delta-phase Kuramoto variants on CAP N3 vs Wake.

The diagnostic study ``docs/studies/cap_kuramoto_diagnostic.md`` showed that the
simple mean-R Kuramoto detector fails on recordings such as ``brux2`` because
both N3 and Wake epochs carry a high *mean* order parameter — the mean discards
the N3-vs-Wake separation of ``R(t)``. The earlier SNR-weighted variant (per
channel, per epoch) did not improve on the mean. This module audits four new
variants that each target that separation from a different angle, alongside the
three established detectors, on the same four-recording CAP panel and at the same
matched false-alarm operating point.

Variants
--------
``amplitude_gated_delta_kuramoto``
    Weight ``R(t)`` by the instantaneous mean delta-band Hilbert amplitude, so
    coherence only contributes when genuine slow-wave power is present. This is a
    *time-resolved* amplitude gate, distinct from the per-epoch SNR weighting.
``sustained_delta_kuramoto``
    The lower quartile of ``R(t)`` within each epoch — the sustained coherence
    floor. N3 slow-wave coherence persists across the epoch; bursty artefact
    coherence dips.
``adaptive_channel_kuramoto``
    The mean ``R(t)`` computed only over channels whose whole-recording
    delta-band SNR is at or above the median, so noisy derivations do not dilute
    the estimate.
``coherent_sustained_kuramoto``
    The lower quartile of the amplitude-gated ``R(t)`` — sustained coherent
    slow-wave power, combining the amplitude gate and the sustained floor.

Raw EDF files are citation-only and never redistributed; only derived sealed
audit records and the aggregate comparison JSON are written.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

from bench.cap_multichannel_n3_vs_wake import (
    DELTA_BAND_HZ,
    EPOCH_SECONDS,
    PERMUTATION_SEED,
    PERMUTATIONS,
    SCORE_PRECISION,
    TARGET_FALSE_ALARM,
    _bandpass,
    _envelope_scores,
    _epoch_stages,
    _kuramoto_scores,
    _load_eeg_channels,
    _parse_remlogic_stages,
    _snr_weighted_kuramoto_scores,
)
from bench.honest_dataset_audit import (
    AuditConfig,
    RecordingSpec,
    default_recommendation,
    run_honest_audit,
)

FloatArray = NDArray[np.float64]

#: Lower-quartile probability used by the sustained-coherence variants.
SUSTAINED_QUANTILE = 0.25

#: The four CAP panel recordings and their clinical condition, in audit order.
CAP_PANEL: tuple[tuple[str, str], ...] = (
    ("n1", "control"),
    ("n2", "control"),
    ("brux2", "bruxism"),
    ("narco2", "narcolepsy"),
)


def _delta_phase_env(data: FloatArray, fs: float) -> tuple[FloatArray, FloatArray]:
    """Return per-channel delta-band Hilbert phase and amplitude envelope.

    Parameters
    ----------
    data : FloatArray
        EEG channels, shape ``(n_channels, n_samples)``.
    fs : float
        Sampling rate in hertz.

    Returns
    -------
    phases : FloatArray
        Instantaneous delta-band phase per channel, shape ``(n_channels, n_samples)``.
    envelope : FloatArray
        Instantaneous delta-band amplitude per channel, same shape.
    """
    n_channels, n_samples = data.shape
    phases = np.empty((n_channels, n_samples), dtype=np.float64)
    envelope = np.empty((n_channels, n_samples), dtype=np.float64)
    for c in range(n_channels):
        delta = _bandpass(data[c], fs, DELTA_BAND_HZ[0], DELTA_BAND_HZ[1])
        analytic = hilbert(delta)
        phases[c, :] = np.angle(analytic)
        envelope[c, :] = np.abs(analytic)
    return phases, envelope


def _epoch_reshape(series: FloatArray, fs: float) -> FloatArray:
    """Reshape a per-sample series into ``(n_epochs, epoch_len)``, trimming the tail."""
    epoch_len = int(EPOCH_SECONDS * fs)
    n_epochs = series.shape[0] // epoch_len
    return series[: n_epochs * epoch_len].reshape(n_epochs, epoch_len)


def amplitude_gated_scores(data: FloatArray, fs: float) -> FloatArray:
    """Amplitude-gated delta-phase Kuramoto scores per epoch.

    The per-sample order parameter ``R(t)`` is weighted by the instantaneous mean
    delta amplitude ``A(t)`` across channels; the epoch score is the mean of
    ``R(t)·A(t)``. Coherence only contributes where real slow-wave power exists,
    suppressing artefact-driven coherence with little delta amplitude.

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
    phases, envelope = _delta_phase_env(data, fs)
    r_t = np.abs(np.exp(1j * phases).mean(axis=0))
    a_t = envelope.mean(axis=0)
    scores = _epoch_reshape(r_t * a_t, fs).mean(axis=1)
    return np.round(scores, SCORE_PRECISION)


def sustained_scores(data: FloatArray, fs: float) -> FloatArray:
    """Sustained delta-phase Kuramoto scores per epoch (lower quartile of R(t)).

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
    phases, _ = _delta_phase_env(data, fs)
    r_t = np.abs(np.exp(1j * phases).mean(axis=0))
    scores = np.quantile(_epoch_reshape(r_t, fs), SUSTAINED_QUANTILE, axis=1)
    return np.round(scores, SCORE_PRECISION)


def adaptive_channel_scores(data: FloatArray, fs: float) -> FloatArray:
    """Adaptive-channel delta-phase Kuramoto scores per epoch.

    Channels whose whole-recording delta-band SNR (delta power / total power) is
    at or above the panel median are retained (at least three); the mean ``R(t)``
    is computed over that subset only, so noisy derivations do not dilute the
    coherence estimate.

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
    delta = np.empty((n_channels, n_samples), dtype=np.float64)
    snr = np.empty(n_channels, dtype=np.float64)
    for c in range(n_channels):
        delta[c] = _bandpass(data[c], fs, DELTA_BAND_HZ[0], DELTA_BAND_HZ[1])
        snr[c] = (delta[c] ** 2).mean() / ((data[c] ** 2).mean() + 1e-12)

    keep = np.flatnonzero(snr >= np.median(snr))
    if keep.size < 3:
        keep = np.argsort(snr)[::-1][:3]

    phases = np.angle(hilbert(delta[keep], axis=1))
    r_t = np.abs(np.exp(1j * phases).mean(axis=0))
    scores = _epoch_reshape(r_t, fs).mean(axis=1)
    return np.round(scores, SCORE_PRECISION)


def coherent_sustained_scores(data: FloatArray, fs: float) -> FloatArray:
    """Coherent-sustained scores per epoch (lower quartile of R(t)·A(t)).

    Combines the amplitude gate and the sustained floor: the lower quartile of
    the amplitude-weighted order parameter within each epoch.

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
    phases, envelope = _delta_phase_env(data, fs)
    r_t = np.abs(np.exp(1j * phases).mean(axis=0))
    a_t = envelope.mean(axis=0)
    scores = np.quantile(_epoch_reshape(r_t * a_t, fs), SUSTAINED_QUANTILE, axis=1)
    return np.round(scores, SCORE_PRECISION)


#: The full detector registry audited by this module: three established
#: detectors plus the four variants, keyed by their sealed-record names.
DETECTORS: dict[str, object] = {
    "normalized_delta_envelope": lambda o: _envelope_scores(o["data"], o["fs"]),
    "multi_channel_delta_kuramoto": lambda o: _kuramoto_scores(o["data"], o["fs"]),
    "snr_weighted_delta_kuramoto": lambda o: _snr_weighted_kuramoto_scores(
        o["data"], o["fs"]
    ),
    "amplitude_gated_delta_kuramoto": lambda o: amplitude_gated_scores(
        o["data"], o["fs"]
    ),
    "sustained_delta_kuramoto": lambda o: sustained_scores(o["data"], o["fs"]),
    "adaptive_channel_kuramoto": lambda o: adaptive_channel_scores(o["data"], o["fs"]),
    "coherent_sustained_kuramoto": lambda o: coherent_sustained_scores(
        o["data"], o["fs"]
    ),
}


def _load_recording(spec: RecordingSpec) -> dict[str, object]:
    """Load a CAP recording into the domain object consumed by the detectors."""
    data, channels, fs = _load_eeg_channels(spec.paths["edf"])
    intervals = _parse_remlogic_stages(spec.paths["txt"])
    epoch_len = int(EPOCH_SECONDS * fs)
    n_epochs = data.shape[1] // epoch_len
    stages = _epoch_stages(n_epochs, intervals)
    return {"data": data, "fs": fs, "stages": stages, "channels": channels}


def _stage_labels(domain_object: dict[str, object]) -> list[str]:
    """Return the per-epoch stage labels for the loaded recording."""
    stages: list[str] = domain_object["stages"]  # type: ignore[assignment]
    return stages


def build_manifest(cap_data_dir: Path) -> list[RecordingSpec]:
    """Return the four-recording CAP panel manifest rooted at ``cap_data_dir``."""
    manifest: list[RecordingSpec] = []
    for recording_id, condition in CAP_PANEL:
        manifest.append(
            RecordingSpec(
                recording_id=recording_id,
                paths={
                    "edf": cap_data_dir / f"{recording_id}.edf",
                    "txt": cap_data_dir / f"{recording_id}.txt",
                },
                metadata={"condition": condition},
            )
        )
    return manifest


def run(cap_data_dir: Path, output_dir: Path, *, captured_at: str) -> dict[str, Any]:
    """Audit every detector on the CAP panel and write sealed evidence.

    Parameters
    ----------
    cap_data_dir : Path
        Directory holding ``{recording_id}.edf`` / ``.txt`` for the panel.
    output_dir : Path
        Directory for sealed audit records and the aggregate JSON.
    captured_at : str
        Provenance timestamp recorded in the sealed records.

    Returns
    -------
    dict
        The aggregate comparison JSON.
    """
    config = AuditConfig(
        target_false_alarm=TARGET_FALSE_ALARM,
        n_permutations=PERMUTATIONS,
        seed=PERMUTATION_SEED,
        captured_at=captured_at,
    )
    return run_honest_audit(
        manifest=build_manifest(cap_data_dir),
        loader=_load_recording,
        detectors=DETECTORS,  # type: ignore[arg-type]
        label_extractor=_stage_labels,
        output_dir=output_dir,
        config=config,
        benchmark="cap_kuramoto_variants",
        corpus="PhysioNet CAP Sleep Database",
        event_label="N3",
        null_label="Wake",
        corpus_id_fn=lambda spec: f"cap-{spec.recording_id}-kuramoto-variants",
        recommendation_fn=default_recommendation,
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the CAP Kuramoto-variant audit.

    Parameters
    ----------
    argv : list of str or None
        Arguments excluding the program name; read from :data:`sys.argv` when
        ``None``.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(description="CAP delta-phase Kuramoto variants.")
    parser.add_argument(
        "--cap-data-dir",
        type=Path,
        default=Path("scratchpad/cap_data"),
        help="Directory with {recording_id}.edf/.txt for the CAP panel.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for sealed audit output.",
    )
    parser.add_argument(
        "--captured-at",
        default="2026-07-10T00:00:00Z",
        help="Provenance timestamp for sealed records.",
    )
    args = parser.parse_args(argv)

    aggregate = run(args.cap_data_dir, args.output_dir, captured_at=args.captured_at)
    for name in DETECTORS:
        stats = aggregate[name]
        print(
            f"{name:32} mean DR={stats['mean_detection_rate']:.3f}  "
            f"beats={stats['fraction_beats_chance']:.2f}"
        )
    print(f"preferred: {aggregate['recommendation']['preferred_variant']}")
    print(f"aggregate written to {args.output_dir / 'cap_kuramoto_variants.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shell over tested logic
    raise SystemExit(main())
