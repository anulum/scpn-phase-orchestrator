# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — regime-adaptive slow-wave detector ensemble

"""Regime-adaptive slow-wave detector audited across CAP and Sleep-EDF.

Two honest audits established a regime split that no single detector spans:

* [CAP variants](../../docs/studies/cap_kuramoto_variants.md) — on a rich (6–8
  channel) montage the amplitude-gated ``coherent_sustained_kuramoto`` beats the
  delta envelope.
* [Sleep-EDF transfer](../../docs/studies/sleepedf_kuramoto_variants.md) — on a
  sparse (2 channel) montage pure coherence is dead and the plain delta envelope
  dominates; amplitude gating transfers as a mechanism but its advantage does
  not.

This module builds a **label-free regime router** that chooses a detector from
observable montage properties alone, and audits it alongside every component
detector on a combined five-recording cross-corpus manifest (four CAP + one
Sleep-EDF). The claim is not that the router beats every detector on every
recording — by construction it returns a component's score — but that it has the
best *cross-corpus* robustness: it selects the winning detector for each montage
regime without seeing the labels.

Two routers are provided:

``regime_adaptive_montage``
    Montage richness only: the delta envelope on a sparse montage (≤ 3 channels),
    ``coherent_sustained_kuramoto`` on a rich montage. This rule is validated
    across both corpora.
``regime_adaptive_full``
    Adds the ``n2`` axis: within a rich montage, a low-delta-SNR / high-coherence
    recording is routed to the plain mean-R detector. The thresholds are read
    from the CAP diagnostic (in-sample), so this router is illustrative and
    awaits out-of-sample validation.

Raw EDF files are citation-only and never redistributed; only derived sealed
audit records and the aggregate comparison JSON are written.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bench.cap_kuramoto_variants import (
    CAP_PANEL,
    _delta_phase_env,
    coherent_sustained_scores,
)
from bench.cap_kuramoto_variants import (
    DETECTORS as _VARIANT_DETECTORS,
)
from bench.cap_kuramoto_variants import _load_recording as _load_cap
from bench.cap_multichannel_n3_vs_wake import (
    DELTA_BAND_HZ,
    _bandpass,
    _envelope_scores,
    _kuramoto_scores,
)
from bench.honest_dataset_audit import (
    AuditConfig,
    RecordingSpec,
    default_recommendation,
    run_honest_audit,
)
from bench.sleepedf_kuramoto_variants import _load_recording as _load_sleepedf

FloatArray = NDArray[np.float64]

#: A montage with at most this many channels is treated as sparse.
SPARSE_MONTAGE_MAX = 3
#: Whole-recording mean delta SNR below this marks the low-SNR regime (from the
#: CAP diagnostic: n2 = 0.445, other CAP recordings 0.58–0.60). In-sample.
LOW_SNR_THRESHOLD = 0.50
#: Whole-recording mean order parameter above this marks the high-coherence
#: regime (from the CAP diagnostic: n2 = 0.728, others 0.60–0.69). In-sample.
HIGH_COHERENCE_THRESHOLD = 0.70


def _mean_delta_snr(data: FloatArray, fs: float) -> float:
    """Return the whole-recording mean delta-band SNR across channels."""
    ratios = []
    for c in range(data.shape[0]):
        delta = _bandpass(data[c], fs, DELTA_BAND_HZ[0], DELTA_BAND_HZ[1])
        ratios.append((delta**2).mean() / ((data[c] ** 2).mean() + 1e-12))
    return float(np.mean(ratios))


def _mean_order_parameter(data: FloatArray, fs: float) -> float:
    """Return the whole-recording mean delta-phase Kuramoto order parameter."""
    phases, _ = _delta_phase_env(data, fs)
    r_t = np.abs(np.exp(1j * phases).mean(axis=0))
    return float(r_t.mean())


def regime_adaptive_montage_scores(data: FloatArray, fs: float) -> FloatArray:
    """Route by montage richness alone (cross-corpus-validated rule).

    Returns the delta-envelope score on a sparse montage (``n_channels <=
    SPARSE_MONTAGE_MAX``) and the amplitude-gated sustained coherence score on a
    rich montage. The routing uses only the observable channel count, never the
    labels.

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
    if data.shape[0] <= SPARSE_MONTAGE_MAX:
        return _envelope_scores(data, fs)
    return coherent_sustained_scores(data, fs)


def regime_adaptive_full_scores(data: FloatArray, fs: float) -> FloatArray:
    """Route by montage richness and the low-SNR/high-coherence axis.

    On a sparse montage the delta envelope is used. On a rich montage a
    low-delta-SNR, high-coherence recording (the ``n2`` regime) is routed to the
    plain mean-R detector, otherwise to amplitude-gated sustained coherence. The
    SNR/coherence thresholds are read from the CAP diagnostic (in-sample), so
    this router is illustrative pending out-of-sample validation.

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
    if data.shape[0] <= SPARSE_MONTAGE_MAX:
        return _envelope_scores(data, fs)
    if (
        _mean_delta_snr(data, fs) < LOW_SNR_THRESHOLD
        and _mean_order_parameter(data, fs) > HIGH_COHERENCE_THRESHOLD
    ):
        return _kuramoto_scores(data, fs)
    return coherent_sustained_scores(data, fs)


#: The component detectors plus the two regime-adaptive routers.
DETECTORS: dict[str, Any] = {
    **_VARIANT_DETECTORS,
    "regime_adaptive_montage": lambda o: regime_adaptive_montage_scores(
        o["data"], o["fs"]
    ),
    "regime_adaptive_full": lambda o: regime_adaptive_full_scores(o["data"], o["fs"]),
}


def _load_dispatch(spec: RecordingSpec) -> dict[str, Any]:
    """Load a recording via the loader matching its corpus tag."""
    corpus = (spec.metadata or {}).get("corpus")
    if corpus == "cap":
        return _load_cap(spec)
    if corpus == "sleepedf":
        return _load_sleepedf(spec)
    raise ValueError(f"unknown corpus for {spec.recording_id!r}: {corpus!r}")


def _stage_labels(domain_object: dict[str, Any]) -> list[str]:
    """Return the per-epoch stage labels for the loaded recording."""
    stages: list[str] = domain_object["stages"]
    return stages


def build_manifest(
    cap_data_dir: Path,
    sleepedf_data_dir: Path,
    sleepedf_recording: str,
) -> list[RecordingSpec]:
    """Return the combined CAP + Sleep-EDF cross-corpus manifest."""
    manifest: list[RecordingSpec] = []
    for recording_id, condition in CAP_PANEL:
        manifest.append(
            RecordingSpec(
                recording_id=recording_id,
                paths={
                    "edf": cap_data_dir / f"{recording_id}.edf",
                    "txt": cap_data_dir / f"{recording_id}.txt",
                },
                metadata={"corpus": "cap", "condition": condition},
            )
        )
    manifest.append(
        RecordingSpec(
            recording_id=sleepedf_recording,
            paths={
                "psg": sleepedf_data_dir / f"{sleepedf_recording}-PSG.edf",
                "hypnogram": sleepedf_data_dir
                / f"{sleepedf_recording[:6]}EC-Hypnogram.edf",
            },
            metadata={"corpus": "sleepedf"},
        )
    )
    return manifest


def run(
    cap_data_dir: Path,
    sleepedf_data_dir: Path,
    output_dir: Path,
    *,
    sleepedf_recording: str,
    captured_at: str,
) -> dict[str, Any]:
    """Audit every detector and both routers across the cross-corpus manifest.

    Parameters
    ----------
    cap_data_dir : Path
        Directory with the CAP ``{recording_id}.edf`` / ``.txt`` files.
    sleepedf_data_dir : Path
        Directory with the Sleep-EDF PSG and hypnogram files.
    output_dir : Path
        Directory for sealed audit records and the aggregate JSON.
    sleepedf_recording : str
        Sleep-EDF PSG stem, e.g. ``SC4001E0``.
    captured_at : str
        Provenance timestamp recorded in the sealed records.

    Returns
    -------
    dict
        The aggregate comparison JSON.
    """
    config = AuditConfig(captured_at=captured_at)
    return run_honest_audit(
        manifest=build_manifest(cap_data_dir, sleepedf_data_dir, sleepedf_recording),
        loader=_load_dispatch,
        detectors=DETECTORS,  # type: ignore[arg-type]
        label_extractor=_stage_labels,
        output_dir=output_dir,
        config=config,
        benchmark="regime_adaptive_ensemble",
        corpus="PhysioNet CAP + Sleep-EDF",
        event_label="N3",
        null_label="Wake",
        corpus_id_fn=lambda spec: f"regime-{spec.recording_id}",
        recommendation_fn=default_recommendation,
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the regime-adaptive ensemble audit.

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
    parser = argparse.ArgumentParser(
        description="Regime-adaptive slow-wave detector cross-corpus audit."
    )
    parser.add_argument(
        "--cap-data-dir", type=Path, default=Path("scratchpad/cap_data")
    )
    parser.add_argument(
        "--sleepedf-data-dir", type=Path, default=Path("scratchpad/sleepedf_data")
    )
    parser.add_argument("--sleepedf-recording", default="SC4001E0")
    parser.add_argument("output_dir", type=Path, help="Directory for sealed output.")
    parser.add_argument("--captured-at", default="2026-07-10T00:00:00Z")
    args = parser.parse_args(argv)

    aggregate = run(
        args.cap_data_dir,
        args.sleepedf_data_dir,
        args.output_dir,
        sleepedf_recording=args.sleepedf_recording,
        captured_at=args.captured_at,
    )
    for name in DETECTORS:
        stats = aggregate[name]
        print(f"{name:34} cross-corpus mean DR={stats['mean_detection_rate']:.3f}")
    print(f"preferred: {aggregate['recommendation']['preferred_variant']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shell over tested logic
    raise SystemExit(main())
