# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sleep-EDF delta-phase Kuramoto variant transfer audit

"""Transfer audit of the CAP Kuramoto variants on PhysioNet Sleep-EDF.

The [CAP variant panel](../../docs/studies/cap_kuramoto_variants.md) showed that
time-resolved amplitude gating rescues the delta-phase Kuramoto detector on a
multi-channel montage. This module tests whether that mechanism *transfers* to a
different corpus and channel regime: PhysioNet Sleep-EDF Expanded, using the two
EEG derivations it provides (Fpz-Cz and Pz-Oz) as a two-oscillator montage.

The same seven detectors and the same matched false-alarm protocol
(``target_false_alarm = 0.10``, 10 000-permutation test, seed 42) are applied.
The transfer question is not whether the variants beat the envelope here — the
single-channel delta envelope already scores ~0.96 on this recording — but
whether amplitude gating keeps the Kuramoto family strong while the plain mean-R
detector collapses, as it did on CAP.

Raw EDF files are citation-only and never redistributed; only derived sealed
audit records and the aggregate comparison JSON are written.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bench.cap_kuramoto_variants import DETECTORS
from bench.honest_dataset_audit import (
    AuditConfig,
    RecordingSpec,
    default_recommendation,
    run_honest_audit,
)
from bench.sleep_staging_sleepedf import (
    EPOCH_SECONDS,
    PERMUTATION_SEED,
    PERMUTATIONS,
    TARGET_FALSE_ALARM,
    _epoch_stages,
    _load_annotations,
)

FloatArray = NDArray[np.float64]

#: The two Sleep-EDF EEG derivations used as a two-oscillator montage.
EEG_CHANNELS: tuple[str, ...] = ("EEG Fpz-Cz", "EEG Pz-Oz")


def _load_two_eeg(psg_path: Path) -> tuple[FloatArray, float]:
    """Load the two Sleep-EDF EEG channels into a ``(2, n_samples)`` array.

    Parameters
    ----------
    psg_path : Path
        Path to the Sleep-EDF PSG recording.

    Returns
    -------
    data : FloatArray
        The two EEG channels, shape ``(2, n_samples)``, trimmed to a common length.
    fs : float
        The common sampling rate in hertz.

    Raises
    ------
    ValueError
        If either EEG derivation is missing from the recording.
    """
    import pyedflib

    reader = pyedflib.EdfReader(str(psg_path))
    try:
        labels = reader.getSignalLabels()
        channels: list[FloatArray] = []
        fs: float | None = None
        for target in EEG_CHANNELS:
            idx = next((i for i, lab in enumerate(labels) if target in lab), None)
            if idx is None:
                raise ValueError(f"no channel matching {target!r} in {psg_path}")
            fs = float(reader.getSampleFrequency(idx))
            channels.append(np.asarray(reader.readSignal(idx), dtype=np.float64))
    finally:
        reader.close()

    if fs is None:
        raise ValueError("no EEG channels loaded")
    n = min(len(c) for c in channels)
    data = np.stack([c[:n] for c in channels])
    return data, fs


def _load_recording(spec: RecordingSpec) -> dict[str, Any]:
    """Load a Sleep-EDF recording into the domain object consumed by the detectors."""
    data, fs = _load_two_eeg(spec.paths["psg"])
    onsets, durations, descriptions = _load_annotations(spec.paths["hypnogram"])
    epoch_len = int(EPOCH_SECONDS * fs)
    n_epochs = data.shape[1] // epoch_len
    stages = _epoch_stages(n_epochs, onsets, durations, descriptions)
    return {"data": data, "fs": fs, "stages": stages}


def _stage_labels(domain_object: dict[str, Any]) -> list[str]:
    """Return the per-epoch stage labels for the loaded recording."""
    stages: list[str] = domain_object["stages"]
    return stages


def build_manifest(sleepedf_data_dir: Path, recording_id: str) -> list[RecordingSpec]:
    """Return a single-recording Sleep-EDF manifest rooted at ``sleepedf_data_dir``."""
    return [
        RecordingSpec(
            recording_id=recording_id,
            paths={
                "psg": sleepedf_data_dir / f"{recording_id}-PSG.edf",
                "hypnogram": sleepedf_data_dir / f"{recording_id[:6]}EC-Hypnogram.edf",
            },
            metadata={"corpus": "sleepedf"},
        )
    ]


def run(
    sleepedf_data_dir: Path,
    output_dir: Path,
    *,
    recording_id: str,
    captured_at: str,
) -> dict[str, Any]:
    """Audit every detector on the Sleep-EDF recording and write sealed evidence.

    Parameters
    ----------
    sleepedf_data_dir : Path
        Directory holding the ``{recording_id}-PSG.edf`` and hypnogram files.
    output_dir : Path
        Directory for sealed audit records and the aggregate JSON.
    recording_id : str
        Sleep-EDF PSG stem, e.g. ``SC4001E0``.
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
        manifest=build_manifest(sleepedf_data_dir, recording_id),
        loader=_load_recording,
        detectors=DETECTORS,  # type: ignore[arg-type]
        label_extractor=_stage_labels,
        output_dir=output_dir,
        config=config,
        benchmark="sleepedf_kuramoto_variants",
        corpus="PhysioNet Sleep-EDF Expanded",
        event_label="N3",
        null_label="Wake",
        corpus_id_fn=lambda spec: f"sleepedf-{spec.recording_id}-kuramoto-variants",
        recommendation_fn=default_recommendation,
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the Sleep-EDF Kuramoto-variant transfer audit.

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
        description="Sleep-EDF delta-phase Kuramoto-variant transfer audit."
    )
    parser.add_argument(
        "--sleepedf-data-dir",
        type=Path,
        default=Path("scratchpad/sleepedf_data"),
        help="Directory with {recording_id}-PSG.edf and the hypnogram.",
    )
    parser.add_argument(
        "--recording-id",
        default="SC4001E0",
        help="Sleep-EDF PSG stem (default: SC4001E0).",
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

    aggregate = run(
        args.sleepedf_data_dir,
        args.output_dir,
        recording_id=args.recording_id,
        captured_at=args.captured_at,
    )
    for name in DETECTORS:
        stats = aggregate[name]
        print(
            f"{name:32} DR={stats['mean_detection_rate']:.3f}  "
            f"beats={stats['fraction_beats_chance']:.2f}"
        )
    print(f"preferred: {aggregate['recommendation']['preferred_variant']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shell over tested logic
    raise SystemExit(main())
