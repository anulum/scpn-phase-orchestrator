# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — synthetic honest-audit harness demo

"""Synthetic cross-domain demo of the reusable honest-audit harness.

Generates a tiny corpus of event/null windows and audits two detectors through
``bench.honest_dataset_audit``. Events are AR(1) windows with rising
autocorrelation (the canonical critical-slowing-down signature); nulls are white
noise. The autocorrelation detector carries real skill, while the window-mean
detector is a no-skill control because both arms are zero-mean with unit
marginal variance.

This is intentionally not a clinical or scientific claim; it is a proof that the
harness generalises beyond sleep EEG to any domain that can provide per-recording
event/null labels and per-epoch scores.

Usage::

    python bench/synthetic_honest_audit_demo.py \
      examples/real_data/synthetic_honest_audit_demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bench.honest_dataset_audit import (
    AuditConfig,
    RecordingSpec,
    run_honest_audit,
)

FloatArray = NDArray[np.float64]

#: Samples per window.
WINDOW = 64
#: Event windows per recording.
N_EVENTS = 40
#: Null windows per recording.
N_NULLS = 200
#: AR(1) coefficient for event windows.
EVENT_AR1 = 0.8
#: Recording identifiers for the synthetic panel.
RECORDING_IDS = ("run_a", "run_b", "run_c")


def _ar1_window(rng: np.random.Generator, phi: float, length: int) -> FloatArray:
    """Return one zero-mean, unit-marginal-variance AR(1) window."""
    innovation = float(np.sqrt(1.0 - phi**2))
    values = np.empty(length)
    values[0] = rng.normal(0.0, 1.0)
    for index in range(1, length):
        values[index] = phi * values[index - 1] + rng.normal(0.0, innovation)
    return np.asarray(values - values.mean(), dtype=np.float64)


def _make_windows(seed: int) -> tuple[list[FloatArray], list[FloatArray]]:
    """Return ``(event_windows, null_windows)`` for one recording."""
    rng = np.random.default_rng(seed)
    events = [_ar1_window(rng, EVENT_AR1, WINDOW) for _ in range(N_EVENTS)]
    nulls = [_ar1_window(rng, 0.0, WINDOW) for _ in range(N_NULLS)]
    return events, nulls


def _recording_object(spec: RecordingSpec) -> dict[str, Any]:
    """Load the synthetic windows and labels for one recording."""
    seed = int(spec.metadata["seed"])
    events, nulls = _make_windows(seed)
    windows = events + nulls
    labels = ["event"] * N_EVENTS + ["null"] * N_NULLS
    return {"windows": windows, "labels": labels, "seed": seed}


def _labels(obj: dict[str, Any]) -> list[str]:
    """Return the per-window labels."""
    return obj["labels"]


def lag1_autocorrelation_score(obj: dict[str, Any]) -> FloatArray:
    """Skilful detector: lag-1 autocorrelation of each window."""
    scores = []
    for window in obj["windows"]:
        centred = window - window.mean()
        denominator = float(np.dot(centred, centred))
        if denominator == 0.0:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(centred[:-1], centred[1:]) / denominator))
    return np.round(np.asarray(scores, dtype=np.float64), 6)


def window_mean_score(obj: dict[str, Any]) -> FloatArray:
    """No-skill control: window mean (both arms are zero-mean)."""
    return np.round(np.asarray([w.mean() for w in obj["windows"]], dtype=np.float64), 6)


def main(output_dir: str | Path) -> None:
    """Generate the synthetic corpus and run the honest-audit harness."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = [
        RecordingSpec(
            recording_id=rec_id,
            paths={"manifest": out / f"{rec_id}_synthetic_manifest.json"},
            metadata={"seed": seed},
        )
        for rec_id, seed in zip(RECORDING_IDS, (0, 1, 2), strict=True)
    ]
    for spec in manifest:
        spec.paths["manifest"].write_text(
            json.dumps(
                {
                    "recording_id": spec.recording_id,
                    "seed": spec.metadata["seed"],
                    "n_events": N_EVENTS,
                    "n_nulls": N_NULLS,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    aggregate = run_honest_audit(
        manifest=manifest,
        loader=_recording_object,
        detectors={
            "lag1_autocorrelation": lag1_autocorrelation_score,
            "window_mean_control": window_mean_score,
        },
        label_extractor=_labels,
        output_dir=out,
        config=AuditConfig(n_permutations=10_000, seed=42),
        benchmark="synthetic_honest_audit_demo",
        corpus="Synthetic AR(1) critical-slowing-down corpus",
        event_label="event",
        null_label="null",
        corpus_id_fn=lambda spec: f"synthetic-{spec.recording_id}",
    )

    auto_dr = aggregate["lag1_autocorrelation"]["mean_detection_rate"]
    mean_dr = aggregate["window_mean_control"]["mean_detection_rate"]
    preferred = aggregate["recommendation"]["preferred_variant"]
    print(
        f"Synthetic honest-audit demo:\n"
        f"  recordings={aggregate['recording_ids']}\n"
        f"  lag1_autocorrelation mean DR={auto_dr:.3f}\n"
        f"  window_mean_control mean DR={mean_dr:.3f}\n"
        f"  preferred_variant={preferred}"
    )
    print(f"Artefacts written to {out}")


if __name__ == "__main__":  # pragma: no cover - CLI shell over tested logic
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        help="Directory for the manifest and sealed audit output",
    )
    args = parser.parse_args()
    main(args.output_dir)
