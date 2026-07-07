#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — worked example: honest early-warning auditor

"""Audit two detectors on one corpus with the public evaluation API.

Runnable demonstration of :mod:`scpn_phase_orchestrator.evaluation`: a single
synthetic labelled corpus of pre-transition *event* windows and transition-free
*null* windows is scored by two detectors — a lag-1 autocorrelation detector that
carries real skill and a window-mean detector that carries none — and each is
audited at the same matched false-alarm rate. The auditor reads only the
per-window scores, so the two are judged on identical footing; the skilful one
beats chance and the null one does not, and both verdicts are sealed into
hash-addressed records.

The event windows are an AR(1) process with rising autocorrelation — the textbook
critical-slowing-down signature (Scheffer et al. 2009) — while the null windows
are white noise. Crucially both arms are zero-mean with the same marginal
variance, so the window mean is genuinely uninformative: a detector reading it
*must* land at chance, which is exactly what makes it the honest negative control.

Run with ``python -m bench.honest_auditor_worked_example`` (or execute the file).
It prints a two-line verdict and the sealed record hashes; nothing is written to
disk.
"""

from __future__ import annotations

import json

import numpy as np

from scpn_phase_orchestrator.evaluation import (
    DetectorAudit,
    audit_scoring_detector,
    seal_detector_audit,
)

#: Windows per corpus arm and samples per window — small, enough for a clear signal.
N_EVENTS = 40
N_NULLS = 200
WINDOW = 64
#: AR(1) coefficient of the event windows — the rising-autocorrelation signature.
EVENT_AR1 = 0.8
#: Caller-supplied timestamp; the auditor never reads a wall clock itself.
CAPTURED_AT = "2026-07-07T15:00:00+02:00"


def _ar1_window(rng: np.random.Generator, phi: float, length: int) -> np.ndarray:
    """Return one zero-mean, unit-marginal-variance AR(1) window.

    The innovation scale ``sqrt(1 - phi**2)`` holds the marginal variance at one
    for any ``phi`` in ``(-1, 1)``, so a change in ``phi`` shifts autocorrelation
    without shifting variance — the mean and variance stay uninformative.
    """
    innovation = float(np.sqrt(1.0 - phi**2))
    values = np.empty(length)
    values[0] = rng.normal(0.0, 1.0)
    for index in range(1, length):
        values[index] = phi * values[index - 1] + rng.normal(0.0, innovation)
    return np.asarray(values - values.mean(), dtype=np.float64)


def build_corpus(
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return ``(event_windows, null_windows)`` of a synthetic transition corpus.

    Event windows are AR(1) with autocorrelation ``EVENT_AR1`` (critical slowing
    down); null windows are white noise (``phi = 0``). Both arms are zero-mean and
    unit marginal variance, so only the autocorrelation separates them.
    """
    rng = np.random.default_rng(seed)
    events = [_ar1_window(rng, EVENT_AR1, WINDOW) for _ in range(N_EVENTS)]
    nulls = [_ar1_window(rng, 0.0, WINDOW) for _ in range(N_NULLS)]
    return events, nulls


def lag1_autocorrelation_score(window: np.ndarray) -> float:
    """Lag-1 autocorrelation of a window — the skilful critical-slowing detector."""
    values = np.asarray(window, dtype=float)
    centred = values - values.mean()
    denominator = float(np.dot(centred, centred))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(centred[:-1], centred[1:]) / denominator)


def window_mean_score(window: np.ndarray) -> float:
    """Mean of a window — a no-skill control, since both arms are zero-mean."""
    return float(np.mean(np.asarray(window, dtype=float)))


def audit_both(seed: int = 0) -> dict[str, DetectorAudit]:
    """Audit the autocorrelation and window-mean detectors on one shared corpus."""
    events, nulls = build_corpus(seed)
    skilful = audit_scoring_detector(
        score=lag1_autocorrelation_score,
        event_series=events,
        null_series=nulls,
        detector_name="lag1-autocorrelation",
        target_false_alarm=0.10,
    )
    no_skill = audit_scoring_detector(
        score=window_mean_score,
        event_series=events,
        null_series=nulls,
        detector_name="window-mean-control",
        target_false_alarm=0.10,
    )
    return {"lag1-autocorrelation": skilful, "window-mean-control": no_skill}


def main() -> None:
    """Run the worked audit and print the sealed verdicts."""
    audits = audit_both()
    for name, audit in audits.items():
        record = seal_detector_audit(
            audit, corpus_id="worked-example-synthetic", captured_at=CAPTURED_AT
        )
        print(
            f"{name:22s} "
            f"target_fa={audit.target_false_alarm:.2f} "
            f"achieved_fa={audit.achieved_false_alarm:.3f} "
            f"detect={audit.detection_rate:.3f} "
            f"p={audit.p_value:.4g} "
            f"beats_chance={audit.beats_chance} "
            f"hash={record.content_hash[:12]}"
        )
    print(
        json.dumps(audits["lag1-autocorrelation"].to_record(), indent=2, sort_keys=True)
    )


if __name__ == "__main__":  # pragma: no cover - CLI shell over the tested logic
    main()
