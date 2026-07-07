#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real-detector head-to-head through the honest auditor

"""Audit two real early-warning detectors head-to-head through the public auditor.

This is the integration proof for :mod:`scpn_phase_orchestrator.evaluation`: the
productised auditor scores the *real* detector code — the SCPN modal
envelope-growth detector
(:func:`~scpn_phase_orchestrator.monitor.grid_modal_growth.modal_growth_score`) and
a published competitor, the Dakos et al. 2008 AR(1)/Kendall-τ rising-autocorrelation
trend (:func:`bench.competitor_ar1_kendall.ar1_trend_tau`) — from their per-segment
scores alone, at a matched false-alarm rate, and returns a sealed verdict for each.

The corpora are **synthetic but physically honest**, not field PMU data (real
measured-damping data is owner-gated and out of scope here). Two regimes are built
so the comparison is not cherry-picked:

* an **oscillatory instability** — a growing complex mode (positive eigenvalue real
  part) versus a damped one, where the envelope-growth estimator is the
  magnitude-correct one; and
* a **monotone instability** — a rising lag-one autocorrelation (an AR(1) approaching
  unit root, no oscillation) versus white noise, where the AR(1)/Kendall-τ trend is
  the magnitude-correct one.

Auditing both detectors on both regimes yields a 2×2 skill table. With a clearly
skilful detector on a corpus this size the label-permutation p-value **saturates at
its floor** — every cell "beats chance" decisively, so the p-value alone cannot
rank the detectors. The honest discriminator is then the **detection rate at the
matched false alarm** (the skill magnitude the auditor reports alongside the
p-value), and *that* is regime-dependent: the envelope-growth detector leads on the
oscillatory regime, the AR(1)/Kendall-τ competitor leads on the monotone regime —
exactly the eigenvalue-regime-map finding, adjudicated without bias by a single
matched-false-alarm + permutation test. The point is not that one detector wins
overall; it is that the auditor surfaces *which* detector has more skill *where*,
objectively. Real field data would replace the synthetic corpora without changing
the auditor.

Run with ``python -m bench.auditor_detector_head_to_head``.
"""

from __future__ import annotations

import json

import numpy as np
from numpy.typing import NDArray

from bench.competitor_ar1_kendall import ar1_trend_tau
from scpn_phase_orchestrator.evaluation import (
    DetectorAudit,
    audit_detector,
    seal_detector_audit,
)
from scpn_phase_orchestrator.monitor.grid_modal_growth import (
    cross_bus_deviation,
    modal_growth_score,
)

FloatArray = NDArray[np.float64]

#: PMU-like sampling rate and segment geometry.
RATE_HZ = 50.0
SEGMENT_SAMPLES = 200
N_BUSES = 8
#: Inter-area oscillation frequency of the complex mode, in hertz.
MODE_HZ = 0.3
#: Segments per corpus arm.
N_EVENTS = 40
N_NULLS = 120
#: AR(1)/Kendall-τ sliding-window geometry, in samples.
COMPETITOR_WINDOW = 50
COMPETITOR_STEP = 12
#: Caller-supplied timestamp; the auditor never reads a wall clock itself.
CAPTURED_AT = "2026-07-07T18:00:00+02:00"


def oscillatory_block(rng: np.random.Generator, growth_rate: float) -> FloatArray:
    """Return one ``(buses, samples)`` block of a growing/damped complex mode.

    A single collective oscillation at :data:`MODE_HZ` whose amplitude scales as
    ``exp(growth_rate · t)`` — an unstable mode for ``growth_rate > 0``, a damped one
    for ``growth_rate < 0`` — projected onto the buses with random per-bus weights and
    phases, on a nominal 1.0 pu base plus measurement noise.
    """
    t = np.arange(SEGMENT_SAMPLES) / RATE_HZ
    shape = rng.normal(0.0, 1.0, N_BUSES)
    phase = rng.uniform(0.0, 2.0 * np.pi, N_BUSES)
    envelope = 0.01 * np.exp(growth_rate * t)
    oscillation = np.cos(2.0 * np.pi * MODE_HZ * t[None, :] + phase[:, None])
    signal = shape[:, None] * envelope[None, :] * oscillation
    noise = rng.normal(0.0, 0.002, (N_BUSES, SEGMENT_SAMPLES))
    return np.asarray(1.0 + signal + noise, dtype=np.float64)


def monotone_block(rng: np.random.Generator, phi_end: float) -> FloatArray:
    """Return one ``(buses, samples)`` block of an AR(1) with rising autocorrelation.

    Each bus is an AR(1) process whose coefficient ramps linearly from zero to
    ``phi_end`` across the segment — critical slowing down without oscillation for a
    high ``phi_end``, white noise for ``phi_end == 0`` — on a nominal 1.0 pu base. The
    innovation scale holds the marginal variance roughly constant, so only the
    autocorrelation, not the amplitude, separates the arms.
    """
    block = np.empty((N_BUSES, SEGMENT_SAMPLES))
    phi = np.linspace(0.0, phi_end, SEGMENT_SAMPLES)
    innovation = np.sqrt(np.maximum(1.0e-6, 1.0 - phi**2))
    for bus in range(N_BUSES):
        values = np.empty(SEGMENT_SAMPLES)
        values[0] = rng.normal(0.0, 1.0)
        for index in range(1, SEGMENT_SAMPLES):
            values[index] = phi[index] * values[index - 1] + rng.normal(
                0.0, innovation[index]
            )
        block[bus] = 1.0 + 0.01 * values
    return np.asarray(block, dtype=np.float64)


def scpn_growth_score(block: FloatArray) -> float:
    """Score a block with the real SCPN modal envelope-growth detector (focal σ)."""
    return modal_growth_score(block, rate=RATE_HZ, aggregation="focal")


def competitor_ar1_score(block: FloatArray) -> float:
    """Score a block with the real AR(1)/Kendall-τ competitor on its envelope."""
    return ar1_trend_tau(
        cross_bus_deviation(block), window=COMPETITOR_WINDOW, step=COMPETITOR_STEP
    )


def oscillatory_corpus(
    seed: int = 0,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Return ``(event_blocks, null_blocks)`` for the oscillatory-instability regime."""
    rng = np.random.default_rng(seed)
    events = [
        oscillatory_block(rng, float(rate)) for rate in rng.uniform(0.4, 0.8, N_EVENTS)
    ]
    nulls = [
        oscillatory_block(rng, float(rate))
        for rate in rng.uniform(-0.05, 0.05, N_NULLS)
    ]
    return events, nulls


def monotone_corpus(
    seed: int = 1,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Return ``(event_blocks, null_blocks)`` for the monotone-instability regime."""
    rng = np.random.default_rng(seed)
    events = [
        monotone_block(rng, float(phi)) for phi in rng.uniform(0.9, 0.98, N_EVENTS)
    ]
    nulls = [monotone_block(rng, 0.0) for _ in range(N_NULLS)]
    return events, nulls


def audit_regime(
    events: list[FloatArray],
    nulls: list[FloatArray],
    *,
    n_permutations: int = 10000,
) -> dict[str, DetectorAudit]:
    """Audit both real detectors on one regime's corpus through the public auditor."""
    scpn = audit_detector(
        event_scores=[scpn_growth_score(block) for block in events],
        null_scores=[scpn_growth_score(block) for block in nulls],
        detector_name="scpn-modal-growth",
        n_permutations=n_permutations,
    )
    competitor = audit_detector(
        event_scores=[competitor_ar1_score(block) for block in events],
        null_scores=[competitor_ar1_score(block) for block in nulls],
        detector_name="ar1-kendall-tau",
        n_permutations=n_permutations,
    )
    return {"scpn-modal-growth": scpn, "ar1-kendall-tau": competitor}


def run(*, n_permutations: int = 10000) -> dict[str, dict[str, DetectorAudit]]:
    """Audit both detectors on both regimes; return ``{regime: {detector: audit}}``."""
    return {
        "oscillatory": audit_regime(
            *oscillatory_corpus(), n_permutations=n_permutations
        ),
        "monotone": audit_regime(*monotone_corpus(), n_permutations=n_permutations),
    }


def main() -> None:
    """Run the 2×2 head-to-head and print the sealed skill table."""
    table = run()
    for regime, audits in table.items():
        for name, audit in audits.items():
            record = seal_detector_audit(
                audit,
                corpus_id=f"synthetic-{regime}",
                captured_at=CAPTURED_AT,
            )
            print(
                f"{regime:12s} {name:20s} "
                f"achieved_fa={audit.achieved_false_alarm:.3f} "
                f"detect={audit.detection_rate:.3f} "
                f"p={audit.p_value:.4g} "
                f"beats_chance={audit.beats_chance} "
                f"hash={record.content_hash[:12]}"
            )
    # The p-value saturates when both beat chance, so rank by the skill magnitude:
    # detection rate at the matched false alarm.
    more_skilful = {
        regime: max(
            audits.values(), key=lambda audit: audit.detection_rate
        ).detector_name
        for regime, audits in table.items()
    }
    print(json.dumps(more_skilful, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI shell over the tested logic
    main()
