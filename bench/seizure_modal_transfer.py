# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — does the grid modal-growth moat transfer to scalp EEG?

"""Does the grid's modal-growth + fit-quality-gate moat transfer to scalp EEG?

The power-grid detector wins because an under-damped electromechanical mode makes the
cross-bus voltage deviation grow *exponentially* — its ``σ`` is a genuine eigenvalue
real part — and the R²-gate then separates that smooth exponential from a fault's
step-like transient. This runner asks, honestly, whether that same winning form
transfers to scalp-EEG seizure prediction: it scores each pre-onset segment by the
**exponential growth rate of the beta-to-delta band-power ratio trajectory** (the
a-priori seizure feature of :mod:`bench.seizure_detector`, scored with the product
primitive
:func:`~scpn_phase_orchestrator.monitor.grid_modal_growth.fit_gated_growth_rate` in
place of a rank trend), head-to-head against that detector's Kendall-τ rise, through the
identical matched-false-alarm + label-permutation moat, on the same real CHB-MIT
segments.

The honest answer, sealed here, is **no**: the modal-growth score leads none of the six
usable chb01 preictal segments (the fit-quality gate, built to reject non-exponential
transients, rejects the *signal* too, because the preictal rise is not exponential),
below even the a-priori rank-trend. The exponential-growth generative model is
grid-specific — it presupposes a linear instability's eigenvalue, which EEG preictal
dynamics do not exhibit — so the moat's winning form does not carry across. A robustness
grid (window, recency, gate) keeps the modal score at chance at every setting, so the
result is a model mismatch, not a parameter artefact; a shorter-horizon spectral
observation is disclosed as exploratory and explicitly *not* claimed (one subject,
``n=6``, a single forking path that does not survive multiplicity).

References
----------
* Mormann, Andrzejak, Elger & Lehnertz 2007, *Brain* 130:314 — the rigorous,
  above-chance-only standard for preictal features this protocol shares.
* Kundur 1994, *Power System Stability and Control* — small-signal (modal) stability:
  the eigenvalue real part is the growth rate, the model the grid detector is built on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bench.early_warning_domain import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
    calibrate_score_threshold,
    permutation_significance_from_alarms,
)
from bench.seizure_detector import (
    BETA_BAND,
    DELTA_BAND,
    channel_ratio_trajectories,
    segment_rise_score,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.monitor.grid_modal_growth import fit_gated_growth_rate

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]

#: The sealed-artefact identifier.
BENCHMARK = "seizure_modal_transfer"
#: A-priori band-power window in samples (4 s at 256 Hz), a standard beta-band estimate.
WINDOW = 1024
#: A-priori hop in samples (2 s, 50 % overlap).
STEP = 512
#: Recency weighting, carried unchanged from the certified grid operating point.
RECENCY_TOP = 3.0
#: The certified grid fit-quality gate, tested here for transfer.
GATE_R2 = 0.5
#: The a-priori analysis-segment length in seconds (baseline + horizon), the codebase
#: default in :mod:`bench.early_warning_leadtime_eeg`.
SEGMENT_SECONDS = 900.0
#: A shorter horizon disclosed as an exploratory, explicitly unclaimed observation.
EXPLORATORY_SEGMENT_SECONDS = 600.0

__all__ = [
    "BENCHMARK",
    "detector_record",
    "modal_rise_score",
    "modal_transfer_payload",
    "modal_transfer_verdict",
]


def modal_rise_score(
    signals: FloatArray,
    *,
    rate: float,
    window: int = WINDOW,
    step: int = STEP,
    aggregation: str = "focal",
    recency_top: float = RECENCY_TOP,
    r2_gate: float = 0.0,
    low_band: tuple[float, float] = DELTA_BAND,
    high_band: tuple[float, float] = BETA_BAND,
) -> float:
    """Return one segment's modal-growth rise score of the beta/delta ratio.

    The beta-to-delta band-power ratio trajectory (per channel, via
    :func:`~bench.seizure_detector.channel_ratio_trajectories`) is scored by the
    exponential growth rate of its envelope
    (:func:`~scpn_phase_orchestrator.monitor.grid_modal_growth.fit_gated_growth_rate`) —
    the grid detector's statistic, in place of the rank trend. ``"mean"`` scores the
    whole-head channel-averaged trajectory; ``"focal"`` the most-growing channel. An
    ``r2_gate`` above zero applies the fit-quality gate.

    Parameters
    ----------
    signals : FloatArray
        The segment's multichannel EEG, shape ``(channels, samples)``.
    rate : float
        Sampling rate in Hz.
    window, step : int
        Band-power analysis window length and hop in samples.
    aggregation : str
        ``"mean"`` (whole head) or ``"focal"`` (most-growing channel).
    recency_top : float
        Recency weighting passed to the growth-rate fit.
    r2_gate : float
        Fit-quality gate in ``[0, 1]``; ``0.0`` disables it.
    low_band, high_band : tuple[float, float]
        The delta and beta bands (Hz).

    Returns
    -------
    float
        The segment's modal-growth rise score.

    Raises
    ------
    ValueError
        If ``aggregation`` is neither ``"mean"`` nor ``"focal"``.
    """
    trajectories = channel_ratio_trajectories(
        signals,
        rate=rate,
        window=window,
        step=step,
        low_band=low_band,
        high_band=high_band,
    )
    trajectory_rate = rate / step
    if aggregation == "mean":
        return fit_gated_growth_rate(
            trajectories.mean(axis=0),
            rate=trajectory_rate,
            recency_top=recency_top,
            r2_gate=r2_gate,
        )
    if aggregation == "focal":
        return max(
            fit_gated_growth_rate(
                channel, rate=trajectory_rate, recency_top=recency_top, r2_gate=r2_gate
            )
            for channel in trajectories
        )
    raise ValueError(f"aggregation must be 'mean' or 'focal', got {aggregation!r}")


def detector_record(
    transition_scores: Sequence[float],
    null_scores: Sequence[float],
    *,
    detector: str,
    aggregation: str,
    target_fa: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> dict[str, object]:
    """Score one detector's transition/null segment scores through the shared moat.

    The threshold is calibrated on the null scores to a matched false alarm; a segment
    alarms when its score meets it; and the transition alarm count is tested by the
    label-permutation core — the identical moat the grid and spectral detectors use, so
    the p-value is directly comparable.

    Parameters
    ----------
    transition_scores, null_scores : sequence of float
        The per-segment scores of the preictal and interictal segments.
    detector : str
        The detector label sealed into the record.
    aggregation : str
        The channel aggregation the scores were computed under.
    target_fa : float
        Matched false-alarm rate the threshold is held at or below.
    n_permutations : int
        Number of random relabellings behind the p-value.
    seed : int
        Seed of the resampling, so the p-value is reproducible.

    Returns
    -------
    dict
        A JSON-safe record: detector, aggregation, threshold, achieved false alarm, lead
        count, transition count, and the permutation-significance sub-record.

    Raises
    ------
    ValueError
        If either score set is empty.
    """
    if not transition_scores:
        raise ValueError("transition_scores must not be empty")
    if not null_scores:
        raise ValueError("null_scores must not be empty")
    threshold = calibrate_score_threshold(list(null_scores), target_fa=target_fa)
    transition_alarms = [score >= threshold for score in transition_scores]
    null_alarms = [score >= threshold for score in null_scores]
    significance = permutation_significance_from_alarms(
        transition_alarms, null_alarms, n_permutations=n_permutations, seed=seed
    )
    return {
        "detector": detector,
        "aggregation": aggregation,
        "score_threshold": float(threshold),
        "achieved_false_alarm": float(np.mean(null_alarms)),
        "led": int(sum(transition_alarms)),
        "n_transitions": len(transition_scores),
        "significance": significance.to_audit_record(),
    }


def modal_transfer_verdict(
    spectral: Mapping[str, object],
    modal: Mapping[str, object],
    gated: Mapping[str, object],
) -> str:
    """Return a one-line honest verdict of the transfer, from the focal records.

    Parameters
    ----------
    spectral : mapping
        The a-priori Kendall-τ spectral detector's focal record.
    modal : mapping
        The modal-growth detector's focal record (gate off).
    gated : mapping
        The modal-growth detector's focal record (fit-quality gate on).

    Returns
    -------
    str
        A factual sentence contrasting the modal-growth transfer with the spectral rise.
    """
    n = spectral["n_transitions"]
    return (
        f"The grid modal-growth moat does not transfer to scalp EEG: on the same "
        f"chb01 segments the exponential-growth score leads {modal['led']}/{n} "
        f"preictal segments and the fit-quality gate {gated['led']}/{n} (it rejects a "
        f"non-exponential preictal rise as a transient), below the a-priori "
        f"Kendall-τ spectral trend at {spectral['led']}/{n}, itself at chance. The "
        f"exponential-growth model is grid-specific — it presupposes a linear "
        f"instability's eigenvalue that EEG preictal dynamics do not exhibit."
    )


def modal_transfer_payload(
    *,
    spectral_records: Mapping[str, Mapping[str, object]],
    modal_records: Mapping[str, Mapping[str, object]],
    gated_records: Mapping[str, Mapping[str, object]],
    robustness: Sequence[Mapping[str, object]],
    exploratory: Mapping[str, object],
    corpus: Mapping[str, object],
    operating_point: Mapping[str, object],
    target_fa: float,
    n_permutations: int,
    seed: int,
) -> dict[str, object]:
    """Assemble and hash-seal the EEG modal-growth transfer result.

    Parameters
    ----------
    spectral_records, modal_records, gated_records : mapping
        The ``"mean"``/``"focal"`` records of the a-priori spectral detector, the
        modal-growth score, and the fit-quality-gated modal-growth score.
    robustness : sequence of mapping
        The modal focal lead count across a window/recency/gate grid — the disclosure
        that the failure is not a parameter artefact.
    exploratory : mapping
        The shorter-horizon spectral observation, explicitly flagged as unclaimed.
    corpus : mapping
        Corpus metadata (source, transition and null counts, sampling rate, segment).
    operating_point : mapping
        The a-priori window, step, recency weighting, and gate under test.
    target_fa : float
        The matched false-alarm rate every detector is calibrated to.
    n_permutations : int
        Number of random relabellings behind every p-value.
    seed : int
        Seed of the resampling.

    Returns
    -------
    dict
        The JSON-safe payload with a ``content_hash`` field sealing the record.
    """
    payload: dict[str, object] = {
        "benchmark": BENCHMARK,
        "question": (
            "Does the grid modal-growth + fit-quality-gate moat — exponential envelope "
            "growth rate, focal, R²-gated — transfer to scalp-EEG seizure prediction?"
        ),
        "corpus": dict(corpus),
        "operating_point": dict(operating_point),
        "target_false_alarm": target_fa,
        "n_permutations": n_permutations,
        "seed": seed,
        "spectral_kendall_tau": {k: dict(v) for k, v in spectral_records.items()},
        "modal_growth": {k: dict(v) for k, v in modal_records.items()},
        "modal_growth_r2_gated": {k: dict(v) for k, v in gated_records.items()},
        "robustness_modal_focal": [dict(row) for row in robustness],
        "exploratory_shorter_horizon": dict(exploratory),
        "verdict": modal_transfer_verdict(
            spectral_records["focal"], modal_records["focal"], gated_records["focal"]
        ),
    }
    payload["content_hash"] = canonical_record_hash(payload)
    return payload


def _load_segments(  # pragma: no cover - EDF I/O shell over the tested pure core
    data_dir: str,
    *,
    rate: float,
    segment_seconds: float,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Load the chb01 preictal and interictal null segments from the EDF corpus."""
    from pathlib import Path

    from bench.early_warning_leadtime_eeg import (
        INTERICTAL_RECORDS,
        SEIZURE_ONSETS_S,
        load_edf_channels,
    )

    segment = int(segment_seconds * rate)
    preictal: list[FloatArray] = []
    for record, onset in SEIZURE_ONSETS_S.items():
        start = onset * int(rate) - segment
        if start < 0:
            continue
        signals = load_edf_channels(str(Path(data_dir) / f"{record}.edf"))
        preictal.append(signals[:, start : onset * int(rate)])
    nulls: list[FloatArray] = []
    for record in INTERICTAL_RECORDS:
        signals = load_edf_channels(str(Path(data_dir) / f"{record}.edf"))
        for start in range(0, signals.shape[1] - segment + 1, segment):
            nulls.append(signals[:, start : start + segment])
    return preictal, nulls


def evaluate_modal_transfer(  # pragma: no cover - I/O shell over the tested pure core
    data_dir: str,
    *,
    rate: float = 256.0,
    target_fa: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> dict[str, object]:
    """Run the transfer head-to-head over the real chb01 corpus and seal the result."""
    preictal, nulls = _load_segments(
        data_dir, rate=rate, segment_seconds=SEGMENT_SECONDS
    )

    def _records(
        scorer: Callable[[FloatArray, str], float], label: str
    ) -> dict[str, Mapping[str, object]]:
        records: dict[str, Mapping[str, object]] = {}
        for aggregation in ("mean", "focal"):
            t_scores = [scorer(s, aggregation) for s in preictal]
            n_scores = [scorer(s, aggregation) for s in nulls]
            records[aggregation] = detector_record(
                t_scores,
                n_scores,
                detector=label,
                aggregation=aggregation,
                target_fa=target_fa,
                n_permutations=n_permutations,
                seed=seed,
            )
        return records

    spectral = _records(
        lambda s, agg: segment_rise_score(
            s, rate=rate, window=WINDOW, step=STEP, aggregation=agg
        ),
        "spectral_beta_delta_rise",
    )
    modal = _records(
        lambda s, agg: modal_rise_score(s, rate=rate, aggregation=agg, r2_gate=0.0),
        "modal_growth_rise",
    )
    gated = _records(
        lambda s, agg: modal_rise_score(s, rate=rate, aggregation=agg, r2_gate=GATE_R2),
        "modal_growth_rise_r2_gated",
    )

    robustness: list[Mapping[str, object]] = []
    for window in (512, 1024, 2048):
        for recency in (1.0, 3.0):
            for gate in (0.0, GATE_R2):
                scores_t = [
                    modal_rise_score(
                        s,
                        rate=rate,
                        window=window,
                        step=window // 2,
                        aggregation="focal",
                        recency_top=recency,
                        r2_gate=gate,
                    )
                    for s in preictal
                ]
                scores_n = [
                    modal_rise_score(
                        s,
                        rate=rate,
                        window=window,
                        step=window // 2,
                        aggregation="focal",
                        recency_top=recency,
                        r2_gate=gate,
                    )
                    for s in nulls
                ]
                record = detector_record(
                    scores_t,
                    scores_n,
                    detector="modal_growth_rise",
                    aggregation="focal",
                    target_fa=target_fa,
                    n_permutations=n_permutations,
                    seed=seed,
                )
                robustness.append(
                    {
                        "window": window,
                        "recency_top": recency,
                        "r2_gate": gate,
                        "led": record["led"],
                        "achieved_false_alarm": record["achieved_false_alarm"],
                    }
                )

    short_pre, short_nulls = _load_segments(
        data_dir, rate=rate, segment_seconds=EXPLORATORY_SEGMENT_SECONDS
    )
    exploratory_record = detector_record(
        [
            segment_rise_score(
                s, rate=rate, window=WINDOW, step=STEP, aggregation="focal"
            )
            for s in short_pre
        ],
        [
            segment_rise_score(
                s, rate=rate, window=WINDOW, step=STEP, aggregation="focal"
            )
            for s in short_nulls
        ],
        detector="spectral_beta_delta_rise",
        aggregation="focal",
        target_fa=target_fa,
        n_permutations=n_permutations,
        seed=seed,
    )

    corpus = {
        "source": "CHB-MIT scalp-EEG (Shoeb 2009), subject chb01",
        "n_transitions": len(preictal),
        "n_nulls": len(nulls),
        "sampling_rate_hz": rate,
        "segment_seconds": SEGMENT_SECONDS,
    }
    operating_point = {
        "window_samples": WINDOW,
        "step_samples": STEP,
        "recency_top": RECENCY_TOP,
        "r2_gate": GATE_R2,
        "feature": "beta/delta band-power ratio trajectory",
    }
    exploratory = {
        "segment_seconds": EXPLORATORY_SEGMENT_SECONDS,
        "spectral_focal": exploratory_record,
        "claim": (
            "not claimed — one subject, n=6, a single horizon forking path whose p "
            "does not survive multiplicity; needs pre-registration and more subjects"
        ),
    }
    return modal_transfer_payload(
        spectral_records=spectral,
        modal_records=modal,
        gated_records=gated,
        robustness=robustness,
        exploratory=exploratory,
        corpus=corpus,
        operating_point=operating_point,
        target_fa=target_fa,
        n_permutations=n_permutations,
        seed=seed,
    )


def main() -> None:  # pragma: no cover - CLI shell
    """Run the transfer head-to-head and write the sealed artefact."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="directory of chb01 EDF recordings")
    parser.add_argument("output", help="path for the sealed JSON artefact")
    args = parser.parse_args()

    payload = evaluate_modal_transfer(args.data_dir)
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}: {payload['verdict']}")


if __name__ == "__main__":  # pragma: no cover - CLI shell
    main()
