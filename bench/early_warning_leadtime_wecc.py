# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — WECC 240-bus cross-dataset modal-growth evaluation

"""Cross-dataset evaluation of the certified modal-growth detector on WECC 240-bus.

This module runs the statistical leg of the cross-dataset generalisation
evaluation: the *frozen* detector shape from
the PSML certification (:mod:`bench.grid_modal_head_to_head`) against the 13
forced-oscillation cases of the 2021 IEEE-NASPI Oscillation Source Location
Contest — synthetic PMU exports of the reduced WECC 240-bus system (NREL model,
OSL Committee scenario design), mirrored openly in the UTK oscillation
test-case library. Raw data stays citation-only and is never committed.

The protocol is pre-registered in the internal plan (appendix A.16) BEFORE the
first detector run on this corpus:

* Ground truth from the contest solution key, independent of any detector
  output: every case introduces its forced oscillation at
  :data:`FORCING_SECONDS`; every documented masking disturbance (faults, one
  line trip, forcing-frequency transitions) happens at 26 s or later.
* Null windows END at or before :data:`NULL_END_SECONDS` — one second clear of
  the earliest documented disturbance. This replaces the 60 s ISO-NE transition
  guard: that guard protected against onset-ESTIMATION error, and here the
  forcing start is exact by simulation design (disclosed protocol revision).
* The scored early-warning region is ``(FORCING_SECONDS, onset_est]`` where
  ``onset_est`` is the in-band envelope onset under the shared ISO-NE rule with
  an ambient-only baseline of :data:`BASELINE_SECONDS`. Alarms at or before the
  forcing start can never count as detections; a case whose transition region
  holds no window reports ``led=False`` with the reason on the record.
* **Frozen transfer** re-uses the PSML operating point verbatim;
  **local calibration** re-uses the frozen shape with the threshold
  calibrated at a matched false alarm on pooled null windows only. Both
  branches are reported whatever the outcome; no variant search runs here.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bench.early_warning_domain import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    PermutationSignificance,
    permutation_significance_from_alarms,
)
from bench.early_warning_leadtime_isone import (
    CYCLES_PER_WINDOW,
    FROZEN_PSML_STEP_SECONDS,
    FROZEN_PSML_THRESHOLD,
    FROZEN_PSML_WINDOW_SECONDS,
    MIN_CHANNELS,
    STEP_FRACTION,
    CaseScores,
    FrozenTransferCase,
    LocalCalibrationResult,
    estimate_onset,
    window_scores,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.monitor.grid_modal_growth import (
    DEFAULT_AGGREGATION,
    DEFAULT_RECENCY_TOP,
    FloatArray,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

__all__ = [
    "BASELINE_SECONDS",
    "DOCUMENTED_MODES",
    "FORCING_SECONDS",
    "NULL_END_SECONDS",
    "CaseDetection",
    "DetectionResult",
    "case_records",
    "case_scores",
    "evaluate_detection",
    "frozen_transfer_case",
    "local_calibration_payload",
    "split_scores",
    "voltage_matrix",
]

#: Documented lowest forcing fundamental per contest case, hertz (solution key).
DOCUMENTED_MODES: dict[str, float] = {
    "WECC_case1": 0.82,
    "WECC_case2": 1.19,
    "WECC_case3": 0.379,
    "WECC_case4": 0.379,
    "WECC_case5": 0.68,
    "WECC_case6": 1.27,
    "WECC_case7": 0.379,
    "WECC_case8": 0.614,
    "WECC_case9": 0.762,
    "WECC_case10": 0.614,
    "WECC_case11": 0.614,
    "WECC_case12": 0.37,
    "WECC_case13": 0.614,
}

#: Forced-oscillation start, seconds — uniform across the corpus (solution key).
FORCING_SECONDS = 30.0

#: Null windows must END at or before this time, seconds. One second clear of
#: the earliest documented disturbance (t=26 s) — pre-registered in A.16.
NULL_END_SECONDS = 25.0

#: Ambient-only baseline for the in-band onset estimate, seconds (the 90 s
#: records leave no room for the 30 s ISO-NE default; disclosed).
BASELINE_SECONDS = 25.0


def voltage_matrix(path: str | Path) -> tuple[float, FloatArray]:
    """Load the clean bus-voltage matrix of one TSAT synthetic PMU export.

    The export is a whitespace-separated text block whose first line names the
    columns in single quotes (``'Time'`` then ``'bus|name'`` per channel) and
    whose remaining lines carry one sample per row. TSAT writes a network
    discontinuity instant twice (pre- and post-event solution at the same
    timestamp); the first row of an exact-duplicate pair is dropped so the
    post-event solution survives, and the axis must then be strictly
    increasing (amendment A.16.2).

    Parameters
    ----------
    path : str | Path
        The case's ``BusVolMag.txt`` export.

    Returns
    -------
    tuple[float, FloatArray]
        The sampling rate in hertz and the matrix of clean channels, shape
        ``(channels, samples)``.

    Raises
    ------
    ValueError
        If the header does not start with a ``'Time'`` column, the numeric
        block is malformed or disagrees with the header, the time axis is not
        strictly increasing, fewer than :data:`MIN_CHANNELS` clean channels
        remain, or the clean channels collapse to fewer than
        :data:`MIN_CHANNELS` distinct signals.
    """
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{source.name}: needs a header row and >=2 sample rows")
    names = re.findall(r"'([^']*)'", lines[0])
    if not names or names[0] != "Time":
        raise ValueError(f"{source.name}: first header column must be 'Time'")
    try:
        data = np.loadtxt(io.StringIO("\n".join(lines[1:])), dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"{source.name}: malformed numeric block: {error}") from None
    if data.ndim != 2 or data.shape[1] != len(names):
        raise ValueError(
            f"{source.name}: numeric block shape {data.shape} disagrees with "
            f"{len(names)} header columns"
        )
    times = data[:, 0]
    duplicate = np.zeros(times.size, dtype=bool)
    duplicate[:-1] = np.diff(times) == 0.0
    if bool(duplicate.any()):
        # TSAT writes the discontinuity instant twice (pre- and post-event
        # solution at the same timestamp); keep the post-event row.
        data = data[~duplicate]
        times = data[:, 0]
    diffs = np.diff(times)
    if not bool(np.all(diffs > 0.0)):
        raise ValueError(f"{source.name}: time axis must be strictly increasing")
    rate = 1.0 / float(np.median(diffs))
    channels = [
        row
        for row in data[:, 1:].T
        if bool(np.all(np.isfinite(row))) and float(np.std(row)) > 0.0
    ]
    if len(channels) < MIN_CHANNELS:
        raise ValueError(
            f"{source.name}: only {len(channels)} clean channels of "
            f"{len(names) - 1}; need at least {MIN_CHANNELS}"
        )
    matrix = np.vstack(channels)
    distinct = np.unique(matrix, axis=0).shape[0]
    if distinct < MIN_CHANNELS:
        raise ValueError(
            f"{source.name}: clean channels collapse to {distinct} distinct "
            "signal(s); the focal detector needs per-bus structure"
        )
    return rate, matrix


def split_scores(
    ends: FloatArray,
    scores: FloatArray,
    *,
    onset_seconds: float,
) -> tuple[FloatArray, FloatArray]:
    """Split window scores into ambient nulls and the early-warning region.

    Null windows end at or before :data:`NULL_END_SECONDS`; scored transition
    windows end inside ``(FORCING_SECONDS, onset_seconds]``. Windows in the
    guard gap ``(NULL_END_SECONDS, FORCING_SECONDS]`` or after the estimated
    onset belong to neither set.

    Parameters
    ----------
    ends : FloatArray
        Window end times in seconds.
    scores : FloatArray
        Modal growth score of each window.
    onset_seconds : float
        Estimated in-band envelope onset in seconds.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Null scores and transition scores.

    Raises
    ------
    ValueError
        If ``ends`` and ``scores`` differ in shape, or ``onset_seconds`` is
        not a positive finite number.
    """
    end_times = np.asarray(ends, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if end_times.shape != values.shape:
        raise ValueError("ends and scores must have identical shape")
    if isinstance(onset_seconds, bool) or not isinstance(onset_seconds, (int, float)):
        raise ValueError("onset_seconds must be a positive finite number")
    if not np.isfinite(onset_seconds) or onset_seconds <= 0.0:
        raise ValueError("onset_seconds must be a positive finite number")
    null_mask = end_times <= NULL_END_SECONDS
    transition_mask = (end_times > FORCING_SECONDS) & (end_times <= onset_seconds)
    return values[null_mask], values[transition_mask]


def _estimate_case_onset(
    matrix: FloatArray,
    *,
    rate: float,
    mode_hz: float,
    allow_unresolved_onset: bool,
) -> float:
    """Estimate the in-band onset, optionally absorbing the fail-closed signal.

    Parameters
    ----------
    matrix : FloatArray
        Clean channel matrix, shape ``(channels, samples)``.
    rate : float
        Sampling rate in hertz.
    mode_hz : float
        Documented forcing fundamental in hertz.
    allow_unresolved_onset : bool
        When true, a capture without a sustained in-band exceedance returns
        ``nan`` instead of raising — the pre-registered ``led=False`` path
        for a case whose oscillation never surfaces in the envelope.

    Returns
    -------
    float
        Onset time in seconds, or ``nan`` when unresolved and allowed.

    Raises
    ------
    ValueError
        If the onset is unresolved and ``allow_unresolved_onset`` is false,
        or the estimation inputs are invalid.
    """
    try:
        return estimate_onset(
            matrix,
            rate=rate,
            mode_hz=mode_hz,
            baseline_seconds=BASELINE_SECONDS,
            search_start_seconds=FORCING_SECONDS,
        )
    except ValueError as error:
        if allow_unresolved_onset and "no separable onset" in str(error):
            return float("nan")
        raise


def _split_with_onset(
    ends: FloatArray,
    scores: FloatArray,
    onset: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Split scores under the WECC segmentation, honouring an unresolved onset.

    Parameters
    ----------
    ends : FloatArray
        Window end times in seconds.
    scores : FloatArray
        Modal growth score of each window.
    onset : float
        Estimated onset in seconds, or ``nan`` when unresolved (the
        transition region is then empty by construction).

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray]
        Null scores, transition scores, and transition window end times.
    """
    end_times = np.asarray(ends, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if np.isnan(onset):
        empty = values[:0]
        return values[end_times <= NULL_END_SECONDS], empty, end_times[:0]
    nulls, transitions = split_scores(end_times, values, onset_seconds=onset)
    transition_mask = (end_times > FORCING_SECONDS) & (end_times <= onset)
    return nulls, transitions, end_times[transition_mask]


@dataclass(frozen=True)
class CaseDetection:
    """Post-forcing scoring windows of one case, for the secondary branch.

    Attributes
    ----------
    case_id : str
        Corpus case identifier.
    detection_scores : FloatArray
        Scores of every window ending after :data:`FORCING_SECONDS`.
    detection_ends : FloatArray
        End times of those windows, seconds.
    """

    case_id: str
    detection_scores: FloatArray
    detection_ends: FloatArray


def case_records(
    case_id: str, path: str | Path, *, allow_unresolved_onset: bool = False
) -> tuple[CaseScores, CaseDetection]:
    """Prepare one contest case: load, estimate the onset, score, and split.

    The local-calibration configuration is derived a priori from the documented forcing
    fundamental: the window is :data:`CYCLES_PER_WINDOW` cycles of ``f0`` and
    the step is :data:`STEP_FRACTION` of the window. Nothing here reads the
    scores before fixing the segmentation. The onset search is pinned at
    :data:`FORCING_SECONDS` (amendment A.16.1) so smoothing smear cannot pull
    the estimate before the true forcing start.

    Parameters
    ----------
    case_id : str
        A key of :data:`DOCUMENTED_MODES`.
    path : str | Path
        The case's ``BusVolMag.txt`` export.
    allow_unresolved_onset : bool
        When true, a case without a sustained in-band envelope onset is
        returned with ``onset_seconds = nan`` and an empty transition region
        (the pre-registered ``led=False`` path) instead of raising.

    Returns
    -------
    tuple[CaseScores, CaseDetection]
        The shared cross-dataset case record (primary early-warning branch) and the
        post-forcing windows of the same scoring pass (secondary detection
        branch).

    Raises
    ------
    ValueError
        If ``case_id`` is not in the fixed corpus, or the capture fails the
        channel, onset, or window validations.
    """
    if case_id not in DOCUMENTED_MODES:
        raise ValueError(
            f"case_id must be one of {sorted(DOCUMENTED_MODES)}, got {case_id!r}"
        )
    mode_hz = DOCUMENTED_MODES[case_id]
    rate, matrix = voltage_matrix(path)
    onset = _estimate_case_onset(
        matrix,
        rate=rate,
        mode_hz=mode_hz,
        allow_unresolved_onset=allow_unresolved_onset,
    )
    window_seconds = CYCLES_PER_WINDOW / mode_hz
    step_seconds = window_seconds * STEP_FRACTION
    ends, scores = window_scores(
        matrix, rate=rate, window_seconds=window_seconds, step_seconds=step_seconds
    )
    nulls, transitions, transition_ends = _split_with_onset(ends, scores, onset)
    end_times = np.asarray(ends, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    detection_mask = end_times > FORCING_SECONDS
    prepared = CaseScores(
        case_id=case_id,
        mode_hz=mode_hz,
        rate_hz=rate,
        n_channels=int(matrix.shape[0]),
        onset_seconds=onset,
        window_seconds=window_seconds,
        null_scores=nulls,
        transition_scores=transitions,
        transition_ends=transition_ends,
    )
    detection = CaseDetection(
        case_id=case_id,
        detection_scores=values[detection_mask],
        detection_ends=end_times[detection_mask],
    )
    return prepared, detection


def case_scores(
    case_id: str, path: str | Path, *, allow_unresolved_onset: bool = False
) -> CaseScores:
    """Prepare the primary early-warning record of one contest case.

    A convenience wrapper over :func:`case_records` returning only the shared
    cross-dataset case record.

    Parameters
    ----------
    case_id : str
        A key of :data:`DOCUMENTED_MODES`.
    path : str | Path
        The case's ``BusVolMag.txt`` export.
    allow_unresolved_onset : bool
        Forwarded to :func:`case_records`.

    Returns
    -------
    CaseScores
        The prepared case (the shared cross-dataset case record).

    Raises
    ------
    ValueError
        If ``case_id`` is not in the fixed corpus, or the capture fails the
        channel, onset, or window validations.
    """
    return case_records(case_id, path, allow_unresolved_onset=allow_unresolved_onset)[0]


def frozen_transfer_case(
    case_id: str, path: str | Path, *, allow_unresolved_onset: bool = False
) -> FrozenTransferCase:
    """Run the frozen-transfer branch on one case: frozen PSML operating point verbatim.

    Parameters
    ----------
    case_id : str
        A key of :data:`DOCUMENTED_MODES`.
    path : str | Path
        The case's ``BusVolMag.txt`` export.
    allow_unresolved_onset : bool
        When true, a case without a sustained in-band envelope onset reports
        an empty transition region instead of raising.

    Returns
    -------
    FrozenTransferCase
        Crossing counts at the frozen threshold in ambient and transition
        windows under the WECC segmentation.

    Raises
    ------
    ValueError
        If ``case_id`` is not in the fixed corpus or the capture fails
        validation.
    """
    if case_id not in DOCUMENTED_MODES:
        raise ValueError(
            f"case_id must be one of {sorted(DOCUMENTED_MODES)}, got {case_id!r}"
        )
    rate, matrix = voltage_matrix(path)
    onset = _estimate_case_onset(
        matrix,
        rate=rate,
        mode_hz=DOCUMENTED_MODES[case_id],
        allow_unresolved_onset=allow_unresolved_onset,
    )
    ends, scores = window_scores(
        matrix,
        rate=rate,
        window_seconds=FROZEN_PSML_WINDOW_SECONDS,
        step_seconds=FROZEN_PSML_STEP_SECONDS,
    )
    nulls, transitions, _ = _split_with_onset(ends, scores, onset)
    return FrozenTransferCase(
        case_id=case_id,
        n_null=int(nulls.size),
        null_crossings=int((nulls > FROZEN_PSML_THRESHOLD).sum()),
        n_transition=int(transitions.size),
        transition_crossings=int((transitions > FROZEN_PSML_THRESHOLD).sum()),
    )


@dataclass(frozen=True)
class DetectionResult:
    """Secondary detection-branch outcome (amendment A.16.1).

    Attributes
    ----------
    threshold : float
        The matched-false-alarm threshold the branch re-uses.
    detected : tuple of bool
        Per-case flags: any post-forcing window at or above the threshold.
    latency_seconds : tuple of float
        Per-case first-alarm latency from the exact forcing start; ``nan``
        where the case never alarmed.
    significance : PermutationSignificance
        Label-permutation significance of the detection count.
    """

    threshold: float
    detected: tuple[bool, ...]
    latency_seconds: tuple[float, ...]
    significance: PermutationSignificance


def evaluate_detection(
    cases: Sequence[CaseScores],
    detections: Sequence[CaseDetection],
    *,
    threshold: float,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> DetectionResult:
    """Run the secondary detection branch at the locally calibrated operating point.

    Parameters
    ----------
    cases : sequence of CaseScores
        Prepared corpus cases (their pooled null windows anchor the
        permutation null).
    detections : sequence of CaseDetection
        Post-forcing windows of the same cases, in the same order.
    threshold : float
        The matched-false-alarm threshold from the local calibration.
    n_permutations : int
        Permutations for the significance test.
    seed : int
        Seed of the permutation draw (byte-reproducibility).

    Returns
    -------
    DetectionResult
        Per-case detections, latencies, and the permutation significance.

    Raises
    ------
    ValueError
        If ``cases`` and ``detections`` disagree in length or case order,
        or ``threshold`` is not a finite number.
    """
    if len(cases) != len(detections) or any(
        c.case_id != d.case_id for c, d in zip(cases, detections, strict=True)
    ):
        raise ValueError("cases and detections must align one-to-one")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a finite number")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be a finite number")
    null_alarms = [
        bool(value >= threshold) for case in cases for value in case.null_scores
    ]
    detected: list[bool] = []
    latencies: list[float] = []
    for detection in detections:
        crossing = detection.detection_scores >= threshold
        if bool(crossing.any()):
            first_end = float(detection.detection_ends[np.argmax(crossing)])
            detected.append(True)
            latencies.append(first_end - FORCING_SECONDS)
        else:
            detected.append(False)
            latencies.append(float("nan"))
    significance = permutation_significance_from_alarms(
        detected, null_alarms, n_permutations=n_permutations, seed=seed
    )
    return DetectionResult(
        threshold=float(threshold),
        detected=tuple(detected),
        latency_seconds=tuple(latencies),
        significance=significance,
    )


def local_calibration_payload(
    cases: Sequence[CaseScores],
    frozen: Sequence[FrozenTransferCase],
    result: LocalCalibrationResult,
    *,
    detections: Sequence[CaseDetection],
    detection_result: DetectionResult,
    source_digests: dict[str, str],
) -> dict[str, object]:
    """Assemble the sealed JSON-safe payload for the WECC cross-dataset artefact.

    Parameters
    ----------
    cases : sequence of CaseScores
        Prepared corpus cases, in evaluation order.
    frozen : sequence of FrozenTransferCase
        frozen-transfer frozen-transfer outcomes, in the same order.
    result : LocalCalibrationResult
        local-calibration outcome over the same corpus.
    detections : sequence of CaseDetection
        Post-forcing windows of the same cases, in the same order.
    detection_result : DetectionResult
        Secondary detection-branch outcome at the locally calibrated operating point.
    source_digests : dict[str, str]
        SHA-256 of each case's raw ``BusVolMag.txt``, keyed by case id
        (provenance chain; the raw files themselves stay citation-only).

    Returns
    -------
    dict[str, object]
        The payload with a ``content_hash`` field sealing the record.

    Raises
    ------
    ValueError
        If ``cases``, ``frozen``, and ``detections`` disagree in length or
        case order.
    """
    if len(cases) != len(frozen) or any(
        c.case_id != f.case_id for c, f in zip(cases, frozen, strict=True)
    ):
        raise ValueError("cases and frozen outcomes must align one-to-one")
    if len(cases) != len(detections) or any(
        c.case_id != d.case_id for c, d in zip(cases, detections, strict=True)
    ):
        raise ValueError("cases and detections must align one-to-one")
    payload: dict[str, object] = {
        "benchmark": "wecc_240_osl_modal_growth_cross_dataset",
        "evaluation": "cross-dataset generalisation",
        "detector": {
            "shape": "modal envelope growth, focal per-bus deviation",
            "aggregation": DEFAULT_AGGREGATION,
            "recency_top": DEFAULT_RECENCY_TOP,
            "certified_on": "PSML 23-bus (Zheng et al. 2021)",
        },
        "protocol": {
            "pre_registered": (
                "internal plan appendix A.16, fixed before the first detector "
                "run on this corpus"
            ),
            "forcing_seconds": FORCING_SECONDS,
            "null_end_seconds": NULL_END_SECONDS,
            "baseline_seconds": BASELINE_SECONDS,
            "cycles_per_window": CYCLES_PER_WINDOW,
            "step_fraction": STEP_FRACTION,
        },
        "corpus": {
            "source": (
                "2021 IEEE-NASPI Oscillation Source Location Contest, WECC "
                "240-bus synthetic PMU (NREL reduced model, OSL Committee "
                "scenarios; Maslennikov et al.), UTK oscillation library "
                "mirror; raw citation-only, never committed"
            ),
            "transitions": [
                {
                    "case": case.case_id,
                    "mode_hz": case.mode_hz,
                    "rate_hz": case.rate_hz,
                    "n_channels": case.n_channels,
                    "onset_resolved": not bool(np.isnan(case.onset_seconds)),
                    "onset_seconds": (
                        None if np.isnan(case.onset_seconds) else case.onset_seconds
                    ),
                    "window_seconds": case.window_seconds,
                    "n_null_windows": int(case.null_scores.size),
                    "n_transition_windows": int(case.transition_scores.size),
                    "n_detection_windows": int(detection.detection_scores.size),
                    "source_sha256": source_digests.get(case.case_id, ""),
                }
                for case, detection in zip(cases, detections, strict=True)
            ],
        },
        "frozen_transfer": [
            {
                "case": outcome.case_id,
                "n_null": outcome.n_null,
                "null_crossings": outcome.null_crossings,
                "n_transition": outcome.n_transition,
                "transition_crossings": outcome.transition_crossings,
                "threshold": FROZEN_PSML_THRESHOLD,
                "window_seconds": FROZEN_PSML_WINDOW_SECONDS,
            }
            for outcome in frozen
        ],
        "local_calibration": {
            "threshold": result.threshold,
            "target_false_alarm": result.target_false_alarm,
            "achieved_false_alarm": result.achieved_false_alarm,
            "n_null": result.n_null,
            "led": list(result.led),
            "lead_seconds": [
                None if np.isnan(value) else value for value in result.lead_seconds
            ],
            "significance": {
                "observed_led": result.significance.observed_led,
                "n_transitions": result.significance.n_transitions,
                "pooled_alarm_rate": result.significance.pooled_alarm_rate,
                "expected_led": result.significance.expected_led,
                "p_value": result.significance.p_value,
                "n_permutations": result.significance.n_permutations,
                "seed": result.significance.seed,
            },
        },
        "detection_secondary": {
            "note": (
                "secondary branch, amendment A.16.1: detection power after "
                "the exact forcing start at the same matched-FA operating "
                "point; does not replace the primary early-warning metric"
            ),
            "threshold": detection_result.threshold,
            "detected": list(detection_result.detected),
            "latency_seconds": [
                None if np.isnan(value) else value
                for value in detection_result.latency_seconds
            ],
            "significance_caveat": (
                "DESCRIPTIVE ONLY (disclosure A.16.3): the permutation core "
                "treats a whole multi-window detection region as one unit "
                "against single-window nulls, which is anti-conservative for "
                "long regions — the permutation p-value below must not be "
                "read as significance. Under the independence upper bound, "
                "per-case chance detection is 1-(1-FA)^n_windows (see "
                "chance_detection_upper_bound); the informative quantity is "
                "the first-alarm latency, reported per case."
            ),
            "chance_detection_upper_bound": [
                {
                    "case": detection.case_id,
                    "n_windows": int(detection.detection_scores.size),
                    "p_chance_upper": 1.0
                    - (1.0 - result.achieved_false_alarm)
                    ** int(detection.detection_scores.size),
                }
                for detection in detections
            ],
            "significance": {
                "observed_led": detection_result.significance.observed_led,
                "n_transitions": detection_result.significance.n_transitions,
                "pooled_alarm_rate": (detection_result.significance.pooled_alarm_rate),
                "expected_led": detection_result.significance.expected_led,
                "p_value": detection_result.significance.p_value,
                "n_permutations": detection_result.significance.n_permutations,
                "seed": detection_result.significance.seed,
            },
        },
        "caveats": [
            "synthetic corpus: TSAT time-domain simulation of the WECC "
            "240-bus model, not field PMU; the real-data leg of the "
            "cross-dataset evaluation is the ISO-NE case study",
            "the secondary detection branch is descriptive only: at a 10% "
            "window false alarm a 60 s region alarms by chance with high "
            "probability, so the detection COUNT carries no significance "
            "claim (disclosure A.16.3); the latency distribution is the "
            "informative quantity",
            "forcing start is exact by simulation design (t=30 s); the "
            "early-warning onset is estimated in-band with an ambient-only "
            "baseline",
            "observable is per-bus voltage magnitude — the same observable "
            "family the detector was certified on",
        ],
    }
    payload["content_hash"] = canonical_record_hash(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI shell
    """Run both branches over a local raw-data directory and print the payload.

    Parameters
    ----------
    argv : sequence of str | None
        Command-line arguments; ``--data-dir`` points at the extracted
        ``All_cases`` directory, ``--output`` optionally writes the sealed
        payload JSON.

    Returns
    -------
    int
        Process exit status (zero on success).
    """
    import argparse
    import hashlib
    import json

    from bench.early_warning_leadtime_isone import evaluate_local_calibration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default=None)
    options = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(options.data_dir)
    paths = {
        f"WECC_case{index}": (
            root / f"Case{index}" / f"Case{index}_PMU" / "BusVolMag.txt"
        )
        for index in range(1, 14)
    }
    digests = {
        case_id: hashlib.sha256(path.read_bytes()).hexdigest()
        for case_id, path in paths.items()
    }
    records = [
        case_records(case_id, path, allow_unresolved_onset=True)
        for case_id, path in paths.items()
    ]
    prepared = [record[0] for record in records]
    detections = [record[1] for record in records]
    frozen = [
        frozen_transfer_case(case_id, path, allow_unresolved_onset=True)
        for case_id, path in paths.items()
    ]
    result = evaluate_local_calibration(prepared)
    detection_result = evaluate_detection(
        prepared, detections, threshold=result.threshold
    )
    payload = local_calibration_payload(
        prepared,
        frozen,
        result,
        detections=detections,
        detection_result=detection_result,
        source_digests=digests,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if options.output:
        Path(options.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI entry
    raise SystemExit(main())
