# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — ISO-NE cross-dataset modal-growth evaluation (E2.G)

"""Cross-dataset evaluation of the certified grid modal-growth detector on ISO-NE.

The PSML head-to-head certified the modal envelope-growth detector inside one
dataset (:mod:`bench.grid_modal_head_to_head`). This module runs the E2.G leg of
the External Validation Program: the *frozen* detector shape — focal per-bus
deviation envelope, recency weighting, growth rate ``σ`` in inverse seconds —
against real ISO-NE PMU captures of documented sustained oscillations from the
UTK oscillation test-case library (Maslennikov et al. 2016, citation-only, never
committed).

Two pre-registered branches, both reported:

* **G-a frozen transfer** — the PSML offline operating point verbatim (threshold
  and two-second windows). A negative is a finding about operating-point
  portability, not a failure to hide.
* **G-b frozen shape, local calibration** — the window scaled to the documented
  mode a priori (:data:`CYCLES_PER_WINDOW` cycles of ``f0``), the threshold
  calibrated at a matched false alarm ONLY on pre-onset ambient (null) windows —
  the product's own per-system calibration step. Lead time is reported honestly
  and significance uses the shared label-permutation core with the small-corpus
  power caveat disclosed.

The corpus is fixed before any detector run (recon 2026-08-26, plan Appendix A):
cases 1-3 carry separable in-capture onsets and form the transitions; cases 4-6
are excluded for disclosed reasons (:data:`EXCLUDED_CASES`). No variant search
runs here — any change to the detector shape disqualifies the run as E2.G.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import hilbert

from bench.analytic_phase_pipeline import bandpass
from bench.early_warning_domain import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
    PermutationSignificance,
    permutation_significance_from_alarms,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.evaluation.skill import calibrate_score_threshold
from scpn_phase_orchestrator.monitor.grid_modal_growth import (
    DEFAULT_AGGREGATION,
    DEFAULT_RECENCY_TOP,
    FloatArray,
    modal_growth_score,
)
from scpn_phase_orchestrator.runtime.pmu_ieee_adapter import read_ieee_pmu_recording

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

__all__ = [
    "CYCLES_PER_WINDOW",
    "DOCUMENTED_MODES",
    "EXCLUDED_CASES",
    "FROZEN_PSML_STEP_SECONDS",
    "FROZEN_PSML_THRESHOLD",
    "FROZEN_PSML_WINDOW_SECONDS",
    "CaseScores",
    "FrozenTransferCase",
    "LocalCalibrationResult",
    "case_scores",
    "estimate_onset",
    "evaluate_local_calibration",
    "frequency_matrix",
    "frozen_transfer_case",
    "local_calibration_payload",
    "split_scores",
    "window_scores",
]

#: Documented dominant oscillation mode per corpus case, hertz (UTK descriptors).
DOCUMENTED_MODES: dict[str, float] = {
    "ISO-NE_case1": 0.27,
    "ISO-NE_case2": 0.15,
    "ISO-NE_case3": 1.13,
}

#: Cases excluded from the corpus BEFORE any detector run, with the reason.
EXCLUDED_CASES: dict[str, str] = {
    "ISO-NE_case4": "no separable in-band onset (<=10 MW event, flat envelope)",
    "ISO-NE_case5": "no separable in-band onset (15 MW event, flat envelope)",
    "ISO-NE_case6": (
        "all four frequency channels are duplicates of one substation "
        "measurement; no per-bus structure and no in-band onset"
    ),
}

#: PSML offline per-window operating point, frozen for the G-a transfer branch.
FROZEN_PSML_THRESHOLD = 1.3203407954771857
FROZEN_PSML_WINDOW_SECONDS = 2.0
FROZEN_PSML_STEP_SECONDS = 0.5

#: G-b window scaling: cycles of the documented mode per scoring window.
CYCLES_PER_WINDOW = 5.0
#: G-b step, as a fraction of the window.
STEP_FRACTION = 0.25

#: Onset estimation controls (formalising the reconnaissance rule).
BASELINE_SECONDS = 30.0
ONSET_FACTOR = 3.0
ONSET_SUSTAIN_SECONDS = 10.0
SMOOTH_SECONDS = 5.0

#: Scoring region before the onset, seconds; alarms in it count as detections.
TRANSITION_SECONDS = 60.0

#: Fail-closed floors for a meaningful evaluation.
MIN_CHANNELS = 2
MIN_NULL_WINDOWS = 5


def frequency_matrix(path: str | Path) -> tuple[float, FloatArray]:
    """Load the clean frequency-channel matrix of one IEEE PMU capture.

    Parameters
    ----------
    path : str | Path
        The IEEE-format multi-header PMU CSV.

    Returns
    -------
    tuple[float, FloatArray]
        The sampling rate in hertz and the matrix of clean channels, shape
        ``(channels, samples)``.

    Raises
    ------
    ValueError
        If fewer than :data:`MIN_CHANNELS` clean channels remain, or the clean
        channels carry fewer than :data:`MIN_CHANNELS` distinct signals (the
        case-6 degeneracy: one substation duplicated across lines).
    """
    recording = read_ieee_pmu_recording(path)
    clean = [channel for channel in recording.channels if channel.is_clean]
    if len(clean) < MIN_CHANNELS:
        raise ValueError(
            f"{recording.source_name}: only {len(clean)} clean channels of "
            f"{len(recording.channels)}; need at least {MIN_CHANNELS}"
        )
    matrix = np.vstack(
        [np.asarray(channel.samples, dtype=np.float64) for channel in clean]
    )
    distinct = np.unique(matrix, axis=0).shape[0]
    if distinct < MIN_CHANNELS:
        raise ValueError(
            f"{recording.source_name}: clean channels collapse to {distinct} "
            "distinct signal(s); the focal detector needs per-bus structure"
        )
    rate = 1.0 / float(np.median(np.diff(recording.times)))
    return rate, matrix


def _mode_band(mode_hz: float, rate: float) -> tuple[float, float]:
    """Return the analysis band around one documented mode, clipped to Nyquist.

    Parameters
    ----------
    mode_hz : float
        Documented oscillation frequency in hertz.
    rate : float
        Sampling rate in hertz.

    Returns
    -------
    tuple[float, float]
        Low and high band edges in hertz.

    Raises
    ------
    ValueError
        If ``mode_hz`` or ``rate`` is not a positive finite number, or the band
        collapses under the Nyquist clip.
    """
    for name, value in (("mode_hz", mode_hz), ("rate", rate)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a positive finite number")
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    low = 0.5 * float(mode_hz)
    high = min(1.5 * float(mode_hz), 0.45 * float(rate))
    if high <= low:
        raise ValueError(
            f"mode band [{low}, {high}] Hz collapses at sampling rate {rate} Hz"
        )
    return low, high


def estimate_onset(
    matrix: FloatArray,
    *,
    rate: float,
    mode_hz: float,
    baseline_seconds: float = BASELINE_SECONDS,
    factor: float = ONSET_FACTOR,
    sustain_seconds: float = ONSET_SUSTAIN_SECONDS,
    smooth_seconds: float = SMOOTH_SECONDS,
) -> float:
    """Estimate the oscillation onset from the in-band envelope, fail-closed.

    The reconnaissance rule, formalised: band-pass every channel around the
    documented mode, average the analytic envelopes, smooth, and take the first
    time the smoothed envelope exceeds ``factor`` times the baseline median for
    ``sustain_seconds`` without interruption.

    Parameters
    ----------
    matrix : FloatArray
        Clean channel matrix, shape ``(channels, samples)``.
    rate : float
        Sampling rate in hertz.
    mode_hz : float
        Documented oscillation frequency in hertz.
    baseline_seconds : float
        Length of the leading stretch whose median envelope is the baseline.
    factor : float
        Multiple of the baseline the envelope must exceed.
    sustain_seconds : float
        How long the exceedance must hold without interruption.
    smooth_seconds : float
        Moving-average length applied to the envelope before thresholding.

    Returns
    -------
    float
        Onset time in seconds from the start of the capture.

    Raises
    ------
    ValueError
        If the controls are not positive finite numbers, or no sustained
        exceedance exists (a capture without a separable onset cannot be a
        transition — fail closed, never a silent zero).
    """
    for name, value in (
        ("baseline_seconds", baseline_seconds),
        ("factor", factor),
        ("sustain_seconds", sustain_seconds),
        ("smooth_seconds", smooth_seconds),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a positive finite number")
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    band = _mode_band(mode_hz, rate)
    centred = np.asarray(matrix, dtype=np.float64)
    centred = centred - centred.mean(axis=1, keepdims=True)
    filtered = bandpass(centred, sampling_rate_hz=rate, band_hz=band)
    envelope = np.abs(hilbert(filtered, axis=1)).mean(axis=0)
    smooth_n = max(int(smooth_seconds * rate), 1)
    kernel = np.full(smooth_n, 1.0 / smooth_n)
    smoothed = np.convolve(envelope, kernel, mode="same")
    baseline_n = min(int(baseline_seconds * rate), envelope.size // 4)
    baseline = float(np.median(envelope[: max(baseline_n, 1)]))
    above = smoothed > factor * baseline
    run = max(int(sustain_seconds * rate), 1)
    for start in range(above.size - run + 1):
        if bool(above[start : start + run].all()):
            return float(start) / rate
    raise ValueError(
        f"no sustained {factor}x baseline exceedance of {sustain_seconds}s "
        "in band; the capture has no separable onset"
    )


def window_scores(
    matrix: FloatArray,
    *,
    rate: float,
    window_seconds: float,
    step_seconds: float,
) -> tuple[FloatArray, FloatArray]:
    """Score sliding windows with the frozen modal-growth detector shape.

    Parameters
    ----------
    matrix : FloatArray
        Clean channel matrix, shape ``(channels, samples)``.
    rate : float
        Sampling rate in hertz.
    window_seconds : float
        Window length in seconds.
    step_seconds : float
        Step between window starts in seconds.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Window END times in seconds from capture start, and the modal growth
        score ``σ`` (inverse seconds) of each window under the certified shape
        (focal aggregation, certified recency weighting).

    Raises
    ------
    ValueError
        If the window controls are not positive finite numbers or the capture
        is shorter than one window.
    """
    for name, value in (
        ("window_seconds", window_seconds),
        ("step_seconds", step_seconds),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a positive finite number")
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    samples = np.asarray(matrix, dtype=np.float64)
    window_n = int(window_seconds * rate)
    step_n = max(int(step_seconds * rate), 1)
    if window_n < 2 or samples.shape[1] < window_n:
        raise ValueError(
            f"capture of {samples.shape[1]} samples cannot hold a "
            f"{window_seconds}s window at {rate} Hz"
        )
    ends: list[float] = []
    scores: list[float] = []
    for start in range(0, samples.shape[1] - window_n + 1, step_n):
        segment = samples[:, start : start + window_n]
        scores.append(
            modal_growth_score(
                segment,
                rate=rate,
                aggregation=DEFAULT_AGGREGATION,
                recency_top=DEFAULT_RECENCY_TOP,
            )
        )
        ends.append((start + window_n) / rate)
    return np.asarray(ends), np.asarray(scores)


def split_scores(
    ends: FloatArray,
    scores: FloatArray,
    *,
    onset_seconds: float,
    window_seconds: float,
    transition_seconds: float = TRANSITION_SECONDS,
) -> tuple[FloatArray, FloatArray]:
    """Split window scores into pre-onset nulls and the transition region.

    Null windows END before the transition region opens, separated by one full
    window as a leakage guard; transition windows end inside
    ``(onset - transition_seconds, onset]``. Windows in the guard gap or after
    the onset belong to neither set.

    Parameters
    ----------
    ends : FloatArray
        Window end times in seconds.
    scores : FloatArray
        Modal growth score of each window.
    onset_seconds : float
        Estimated onset time in seconds.
    window_seconds : float
        Scoring window length in seconds (the guard width).
    transition_seconds : float
        Length of the scored region ending at the onset.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Null scores and transition scores.

    Raises
    ------
    ValueError
        If ``ends`` and ``scores`` differ in length.
    """
    end_times = np.asarray(ends, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if end_times.shape != values.shape:
        raise ValueError("ends and scores must have identical shape")
    transition_open = onset_seconds - transition_seconds
    null_mask = end_times <= transition_open - window_seconds
    transition_mask = (end_times > transition_open) & (end_times <= onset_seconds)
    return values[null_mask], values[transition_mask]


@dataclass(frozen=True)
class CaseScores:
    """One corpus case prepared for evaluation.

    Attributes
    ----------
    case_id : str
        Corpus case identifier (a :data:`DOCUMENTED_MODES` key).
    mode_hz : float
        Documented oscillation mode in hertz.
    rate_hz : float
        Sampling rate of the capture in hertz.
    n_channels : int
        Number of clean channels in the scored matrix.
    onset_seconds : float
        Estimated onset time in seconds from capture start.
    window_seconds : float
        Scoring window used for this case, seconds.
    null_scores : FloatArray
        Pre-onset ambient window scores.
    transition_scores : FloatArray
        Scores of windows ending inside the transition region.
    transition_ends : FloatArray
        End times of the transition-region windows, seconds.
    """

    case_id: str
    mode_hz: float
    rate_hz: float
    n_channels: int
    onset_seconds: float
    window_seconds: float
    null_scores: FloatArray
    transition_scores: FloatArray
    transition_ends: FloatArray


def case_scores(case_id: str, path: str | Path) -> CaseScores:
    """Prepare one corpus case: load, estimate the onset, score, and split.

    The G-b configuration is derived a priori from the documented mode: the
    window is :data:`CYCLES_PER_WINDOW` cycles of ``f0`` and the step is
    :data:`STEP_FRACTION` of the window. Nothing here reads the scores before
    fixing the segmentation.

    Parameters
    ----------
    case_id : str
        A key of :data:`DOCUMENTED_MODES`.
    path : str | Path
        The case's IEEE PMU CSV.

    Returns
    -------
    CaseScores
        The prepared case.

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
    rate, matrix = frequency_matrix(path)
    onset = estimate_onset(matrix, rate=rate, mode_hz=mode_hz)
    window_seconds = CYCLES_PER_WINDOW / mode_hz
    step_seconds = window_seconds * STEP_FRACTION
    ends, scores = window_scores(
        matrix, rate=rate, window_seconds=window_seconds, step_seconds=step_seconds
    )
    nulls, transitions = split_scores(
        ends, scores, onset_seconds=onset, window_seconds=window_seconds
    )
    transition_open = onset - TRANSITION_SECONDS
    transition_mask = (ends > transition_open) & (ends <= onset)
    return CaseScores(
        case_id=case_id,
        mode_hz=mode_hz,
        rate_hz=rate,
        n_channels=int(matrix.shape[0]),
        onset_seconds=onset,
        window_seconds=window_seconds,
        null_scores=nulls,
        transition_scores=transitions,
        transition_ends=np.asarray(ends)[transition_mask],
    )


@dataclass(frozen=True)
class FrozenTransferCase:
    """G-a frozen-transfer outcome for one case.

    Attributes
    ----------
    case_id : str
        Corpus case identifier.
    n_null : int
        Number of pre-onset ambient windows.
    null_crossings : int
        Ambient windows whose score crosses the frozen PSML threshold.
    n_transition : int
        Number of transition-region windows.
    transition_crossings : int
        Transition windows crossing the frozen PSML threshold.
    """

    case_id: str
    n_null: int
    null_crossings: int
    n_transition: int
    transition_crossings: int


def frozen_transfer_case(case_id: str, path: str | Path) -> FrozenTransferCase:
    """Run the G-a branch on one case: frozen PSML operating point verbatim.

    Parameters
    ----------
    case_id : str
        A key of :data:`DOCUMENTED_MODES`.
    path : str | Path
        The case's IEEE PMU CSV.

    Returns
    -------
    FrozenTransferCase
        Crossing counts at the frozen threshold in ambient and transition
        windows.

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
    rate, matrix = frequency_matrix(path)
    onset = estimate_onset(matrix, rate=rate, mode_hz=DOCUMENTED_MODES[case_id])
    ends, scores = window_scores(
        matrix,
        rate=rate,
        window_seconds=FROZEN_PSML_WINDOW_SECONDS,
        step_seconds=FROZEN_PSML_STEP_SECONDS,
    )
    nulls, transitions = split_scores(
        ends,
        scores,
        onset_seconds=onset,
        window_seconds=FROZEN_PSML_WINDOW_SECONDS,
    )
    return FrozenTransferCase(
        case_id=case_id,
        n_null=int(nulls.size),
        null_crossings=int((nulls > FROZEN_PSML_THRESHOLD).sum()),
        n_transition=int(transitions.size),
        transition_crossings=int((transitions > FROZEN_PSML_THRESHOLD).sum()),
    )


@dataclass(frozen=True)
class LocalCalibrationResult:
    """G-b outcome: frozen shape, locally calibrated threshold, honest leads.

    Attributes
    ----------
    threshold : float
        Matched-false-alarm threshold calibrated on pooled null windows only.
    target_false_alarm : float
        The false-alarm target the calibration held.
    achieved_false_alarm : float
        Fraction of pooled null windows at or above the threshold.
    n_null : int
        Pooled null window count across the corpus.
    led : tuple[bool, ...]
        Per-case detection flags (an alarm inside the transition region).
    lead_seconds : tuple[float, ...]
        Per-case lead from the FIRST alarming window's end to the onset;
        ``nan`` where the case did not alarm.
    significance : PermutationSignificance
        Label-permutation significance of the detection count.
    """

    threshold: float
    target_false_alarm: float
    achieved_false_alarm: float
    n_null: int
    led: tuple[bool, ...]
    lead_seconds: tuple[float, ...]
    significance: PermutationSignificance


def evaluate_local_calibration(
    cases: Sequence[CaseScores],
    *,
    target_false_alarm: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> LocalCalibrationResult:
    """Run the G-b branch over the prepared corpus.

    The threshold is calibrated on the POOLED null scores of every case — the
    growth rate ``σ`` is in inverse seconds, so scores are physically
    comparable across cases despite per-case windows — and detections are
    counted per case against that single system-level operating point.

    Parameters
    ----------
    cases : sequence of CaseScores
        Prepared corpus cases.
    target_false_alarm : float
        Matched false-alarm target for the null calibration.
    n_permutations : int
        Permutations for the significance test.
    seed : int
        Seed of the permutation draw (byte-reproducibility).

    Returns
    -------
    LocalCalibrationResult
        Threshold, achieved false alarm, per-case detections and leads, and
        the permutation significance.

    Raises
    ------
    ValueError
        If ``cases`` is empty or the pooled null count is below
        :data:`MIN_NULL_WINDOWS` (calibration would be meaningless — fail
        closed rather than certify from nothing).
    """
    if not cases:
        raise ValueError("cases must not be empty")
    pooled = np.concatenate([case.null_scores for case in cases])
    if pooled.size < MIN_NULL_WINDOWS:
        raise ValueError(
            f"pooled null windows {pooled.size} < floor {MIN_NULL_WINDOWS}; "
            "refusing to calibrate a threshold from that little ambient data"
        )
    threshold = calibrate_score_threshold(
        [float(value) for value in pooled], target_fa=target_false_alarm
    )
    achieved = float((pooled >= threshold).sum() / pooled.size)
    led: list[bool] = []
    leads: list[float] = []
    null_alarms = [bool(value >= threshold) for value in pooled]
    for case in cases:
        crossing = case.transition_scores >= threshold
        if bool(crossing.any()):
            first_end = float(case.transition_ends[np.argmax(crossing)])
            led.append(True)
            leads.append(case.onset_seconds - first_end)
        else:
            led.append(False)
            leads.append(float("nan"))
    significance = permutation_significance_from_alarms(
        led, null_alarms, n_permutations=n_permutations, seed=seed
    )
    return LocalCalibrationResult(
        threshold=float(threshold),
        target_false_alarm=float(target_false_alarm),
        achieved_false_alarm=achieved,
        n_null=int(pooled.size),
        led=tuple(led),
        lead_seconds=tuple(leads),
        significance=significance,
    )


def local_calibration_payload(
    cases: Sequence[CaseScores],
    frozen: Sequence[FrozenTransferCase],
    result: LocalCalibrationResult,
    *,
    source_digests: dict[str, str],
) -> dict[str, object]:
    """Assemble the sealed JSON-safe payload for the ISO-NE E2.G artefact.

    Parameters
    ----------
    cases : sequence of CaseScores
        Prepared corpus cases, in evaluation order.
    frozen : sequence of FrozenTransferCase
        G-a frozen-transfer outcomes, in the same order.
    result : LocalCalibrationResult
        G-b outcome over the same corpus.
    source_digests : dict[str, str]
        SHA-256 of each case's raw CSV, keyed by case id (provenance chain;
        the raw files themselves stay citation-only).

    Returns
    -------
    dict[str, object]
        The payload with a ``content_hash`` field sealing the record.

    Raises
    ------
    ValueError
        If ``cases`` and ``frozen`` disagree in length or case order.
    """
    if len(cases) != len(frozen) or any(
        c.case_id != f.case_id for c, f in zip(cases, frozen, strict=True)
    ):
        raise ValueError("cases and frozen outcomes must align one-to-one")
    payload: dict[str, object] = {
        "benchmark": "iso_ne_modal_growth_cross_dataset",
        "program": "E2.G",
        "detector": {
            "shape": "modal envelope growth, focal per-bus deviation",
            "aggregation": DEFAULT_AGGREGATION,
            "recency_top": DEFAULT_RECENCY_TOP,
            "certified_on": "PSML 23-bus (Zheng et al. 2021)",
        },
        "corpus": {
            "source": (
                "UTK oscillation test-case library, actual ISO-NE PMU events "
                "(Maslennikov et al. 2016); raw citation-only, never committed"
            ),
            "transitions": [
                {
                    "case": case.case_id,
                    "mode_hz": case.mode_hz,
                    "rate_hz": case.rate_hz,
                    "n_channels": case.n_channels,
                    "onset_seconds": case.onset_seconds,
                    "window_seconds": case.window_seconds,
                    "n_null_windows": int(case.null_scores.size),
                    "n_transition_windows": int(case.transition_scores.size),
                    "source_sha256": source_digests.get(case.case_id, ""),
                }
                for case in cases
            ],
            "excluded": dict(EXCLUDED_CASES),
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
        "caveats": [
            "n=3 transition corpus: a case study, not a powered significance "
            "leg; WECC 240-bus remains the statistical leg",
            "observable is per-substation frequency (PSML certified on bus "
            "voltages); disclosed mapping",
            "onsets are estimated in-band, not operator-annotated",
        ],
    }
    payload["content_hash"] = canonical_record_hash(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI shell
    """Run both branches over a local raw-data directory and print the payload.

    Parameters
    ----------
    argv : sequence of str | None
        Command-line arguments; ``--data-dir`` points at the extracted UTK
        cases, ``--output`` optionally writes the sealed payload JSON.

    Returns
    -------
    int
        Process exit status (zero on success).
    """
    import argparse
    import hashlib
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default=None)
    options = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(options.data_dir)
    paths = {
        "ISO-NE_case1": root / "case1" / "ISO-NE_case1.csv",
        "ISO-NE_case2": root / "case2" / "ISO-NE_case2.csv",
        "ISO-NE_case3": root / "case3" / "ISO-NE_case3.csv",
    }
    digests = {
        case_id: hashlib.sha256(path.read_bytes()).hexdigest()
        for case_id, path in paths.items()
    }
    prepared = [case_scores(case_id, path) for case_id, path in paths.items()]
    frozen = [frozen_transfer_case(case_id, path) for case_id, path in paths.items()]
    result = evaluate_local_calibration(prepared)
    payload = local_calibration_payload(
        prepared, frozen, result, source_digests=digests
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if options.output:
        Path(options.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI entry
    raise SystemExit(main())
