# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real-time PMU capture replay (R1 sentinel demo)

"""Causal real-time replay of a recorded PMU capture through the live monitor.

The R1 demo of record: a real recorded oscillation event is replayed sample by
sample — optionally paced at wall-clock speed — through the certified
:class:`~scpn_phase_orchestrator.monitor.grid_modal_stream.GridModalStreamMonitor`,
whose operating point is read from a SEALED evidence artefact, never hand-set.
The seal is verified before any value is trusted; a tampered artefact is
rejected, not partially honoured. Every replay produces a sealed record of its
own: the operating-point provenance (including the calibration's disclosed
null-window count and p-value), the alarms with their live scores, and the
honest lead against the estimated onset — negative when the alarm came late,
absent when it never fired.

The monitor sees only past samples (its own causality guarantee); this module
adds no lookahead. The observable is the capture's clean per-substation
frequency channels — the same disclosed mapping the E2.G evaluation sealed.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bench.early_warning_leadtime_isone import estimate_onset, frequency_matrix
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.monitor.grid_modal_growth import FloatArray
from scpn_phase_orchestrator.monitor.grid_modal_stream import (
    GridModalStreamMonitor,
    StreamAlarm,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

__all__ = [
    "StreamReplayResult",
    "load_sealed_evidence",
    "monitor_from_e2g",
    "replay",
    "replay_case",
    "sealed_replay_record",
]

#: Consecutive above-threshold re-scorings before the live alarm fires.
DEFAULT_PERSISTENCE = 1


def load_sealed_evidence(evidence_path: str | Path) -> dict[str, object]:
    """Load a sealed evidence payload and verify its content hash, fail-closed.

    Parameters
    ----------
    evidence_path : str | Path
        Path to a sealed JSON artefact carrying a ``content_hash`` field.

    Returns
    -------
    dict[str, object]
        The verified payload.

    Raises
    ------
    ValueError
        If the payload carries no ``content_hash`` or the hash does not
        recompute from the record — a tampered or hand-edited artefact must
        never configure a live monitor.
    """
    payload = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    record = copy.deepcopy(payload)
    sealed = record.pop("content_hash", None)
    if not isinstance(sealed, str):
        raise ValueError("evidence carries no content_hash; refusing to trust it")
    if canonical_record_hash(record) != sealed:
        raise ValueError(
            "evidence content_hash does not recompute from the record; "
            "refusing to configure a live monitor from a tampered artefact"
        )
    return payload  # type: ignore[no-any-return]


def _case_entry(payload: dict[str, object], case_id: str) -> dict[str, object]:
    """Return one corpus transition entry of a sealed E2.G payload.

    Parameters
    ----------
    payload : dict[str, object]
        A verified E2.G payload.
    case_id : str
        The corpus case to configure for.

    Returns
    -------
    dict[str, object]
        The transition entry for ``case_id``.

    Raises
    ------
    ValueError
        If the payload structure is not an E2.G record or the case is not in
        its sealed corpus.
    """
    corpus = payload.get("corpus")
    if not isinstance(corpus, dict) or "transitions" not in corpus:
        raise ValueError("evidence carries no corpus.transitions block")
    for entry in corpus["transitions"]:
        if entry.get("case") == case_id:
            return dict(entry)
    raise ValueError(f"case {case_id!r} is not in the sealed corpus")


def monitor_from_e2g(
    evidence_path: str | Path,
    *,
    case_id: str,
    rate: float,
    persistence: int = DEFAULT_PERSISTENCE,
) -> GridModalStreamMonitor:
    """Build the live monitor from a sealed E2.G artefact, no hand-set constants.

    The threshold comes from the sealed G-b local calibration, the aggregation
    and recency weighting from the sealed detector block, and the window from
    the case's sealed per-mode configuration; the step is a quarter window, as
    evaluated.

    Parameters
    ----------
    evidence_path : str | Path
        Path to the sealed ``iso_ne_modal_growth_cross_dataset.json`` artefact.
    case_id : str
        The corpus case whose sealed window configuration to carry.
    rate : float
        The live stream's sampling rate in hertz.
    persistence : int
        Consecutive above-threshold re-scorings before the alarm fires.

    Returns
    -------
    GridModalStreamMonitor
        A causal monitor at the sealed operating point.

    Raises
    ------
    ValueError
        If the seal fails to verify, the payload is not an E2.G record, or the
        case is not in its sealed corpus.
    """
    payload = load_sealed_evidence(evidence_path)
    entry = _case_entry(payload, case_id)
    detector = payload.get("detector")
    calibration = payload.get("local_calibration")
    if not isinstance(detector, dict) or not isinstance(calibration, dict):
        raise ValueError("evidence carries no detector/local_calibration blocks")
    window_value = entry.get("window_seconds")
    if isinstance(window_value, bool) or not isinstance(window_value, (int, float)):
        raise ValueError("sealed case entry carries no numeric window_seconds")
    window_seconds = float(window_value)
    return GridModalStreamMonitor(
        rate=rate,
        threshold=float(calibration["threshold"]),
        window_seconds=window_seconds,
        step_seconds=window_seconds / 4.0,
        aggregation=str(detector["aggregation"]),
        recency_top=float(detector["recency_top"]),
        persistence=persistence,
    )


def replay(
    matrix: FloatArray,
    *,
    monitor: GridModalStreamMonitor,
    pace_seconds: float | None = None,
) -> list[StreamAlarm]:
    """Feed a recorded capture through the monitor causally, sample by sample.

    Parameters
    ----------
    matrix : FloatArray
        Clean channel matrix, shape ``(channels, samples)``, in recorded order.
    monitor : GridModalStreamMonitor
        The live monitor; it sees one sample per step and never the future.
    pace_seconds : float | None
        Optional wall-clock pacing between samples (``1/rate`` replays the
        capture in real time); ``None`` replays as fast as possible.

    Returns
    -------
    list[StreamAlarm]
        Every alarm the monitor raised, in stream order.

    Raises
    ------
    ValueError
        If ``pace_seconds`` is not ``None`` or a non-negative finite number.
    """
    if pace_seconds is not None and (
        isinstance(pace_seconds, bool)
        or not isinstance(pace_seconds, (int, float))
        or not np.isfinite(pace_seconds)
        or pace_seconds < 0.0
    ):
        raise ValueError("pace_seconds must be None or a non-negative number")
    samples = np.asarray(matrix, dtype=np.float64)
    alarms: list[StreamAlarm] = []
    for index in range(samples.shape[1]):
        alarm = monitor.update(samples[:, index])
        if alarm is not None:
            alarms.append(alarm)
        if pace_seconds is not None:
            time.sleep(pace_seconds)
    return alarms


@dataclass(frozen=True)
class StreamReplayResult:
    """One capture replayed through the sealed operating point.

    Attributes
    ----------
    case_id : str
        The corpus case replayed.
    rate_hz : float
        Sampling rate of the capture in hertz.
    n_samples : int
        Samples replayed.
    onset_seconds : float
        Estimated oscillation onset, seconds from capture start.
    alarms : tuple[StreamAlarm, ...]
        Every alarm raised, in stream order.
    first_alarm_seconds : float | None
        Time of the first alarm, or ``None`` when the monitor stayed silent.
    lead_seconds : float | None
        ``onset - first alarm`` — positive when the alarm preceded the onset,
        negative when it came late, ``None`` without an alarm.
    wall_seconds : float
        Wall-clock duration of the replay loop.
    """

    case_id: str
    rate_hz: float
    n_samples: int
    onset_seconds: float
    alarms: tuple[StreamAlarm, ...]
    first_alarm_seconds: float | None
    lead_seconds: float | None
    wall_seconds: float


def replay_case(
    case_id: str,
    csv_path: str | Path,
    evidence_path: str | Path,
    *,
    mode_hz: float,
    pace_seconds: float | None = None,
    persistence: int = DEFAULT_PERSISTENCE,
) -> StreamReplayResult:
    """Replay one recorded capture at the sealed operating point, honestly timed.

    Parameters
    ----------
    case_id : str
        The corpus case being replayed (must be in the sealed corpus).
    csv_path : str | Path
        The case's IEEE PMU CSV.
    evidence_path : str | Path
        The sealed E2.G artefact carrying the operating point.
    mode_hz : float
        Documented oscillation mode, hertz, for the onset estimate.
    pace_seconds : float | None
        Optional wall-clock pacing between samples; ``None`` replays unpaced.
    persistence : int
        Consecutive above-threshold re-scorings before the alarm fires.

    Returns
    -------
    StreamReplayResult
        Alarms, first-alarm time, and the honest lead against the estimated
        onset.

    Raises
    ------
    ValueError
        If the capture, evidence seal, or onset estimation fails validation.
    """
    rate, matrix = frequency_matrix(csv_path)
    onset = estimate_onset(matrix, rate=rate, mode_hz=mode_hz)
    monitor = monitor_from_e2g(
        evidence_path, case_id=case_id, rate=rate, persistence=persistence
    )
    started = time.perf_counter()
    alarms = replay(matrix, monitor=monitor, pace_seconds=pace_seconds)
    wall = time.perf_counter() - started
    first = alarms[0].time_s if alarms else None
    lead = (onset - first) if first is not None else None
    return StreamReplayResult(
        case_id=case_id,
        rate_hz=rate,
        n_samples=int(matrix.shape[1]),
        onset_seconds=onset,
        alarms=tuple(alarms),
        first_alarm_seconds=first,
        lead_seconds=lead,
        wall_seconds=wall,
    )


def sealed_replay_record(
    result: StreamReplayResult,
    *,
    evidence_path: str | Path,
    source_sha256: str,
) -> dict[str, object]:
    """Seal one replay into a hash-addressed record with full provenance.

    Parameters
    ----------
    result : StreamReplayResult
        The replay to seal.
    evidence_path : str | Path
        The sealed artefact that configured the monitor; its hash and the
        calibration's disclosed limits are copied into the provenance block.
    source_sha256 : str
        SHA-256 of the raw capture replayed.

    Returns
    -------
    dict[str, object]
        A JSON-safe record with a ``content_hash`` field sealing it.

    Raises
    ------
    ValueError
        If the evidence seal fails to verify or the payload carries no
        calibration/significance blocks.
    """
    payload = load_sealed_evidence(evidence_path)
    calibration = payload.get("local_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("evidence carries no local_calibration block")
    significance = calibration.get("significance")
    if not isinstance(significance, dict):
        raise ValueError("evidence carries no calibration significance block")
    record: dict[str, object] = {
        "benchmark": "pmu_stream_replay",
        "program": "R1",
        "case": result.case_id,
        "operating_point_provenance": {
            "evidence_content_hash": payload["content_hash"],
            "threshold": calibration["threshold"],
            "calibration_n_null": calibration["n_null"],
            "calibration_p_value": significance["p_value"],
            "caveat": (
                "operating point from an n=3 case-study calibration; a "
                "deployment calibrates on its own ambient data"
            ),
        },
        "source_sha256": source_sha256,
        "rate_hz": result.rate_hz,
        "n_samples": result.n_samples,
        "onset_seconds": result.onset_seconds,
        "alarms": [
            {
                "sample_index": alarm.sample_index,
                "time_s": alarm.time_s,
                "score": alarm.score,
                "threshold": alarm.threshold,
                "bus": alarm.bus,
            }
            for alarm in result.alarms
        ],
        "first_alarm_seconds": result.first_alarm_seconds,
        "lead_seconds": result.lead_seconds,
    }
    record["content_hash"] = canonical_record_hash(record)
    return record


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI shell
    """Replay one recorded case through the sealed operating point and report.

    Parameters
    ----------
    argv : sequence of str | None
        Command-line arguments: ``--csv``, ``--evidence``, ``--case``,
        ``--mode-hz``, optional ``--realtime`` (wall-clock pacing) and
        ``--output`` for the sealed replay record.

    Returns
    -------
    int
        Process exit status (zero on success).
    """
    import argparse
    import hashlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode-hz", type=float, required=True)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--output", default=None)
    options = parser.parse_args(list(argv) if argv is not None else None)
    rate, _ = frequency_matrix(options.csv)
    result = replay_case(
        options.case,
        options.csv,
        options.evidence,
        mode_hz=options.mode_hz,
        pace_seconds=(1.0 / rate) if options.realtime else None,
    )
    digest = hashlib.sha256(Path(options.csv).read_bytes()).hexdigest()
    record = sealed_replay_record(
        result, evidence_path=options.evidence, source_sha256=digest
    )
    rendered = json.dumps(record, indent=2, sort_keys=True)
    if options.output:
        Path(options.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI entry
    raise SystemExit(main())
