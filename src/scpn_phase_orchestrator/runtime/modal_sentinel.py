# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — bridge-agnostic live modal sentinel

"""Bridge-agnostic wiring from live channel observations to sealed alarms.

The R1 sentinel contract: any runtime bridge that yields one ``Mapping`` of
channel name to a real reading per frame — the MQTT and OPC-UA tag bridges, the
C37118 synchrophasor bridge, or a replayed capture — plugs into
:class:`ModalSentinel`, which assembles the fixed channel vector, drives the
certified
:class:`~scpn_phase_orchestrator.monitor.grid_modal_stream.GridModalStreamMonitor`,
and seals every alarm into a hash-addressed record carrying the operating-point
provenance. The operating point is read ONLY from a sealed evidence artefact,
verified before any value is trusted; the sentinel is review-only — it observes,
records, and never actuates.

Fail-closed observation contract: a frame must carry exactly the declared
channels — a missing channel, an unknown channel, or a non-finite reading
rejects the frame with an explicit error rather than silently degrading the
monitored vector.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.monitor.grid_modal_stream import (
    WHOLE_NETWORK_BUS,
    GridModalStreamMonitor,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Mapping

__all__ = ["ModalSentinel", "load_verified_evidence"]


def load_verified_evidence(evidence_path: str | Path) -> dict[str, object]:
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
        recompute from the record — a tampered artefact must never configure a
        live sentinel.
    """
    payload = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    record = copy.deepcopy(payload)
    sealed = record.pop("content_hash", None)
    if not isinstance(sealed, str):
        raise ValueError("evidence carries no content_hash; refusing to trust it")
    if canonical_record_hash(record) != sealed:
        raise ValueError(
            "evidence content_hash does not recompute from the record; "
            "refusing to configure a live sentinel from a tampered artefact"
        )
    return payload  # type: ignore[no-any-return]


@dataclass
class ModalSentinel:
    """Review-only live sentinel: channel observations in, sealed alarms out.

    Attributes
    ----------
    monitor : GridModalStreamMonitor
        The causal stream monitor carrying the certified operating point.
    channels : tuple[str, ...]
        The declared channel names, in the fixed vector order every frame must
        satisfy.
    provenance : dict[str, object]
        Operating-point provenance copied into every sealed alarm record.
    non_actuating : bool
        Always ``True`` — the sentinel observes and never drives hardware.
    execution_disabled : bool
        Always ``True`` — no control action is emitted from this sentinel.
    """

    monitor: GridModalStreamMonitor
    channels: tuple[str, ...]
    provenance: dict[str, object] = field(default_factory=dict)
    non_actuating: bool = field(default=True, init=False)
    execution_disabled: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("at least one channel must be declared")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channel names must be unique")

    @classmethod
    def from_sealed_evidence(
        cls,
        evidence_path: str | Path,
        *,
        case_id: str,
        rate: float,
        channels: tuple[str, ...],
        persistence: int = 1,
    ) -> ModalSentinel:
        """Build a sentinel whose operating point comes only from sealed evidence.

        The threshold comes from the sealed local calibration, the aggregation
        and recency weighting from the sealed detector block, and the window
        from the named case's sealed configuration; the step is a quarter
        window, as evaluated. The evidence content hash and the calibration's
        disclosed limits are carried into every alarm's provenance.

        Parameters
        ----------
        evidence_path : str | Path
            Path to a sealed E2.G evidence artefact.
        case_id : str
            The sealed corpus case whose window configuration to carry.
        rate : float
            The live stream's sampling rate in hertz.
        channels : tuple[str, ...]
            Declared channel names in fixed vector order.
        persistence : int
            Consecutive above-threshold re-scorings before an alarm fires.

        Returns
        -------
        ModalSentinel
            A sentinel at the sealed operating point.

        Raises
        ------
        ValueError
            If the seal fails to verify, the payload is not an E2.G record,
            the case is not in the sealed corpus, or its window is not
            numeric.
        """
        payload = load_verified_evidence(evidence_path)
        corpus = payload.get("corpus")
        if not isinstance(corpus, dict) or "transitions" not in corpus:
            raise ValueError("evidence carries no corpus.transitions block")
        entry: dict[str, object] | None = None
        for candidate in corpus["transitions"]:
            if candidate.get("case") == case_id:
                entry = dict(candidate)
                break
        if entry is None:
            raise ValueError(f"case {case_id!r} is not in the sealed corpus")
        detector = payload.get("detector")
        calibration = payload.get("local_calibration")
        if not isinstance(detector, dict) or not isinstance(calibration, dict):
            raise ValueError("evidence carries no detector/local_calibration blocks")
        window_value = entry.get("window_seconds")
        if isinstance(window_value, bool) or not isinstance(window_value, (int, float)):
            raise ValueError("sealed case entry carries no numeric window_seconds")
        window_seconds = float(window_value)
        monitor = GridModalStreamMonitor(
            rate=rate,
            threshold=float(calibration["threshold"]),
            window_seconds=window_seconds,
            step_seconds=window_seconds / 4.0,
            aggregation=str(detector["aggregation"]),
            recency_top=float(detector["recency_top"]),
            persistence=persistence,
        )
        provenance: dict[str, object] = {
            "evidence_content_hash": payload["content_hash"],
            "case": case_id,
            "threshold": calibration["threshold"],
            "calibration_n_null": calibration.get("n_null"),
        }
        return cls(monitor=monitor, channels=channels, provenance=provenance)

    def observe(self, values: Mapping[str, float]) -> dict[str, object] | None:
        """Consume one frame of channel readings; return a sealed alarm record.

        Parameters
        ----------
        values : Mapping[str, float]
            One reading per declared channel, keyed by channel name. The frame
            must carry exactly the declared channels.

        Returns
        -------
        dict[str, object] | None
            A sealed, hash-addressed alarm record when the monitor raises a
            fresh alarm on this frame, else ``None``.

        Raises
        ------
        ValueError
            If the frame misses a declared channel, carries an unknown
            channel, or any reading is boolean or not a finite real number.
        """
        unknown = set(values) - set(self.channels)
        if unknown:
            raise ValueError(f"unknown channels in frame: {sorted(unknown)}")
        vector = np.empty(len(self.channels), dtype=np.float64)
        for index, name in enumerate(self.channels):
            if name not in values:
                raise ValueError(f"frame misses declared channel {name!r}")
            reading = values[name]
            if isinstance(reading, bool) or not isinstance(reading, (int, float)):
                raise ValueError(f"channel {name!r} reading must be a real number")
            value = float(reading)
            if not np.isfinite(value):
                raise ValueError(f"channel {name!r} reading must be finite")
            vector[index] = value
        alarm = self.monitor.update(vector)
        if alarm is None:
            return None
        focal_channel = (
            None if alarm.bus == WHOLE_NETWORK_BUS else self.channels[alarm.bus]
        )
        record: dict[str, object] = {
            "kind": "modal_sentinel_alarm",
            "provenance": dict(self.provenance),
            "channels": list(self.channels),
            "sample_index": alarm.sample_index,
            "time_s": alarm.time_s,
            "score": alarm.score,
            "threshold": alarm.threshold,
            "focal_channel": focal_channel,
            "review_only": True,
        }
        record["content_hash"] = canonical_record_hash(record)
        return record
