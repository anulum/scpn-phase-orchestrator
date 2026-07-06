# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — causal streaming monitor for the grid modal detector

"""A causal, real-time streaming monitor for the certified grid modal-growth detector.

The offline head-to-head (``bench.grid_modal_head_to_head``) *certifies* the grid modal
detector (``monitor.grid_modal_growth``): on real PMU data, at a matched false alarm,
its growth-rate ``σ`` leads instability transitions far more than chance. That
certification is done on fixed pre-onset segments — the training/validation harness.
This module is the step to the pinnacle: it runs the *same* detector **causally on a
live stream**, so the certified operating point becomes an operational early warning.

The monitor keeps a sliding window of the most recent per-bus voltage samples and, every
``step``, re-scores that window with the **identical primitives** the detector uses
(``per_bus_deviation``, ``envelope_growth_rate``) — so the streaming score on a window
is bit-for-bit the offline ``modal_growth_score`` on that same window. That identity is
what makes the **offline-calibrated threshold valid online**: the monitor never
recalibrates, it carries the certified threshold and fires when the live ``σ`` crosses
it (after an optional persistence debounce), latching until ``σ`` falls back below so
each instability episode raises one lead event.

:meth:`GridModalStreamMonitor.from_evidence` closes the loop: it builds the monitor from
a sealed head-to-head artefact, taking the aggregation, recency weighting, and
matched-false-alarm threshold straight from the certification — the certified detector
becomes the live monitor with no hand-set constants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scpn_phase_orchestrator.monitor.grid_modal_growth import (
    DEFAULT_AGGREGATION,
    DEFAULT_RECENCY_TOP,
    cross_bus_deviation,
    envelope_growth_rate,
    per_bus_deviation,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]

__all__ = ["GridModalStreamMonitor", "StreamAlarm", "WHOLE_NETWORK_BUS"]

#: The whole-network aggregation reports no single bus.
WHOLE_NETWORK_BUS = -1


@dataclass(frozen=True)
class StreamAlarm:
    """A lead event raised when the live growth rate ``σ`` crosses the threshold.

    Attributes
    ----------
    sample_index : int
        The stream sample index the alarm fired at (the last sample in the window).
    time_s : float
        ``sample_index / rate`` — the alarm time in seconds from the stream start.
    score : float
        The modal growth rate ``σ`` at the alarm window.
    threshold : float
        The certified matched-false-alarm threshold ``σ`` crossed.
    bus : int
        The most unstable bus under the focal aggregation (its per-bus ``σ`` is the
        maximum), or :data:`WHOLE_NETWORK_BUS` under the whole-network aggregation.
    """

    sample_index: int
    time_s: float
    score: float
    threshold: float
    bus: int


class GridModalStreamMonitor:
    """A causal sliding-window monitor carrying the certified grid detector online.

    Parameters
    ----------
    rate : float
        Sampling rate in Hz; must be positive and finite.
    threshold : float
        The certified matched-false-alarm growth-rate threshold; ``σ`` at or above it
        fires.
    window_seconds : float
        Sliding-window length in seconds, matching the offline segment length.
    step_seconds : float
        How often the window is re-scored, in seconds; must be positive.
    aggregation : str
        ``"focal"`` (most unstable bus) or ``"mean"`` (whole network), as certified.
    recency_top : float
        The recency weighting, as certified.
    persistence : int
        Consecutive re-scored windows at or above the threshold required before an alarm
        fires; ``1`` fires on the first crossing. Must be a positive integer.

    Raises
    ------
    ValueError
        If ``rate``/``window_seconds``/``step_seconds`` are not positive finite, the
        window is shorter than two samples, ``persistence`` is below one, or
        ``aggregation`` is neither ``"focal"`` nor ``"mean"``.
    """

    def __init__(
        self,
        *,
        rate: float,
        threshold: float,
        window_seconds: float = 2.0,
        step_seconds: float = 0.5,
        aggregation: str = DEFAULT_AGGREGATION,
        recency_top: float = DEFAULT_RECENCY_TOP,
        persistence: int = 1,
    ) -> None:
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("rate must be a positive finite number")
        if not np.isfinite(window_seconds) or window_seconds <= 0.0:
            raise ValueError("window_seconds must be a positive finite number")
        if not np.isfinite(step_seconds) or step_seconds <= 0.0:
            raise ValueError("step_seconds must be a positive finite number")
        if aggregation not in ("focal", "mean"):
            raise ValueError(
                f"aggregation must be 'mean' or 'focal', got {aggregation!r}"
            )
        if persistence < 1:
            raise ValueError("persistence must be a positive integer")
        self._rate = float(rate)
        self._threshold = float(threshold)
        self._window = int(round(window_seconds * rate))
        if self._window < 2:
            raise ValueError("window_seconds is too short for the sampling rate")
        self._step = max(1, int(round(step_seconds * rate)))
        self._aggregation = aggregation
        self._recency_top = float(recency_top)
        self._persistence = int(persistence)
        self._buffer: list[FloatArray] = []
        self._index = 0
        self._since_score = 0
        self._above = 0
        self._latched = False
        self._latest_score = float("nan")

    @classmethod
    def from_evidence(
        cls, evidence_path: str | Path, *, rate: float, **kwargs: object
    ) -> GridModalStreamMonitor:
        """Build a monitor from a sealed head-to-head artefact.

        Reads the aggregation, recency weighting, and matched-false-alarm threshold from
        the certified ``modal`` record, so the live monitor carries exactly the
        certified operating point with no hand-set constants.

        Parameters
        ----------
        evidence_path : str or Path
            Path to a sealed ``grid_modal_head_to_head.json`` artefact.
        rate : float
            The live stream's sampling rate in Hz.
        **kwargs : object
            Extra monitor arguments (``window_seconds``, ``step_seconds``,
            ``persistence``).

        Returns
        -------
        GridModalStreamMonitor
            A monitor at the certified operating point.
        """
        payload = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
        modal = payload["modal"]
        return cls(
            rate=rate,
            threshold=float(modal["score_threshold"]),
            aggregation=str(modal["aggregation"]),
            recency_top=float(modal["recency_top"]),
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def latest_score(self) -> float:
        """The most recent growth rate ``σ``, or NaN before the first score."""
        return self._latest_score

    @property
    def rate(self) -> float:
        """The stream sampling rate in Hz the monitor was constructed with."""
        return self._rate

    @property
    def threshold(self) -> float:
        """The certified matched-false-alarm threshold ``σ`` must reach to alarm."""
        return self._threshold

    @property
    def aggregation(self) -> str:
        """The certified aggregation scored under (``"focal"`` or ``"mean"``)."""
        return self._aggregation

    def reset(self) -> None:
        """Clear the window and alarm state, as if freshly constructed."""
        self._buffer.clear()
        self._index = 0
        self._since_score = 0
        self._above = 0
        self._latched = False
        self._latest_score = float("nan")

    def _score_window(self) -> tuple[float, int]:
        """Return the current window's growth rate ``σ`` and its most unstable bus."""
        window = np.stack(self._buffer, axis=1)  # (buses, window)
        if self._aggregation == "mean":
            score = envelope_growth_rate(
                cross_bus_deviation(window),
                rate=self._rate,
                recency_top=self._recency_top,
            )
            return score, WHOLE_NETWORK_BUS
        per_bus = [
            envelope_growth_rate(
                envelope, rate=self._rate, recency_top=self._recency_top
            )
            for envelope in per_bus_deviation(window)
        ]
        bus = int(np.argmax(per_bus))
        return per_bus[bus], bus

    def update(self, sample: FloatArray) -> StreamAlarm | None:
        """Push one per-bus voltage sample; return an alarm on a threshold crossing.

        Parameters
        ----------
        sample : FloatArray
            The bus-voltage magnitudes at one time step, shape ``(buses,)``.

        Returns
        -------
        StreamAlarm or None
            A :class:`StreamAlarm` on the sample that fires a fresh lead event, else
            ``None`` (still warming up, between re-scorings, below threshold, or already
            latched within the same episode).
        """
        values = np.asarray(sample, dtype=np.float64)
        self._index += 1
        self._buffer.append(values)
        if len(self._buffer) > self._window:
            self._buffer.pop(0)
        self._since_score += 1
        if len(self._buffer) < self._window or self._since_score < self._step:
            return None
        self._since_score = 0
        score, bus = self._score_window()
        self._latest_score = score
        if score < self._threshold:
            self._above = 0
            self._latched = False
            return None
        self._above += 1
        if self._latched or self._above < self._persistence:
            return None
        self._latched = True
        return StreamAlarm(
            sample_index=self._index,
            time_s=self._index / self._rate,
            score=score,
            threshold=self._threshold,
            bus=bus,
        )
