# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid modal-growth stream replay

"""Replay a recorded per-bus voltage stream through the live grid modal monitor.

The streaming monitor (``GridModalStreamMonitor``) is causal — it consumes one per-bus
sample at a time and raises a ``StreamAlarm`` when the live growth rate crosses the
certified threshold. This module drives it from a *recorded* stream, sample by sample in
order, exactly as a live PMU feed would, so the certified detector's operational
behaviour can be measured end-to-end without any hardware: how many transitions it
leads, and by how much.

:func:`replay` feeds every sample and collects the alarms. :func:`replay_lead_time` adds
the transition semantics: given the onset sample, it reports whether an alarm fired
*before* the onset (a genuine lead) and the lead time in seconds of the first such alarm
— the operational quantity an operator cares about. The replay is a thin, deterministic
driver over the already-tested monitor; it adds no detection logic of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from numpy.typing import NDArray

    from scpn_phase_orchestrator.monitor.grid_modal_stream import (
        GridModalStreamMonitor,
        StreamAlarm,
    )

    FloatArray = NDArray[np.float64]

__all__ = ["ReplayResult", "replay", "replay_lead_time"]


@dataclass(frozen=True)
class ReplayResult:
    """The outcome of replaying a recorded stream against a monitor.

    Attributes
    ----------
    alarms : tuple of StreamAlarm
        Every alarm the monitor raised over the replay, in order.
    led : bool
        Whether any alarm fired at or before the onset sample — a genuine lead.
    lead_time_s : float
        The lead time in seconds of the first pre-onset alarm (onset minus the alarm
        sample, over the rate); ``nan`` when no alarm led the onset.
    first_alarm : StreamAlarm or None
        The first pre-onset alarm, or ``None`` when none led the onset.
    """

    alarms: tuple[StreamAlarm, ...]
    led: bool
    lead_time_s: float
    first_alarm: StreamAlarm | None


def replay(voltages: FloatArray, monitor: GridModalStreamMonitor) -> list[StreamAlarm]:
    """Feed every per-bus sample of a recorded stream into the monitor, in order.

    Parameters
    ----------
    voltages : FloatArray
        The recorded per-bus voltages, shape ``(buses, samples)``, replayed column by
        column as a live feed would deliver them.
    monitor : GridModalStreamMonitor
        The causal monitor to drive; it is updated in place.

    Returns
    -------
    list of StreamAlarm
        Every alarm raised over the replay, in sample order.

    Raises
    ------
    ValueError
        If ``voltages`` is not a two-dimensional buses-by-samples array.
    """
    values = np.asarray(voltages, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("voltages must be two-dimensional (buses × samples)")
    alarms: list[StreamAlarm] = []
    for index in range(values.shape[1]):
        alarm = monitor.update(values[:, index])
        if alarm is not None:
            alarms.append(alarm)
    return alarms


def replay_lead_time(
    voltages: FloatArray,
    monitor: GridModalStreamMonitor,
    *,
    onset_sample: int,
) -> ReplayResult:
    """Replay a recorded stream and measure the lead time before a known onset.

    The monitor counts stream samples from one, so ``onset_sample`` is the number of
    samples up to and including the disturbance onset. An alarm at or before that sample
    is a genuine lead; the first such alarm's lead time is ``(onset_sample − its sample
    index) / rate`` seconds.

    Parameters
    ----------
    voltages : FloatArray
        The recorded per-bus voltages, shape ``(buses, samples)``.
    monitor : GridModalStreamMonitor
        The causal monitor to drive.
    onset_sample : int
        The stream sample count at the disturbance onset; must be positive.

    Returns
    -------
    ReplayResult
        The alarms, whether the onset was led, and the first pre-onset lead time.

    Raises
    ------
    ValueError
        If ``onset_sample`` is not positive.
    """
    if onset_sample <= 0:
        raise ValueError("onset_sample must be a positive sample count")
    alarms = replay(voltages, monitor)
    leading = [alarm for alarm in alarms if alarm.sample_index <= onset_sample]
    if not leading:
        return ReplayResult(
            alarms=tuple(alarms), led=False, lead_time_s=float("nan"), first_alarm=None
        )
    first = leading[0]
    lead_time_s = (onset_sample - first.sample_index) / monitor.rate
    return ReplayResult(
        alarms=tuple(alarms), led=True, lead_time_s=lead_time_s, first_alarm=first
    )
