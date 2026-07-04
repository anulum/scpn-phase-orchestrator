# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — IEEE multi-header PMU CSV ingestion adapter

"""Adapt an IEEE-format multi-header PMU concentrator CSV into ingester input.

Phasor-measurement concentrators and the oscillation-detection literature export
captures in a wide, multi-header layout: a channel-label row, a quantity-type row
(``T`` for time, ``F`` for frequency, ``VM``/``VA``/``IM``/``IA`` for the phasor
channels), a unit row, and a secondary-label row, followed by numeric samples with
one time column and five channels per phasor-measurement unit. The ringdown
screener consumes a two-column ``time_s,frequency_hz`` series instead. This module
bridges the two: it parses the multi-header layout, enumerates the frequency
channels with their dropout counts, selects the channel that is free of dropouts
and sits within a plausible band of the nominal grid frequency (breaking ties
toward the largest peak-to-peak swing, which carries the most oscillation
content), and writes the selected channel as the screener's input with a hashed
provenance record linking the derived CSV back to the source capture.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "IEEE_FREQUENCY_QUANTITY",
    "IEEE_FREQUENCY_UNIT",
    "IEEE_TIME_QUANTITY",
    "AdaptedIngesterCSV",
    "IEEEPMURecording",
    "PMUFrequencyChannel",
    "adapt_ieee_pmu_csv",
    "read_ieee_pmu_recording",
    "write_ingester_csv",
]

#: Quantity-type token marking the time column in an IEEE PMU header.
IEEE_TIME_QUANTITY = "T"
#: Quantity-type token marking a frequency channel in an IEEE PMU header.
IEEE_FREQUENCY_QUANTITY = "F"
#: Unit token expected on a frequency channel when a unit row is present.
IEEE_FREQUENCY_UNIT = "HZ"
#: First-cell tokens that identify the unit row by its time-column unit.
_TIME_UNITS = frozenset({"SEC", "S", "SECOND", "SECONDS"})


@dataclass(frozen=True, slots=True)
class PMUFrequencyChannel:
    """One frequency channel extracted from an IEEE-format PMU capture.

    Attributes
    ----------
    label : str
        Channel label from the header's label row (typically a substation and
        line identifier shared by the phasor unit's five channels).
    column_index : int
        Zero-based column index of the channel in the source CSV.
    samples : FloatArray
        Frequency samples in hertz, aligned with the recording's time vector.
    zero_count : int
        Number of exact-zero samples, the concentrator's dropout marker.
    nonfinite_count : int
        Number of non-finite samples (``NaN`` or infinity).
    mean_hz : float
        Mean of the finite samples in hertz, or ``NaN`` if none are finite.
    min_hz : float
        Minimum finite sample in hertz, or ``NaN`` if none are finite.
    max_hz : float
        Maximum finite sample in hertz, or ``NaN`` if none are finite.
    """

    label: str
    column_index: int
    samples: FloatArray
    zero_count: int
    nonfinite_count: int
    mean_hz: float
    min_hz: float
    max_hz: float

    @property
    def peak_to_peak_hz(self) -> float:
        """Return the finite peak-to-peak swing in hertz, or ``NaN`` if empty."""
        return self.max_hz - self.min_hz

    @property
    def identifier(self) -> str:
        """Return a label and column identifier unique within the recording."""
        return f"{self.label}#{self.column_index}"

    @property
    def is_clean(self) -> bool:
        """Return whether the channel is free of dropout and non-finite samples."""
        return self.zero_count == 0 and self.nonfinite_count == 0

    def is_within_band(self, nominal_frequency_hz: float, band_hz: float) -> bool:
        """Return whether the finite mean sits within ``band_hz`` of nominal.

        Parameters
        ----------
        nominal_frequency_hz : float
            Nominal grid frequency the channel is expected to hover around.
        band_hz : float
            Half-width in hertz of the accepted band about the nominal frequency.

        Returns
        -------
        bool
            ``True`` when the finite mean is within the band, ``False`` when it
            is outside the band or no samples are finite.
        """
        if not np.isfinite(self.mean_hz):
            return False
        return abs(self.mean_hz - nominal_frequency_hz) <= band_hz


@dataclass(frozen=True, slots=True)
class IEEEPMURecording:
    """A parsed IEEE-format multi-header PMU capture.

    Attributes
    ----------
    source_name : str
        Basename of the parsed source CSV.
    source_sha256 : str
        SHA-256 digest of the exact source CSV bytes.
    times : FloatArray
        Capture time vector in seconds shared by every channel.
    channels : tuple[PMUFrequencyChannel, ...]
        Frequency channels in source-column order.
    """

    source_name: str
    source_sha256: str
    times: FloatArray
    channels: tuple[PMUFrequencyChannel, ...]

    def select_cleanest_channel(
        self,
        *,
        nominal_frequency_hz: float = 60.0,
        plausible_band_hz: float = 2.0,
    ) -> PMUFrequencyChannel:
        """Return the dropout-free in-band channel with the largest swing.

        A channel qualifies when it carries no dropout or non-finite samples and
        its mean sits within ``plausible_band_hz`` of the nominal frequency,
        which rejects dead channels reading zero and channels reported against a
        different nominal. Among the qualifying channels the one with the largest
        peak-to-peak swing is chosen, since it carries the most oscillation
        content for ringdown screening; ties break toward the lowest column index
        for determinism.

        Parameters
        ----------
        nominal_frequency_hz : float
            Nominal grid frequency the channel is expected to hover around.
        plausible_band_hz : float
            Half-width in hertz of the band about the nominal frequency within
            which a channel mean is accepted.

        Returns
        -------
        PMUFrequencyChannel
            The selected frequency channel.

        Raises
        ------
        ValueError
            If the controls are invalid or no channel qualifies.
        """
        nominal = _positive_float(nominal_frequency_hz, "nominal_frequency_hz")
        band = _positive_float(plausible_band_hz, "plausible_band_hz")
        qualifying = [
            channel
            for channel in self.channels
            if channel.is_clean and channel.is_within_band(nominal, band)
        ]
        if not qualifying:
            dropout = sum(1 for channel in self.channels if not channel.is_clean)
            raise ValueError(
                f"no frequency channel within {band} Hz of {nominal} Hz among "
                f"{len(self.channels)} channels ({dropout} carried dropouts)"
            )
        return max(
            qualifying,
            key=lambda channel: (channel.peak_to_peak_hz, -channel.column_index),
        )


@dataclass(frozen=True, slots=True)
class AdaptedIngesterCSV:
    """Provenance of a screener-ready CSV derived from an IEEE PMU capture.

    Attributes
    ----------
    source_name : str
        Basename of the source IEEE PMU CSV.
    source_sha256 : str
        SHA-256 digest of the source CSV bytes.
    output_name : str
        Basename of the written ingester CSV.
    output_sha256 : str
        SHA-256 digest of the written ingester CSV bytes.
    channel_label : str
        Label of the selected frequency channel.
    channel_column_index : int
        Source-column index of the selected frequency channel.
    time_column : str
        Timestamp column name written to the ingester CSV.
    frequency_column : str
        Frequency column name written to the ingester CSV.
    row_count : int
        Number of sample rows written.
    """

    source_name: str
    source_sha256: str
    output_name: str
    output_sha256: str
    channel_label: str
    channel_column_index: int
    time_column: str
    frequency_column: str
    row_count: int


def read_ieee_pmu_recording(path: str | Path) -> IEEEPMURecording:
    """Parse an IEEE-format multi-header PMU CSV into a recording.

    The header block is located by the quantity-type row — the first row whose
    first cell is the time token ``T`` — with the label row immediately above it
    and the data rows starting at the first row whose first cell parses as a
    number. When a unit row is present it is cross-checked so that every parsed
    frequency channel is reported in hertz.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the IEEE-format PMU concentrator CSV.

    Returns
    -------
    IEEEPMURecording
        The time vector and the frequency channels with their dropout counts.

    Raises
    ------
    ValueError
        If the header block, time column, frequency channels, or numeric samples
        cannot be parsed.
    """
    csv_path = Path(path)
    source_bytes = csv_path.read_bytes()
    rows = list(csv.reader(io.StringIO(source_bytes.decode("utf-8"))))

    quantity_index = _quantity_row_index(rows)
    quantity = rows[quantity_index]
    labels = rows[quantity_index - 1]
    data_start = _data_start_index(rows, quantity_index)
    units = _unit_row(rows, quantity_index, data_start)

    frequency_columns = _frequency_columns(quantity, units)
    max_column = max(frequency_columns)
    times, channel_samples = _read_samples(
        rows, data_start, max_column, frequency_columns
    )
    channels = tuple(
        _build_channel(labels, column, channel_samples[column])
        for column in frequency_columns
    )
    return IEEEPMURecording(
        source_name=csv_path.name,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        times=times,
        channels=channels,
    )


def write_ingester_csv(
    recording: IEEEPMURecording,
    channel: PMUFrequencyChannel,
    dest: str | Path,
    *,
    time_column: str = "time_s",
    frequency_column: str = "frequency_hz",
) -> AdaptedIngesterCSV:
    """Write one frequency channel as the ringdown screener's input CSV.

    Samples are written with Python's shortest round-tripping decimal so the
    derived CSV is deterministic and reparses to the same values the screener
    would have read from the source.

    Parameters
    ----------
    recording : IEEEPMURecording
        The parsed capture supplying the shared time vector.
    channel : PMUFrequencyChannel
        The frequency channel to write; its samples must align with the time
        vector.
    dest : str | pathlib.Path
        Destination path for the two-column ingester CSV.
    time_column : str
        Timestamp column name written to the CSV.
    frequency_column : str
        Frequency column name written to the CSV.

    Returns
    -------
    AdaptedIngesterCSV
        Provenance linking the written CSV back to the source capture.

    Raises
    ------
    ValueError
        If the column names are blank or the channel is not aligned with the
        recording's time vector.
    """
    time_field = _non_empty_str(time_column, "time_column")
    frequency_field = _non_empty_str(frequency_column, "frequency_column")
    if channel.samples.shape[0] != recording.times.shape[0]:
        raise ValueError(
            "channel samples are not aligned with the recording time vector"
        )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow([time_field, frequency_field])
    for time_value, frequency_value in zip(
        recording.times, channel.samples, strict=True
    ):
        writer.writerow([str(float(time_value)), str(float(frequency_value))])
    output_bytes = buffer.getvalue().encode("utf-8")
    dest_path = Path(dest)
    dest_path.write_bytes(output_bytes)
    return AdaptedIngesterCSV(
        source_name=recording.source_name,
        source_sha256=recording.source_sha256,
        output_name=dest_path.name,
        output_sha256=hashlib.sha256(output_bytes).hexdigest(),
        channel_label=channel.label,
        channel_column_index=channel.column_index,
        time_column=time_field,
        frequency_column=frequency_field,
        row_count=int(recording.times.shape[0]),
    )


def adapt_ieee_pmu_csv(
    source: str | Path,
    dest: str | Path,
    *,
    nominal_frequency_hz: float = 60.0,
    plausible_band_hz: float = 2.0,
    time_column: str = "time_s",
    frequency_column: str = "frequency_hz",
) -> AdaptedIngesterCSV:
    """Adapt an IEEE PMU capture into the screener's input in one call.

    Parameters
    ----------
    source : str | pathlib.Path
        Path to the IEEE-format PMU concentrator CSV.
    dest : str | pathlib.Path
        Destination path for the derived two-column ingester CSV.
    nominal_frequency_hz : float
        Nominal grid frequency used to reject out-of-band channels.
    plausible_band_hz : float
        Half-width in hertz of the accepted band about the nominal frequency.
    time_column : str
        Timestamp column name written to the derived CSV.
    frequency_column : str
        Frequency column name written to the derived CSV.

    Returns
    -------
    AdaptedIngesterCSV
        Provenance linking the derived CSV back to the source capture.

    Raises
    ------
    ValueError
        If the source cannot be parsed or no channel qualifies for selection.
    """
    recording = read_ieee_pmu_recording(source)
    channel = recording.select_cleanest_channel(
        nominal_frequency_hz=nominal_frequency_hz,
        plausible_band_hz=plausible_band_hz,
    )
    return write_ingester_csv(
        recording,
        channel,
        dest,
        time_column=time_column,
        frequency_column=frequency_column,
    )


def _quantity_row_index(rows: list[list[str]]) -> int:
    """Return the index of the quantity-type row, else raise ``ValueError``.

    The quantity-type row is the first row whose first cell is the time token,
    and it must have a label row above it.
    """
    for index, row in enumerate(rows):
        if row and row[0].strip().upper() == IEEE_TIME_QUANTITY:
            if index == 0:
                raise ValueError(
                    "IEEE PMU CSV quantity-type row has no label row above it"
                )
            return index
    raise ValueError(
        f"IEEE PMU CSV has no quantity-type row (first cell {IEEE_TIME_QUANTITY!r})"
    )


def _data_start_index(rows: list[list[str]], quantity_index: int) -> int:
    """Return the first numeric-sample row index, else raise ``ValueError``."""
    for index in range(quantity_index + 1, len(rows)):
        row = rows[index]
        if row and _is_float(row[0]):
            return index
    raise ValueError("IEEE PMU CSV has no numeric sample rows after the header")


def _unit_row(
    rows: list[list[str]], quantity_index: int, data_start: int
) -> list[str] | None:
    """Return the unit row among the header's auxiliary rows, else ``None``.

    The unit row is identified by its time-column unit rather than its position,
    so a capture that omits units — leaving only a secondary-label row between
    the quantity-type row and the data — is not mistaken for one that carries
    them.
    """
    for index in range(quantity_index + 1, data_start):
        row = rows[index]
        if row and row[0].strip().upper() in _TIME_UNITS:
            return row
    return None


def _frequency_columns(quantity: list[str], units: list[str] | None) -> tuple[int, ...]:
    """Return the frequency-channel column indices, else raise ``ValueError``.

    A column is a frequency channel when its quantity type is ``F``; when a unit
    row is present the column's unit must be hertz, which guards against a header
    whose quantity and unit rows are misaligned.
    """
    columns: list[int] = []
    for column, token in enumerate(quantity):
        if column == 0 or token.strip().upper() != IEEE_FREQUENCY_QUANTITY:
            continue
        if units is not None:
            unit = units[column].strip().upper() if column < len(units) else ""
            if unit and unit != IEEE_FREQUENCY_UNIT:
                raise ValueError(
                    f"frequency channel in column {column} is reported in "
                    f"{unit!r}, not {IEEE_FREQUENCY_UNIT!r}"
                )
        columns.append(column)
    if not columns:
        raise ValueError(
            f"IEEE PMU CSV has no frequency channels (quantity "
            f"{IEEE_FREQUENCY_QUANTITY!r})"
        )
    return tuple(columns)


def _read_samples(
    rows: list[list[str]],
    data_start: int,
    max_column: int,
    frequency_columns: tuple[int, ...],
) -> tuple[FloatArray, dict[int, FloatArray]]:
    """Return the time vector and per-column sample arrays from the data rows."""
    times: list[float] = []
    columns: dict[int, list[float]] = {column: [] for column in frequency_columns}
    for offset, row in enumerate(rows[data_start:]):
        if not row or not row[0].strip():
            continue
        row_number = data_start + offset + 1
        if len(row) <= max_column:
            raise ValueError(
                f"IEEE PMU CSV row {row_number} has {len(row)} columns, fewer "
                f"than the {max_column + 1} required by the frequency channels"
            )
        times.append(_parse_float(row[0], f"time row {row_number}"))
        for column in frequency_columns:
            columns[column].append(
                _parse_float(row[column], f"column {column} row {row_number}")
            )
    return (
        np.ascontiguousarray(times, dtype=np.float64),
        {
            column: np.ascontiguousarray(values, dtype=np.float64)
            for column, values in columns.items()
        },
    )


def _build_channel(
    labels: list[str], column: int, samples: FloatArray
) -> PMUFrequencyChannel:
    """Return a frequency channel with its dropout counts and finite statistics."""
    label = labels[column].strip() if column < len(labels) else ""
    finite = np.isfinite(samples)
    finite_samples = samples[finite]
    if finite_samples.size:
        mean_hz = float(np.mean(finite_samples))
        min_hz = float(np.min(finite_samples))
        max_hz = float(np.max(finite_samples))
    else:
        mean_hz = min_hz = max_hz = float("nan")
    return PMUFrequencyChannel(
        label=label or f"column_{column}",
        column_index=column,
        samples=samples,
        zero_count=int(np.count_nonzero(samples == 0.0)),
        nonfinite_count=int(np.count_nonzero(~finite)),
        mean_hz=mean_hz,
        min_hz=min_hz,
        max_hz=max_hz,
    )


def _is_float(value: str) -> bool:
    """Return whether ``value`` parses as a float."""
    try:
        float(value)
    except ValueError:
        return False
    return True


def _parse_float(value: str, name: str) -> float:
    """Return ``value`` parsed as a float, else raise ``ValueError``."""
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc


def _non_empty_str(value: str, name: str) -> str:
    """Return ``value`` if it is a non-empty string, else raise ``ValueError``."""
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_float(value: float, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return scalar
