# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — IEEE C37.118.2 synchrophasor frame codec

"""Pure decoder for IEEE C37.118.2-2011 synchrophasor CONFIG-2 and DATA frames.

This module decodes the binary framing of the IEEE synchrophasor data-transfer
protocol without any network I/O: given the raw bytes of a CONFIG-2 frame it
recovers the per-PMU measurement layout, and given the bytes of a DATA frame
plus that configuration it recovers the phasor, frequency, and analog/digital
measurements. Every frame is checksum-validated (CRC-CCITT) before its body is
read, and malformed input raises a typed :class:`SynchrophasorFrameError`
subclass rather than returning partial data.

The byte layout, CRC parameters, and field semantics were cross-checked against
two independent open-source implementations of the standard: the ``pypmu``
Python library (``iicsys/pypmu``, ``synchrophasor/frame.py``) and the C++
``Open-C37.118`` library (``marsolla/Open-C37.118``, ``src/c37118*.{h,cpp}``).
Both agree on the 14-byte common header (``SYNC`` ``FRAMESIZE`` ``IDCODE``
``SOC`` ``FRACSEC``, big-endian), the FORMAT-word field sizes, and the
CRC-CCITT checksum (polynomial ``0x1021``, initial value ``0xFFFF``, no final
mask, computed over every byte except the trailing two). The FREQ field is a
deviation from the PMU nominal frequency: a signed 16-bit integer in millihertz
when the FORMAT ``freq`` bit is clear, or a 32-bit float in hertz when it is
set. The live-socket ingestion path is
:class:`~scpn_phase_orchestrator.adapters.synchrophasor_client.C37118SessionClient`
(pure-standard-library ``asyncio``, no optional extra required); this module
deliberately handles only bytes already read.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

__all__ = [
    "FRAME_TYPE_CONFIG2",
    "FRAME_TYPE_DATA",
    "ConfigurationFrame2",
    "DataFrame",
    "FrameChecksumError",
    "FrameTruncationError",
    "PhasorUnit",
    "PmuConfiguration",
    "PmuMeasurement",
    "SynchrophasorFrameCodec",
    "SynchrophasorFrameError",
    "SynchrophasorHeader",
    "UnsupportedFrameError",
    "compute_crc_ccitt",
    "data_frames_to_frequency_series",
]

# --- Framing constants (cross-checked pypmu + Open-C37.118) ---------------

#: Leading byte of every synchrophasor frame's SYNC word (``0xAA``).
_SYNC_LEAD = 0xAA
#: Fixed size in bytes of the common frame header (SYNC+FRAMESIZE+IDCODE+SOC+FRACSEC).
_HEADER_SIZE = 14
#: Size in bytes of the trailing CRC-CCITT checksum.
_CRC_SIZE = 2
#: CRC-CCITT generating polynomial (``X^16 + X^12 + X^5 + 1``).
_CRC_POLY = 0x1021
#: CRC-CCITT initial register value.
_CRC_INIT = 0xFFFF

#: Frame-type code (SYNC byte 2, bits 6-4) for DATA frames.
FRAME_TYPE_DATA = 0
#: Frame-type code for CONFIG-2 frames.
FRAME_TYPE_CONFIG2 = 3

#: Size in bytes of a fixed-length station name / channel name field.
_NAME_FIELD_SIZE = 16

# FORMAT word (16-bit) bit masks for the per-PMU data representation.
_FORMAT_PHASOR_POLAR = 0x0001
_FORMAT_PHASOR_FLOAT = 0x0002
_FORMAT_ANALOG_FLOAT = 0x0004
_FORMAT_FREQ_FLOAT = 0x0008

# FRACSEC time-quality flag masks (top byte of the 32-bit FRACSEC word).
_TQ_LEAP_DIRECTION = 0x40
_TQ_LEAP_OCCURRED = 0x20
_TQ_LEAP_PENDING = 0x10
_TQ_MESSAGE_CODE = 0x0F
_FRACSEC_FRACTION_MASK = 0x00FFFFFF


class SynchrophasorFrameError(ValueError):
    """Base class for all synchrophasor frame decoding failures."""


class FrameTruncationError(SynchrophasorFrameError):
    """Raised when a frame is shorter than its declared or required length."""


class FrameChecksumError(SynchrophasorFrameError):
    """Raised when the trailing CRC-CCITT checksum does not match the body."""


class UnsupportedFrameError(SynchrophasorFrameError):
    """Raised for a frame whose SYNC/type is not the expected decodable kind."""


def compute_crc_ccitt(data: bytes) -> int:
    """Compute the IEEE C37.118.2 CRC-CCITT checksum of a byte string.

    The checksum uses the generating polynomial ``0x1021``
    (``X^16 + X^12 + X^5 + 1``), an initial register value of ``0xFFFF``, and no
    final mask, processing each byte most-significant-bit first. This matches the
    checksum both reference implementations apply over every frame byte except
    the trailing two CRC bytes.

    Parameters
    ----------
    data : bytes
        The bytes to checksum (a full frame excluding its trailing CRC field).

    Returns
    -------
    int
        The 16-bit CRC-CCITT value.
    """
    crc = _CRC_INIT
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ _CRC_POLY if crc & 0x8000 else crc << 1) & 0xFFFF
    return crc


class _FrameReader:
    """Sequential big-endian reader over a frame body with bounds checking."""

    def __init__(self, buffer: bytes, *, start: int, end: int) -> None:
        self._buffer = buffer
        self._offset = start
        self._end = end

    def _take(self, size: int) -> bytes:
        stop = self._offset + size
        if stop > self._end:
            raise FrameTruncationError(
                f"frame ended after {self._offset} bytes while reading {size} more"
            )
        chunk = self._buffer[self._offset : stop]
        self._offset = stop
        return chunk

    def u16(self) -> int:
        """Read an unsigned big-endian 16-bit integer."""
        return int(struct.unpack(">H", self._take(2))[0])

    def i16(self) -> int:
        """Read a signed big-endian 16-bit integer."""
        return int(struct.unpack(">h", self._take(2))[0])

    def u32(self) -> int:
        """Read an unsigned big-endian 32-bit integer."""
        return int(struct.unpack(">I", self._take(4))[0])

    def f32(self) -> float:
        """Read a big-endian 32-bit IEEE-754 float."""
        return float(struct.unpack(">f", self._take(4))[0])

    def name(self) -> str:
        """Read a fixed 16-byte station/channel name, trimmed of padding."""
        raw = self._take(_NAME_FIELD_SIZE)
        return raw.decode("ascii", errors="replace").rstrip("\x00 ")


@dataclass(frozen=True)
class SynchrophasorHeader:
    """Decoded 14-byte common header shared by every synchrophasor frame.

    Attributes
    ----------
    frame_type : int
        Frame-type code from SYNC byte 2 (bits 6-4); e.g.
        :data:`FRAME_TYPE_DATA` or :data:`FRAME_TYPE_CONFIG2`.
    version : int
        Protocol version number from SYNC byte 2 (bits 3-0).
    framesize : int
        Declared total frame size in bytes, including SYNC and CRC.
    id_code : int
        Data-stream / PMU identification code.
    soc : int
        Second-of-century timestamp (UNIX seconds).
    fracsec_raw : int
        Raw 32-bit FRACSEC word (time-quality byte plus fraction count).
    """

    frame_type: int
    version: int
    framesize: int
    id_code: int
    soc: int
    fracsec_raw: int

    @property
    def fraction_count(self) -> int:
        """Return the raw fraction-of-second count (lower 24 bits of FRACSEC)."""
        return self.fracsec_raw & _FRACSEC_FRACTION_MASK

    @property
    def message_time_quality(self) -> int:
        """Return the 4-bit message time-quality code from the FRACSEC top byte."""
        return (self.fracsec_raw >> 24) & _TQ_MESSAGE_CODE

    @property
    def leap_second_pending(self) -> bool:
        """Return whether the leap-second-pending flag is set."""
        return bool((self.fracsec_raw >> 24) & _TQ_LEAP_PENDING)

    @property
    def leap_second_occurred(self) -> bool:
        """Return whether the leap-second-occurred flag is set."""
        return bool((self.fracsec_raw >> 24) & _TQ_LEAP_OCCURRED)

    @property
    def leap_second_direction(self) -> str:
        """Return the leap-second direction (``-`` if flagged, else ``+``)."""
        return "-" if (self.fracsec_raw >> 24) & _TQ_LEAP_DIRECTION else "+"

    def seconds_of_second(self, time_base: int) -> float:
        """Return the fractional-second offset as a float given ``time_base``.

        Parameters
        ----------
        time_base : int
            The CONFIG-2 ``TIME_BASE`` resolution of the fractional timestamp.

        Returns
        -------
        float
            The fraction of a second, ``fraction_count / time_base``.

        Raises
        ------
        SynchrophasorFrameError
            If ``time_base`` is not a positive integer.
        """
        if time_base <= 0:
            raise SynchrophasorFrameError("time_base must be a positive integer")
        return self.fraction_count / time_base

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the header fields.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the header fields.
        """
        return {
            "frame_type": self.frame_type,
            "version": self.version,
            "framesize": self.framesize,
            "id_code": self.id_code,
            "soc": self.soc,
            "fraction_count": self.fraction_count,
            "message_time_quality": self.message_time_quality,
            "leap_second_pending": self.leap_second_pending,
            "leap_second_occurred": self.leap_second_occurred,
            "leap_second_direction": self.leap_second_direction,
        }


@dataclass(frozen=True)
class PhasorUnit:
    """Conversion factor for one phasor channel (a decoded PHUNIT word).

    Attributes
    ----------
    is_current : bool
        ``True`` if the channel is a current phasor, ``False`` for voltage
        (PHUNIT most-significant byte).
    scale : int
        Unsigned 24-bit scale factor in ``10**-5`` volts or amperes per bit,
        used to convert 16-bit integer phasor components to engineering units.
        Ignored for floating-point phasors, which are already in engineering
        units.
    """

    is_current: bool
    scale: int

    @property
    def volts_or_amperes_per_bit(self) -> float:
        """Return the engineering-unit scale per integer bit (``scale * 1e-5``)."""
        return self.scale * 1.0e-5

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the phasor unit.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the phasor conversion factor.
        """
        return {"is_current": self.is_current, "scale": self.scale}


@dataclass(frozen=True)
class PmuConfiguration:
    """Per-PMU measurement layout decoded from a CONFIG-2 frame.

    Attributes
    ----------
    station_name : str
        Human-readable station name (trimmed of NUL/space padding).
    id_code : int
        PMU identification code.
    phasor_polar : bool
        ``True`` if phasors are polar (magnitude, angle); ``False`` if rectangular.
    phasor_float : bool
        ``True`` if phasors use 32-bit floats; ``False`` if 16-bit integers.
    analog_float : bool
        ``True`` if analog values use 32-bit floats; ``False`` if 16-bit integers.
    freq_float : bool
        ``True`` if FREQ/DFREQ use 32-bit floats (hertz); ``False`` if 16-bit
        integers (millihertz deviation).
    phasor_count, analog_count, digital_word_count : int
        PHNMR, ANNMR, and DGNMR counts respectively.
    channel_names : tuple[str, ...]
        Phasor, analog, and digital channel labels in declared order.
    nominal_frequency_hz : float
        Nominal line frequency (50.0 or 60.0 Hz) from the FNOM word.
    phasor_units : tuple[PhasorUnit, ...]
        Per-phasor conversion factors (PHUNIT), one per phasor channel.
    """

    station_name: str
    id_code: int
    phasor_polar: bool
    phasor_float: bool
    analog_float: bool
    freq_float: bool
    phasor_count: int
    analog_count: int
    digital_word_count: int
    channel_names: tuple[str, ...]
    nominal_frequency_hz: float
    phasor_units: tuple[PhasorUnit, ...] = ()

    @property
    def phasor_size(self) -> int:
        """Return the byte size of one phasor (8 if float, else 4)."""
        return 8 if self.phasor_float else 4

    @property
    def freq_size(self) -> int:
        """Return the byte size of the FREQ/DFREQ field (4 if float, else 2)."""
        return 4 if self.freq_float else 2

    @property
    def analog_size(self) -> int:
        """Return the byte size of one analog value (4 if float, else 2)."""
        return 4 if self.analog_float else 2

    @property
    def data_block_size(self) -> int:
        """Return the byte size of this PMU's block within a DATA frame."""
        return (
            2  # STAT
            + self.phasor_count * self.phasor_size
            + 2 * self.freq_size  # FREQ + DFREQ
            + self.analog_count * self.analog_size
            + self.digital_word_count * 2
        )

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the PMU configuration.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the PMU configuration fields.
        """
        return {
            "station_name": self.station_name,
            "id_code": self.id_code,
            "phasor_polar": self.phasor_polar,
            "phasor_float": self.phasor_float,
            "analog_float": self.analog_float,
            "freq_float": self.freq_float,
            "phasor_count": self.phasor_count,
            "analog_count": self.analog_count,
            "digital_word_count": self.digital_word_count,
            "channel_names": list(self.channel_names),
            "nominal_frequency_hz": self.nominal_frequency_hz,
            "phasor_units": [unit.to_audit_record() for unit in self.phasor_units],
        }


@dataclass(frozen=True)
class ConfigurationFrame2:
    """Decoded CONFIG-2 frame describing every PMU in the data stream.

    Attributes
    ----------
    header : SynchrophasorHeader
        The decoded common header.
    time_base : int
        Resolution of the fractional-second timestamp (TIME_BASE).
    pmus : tuple[PmuConfiguration, ...]
        Per-PMU measurement layouts in declared order.
    data_rate : int
        Reporting rate: frames per second if positive, seconds per frame if
        negative.
    """

    header: SynchrophasorHeader
    time_base: int
    pmus: tuple[PmuConfiguration, ...]
    data_rate: int

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the configuration frame.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the configuration frame.
        """
        return {
            "header": self.header.to_audit_record(),
            "time_base": self.time_base,
            "pmu_count": len(self.pmus),
            "pmus": [pmu.to_audit_record() for pmu in self.pmus],
            "data_rate": self.data_rate,
        }


@dataclass(frozen=True)
class PmuMeasurement:
    """One PMU's measurements decoded from a DATA frame block.

    Attributes
    ----------
    stat : int
        16-bit STAT flag word.
    phasors : tuple[tuple[float, float], ...]
        Phasor components in the frame's native representation: rectangular
        ``(real, imag)`` or polar ``(magnitude, angle)`` per the PMU's FORMAT.
    frequency_hz : float
        Absolute frequency in hertz (nominal plus the decoded deviation).
    frequency_deviation : float
        Raw FREQ deviation from nominal (millihertz if integer, hertz if float).
    df_dt : float
        Rate-of-change of frequency (DFREQ) in the frame's native units.
    analogs : tuple[float, ...]
        Analog channel values.
    digitals : tuple[int, ...]
        Digital status words.
    """

    stat: int
    phasors: tuple[tuple[float, float], ...]
    frequency_hz: float
    frequency_deviation: float
    df_dt: float
    analogs: tuple[float, ...]
    digitals: tuple[int, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the PMU measurement.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the PMU measurement.
        """
        return {
            "stat": self.stat,
            "phasors": [list(component) for component in self.phasors],
            "frequency_hz": self.frequency_hz,
            "frequency_deviation": self.frequency_deviation,
            "df_dt": self.df_dt,
            "analogs": list(self.analogs),
            "digitals": list(self.digitals),
        }


@dataclass(frozen=True)
class DataFrame:
    """Decoded DATA frame carrying one measurement per configured PMU.

    Attributes
    ----------
    header : SynchrophasorHeader
        The decoded common header.
    measurements : tuple[PmuMeasurement, ...]
        Per-PMU measurements aligned with the configuration's PMU order.
    """

    header: SynchrophasorHeader
    measurements: tuple[PmuMeasurement, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit mapping of the data frame.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe mapping of the data frame.
        """
        return {
            "header": self.header.to_audit_record(),
            "measurements": [m.to_audit_record() for m in self.measurements],
        }


class SynchrophasorFrameCodec:
    """Stateless decoder for IEEE C37.118.2-2011 CONFIG-2 and DATA frames.

    The codec performs no network I/O: each method accepts the raw bytes of a
    single frame, validates its SYNC word, declared size, and CRC-CCITT
    checksum, and returns a fully decoded, immutable frame object. Any structural
    fault raises a :class:`SynchrophasorFrameError` subclass; the codec never
    returns partially decoded data.
    """

    def _validate_common(
        self, frame: bytes, *, expected_type: int
    ) -> SynchrophasorHeader:
        """Validate the SYNC word, framesize, and CRC, returning the header."""
        if len(frame) < _HEADER_SIZE + _CRC_SIZE:
            raise FrameTruncationError(
                f"frame of {len(frame)} bytes is shorter than the minimum "
                f"{_HEADER_SIZE + _CRC_SIZE}"
            )
        if frame[0] != _SYNC_LEAD:
            raise UnsupportedFrameError(
                f"frame does not begin with SYNC lead 0xAA (got 0x{frame[0]:02X})"
            )
        sync_type = frame[1]
        frame_type = (sync_type >> 4) & 0x07
        version = sync_type & 0x0F
        if frame_type != expected_type:
            raise UnsupportedFrameError(
                f"expected frame type {expected_type}, decoded {frame_type}"
            )
        framesize = struct.unpack(">H", frame[2:4])[0]
        if framesize != len(frame):
            raise FrameTruncationError(
                f"declared framesize {framesize} does not match {len(frame)} bytes"
            )
        expected_crc = struct.unpack(">H", frame[-_CRC_SIZE:])[0]
        actual_crc = compute_crc_ccitt(frame[:-_CRC_SIZE])
        if expected_crc != actual_crc:
            raise FrameChecksumError(
                f"CRC mismatch: frame carries 0x{expected_crc:04X}, "
                f"computed 0x{actual_crc:04X}"
            )
        id_code = struct.unpack(">H", frame[4:6])[0]
        soc = struct.unpack(">I", frame[6:10])[0]
        fracsec_raw = struct.unpack(">I", frame[10:14])[0]
        return SynchrophasorHeader(
            frame_type=frame_type,
            version=version,
            framesize=framesize,
            id_code=id_code,
            soc=soc,
            fracsec_raw=fracsec_raw,
        )

    def decode_config2(self, frame: bytes) -> ConfigurationFrame2:
        """Decode a CONFIG-2 frame into its per-PMU measurement layout.

        Parameters
        ----------
        frame : bytes
            The complete CONFIG-2 frame, including SYNC and trailing CRC.

        Returns
        -------
        ConfigurationFrame2
            The decoded configuration.

        Raises
        ------
        SynchrophasorFrameError
            If the frame is truncated, has the wrong SYNC/type, or fails CRC.
        """
        header = self._validate_common(frame, expected_type=FRAME_TYPE_CONFIG2)
        reader = _FrameReader(frame, start=_HEADER_SIZE, end=len(frame) - _CRC_SIZE)
        time_base = reader.u32() & 0x00FFFFFF
        pmu_count = reader.u16()
        pmus = tuple(self._decode_pmu_config(reader) for _ in range(pmu_count))
        data_rate = reader.i16()
        return ConfigurationFrame2(
            header=header,
            time_base=time_base,
            pmus=pmus,
            data_rate=data_rate,
        )

    def _decode_pmu_config(self, reader: _FrameReader) -> PmuConfiguration:
        """Decode one PMU's CONFIG-2 station block from ``reader``."""
        station_name = reader.name()
        id_code = reader.u16()
        fmt = reader.u16()
        phasor_count = reader.u16()
        analog_count = reader.u16()
        digital_word_count = reader.u16()
        name_count = phasor_count + analog_count + 16 * digital_word_count
        channel_names = tuple(reader.name() for _ in range(name_count))
        # PHUNIT conversion factors: MSB selects voltage/current, lower 24 bits
        # are the 10^-5 V/A per-bit scale for integer phasor components.
        phasor_units = tuple(self._decode_phunit(reader) for _ in range(phasor_count))
        # ANUNIT and DIGUNIT (4 bytes each) are not modelled; skip them.
        for _ in range(analog_count + digital_word_count):
            reader.u32()
        fnom = reader.u16()
        reader.u16()  # CFGCNT — configuration change count.
        nominal = 50.0 if fnom & 0x0001 else 60.0
        return PmuConfiguration(
            station_name=station_name,
            id_code=id_code,
            phasor_polar=bool(fmt & _FORMAT_PHASOR_POLAR),
            phasor_float=bool(fmt & _FORMAT_PHASOR_FLOAT),
            analog_float=bool(fmt & _FORMAT_ANALOG_FLOAT),
            freq_float=bool(fmt & _FORMAT_FREQ_FLOAT),
            phasor_count=phasor_count,
            analog_count=analog_count,
            digital_word_count=digital_word_count,
            channel_names=channel_names,
            nominal_frequency_hz=nominal,
            phasor_units=phasor_units,
        )

    def _decode_phunit(self, reader: _FrameReader) -> PhasorUnit:
        """Decode one 4-byte PHUNIT conversion factor from ``reader``."""
        word = reader.u32()
        return PhasorUnit(is_current=bool(word & 0xFF000000), scale=word & 0x00FFFFFF)

    def decode_data(self, frame: bytes, config: ConfigurationFrame2) -> DataFrame:
        """Decode a DATA frame using a previously decoded CONFIG-2 layout.

        Parameters
        ----------
        frame : bytes
            The complete DATA frame, including SYNC and trailing CRC.
        config : ConfigurationFrame2
            The configuration describing each PMU's measurement layout.

        Returns
        -------
        DataFrame
            The decoded measurements, one block per configured PMU.

        Raises
        ------
        SynchrophasorFrameError
            If the frame is truncated, has the wrong SYNC/type, or fails CRC.
        """
        header = self._validate_common(frame, expected_type=FRAME_TYPE_DATA)
        reader = _FrameReader(frame, start=_HEADER_SIZE, end=len(frame) - _CRC_SIZE)
        measurements = tuple(
            self._decode_pmu_measurement(reader, pmu) for pmu in config.pmus
        )
        return DataFrame(header=header, measurements=measurements)

    def _decode_pmu_measurement(
        self, reader: _FrameReader, pmu: PmuConfiguration
    ) -> PmuMeasurement:
        """Decode one PMU's DATA block from ``reader`` using its ``pmu`` layout."""
        stat = reader.u16()
        phasors = tuple(self._read_phasor(reader, pmu) for _ in range(pmu.phasor_count))
        deviation, frequency_hz = self._read_frequency(reader, pmu)
        df_dt = reader.f32() if pmu.freq_float else float(reader.i16())
        analogs = tuple(
            reader.f32() if pmu.analog_float else float(reader.i16())
            for _ in range(pmu.analog_count)
        )
        digitals = tuple(reader.u16() for _ in range(pmu.digital_word_count))
        return PmuMeasurement(
            stat=stat,
            phasors=phasors,
            frequency_hz=frequency_hz,
            frequency_deviation=deviation,
            df_dt=df_dt,
            analogs=analogs,
            digitals=digitals,
        )

    def _read_phasor(
        self, reader: _FrameReader, pmu: PmuConfiguration
    ) -> tuple[float, float]:
        """Read one phasor's two components in the PMU's native representation."""
        if pmu.phasor_float:
            return reader.f32(), reader.f32()
        return float(reader.i16()), float(reader.i16())

    def _read_frequency(
        self, reader: _FrameReader, pmu: PmuConfiguration
    ) -> tuple[float, float]:
        """Read FREQ and return ``(raw_deviation, absolute_hz)``.

        A float FREQ is a deviation in hertz; an integer FREQ is a deviation in
        millihertz. Both are added to the PMU's nominal frequency to yield the
        absolute value.
        """
        if pmu.freq_float:
            deviation = reader.f32()
            absolute = pmu.nominal_frequency_hz + deviation
        else:
            deviation = float(reader.i16())
            absolute = pmu.nominal_frequency_hz + deviation / 1000.0
        return deviation, absolute


def data_frames_to_frequency_series(
    config: ConfigurationFrame2,
    frames: tuple[DataFrame, ...],
    *,
    pmu_index: int = 0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Assemble a ``(time_s, frequency_hz)`` series for one PMU across frames.

    The time vector is relative to the first frame, combining the second-of-
    century count and the fractional-second offset resolved against the
    configuration's ``TIME_BASE``; the frequency vector reports each frame's
    absolute frequency for the selected PMU. The result mirrors the two-column
    ``time_s,frequency_hz`` layout consumed by the PMU ringdown screener, so a
    decoded synchrophasor stream feeds directly into ringdown evidence.

    Parameters
    ----------
    config : ConfigurationFrame2
        The configuration whose ``TIME_BASE`` and PMU order the frames follow.
    frames : tuple[DataFrame, ...]
        The DATA frames in acquisition order.
    pmu_index : int, optional
        Index of the PMU whose frequency series is extracted (default ``0``).

    Returns
    -------
    tuple[tuple[float, ...], tuple[float, ...]]
        The relative-time vector in seconds and the frequency vector in hertz.

    Raises
    ------
    SynchrophasorFrameError
        If ``frames`` is empty or ``pmu_index`` is out of range for a frame.
    """
    if not frames:
        raise SynchrophasorFrameError("at least one DATA frame is required")
    if not 0 <= pmu_index < len(config.pmus):
        raise SynchrophasorFrameError(
            f"pmu_index {pmu_index} out of range for {len(config.pmus)} PMUs"
        )
    times: list[float] = []
    frequencies: list[float] = []
    first = frames[0].header
    base_seconds = first.soc + first.seconds_of_second(config.time_base)
    for frame in frames:
        if pmu_index >= len(frame.measurements):
            raise SynchrophasorFrameError(
                f"pmu_index {pmu_index} out of range for a frame with "
                f"{len(frame.measurements)} measurements"
            )
        absolute = frame.header.soc + frame.header.seconds_of_second(config.time_base)
        times.append(absolute - base_seconds)
        frequencies.append(frame.measurements[pmu_index].frequency_hz)
    return tuple(times), tuple(frequencies)
