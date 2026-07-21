# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — IEEE C37.118.2 live synchrophasor session client

"""Live asynchronous IEEE C37.118.2 synchrophasor session client.

Reads synchrophasor frames from a phasor data concentrator (PDC) or PMU over a
TCP stream using only the standard library's :mod:`asyncio` — no third-party
dependency. The client issues the standard C37.118.2 command frames to drive the
data stream: it requests the CONFIG-2 frame, turns data transmission on, reads
the requested number of DATA frames, and turns transmission off again. These
command frames are a benign protocol handshake that controls only the
measurement data stream; the client never writes device setpoints and cannot
actuate grid equipment (``non_actuating``).

The command-word values were verified at source against two independent
references: the ``iicsys/pypmu`` ``CommandFrame`` table and the Wireshark
synchrophasor dissector (``epan/dissectors/packet-synphasor.c``, which cites the
standard's Table 15): ``0x0001`` data-off, ``0x0002`` data-on, ``0x0003`` send
HDR, ``0x0004`` send CONFIG-1, ``0x0005`` send CONFIG-2, ``0x0006`` send
CONFIG-3. A command frame is the 14-byte common header plus a 2-byte command
word plus the CRC (18 bytes total). Frames are reassembled from the stream using
the SYNC/FRAMESIZE prefix, and decoding is delegated to
:class:`~scpn_phase_orchestrator.adapters.synchrophasor_c37118.SynchrophasorFrameCodec`.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scpn_phase_orchestrator.adapters.synchrophasor_c37118 import (
    FRAME_TYPE_CONFIG2,
    FRAME_TYPE_DATA,
    FrameTruncationError,
    SynchrophasorFrameCodec,
    UnsupportedFrameError,
    compute_crc_ccitt,
)

if TYPE_CHECKING:
    from scpn_phase_orchestrator.adapters.synchrophasor_c37118 import (
        ConfigurationFrame2,
        DataFrame,
    )

__all__ = [
    "COMMAND_DATA_OFF",
    "COMMAND_DATA_ON",
    "COMMAND_SEND_CONFIG1",
    "COMMAND_SEND_CONFIG2",
    "COMMAND_SEND_CONFIG3",
    "COMMAND_SEND_HEADER",
    "C37118SessionClient",
    "build_command_frame",
    "read_frame",
]

# Command-word values (verified: pypmu CommandFrame + Wireshark Table 15).
COMMAND_DATA_OFF = 0x0001
COMMAND_DATA_ON = 0x0002
COMMAND_SEND_HEADER = 0x0003
COMMAND_SEND_CONFIG1 = 0x0004
COMMAND_SEND_CONFIG2 = 0x0005
COMMAND_SEND_CONFIG3 = 0x0006
_VALID_COMMANDS = frozenset(
    {
        COMMAND_DATA_OFF,
        COMMAND_DATA_ON,
        COMMAND_SEND_HEADER,
        COMMAND_SEND_CONFIG1,
        COMMAND_SEND_CONFIG2,
        COMMAND_SEND_CONFIG3,
    }
)

_SYNC_LEAD = 0xAA
_HEADER_SIZE = 14
_CRC_SIZE = 2
_FRAME_TYPE_COMMAND = 4
_PREFIX_SIZE = 4  # SYNC (2) + FRAMESIZE (2)
_UINT16_MAX = 0xFFFF


def build_command_frame(
    id_code: int,
    command: int,
    *,
    soc: int = 0,
    fracsec: int = 0,
    version: int = 1,
) -> bytes:
    """Build a CRC-sealed IEEE C37.118.2 COMMAND frame.

    Parameters
    ----------
    id_code : int
        Destination data-stream identification code (0..65535).
    command : int
        Command word; one of the ``COMMAND_*`` constants.
    soc : int
        Second-of-century timestamp for the command (default ``0``).
    fracsec : int
        Fraction-of-second word for the command (default ``0``).
    version : int
        Protocol version number placed in the SYNC word (default ``1``).

    Returns
    -------
    bytes
        The complete COMMAND frame including SYNC and trailing CRC.

    Raises
    ------
    ValueError
        If ``id_code`` is out of range or ``command`` is not a known command.
    """
    if not 0 <= id_code <= _UINT16_MAX:
        raise ValueError("id_code must be in the range 0..65535")
    if command not in _VALID_COMMANDS:
        raise ValueError(f"unknown command word 0x{command:04X}")
    framesize = _HEADER_SIZE + 2 + _CRC_SIZE
    header = (
        bytes([_SYNC_LEAD, (_FRAME_TYPE_COMMAND << 4) | (version & 0x0F)])
        + struct.pack(">H", framesize)
        + struct.pack(">H", id_code)
        + struct.pack(">I", soc)
        + struct.pack(">I", fracsec)
    )
    frame = header + struct.pack(">H", command)
    return frame + struct.pack(">H", compute_crc_ccitt(frame))


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    """Read one complete synchrophasor frame from an async stream.

    The frame is reassembled by reading the 4-byte SYNC/FRAMESIZE prefix,
    validating the SYNC lead, and reading exactly ``FRAMESIZE`` bytes in total.

    Parameters
    ----------
    reader : asyncio.StreamReader
        The stream to read from.

    Returns
    -------
    bytes
        The complete frame, including SYNC and trailing CRC.

    Raises
    ------
    FrameTruncationError
        If the stream ends before a full frame, or the declared frame size is
        smaller than the minimum header-plus-CRC length.
    UnsupportedFrameError
        If the frame does not begin with the SYNC lead byte ``0xAA``.
    """
    try:
        prefix = await reader.readexactly(_PREFIX_SIZE)
    except asyncio.IncompleteReadError as exc:
        raise FrameTruncationError(
            f"stream ended after {len(exc.partial)} bytes before a frame prefix"
        ) from exc
    if prefix[0] != _SYNC_LEAD:
        raise UnsupportedFrameError(
            f"frame does not begin with SYNC lead 0xAA (got 0x{prefix[0]:02X})"
        )
    framesize = struct.unpack(">H", prefix[2:4])[0]
    if framesize < _HEADER_SIZE + _CRC_SIZE:
        raise FrameTruncationError(
            f"declared framesize {framesize} is below the minimum "
            f"{_HEADER_SIZE + _CRC_SIZE}"
        )
    try:
        rest = await reader.readexactly(framesize - _PREFIX_SIZE)
    except asyncio.IncompleteReadError as exc:
        raise FrameTruncationError(
            f"stream ended after {len(exc.partial)} of {framesize - _PREFIX_SIZE} "
            "body bytes"
        ) from exc
    return prefix + rest


def _frame_type(frame: bytes) -> int:
    """Return the frame-type code from a frame's SYNC word."""
    return (frame[1] >> 4) & 0x07


@dataclass
class C37118SessionClient:
    """Review-only async client that drives a C37.118.2 measurement stream.

    Attributes
    ----------
    id_code : int
        The destination data-stream identification code (0..65535).
    non_actuating : bool
        Always ``True`` — the client issues only stream-control command frames
        and never writes device setpoints.
    """

    id_code: int
    non_actuating: bool = field(default=True, init=False)
    _codec: SynchrophasorFrameCodec = field(
        default_factory=SynchrophasorFrameCodec, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not 0 <= self.id_code <= _UINT16_MAX:
            raise ValueError("id_code must be in the range 0..65535")

    async def _send(self, writer: asyncio.StreamWriter, command: int) -> None:
        """Send one command frame and flush the writer."""
        writer.write(build_command_frame(self.id_code, command))
        await writer.drain()

    async def request_configuration(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> ConfigurationFrame2:
        """Request and decode the CONFIG-2 frame from the stream.

        Parameters
        ----------
        reader : asyncio.StreamReader
            The stream to read frames from.
        writer : asyncio.StreamWriter
            The stream to write the command frame to.

        Returns
        -------
        ConfigurationFrame2
            The decoded configuration.

        Raises
        ------
        FrameTruncationError
            If the stream ends before a CONFIG-2 frame arrives.
        """
        await self._send(writer, COMMAND_SEND_CONFIG2)
        while True:
            frame = await read_frame(reader)
            if _frame_type(frame) == FRAME_TYPE_CONFIG2:
                return self._codec.decode_config2(frame)

    async def collect_data_frames(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        config: ConfigurationFrame2,
        *,
        count: int,
    ) -> list[DataFrame]:
        """Turn data on, collect ``count`` DATA frames, then turn data off.

        Parameters
        ----------
        reader : asyncio.StreamReader
            The stream to read frames from.
        writer : asyncio.StreamWriter
            The stream to write command frames to.
        config : ConfigurationFrame2
            The configuration used to decode DATA frames.
        count : int
            Number of DATA frames to collect; must be a positive integer.

        Returns
        -------
        list[DataFrame]
            The decoded DATA frames in arrival order.

        Raises
        ------
        ValueError
            If ``count`` is not a positive integer.
        FrameTruncationError
            If the stream ends before ``count`` DATA frames arrive.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        await self._send(writer, COMMAND_DATA_ON)
        frames: list[DataFrame] = []
        try:
            while len(frames) < count:
                frame = await read_frame(reader)
                if _frame_type(frame) == FRAME_TYPE_DATA:
                    frames.append(self._codec.decode_data(frame, config))
        finally:
            await self._send(writer, COMMAND_DATA_OFF)
        return frames

    async def open_connection(
        self, host: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open a TCP connection to a PDC/PMU endpoint.

        Parameters
        ----------
        host : str
            Hostname or address of the concentrator/PMU.
        port : int
            TCP port of the C37.118.2 data stream.

        Returns
        -------
        tuple[asyncio.StreamReader, asyncio.StreamWriter]
            The connected stream reader and writer.
        """
        return await asyncio.open_connection(host, port)
