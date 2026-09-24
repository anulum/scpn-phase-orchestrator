# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — IEEE C37.118.2 async client tests

"""Tests for the live async IEEE C37.118.2 synchrophasor session client.

Frame reassembly is exercised against a real :class:`asyncio.StreamReader`, the
session methods against fed streams and a recording writer, and one end-to-end
case runs the client against an in-process :func:`asyncio.start_server` that
answers command frames with canned CONFIG-2 and DATA frames.
"""

from __future__ import annotations

import asyncio
import struct
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

from scpn_phase_orchestrator.adapters.synchrophasor_c37118 import (
    FrameTruncationError,
    UnsupportedFrameError,
    compute_crc_ccitt,
)
from scpn_phase_orchestrator.adapters.synchrophasor_client import (
    COMMAND_DATA_OFF,
    COMMAND_DATA_ON,
    COMMAND_SEND_CONFIG2,
    C37118SessionClient,
    build_command_frame,
    read_frame,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _wrap(frame_type: int, body: bytes, *, id_code: int = 7) -> bytes:
    framesize = 14 + len(body) + 2
    header = (
        bytes([0xAA, (frame_type << 4) | 1])
        + struct.pack(">H", framesize)
        + struct.pack(">H", id_code)
        + struct.pack(">I", 1_700_000_000)
        + struct.pack(">I", 0)
    )
    frame = header + body
    return frame + struct.pack(">H", compute_crc_ccitt(frame))


def _config2_bytes() -> bytes:
    body = (
        struct.pack(">I", 1_000_000)  # TIME_BASE
        + struct.pack(">H", 1)  # NUM_PMU
        + b"STATION".ljust(16, b"\x00")
        + struct.pack(">H", 1)  # IDCODE
        + struct.pack(">H", 0x0000)  # FORMAT
        + struct.pack(">H", 0)  # PHNMR
        + struct.pack(">H", 0)  # ANNMR
        + struct.pack(">H", 0)  # DGNMR
        + struct.pack(">H", 0x0000)  # FNOM 60 Hz
        + struct.pack(">H", 0)  # CFGCNT
        + struct.pack(">h", 30)  # DATA_RATE
    )
    return _wrap(3, body)  # frame type 3 = CONFIG-2


def _data_bytes(freq_mhz: int) -> bytes:
    body = struct.pack(">H", 0) + struct.pack(">h", freq_mhz) + struct.pack(">h", 0)
    return _wrap(0, body)  # frame type 0 = DATA


def _header_frame() -> bytes:
    return _wrap(1, b"INFO")  # frame type 1 = HEADER (should be skipped)


async def _reader_from(*chunks: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    for chunk in chunks:
        reader.feed_data(chunk)
    reader.feed_eof()
    return reader


class _RecordingWriter:
    """Minimal StreamWriter stand-in that records written command frames."""

    def __init__(self) -> None:
        self.written: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.written.append(bytes(data))

    async def drain(self) -> None:
        return None


# --- build_command_frame --------------------------------------------------


def test_build_command_frame_structure_and_crc() -> None:
    frame = build_command_frame(7, COMMAND_SEND_CONFIG2)
    assert len(frame) == 18
    assert frame[0] == 0xAA
    assert (frame[1] >> 4) & 0x07 == 4  # command frame type
    assert struct.unpack(">H", frame[2:4])[0] == 18
    assert struct.unpack(">H", frame[4:6])[0] == 7  # id_code
    assert struct.unpack(">H", frame[14:16])[0] == COMMAND_SEND_CONFIG2
    assert struct.unpack(">H", frame[-2:])[0] == compute_crc_ccitt(frame[:-2])


def test_build_command_frame_rejects_bad_id_code() -> None:
    with pytest.raises(ValueError, match="id_code"):
        build_command_frame(99999, COMMAND_DATA_ON)


def test_build_command_frame_rejects_unknown_command() -> None:
    with pytest.raises(ValueError, match="unknown command"):
        build_command_frame(7, 0x00FF)


# --- read_frame -----------------------------------------------------------


async def test_read_frame_returns_complete_frame() -> None:
    payload = _data_bytes(100)
    reader = await _reader_from(payload)
    assert await read_frame(reader) == payload


async def test_read_frame_rejects_bad_sync_lead() -> None:
    reader = await _reader_from(b"\xbb\x01\x00\x12")
    with pytest.raises(UnsupportedFrameError, match="SYNC lead"):
        await read_frame(reader)


async def test_read_frame_rejects_undersized_framesize() -> None:
    reader = await _reader_from(b"\xaa\x01\x00\x04")  # framesize 4 < 16
    with pytest.raises(FrameTruncationError, match="below the minimum"):
        await read_frame(reader)


async def test_read_frame_incomplete_prefix_raises() -> None:
    reader = await _reader_from(b"\xaa\x01")  # only 2 of 4 prefix bytes
    with pytest.raises(FrameTruncationError, match="before a frame prefix"):
        await read_frame(reader)


async def test_read_frame_incomplete_body_raises() -> None:
    # Prefix declares a 40-byte frame but only a few body bytes follow.
    reader = await _reader_from(b"\xaa\x01\x00\x28\x00\x07")
    with pytest.raises(FrameTruncationError, match="body bytes"):
        await read_frame(reader)


# --- C37118SessionClient --------------------------------------------------


def test_client_rejects_bad_id_code() -> None:
    with pytest.raises(ValueError, match="id_code"):
        C37118SessionClient(70000)


async def test_request_configuration_skips_non_config_frames() -> None:
    client = C37118SessionClient(7)
    reader = await _reader_from(_header_frame(), _config2_bytes())
    writer = _RecordingWriter()
    config = await client.request_configuration(reader, writer)  # type: ignore[arg-type]
    assert config.time_base == 1_000_000
    assert len(config.pmus) == 1
    # One command frame was sent: send-CONFIG-2.
    assert len(writer.written) == 1
    assert struct.unpack(">H", writer.written[0][14:16])[0] == COMMAND_SEND_CONFIG2


async def test_collect_data_frames_gathers_count_and_stops() -> None:
    client = C37118SessionClient(7)
    config = client._codec.decode_config2(_config2_bytes())
    reader = await _reader_from(
        _data_bytes(100), _header_frame(), _data_bytes(-50), _data_bytes(200)
    )
    writer = _RecordingWriter()
    frames = await client.collect_data_frames(reader, writer, config, count=2)  # type: ignore[arg-type]
    assert len(frames) == 2
    assert frames[0].measurements[0].frequency_hz == pytest.approx(60.1)
    assert frames[1].measurements[0].frequency_hz == pytest.approx(59.95)
    # Data-on sent first, data-off sent last (in the finally block).
    assert struct.unpack(">H", writer.written[0][14:16])[0] == COMMAND_DATA_ON
    assert struct.unpack(">H", writer.written[-1][14:16])[0] == COMMAND_DATA_OFF


async def test_collect_data_frames_rejects_nonpositive_count() -> None:
    client = C37118SessionClient(7)
    config = client._codec.decode_config2(_config2_bytes())
    reader = await _reader_from(_data_bytes(0))
    with pytest.raises(ValueError, match="positive integer"):
        await client.collect_data_frames(reader, _RecordingWriter(), config, count=0)  # type: ignore[arg-type]


async def test_collect_data_frames_rejects_bool_count() -> None:
    client = C37118SessionClient(7)
    config = client._codec.decode_config2(_config2_bytes())
    reader = await _reader_from(_data_bytes(0))
    with pytest.raises(ValueError, match="positive integer"):
        await client.collect_data_frames(
            reader,  # type: ignore[arg-type]
            _RecordingWriter(),  # type: ignore[arg-type]
            config,
            count=True,
        )


# --- end-to-end against an in-process server ------------------------------


@asynccontextmanager
async def _pdc_server(port: int) -> AsyncIterator[None]:
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # Command 1: request CONFIG-2.
        await read_frame(reader)
        writer.write(_config2_bytes())
        await writer.drain()
        # Command 2: data on -> stream three DATA frames.
        await read_frame(reader)
        for freq in (100, 150, 200):
            writer.write(_data_bytes(freq))
        await writer.drain()
        # Command 3: data off -> close.
        await read_frame(reader)
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", port)
    async with server:
        await server.start_serving()
        yield


async def test_end_to_end_against_in_process_server(unused_tcp_port: int) -> None:
    async with _pdc_server(unused_tcp_port):
        client = C37118SessionClient(7)
        reader, writer = await client.open_connection("127.0.0.1", unused_tcp_port)
        try:
            config = await client.request_configuration(reader, writer)
            frames = await client.collect_data_frames(reader, writer, config, count=3)
        finally:
            writer.close()
    assert len(frames) == 3
    assert frames[0].measurements[0].frequency_hz == pytest.approx(60.1)
    assert frames[2].measurements[0].frequency_hz == pytest.approx(60.2)


# --- hostile or mismatched peers (real in-process server) -----------------


@asynccontextmanager
async def _scripted_server(port: int, replies: list[bytes]) -> AsyncIterator[None]:
    """Serve ``replies`` after the first command, then keep the socket open."""

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await read_frame(reader)
        for reply in replies:
            writer.write(reply)
        await writer.drain()
        try:
            while True:
                await read_frame(reader)
        except Exception:  # the client closing the socket ends the peer
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", port)
    async with server:
        await server.start_serving()
        yield


async def _run_against(port: int, replies: list[bytes], call) -> object:
    async with _scripted_server(port, replies):
        client = C37118SessionClient(7, max_skipped_frames=3)
        reader, writer = await client.open_connection("127.0.0.1", port)
        try:
            return await asyncio.wait_for(call(client, reader, writer), timeout=10.0)
        finally:
            writer.close()


async def test_peer_that_never_sends_config_cannot_hold_the_session(
    unused_tcp_port: int,
) -> None:
    """Before the skip budget a DATA-only peer kept the client waiting forever."""
    replies = [_data_bytes(100)] * 10
    with pytest.raises(UnsupportedFrameError, match="no CONFIG-2 frame after"):
        await _run_against(
            unused_tcp_port,
            replies,
            lambda client, reader, writer: client.request_configuration(reader, writer),
        )


async def test_config_for_another_stream_is_rejected(unused_tcp_port: int) -> None:
    other_stream = _wrap(3, _config2_bytes()[14:-2], id_code=8)
    with pytest.raises(UnsupportedFrameError, match="IDCODE 8 does not match"):
        await _run_against(
            unused_tcp_port,
            [other_stream],
            lambda client, reader, writer: client.request_configuration(reader, writer),
        )


async def test_peer_that_never_sends_data_cannot_hold_the_session(
    unused_tcp_port: int,
) -> None:
    replies = [_config2_bytes()] + [_header_frame()] * 10

    async def call(client, reader, writer):
        config = await client.request_configuration(reader, writer)
        return await client.collect_data_frames(reader, writer, config, count=1)

    with pytest.raises(UnsupportedFrameError, match="no DATA frame after"):
        await _run_against(unused_tcp_port, replies, call)


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_skip_budget_must_be_a_non_negative_integer(value) -> None:
    with pytest.raises(ValueError, match="max_skipped_frames"):
        C37118SessionClient(7, max_skipped_frames=value)


def test_boolean_id_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="id_code"):
        C37118SessionClient(True)
    with pytest.raises(ValueError, match="id_code"):
        build_command_frame(True, COMMAND_DATA_ON)
