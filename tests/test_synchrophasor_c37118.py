# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — IEEE C37.118.2 synchrophasor codec tests

"""Tests for the IEEE C37.118.2-2011 synchrophasor frame codec.

The builders below assemble byte-exact CONFIG-2 and DATA frames (matching the
layout cross-checked against ``pypmu`` and ``Open-C37.118``) with correct
CRC-CCITT trailers, so decoding is verified against real framing rather than
mocks. Malformed-frame paths assert the codec raises typed errors instead of
returning partial data.
"""

from __future__ import annotations

import csv
import math
import struct
from pathlib import Path

import pytest

from scpn_phase_orchestrator.adapters.synchrophasor_c37118 import (
    FRAME_TYPE_CONFIG2,
    FRAME_TYPE_DATA,
    ConfigurationFrame2,
    DataFrame,
    FrameChecksumError,
    FrameTruncationError,
    PmuConfiguration,
    PmuMeasurement,
    SynchrophasorFrameCodec,
    SynchrophasorFrameError,
    SynchrophasorHeader,
    UnsupportedFrameError,
    compute_crc_ccitt,
    data_frames_to_frequency_series,
)

TIME_BASE = 1_000_000


def _name16(text: str) -> bytes:
    return text.encode("ascii")[:16].ljust(16, b"\x00")


def _wrap(
    frame_type: int,
    body: bytes,
    *,
    id_code: int = 7,
    soc: int = 1_700_000_000,
    fracsec: int = 0,
    version: int = 1,
) -> bytes:
    framesize = 14 + len(body) + 2
    header = (
        bytes([0xAA, (frame_type << 4) | version])
        + struct.pack(">H", framesize)
        + struct.pack(">H", id_code)
        + struct.pack(">I", soc)
        + struct.pack(">I", fracsec)
    )
    frame = header + body
    return frame + struct.pack(">H", compute_crc_ccitt(frame))


def _pmu_config_block(
    *,
    station: str,
    fmt: int,
    phnmr: int,
    annmr: int,
    dgnmr: int,
    fnom: int,
    id_code: int = 1,
) -> bytes:
    names = [f"PH{i}" for i in range(phnmr)]
    names += [f"AN{i}" for i in range(annmr)]
    names += [f"DIG{i}" for i in range(16 * dgnmr)]
    block = _name16(station)
    block += struct.pack(">H", id_code)
    block += struct.pack(">H", fmt)
    block += struct.pack(">H", phnmr)
    block += struct.pack(">H", annmr)
    block += struct.pack(">H", dgnmr)
    for name in names:
        block += _name16(name)
    for _ in range(phnmr + annmr + dgnmr):
        block += struct.pack(">I", 0)
    block += struct.pack(">H", fnom)
    block += struct.pack(">H", 0)  # CFGCNT
    return block


def _config2_frame(blocks: list[bytes], *, data_rate: int = 30) -> bytes:
    body = struct.pack(">I", TIME_BASE) + struct.pack(">H", len(blocks))
    for block in blocks:
        body += block
    body += struct.pack(">h", data_rate)
    return _wrap(FRAME_TYPE_CONFIG2, body)


def _int_data_block(
    *,
    phasors: list[tuple[int, int]],
    freq_mhz: int,
    df_dt: int,
    analogs: list[int],
    digitals: list[int],
    stat: int = 0,
) -> bytes:
    block = struct.pack(">H", stat)
    for real, imag in phasors:
        block += struct.pack(">hh", real, imag)
    block += struct.pack(">h", freq_mhz)
    block += struct.pack(">h", df_dt)
    for value in analogs:
        block += struct.pack(">h", value)
    for word in digitals:
        block += struct.pack(">H", word)
    return block


def _float_data_block(
    *,
    phasors: list[tuple[float, float]],
    freq_hz_dev: float,
    df_dt: float,
    analogs: list[float],
    digitals: list[int],
    stat: int = 0,
) -> bytes:
    block = struct.pack(">H", stat)
    for a, b in phasors:
        block += struct.pack(">ff", a, b)
    block += struct.pack(">f", freq_hz_dev)
    block += struct.pack(">f", df_dt)
    for value in analogs:
        block += struct.pack(">f", value)
    for word in digitals:
        block += struct.pack(">H", word)
    return block


# --- compute_crc_ccitt ----------------------------------------------------


def test_crc_known_vector() -> None:
    # CRC-CCITT (0xFFFF) of the ASCII "123456789" check string is 0x29B1.
    assert compute_crc_ccitt(b"123456789") == 0x29B1


def test_crc_empty_is_init() -> None:
    assert compute_crc_ccitt(b"") == 0xFFFF


# --- SynchrophasorHeader --------------------------------------------------


def _header(fracsec: int = 0, soc: int = 100) -> SynchrophasorHeader:
    return SynchrophasorHeader(
        frame_type=FRAME_TYPE_DATA,
        version=1,
        framesize=16,
        id_code=7,
        soc=soc,
        fracsec_raw=fracsec,
    )


def test_header_fraction_and_quality() -> None:
    fracsec = (0x1A << 24) | 500_000  # leap flags + message quality nibble 0x0A
    header = _header(fracsec=fracsec)
    assert header.fraction_count == 500_000
    assert header.message_time_quality == 0x0A
    assert header.leap_second_pending is True
    assert header.leap_second_occurred is False
    assert header.leap_second_direction == "+"


def test_header_leap_direction_and_occurred_set() -> None:
    fracsec = (0x60 << 24) | 1  # leap direction + occurred bits set
    header = _header(fracsec=fracsec)
    assert header.leap_second_direction == "-"
    assert header.leap_second_occurred is True
    assert header.leap_second_pending is False


def test_header_seconds_of_second() -> None:
    header = _header(fracsec=250_000)
    assert header.seconds_of_second(TIME_BASE) == pytest.approx(0.25)


def test_header_seconds_of_second_rejects_nonpositive_time_base() -> None:
    with pytest.raises(SynchrophasorFrameError, match="time_base"):
        _header(fracsec=1).seconds_of_second(0)


def test_header_audit_record() -> None:
    record = _header(fracsec=(0x10 << 24) | 42, soc=99).to_audit_record()
    assert record["soc"] == 99
    assert record["fraction_count"] == 42
    assert record["leap_second_pending"] is True
    assert record["leap_second_direction"] == "+"


# --- PmuConfiguration sizing ---------------------------------------------


def test_pmu_config_int_sizes() -> None:
    pmu = PmuConfiguration(
        station_name="S",
        id_code=1,
        phasor_polar=False,
        phasor_float=False,
        analog_float=False,
        freq_float=False,
        phasor_count=2,
        analog_count=1,
        digital_word_count=1,
        channel_names=(),
        nominal_frequency_hz=60.0,
    )
    assert pmu.phasor_size == 4
    assert pmu.freq_size == 2
    assert pmu.analog_size == 2
    # 2(STAT) + 2*4 + 2*2 + 1*2 + 1*2 = 18
    assert pmu.data_block_size == 18


def test_pmu_config_float_sizes_and_audit() -> None:
    pmu = PmuConfiguration(
        station_name="Grid",
        id_code=3,
        phasor_polar=True,
        phasor_float=True,
        analog_float=True,
        freq_float=True,
        phasor_count=1,
        analog_count=1,
        digital_word_count=0,
        channel_names=("VA",),
        nominal_frequency_hz=50.0,
    )
    assert pmu.phasor_size == 8
    assert pmu.freq_size == 4
    assert pmu.analog_size == 4
    # 2 + 1*8 + 2*4 + 1*4 + 0 = 22
    assert pmu.data_block_size == 22
    record = pmu.to_audit_record()
    assert record["nominal_frequency_hz"] == 50.0
    assert record["channel_names"] == ["VA"]
    assert record["phasor_polar"] is True


# --- decode_config2 -------------------------------------------------------


def test_decode_config2_single_int_pmu() -> None:
    codec = SynchrophasorFrameCodec()
    block = _pmu_config_block(
        station="Station A", fmt=0x0000, phnmr=1, annmr=0, dgnmr=0, fnom=0x0000
    )
    config = codec.decode_config2(_config2_frame([block]))
    assert isinstance(config, ConfigurationFrame2)
    assert config.time_base == TIME_BASE
    assert config.data_rate == 30
    assert len(config.pmus) == 1
    pmu = config.pmus[0]
    assert pmu.station_name == "Station A"
    assert pmu.nominal_frequency_hz == 60.0
    assert pmu.phasor_float is False
    assert pmu.channel_names == ("PH0",)


def test_decode_config2_float_polar_50hz_with_channels() -> None:
    codec = SynchrophasorFrameCodec()
    block = _pmu_config_block(
        station="EU", fmt=0x000F, phnmr=1, annmr=1, dgnmr=1, fnom=0x0001
    )
    config = codec.decode_config2(_config2_frame([block], data_rate=-2))
    pmu = config.pmus[0]
    assert pmu.nominal_frequency_hz == 50.0
    assert pmu.phasor_polar is True
    assert pmu.phasor_float is True
    assert pmu.analog_float is True
    assert pmu.freq_float is True
    assert pmu.phasor_count == 1
    assert pmu.analog_count == 1
    assert pmu.digital_word_count == 1
    assert len(pmu.channel_names) == 1 + 1 + 16
    assert config.data_rate == -2


def test_decode_config2_audit_record() -> None:
    codec = SynchrophasorFrameCodec()
    block = _pmu_config_block(
        station="S", fmt=0x0000, phnmr=1, annmr=0, dgnmr=0, fnom=0x0000
    )
    record = codec.decode_config2(_config2_frame([block])).to_audit_record()
    assert record["time_base"] == TIME_BASE
    assert record["pmu_count"] == 1
    assert record["data_rate"] == 30


# --- decode_data ----------------------------------------------------------


def _int_config() -> tuple[SynchrophasorFrameCodec, ConfigurationFrame2]:
    codec = SynchrophasorFrameCodec()
    block = _pmu_config_block(
        station="Int", fmt=0x0000, phnmr=1, annmr=1, dgnmr=1, fnom=0x0000
    )
    return codec, codec.decode_config2(_config2_frame([block]))


def test_decode_data_int_format() -> None:
    codec, config = _int_config()
    body = _int_data_block(
        phasors=[(100, -50)],
        freq_mhz=250,
        df_dt=-3,
        analogs=[7],
        digitals=[0xABCD],
        stat=0x0002,
    )
    frame = codec.decode_data(_wrap(FRAME_TYPE_DATA, body), config)
    assert isinstance(frame, DataFrame)
    meas = frame.measurements[0]
    assert meas.stat == 0x0002
    assert meas.phasors == ((100.0, -50.0),)
    assert meas.frequency_deviation == 250.0
    # 60 Hz nominal + 250 mHz deviation
    assert meas.frequency_hz == pytest.approx(60.25)
    assert meas.df_dt == -3.0
    assert meas.analogs == (7.0,)
    assert meas.digitals == (0xABCD,)


def test_decode_data_float_format() -> None:
    codec = SynchrophasorFrameCodec()
    block = _pmu_config_block(
        station="Flt", fmt=0x000F, phnmr=1, annmr=1, dgnmr=0, fnom=0x0001
    )
    config = codec.decode_config2(_config2_frame([block]))
    body = _float_data_block(
        phasors=[(230.0, 1.57)],
        freq_hz_dev=-0.1,
        df_dt=0.02,
        analogs=[3.5],
        digitals=[],
    )
    frame = codec.decode_data(_wrap(FRAME_TYPE_DATA, body), config)
    meas = frame.measurements[0]
    assert meas.phasors[0][0] == pytest.approx(230.0)
    assert meas.frequency_deviation == pytest.approx(-0.1)
    assert meas.frequency_hz == pytest.approx(49.9)
    assert meas.df_dt == pytest.approx(0.02)
    assert meas.analogs[0] == pytest.approx(3.5)


def test_decode_data_audit_record() -> None:
    codec, config = _int_config()
    body = _int_data_block(
        phasors=[(1, 2)], freq_mhz=0, df_dt=0, analogs=[0], digitals=[1]
    )
    record = codec.decode_data(_wrap(FRAME_TYPE_DATA, body), config).to_audit_record()
    assert record["measurements"][0]["frequency_hz"] == pytest.approx(60.0)
    assert record["measurements"][0]["digitals"] == [1]


def test_decode_data_multi_pmu() -> None:
    codec = SynchrophasorFrameCodec()
    block_a = _pmu_config_block(
        station="A", fmt=0x0000, phnmr=1, annmr=0, dgnmr=0, fnom=0x0000, id_code=1
    )
    block_b = _pmu_config_block(
        station="B", fmt=0x0000, phnmr=1, annmr=0, dgnmr=0, fnom=0x0001, id_code=2
    )
    config = codec.decode_config2(_config2_frame([block_a, block_b]))
    body = _int_data_block(
        phasors=[(10, 0)], freq_mhz=100, df_dt=0, analogs=[], digitals=[]
    ) + _int_data_block(
        phasors=[(20, 0)], freq_mhz=-100, df_dt=0, analogs=[], digitals=[]
    )
    frame = codec.decode_data(_wrap(FRAME_TYPE_DATA, body), config)
    assert len(frame.measurements) == 2
    assert frame.measurements[0].frequency_hz == pytest.approx(60.1)
    assert frame.measurements[1].frequency_hz == pytest.approx(49.9)


# --- malformed-frame paths ------------------------------------------------


def test_decode_rejects_short_frame() -> None:
    codec = SynchrophasorFrameCodec()
    with pytest.raises(FrameTruncationError, match="shorter than the minimum"):
        codec.decode_config2(b"\xaa\x31\x00")


def test_decode_rejects_bad_sync_lead() -> None:
    codec = SynchrophasorFrameCodec()
    bad = bytearray(
        _config2_frame(
            [_pmu_config_block(station="S", fmt=0, phnmr=1, annmr=0, dgnmr=0, fnom=0)]
        )
    )
    bad[0] = 0xBB
    with pytest.raises(UnsupportedFrameError, match="SYNC lead"):
        codec.decode_config2(bytes(bad))


def test_decode_rejects_wrong_frame_type() -> None:
    codec = SynchrophasorFrameCodec()
    block = _pmu_config_block(station="S", fmt=0, phnmr=1, annmr=0, dgnmr=0, fnom=0)
    config_frame = _config2_frame([block])
    with pytest.raises(UnsupportedFrameError, match="expected frame type 0"):
        codec.decode_data(
            config_frame,
            ConfigurationFrame2(
                header=_header(), time_base=TIME_BASE, pmus=(), data_rate=30
            ),
        )


def test_decode_rejects_framesize_mismatch() -> None:
    codec = SynchrophasorFrameCodec()
    good = bytearray(
        _config2_frame(
            [_pmu_config_block(station="S", fmt=0, phnmr=1, annmr=0, dgnmr=0, fnom=0)]
        )
    )
    struct.pack_into(">H", good, 2, len(good) + 5)  # corrupt declared framesize
    with pytest.raises(FrameTruncationError, match="declared framesize"):
        codec.decode_config2(bytes(good))


def test_decode_rejects_crc_mismatch() -> None:
    codec = SynchrophasorFrameCodec()
    good = bytearray(
        _config2_frame(
            [_pmu_config_block(station="S", fmt=0, phnmr=1, annmr=0, dgnmr=0, fnom=0)]
        )
    )
    good[-1] ^= 0xFF  # flip the CRC
    with pytest.raises(FrameChecksumError, match="CRC mismatch"):
        codec.decode_config2(bytes(good))


def test_decode_data_truncated_body_raises() -> None:
    codec, config = _int_config()
    # A DATA frame whose body carries only STAT cannot satisfy the config's
    # declared phasor/frequency/analog/digital fields, so the reader runs off
    # the frame end and raises rather than returning partial data.
    tiny_body = struct.pack(">H", 0)
    with pytest.raises(FrameTruncationError):
        codec.decode_data(_wrap(FRAME_TYPE_DATA, tiny_body), config)


# --- data_frames_to_frequency_series --------------------------------------


def _series_config() -> tuple[SynchrophasorFrameCodec, ConfigurationFrame2]:
    codec = SynchrophasorFrameCodec()
    block = _pmu_config_block(
        station="Ser", fmt=0x0000, phnmr=1, annmr=0, dgnmr=0, fnom=0x0000
    )
    return codec, codec.decode_config2(_config2_frame([block]))


def test_frequency_series_relative_time_and_hz() -> None:
    codec, config = _series_config()
    body0 = _int_data_block(
        phasors=[(0, 0)], freq_mhz=0, df_dt=0, analogs=[], digitals=[]
    )
    body1 = _int_data_block(
        phasors=[(0, 0)], freq_mhz=500, df_dt=0, analogs=[], digitals=[]
    )
    f0 = codec.decode_data(_wrap(FRAME_TYPE_DATA, body0, soc=1000, fracsec=0), config)
    f1 = codec.decode_data(
        _wrap(FRAME_TYPE_DATA, body1, soc=1000, fracsec=TIME_BASE // 2), config
    )
    times, freqs = data_frames_to_frequency_series(config, (f0, f1))
    assert times == pytest.approx((0.0, 0.5))
    assert freqs == pytest.approx((60.0, 60.5))


def test_frequency_series_rejects_empty() -> None:
    _, config = _series_config()
    with pytest.raises(SynchrophasorFrameError, match="at least one"):
        data_frames_to_frequency_series(config, ())


def test_frequency_series_rejects_pmu_index_out_of_config_range() -> None:
    codec, config = _series_config()
    body = _int_data_block(
        phasors=[(0, 0)], freq_mhz=0, df_dt=0, analogs=[], digitals=[]
    )
    frame = codec.decode_data(_wrap(FRAME_TYPE_DATA, body), config)
    with pytest.raises(SynchrophasorFrameError, match="out of range for 1 PMUs"):
        data_frames_to_frequency_series(config, (frame,), pmu_index=5)


def test_frequency_series_rejects_pmu_index_missing_in_frame() -> None:
    # A configuration declaring two PMUs but a frame carrying only one measurement
    # exercises the per-frame guard.
    config = ConfigurationFrame2(
        header=_header(),
        time_base=TIME_BASE,
        pmus=(
            PmuConfiguration(
                station_name="A",
                id_code=1,
                phasor_polar=False,
                phasor_float=False,
                analog_float=False,
                freq_float=False,
                phasor_count=0,
                analog_count=0,
                digital_word_count=0,
                channel_names=(),
                nominal_frequency_hz=60.0,
            ),
        )
        * 2,
        data_rate=30,
    )
    frame = DataFrame(
        header=_header(soc=5),
        measurements=(
            PmuMeasurement(
                stat=0,
                phasors=(),
                frequency_hz=60.0,
                frequency_deviation=0.0,
                df_dt=0.0,
                analogs=(),
                digitals=(),
            ),
        ),
    )
    with pytest.raises(SynchrophasorFrameError, match="a frame with 1 measurements"):
        data_frames_to_frequency_series(config, (frame,), pmu_index=1)


# --- end-to-end: decoded frames feed the ringdown screener ----------------


def test_decoded_stream_feeds_ringdown_screener(tmp_path: Path) -> None:
    from scpn_phase_orchestrator.runtime.pmu_ringdown import (
        PMURingdownEvidence,
        screen_pmu_ringdown_csv,
    )

    codec, config = _series_config()
    rate_hz = 30
    n_samples = 60
    frames = []
    for k in range(n_samples):
        t = k / rate_hz
        # Decaying 0.5 Hz electromechanical ringdown about 60 Hz, in millihertz.
        deviation_mhz = round(
            200.0 * math.exp(-0.4 * t) * math.sin(2 * math.pi * 0.5 * t)
        )
        body = _int_data_block(
            phasors=[(0, 0)], freq_mhz=deviation_mhz, df_dt=0, analogs=[], digitals=[]
        )
        fracsec = round(k * TIME_BASE / rate_hz)
        frames.append(
            codec.decode_data(
                _wrap(FRAME_TYPE_DATA, body, soc=1_700_000_000, fracsec=fracsec), config
            )
        )

    times, freqs = data_frames_to_frequency_series(config, tuple(frames))
    csv_path = tmp_path / "ringdown.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "frequency_hz"])
        for t, f in zip(times, freqs, strict=True):
            writer.writerow([f"{t:.6f}", f"{f:.6f}"])

    evidence = screen_pmu_ringdown_csv(
        csv_path,
        event_id="c37118-e2e",
        captured_at="2026-07-21T00:00:00Z",
        signal_source="synchrophasor-codec",
        nominal_frequency_hz=60.0,
    )
    assert isinstance(evidence, PMURingdownEvidence)
    assert evidence.sample_count == n_samples
    assert evidence.sampling_rate_hz == pytest.approx(float(rate_hz), rel=1e-3)
