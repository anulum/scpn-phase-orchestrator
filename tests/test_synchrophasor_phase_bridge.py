# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — IEEE C37.118.2 phase bridge tests

"""Tests for the IEEE C37.118.2 synchrophasor phase bridge.

The bridge maps decoded PMU phasors to oscillator phase states. Most cases
construct decoded frame objects directly for clarity; one end-to-end case
decodes real frame bytes with the codec and feeds the bridge, proving the
byte-to-phase-state path.
"""

from __future__ import annotations

import math

import pytest

from scpn_phase_orchestrator.adapters.synchrophasor_c37118 import (
    ConfigurationFrame2,
    DataFrame,
    PhasorUnit,
    PmuConfiguration,
    PmuMeasurement,
    SynchrophasorHeader,
)
from scpn_phase_orchestrator.adapters.synchrophasor_phase_bridge import (
    C37118PhaseBridge,
    PhasorBinding,
)


def _header(soc: int = 100, fracsec: int = 0) -> SynchrophasorHeader:
    return SynchrophasorHeader(
        frame_type=0,
        version=1,
        framesize=32,
        id_code=7,
        soc=soc,
        fracsec_raw=fracsec,
    )


def _pmu(
    *,
    phasor_polar: bool = False,
    phasor_float: bool = False,
    phasor_count: int = 1,
    phasor_units: tuple[PhasorUnit, ...] = (),
    nominal: float = 60.0,
) -> PmuConfiguration:
    return PmuConfiguration(
        station_name="S",
        id_code=1,
        phasor_polar=phasor_polar,
        phasor_float=phasor_float,
        analog_float=False,
        freq_float=False,
        phasor_count=phasor_count,
        analog_count=0,
        digital_word_count=0,
        channel_names=(),
        nominal_frequency_hz=nominal,
        phasor_units=phasor_units,
    )


def _measurement(
    *,
    phasors: tuple[tuple[float, float], ...],
    frequency_hz: float = 60.0,
    stat: int = 0,
) -> PmuMeasurement:
    return PmuMeasurement(
        stat=stat,
        phasors=phasors,
        frequency_hz=frequency_hz,
        frequency_deviation=0.0,
        df_dt=0.0,
        analogs=(),
        digitals=(),
    )


def _config(pmu: PmuConfiguration) -> ConfigurationFrame2:
    return ConfigurationFrame2(
        header=_header(), time_base=1_000_000, pmus=(pmu,), data_rate=30
    )


def _frame(measurement: PmuMeasurement) -> DataFrame:
    return DataFrame(header=_header(), measurements=(measurement,))


# --- PhasorBinding --------------------------------------------------------


def test_binding_defaults_and_audit() -> None:
    binding = PhasorBinding("gen1")
    assert binding.pmu_index == 0
    assert binding.phasor_index == 0
    assert binding.to_audit_record() == {
        "oscillator": "gen1",
        "pmu_index": 0,
        "phasor_index": 0,
    }


def test_binding_rejects_empty_oscillator() -> None:
    with pytest.raises(ValueError, match="oscillator"):
        PhasorBinding("   ")


def test_binding_rejects_negative_pmu_index() -> None:
    with pytest.raises(ValueError, match="pmu_index"):
        PhasorBinding("gen1", pmu_index=-1)


def test_binding_rejects_non_int_phasor_index() -> None:
    with pytest.raises(ValueError, match="phasor_index"):
        PhasorBinding("gen1", phasor_index=True)


# --- bridge construction --------------------------------------------------


def test_bridge_review_only_posture_and_audit() -> None:
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("g1")])
    assert bridge.non_actuating is True
    assert bridge.execution_disabled is True
    record = bridge.to_audit_record()
    assert record["non_actuating"] is True
    assert record["execution_disabled"] is True
    assert record["bindings"] == [
        {"oscillator": "g1", "pmu_index": 0, "phasor_index": 0}
    ]


def test_bridge_rejects_empty_bindings() -> None:
    with pytest.raises(ValueError, match="at least one binding"):
        C37118PhaseBridge(bindings=())


def test_bridge_rejects_duplicate_oscillator_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        C37118PhaseBridge.from_bindings(
            [PhasorBinding("g1"), PhasorBinding("g1", pmu_index=1)]
        )


# --- extract_phases: representations --------------------------------------


def test_extract_rectangular_integer_scales_amplitude() -> None:
    pmu = _pmu(phasor_units=(PhasorUnit(is_current=False, scale=100000),))
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("g1")])
    frame = _frame(_measurement(phasors=((100.0, -50.0),), frequency_hz=60.2))
    states = bridge.extract_phases(_config(pmu), [frame])
    state = states["g1"]
    assert state.theta == pytest.approx(math.atan2(-50.0, 100.0) % (2 * math.pi))
    # PHUNIT scale 100000 * 1e-5 = 1.0 V/bit -> magnitude in volts.
    assert state.amplitude == pytest.approx(math.hypot(100.0, 50.0))
    assert state.omega == pytest.approx(2 * math.pi * 60.2)
    assert state.quality == 1.0
    assert state.channel == "P"
    assert state.node_id == "g1"


def test_extract_rectangular_float_amplitude_unscaled() -> None:
    pmu = _pmu(phasor_float=True)
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("g1")])
    frame = _frame(_measurement(phasors=((3.0, 4.0),)))
    state = bridge.extract_phases(_config(pmu), [frame])["g1"]
    assert state.amplitude == pytest.approx(5.0)
    assert state.theta == pytest.approx(math.atan2(4.0, 3.0))


def test_extract_polar_float_uses_angle_directly() -> None:
    pmu = _pmu(phasor_polar=True, phasor_float=True)
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("g1")])
    frame = _frame(_measurement(phasors=((230.0, 1.2),)))
    state = bridge.extract_phases(_config(pmu), [frame])["g1"]
    assert state.amplitude == pytest.approx(230.0)
    assert state.theta == pytest.approx(1.2)


def test_extract_polar_integer_is_an_honest_boundary() -> None:
    pmu = _pmu(phasor_polar=True, phasor_float=False)
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("g1")])
    frame = _frame(_measurement(phasors=((230.0, 1000.0),)))
    with pytest.raises(ValueError, match="integer polar phasor angle scaling"):
        bridge.extract_phases(_config(pmu), [frame])


def test_extract_rectangular_integer_default_scale_when_no_phunit() -> None:
    # phasor_units empty -> _phasor_scale falls back to 1.0.
    pmu = _pmu(phasor_units=())
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("g1")])
    frame = _frame(_measurement(phasors=((30.0, 40.0),)))
    state = bridge.extract_phases(_config(pmu), [frame])["g1"]
    assert state.amplitude == pytest.approx(50.0)


# --- extract_phases: quality mapping --------------------------------------


@pytest.mark.parametrize(
    ("stat", "expected"),
    [(0x0000, 1.0), (0x2000, 0.5), (0x8000, 0.0), (0xC000, 0.0)],
)
def test_extract_quality_from_stat(stat: int, expected: float) -> None:
    pmu = _pmu(phasor_float=True)
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("g1")])
    frame = _frame(_measurement(phasors=((1.0, 0.0),), stat=stat))
    state = bridge.extract_phases(_config(pmu), [frame])["g1"]
    assert state.quality == expected


# --- extract_phases: error paths ------------------------------------------


def test_extract_rejects_empty_frames() -> None:
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("g1")])
    with pytest.raises(ValueError, match="at least one DATA frame"):
        bridge.extract_phases(_config(_pmu(phasor_float=True)), [])


def test_extract_rejects_pmu_index_out_of_config_range() -> None:
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("g1", pmu_index=3)])
    frame = _frame(_measurement(phasors=((1.0, 0.0),)))
    with pytest.raises(ValueError, match="out of range for 1 PMUs"):
        bridge.extract_phases(_config(_pmu(phasor_float=True)), [frame])


def test_extract_rejects_pmu_index_missing_in_frame() -> None:
    pmu = _pmu(phasor_float=True)
    config = ConfigurationFrame2(
        header=_header(), time_base=1_000_000, pmus=(pmu, pmu), data_rate=30
    )
    frame = _frame(_measurement(phasors=((1.0, 0.0),)))  # one measurement only
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("g1", pmu_index=1)])
    with pytest.raises(ValueError, match="a frame with 1 measurements"):
        bridge.extract_phases(config, [frame])


def test_extract_rejects_phasor_index_out_of_range() -> None:
    pmu = _pmu(phasor_float=True, phasor_count=1)
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("g1", phasor_index=2)])
    frame = _frame(_measurement(phasors=((1.0, 0.0),)))
    with pytest.raises(ValueError, match="phasor_index 2 out of range"):
        bridge.extract_phases(_config(pmu), [frame])


def test_extract_uses_latest_frame() -> None:
    pmu = _pmu(phasor_float=True)
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("g1")])
    f0 = _frame(_measurement(phasors=((1.0, 0.0),), frequency_hz=59.9))
    f1 = _frame(_measurement(phasors=((0.0, 1.0),), frequency_hz=60.1))
    state = bridge.extract_phases(_config(pmu), [f0, f1])["g1"]
    assert state.omega == pytest.approx(2 * math.pi * 60.1)
    assert state.theta == pytest.approx(math.pi / 2)


# --- end-to-end: real bytes through the codec into the bridge -------------


def test_bytes_through_codec_into_bridge() -> None:
    import struct

    from scpn_phase_orchestrator.adapters.synchrophasor_c37118 import (
        FRAME_TYPE_CONFIG2,
        FRAME_TYPE_DATA,
        SynchrophasorFrameCodec,
        compute_crc_ccitt,
    )

    def wrap(frame_type: int, body: bytes) -> bytes:
        framesize = 14 + len(body) + 2
        header = (
            bytes([0xAA, (frame_type << 4) | 1])
            + struct.pack(">H", framesize)
            + struct.pack(">H", 7)
            + struct.pack(">I", 1_700_000_000)
            + struct.pack(">I", 0)
        )
        frame = header + body
        return frame + struct.pack(">H", compute_crc_ccitt(frame))

    # CONFIG-2: one PMU, one rectangular integer phasor, 60 Hz, unit scale 1e5.
    name = b"BUS1".ljust(16, b"\x00")
    ch = b"VA".ljust(16, b"\x00")
    config_body = (
        struct.pack(">I", 1_000_000)  # TIME_BASE
        + struct.pack(">H", 1)  # NUM_PMU
        + name
        + struct.pack(">H", 1)  # IDCODE
        + struct.pack(">H", 0x0000)  # FORMAT: rect int
        + struct.pack(">H", 1)  # PHNMR
        + struct.pack(">H", 0)  # ANNMR
        + struct.pack(">H", 0)  # DGNMR
        + ch
        + struct.pack(">I", 100000)  # PHUNIT voltage, scale 1e5 -> 1.0 V/bit
        + struct.pack(">H", 0x0000)  # FNOM 60 Hz
        + struct.pack(">H", 0)  # CFGCNT
        + struct.pack(">h", 30)  # DATA_RATE
    )
    data_body = (
        struct.pack(">H", 0)  # STAT good
        + struct.pack(">hh", 200, 0)  # phasor (200, 0)
        + struct.pack(">h", 100)  # FREQ +100 mHz -> 60.1 Hz
        + struct.pack(">h", 0)  # DFREQ
    )

    codec = SynchrophasorFrameCodec()
    config = codec.decode_config2(wrap(FRAME_TYPE_CONFIG2, config_body))
    frame = codec.decode_data(wrap(FRAME_TYPE_DATA, data_body), config)
    bridge = C37118PhaseBridge.from_bindings([PhasorBinding("bus1_va")])
    state = bridge.extract_phases(config, [frame])["bus1_va"]
    assert state.theta == pytest.approx(0.0)
    assert state.amplitude == pytest.approx(200.0)  # 200 counts * 1.0 V/bit
    assert state.omega == pytest.approx(2 * math.pi * 60.1)
    assert state.quality == 1.0
