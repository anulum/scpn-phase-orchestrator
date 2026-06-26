# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Adapter network schema fuzz tests

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scpn_phase_orchestrator.adapters._schema import (
    require_non_empty_str,
    require_non_negative_int,
    require_tcp_port,
    require_waveform_extractor_type,
)


@given(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.binary(),
        st.text(max_size=20).filter(lambda value: not value.strip()),
    )
)
def test_require_non_empty_str_fuzzes_invalid_inputs(value: object) -> None:
    with pytest.raises(ValueError):
        require_non_empty_str(value, field="field")


def test_require_non_empty_str_rejects_blank_string() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        require_non_empty_str("   ", field="field")


@given(st.one_of(st.booleans(), st.none(), st.text(max_size=20), st.floats(width=32)))
def test_require_non_negative_int_rejects_invalid_inputs(value: object) -> None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return
    with pytest.raises(ValueError):
        require_non_negative_int(value, field="field")


@given(
    st.one_of(
        st.integers(max_value=0),
        st.integers(min_value=65536),
    )
)
def test_require_tcp_port_rejects_out_of_range(port: int) -> None:
    with pytest.raises(ValueError, match="1..65535"):
        require_tcp_port(port, field="field")


def test_require_tcp_port_rejects_non_integer() -> None:
    with pytest.raises(ValueError, match="integer"):
        require_tcp_port(70000.1, field="field")  # type: ignore[arg-type]


def test_require_waveform_extractor_type_canonicalises_physical_aliases() -> None:
    assert require_waveform_extractor_type("physical", field="extractor_type") == (
        "hilbert"
    )
    assert require_waveform_extractor_type("wavelet", field="extractor_type") == (
        "wavelet"
    )
    assert require_waveform_extractor_type("zero_crossing", field="extractor_type") == (
        "zero_crossing"
    )


@pytest.mark.parametrize("extractor_type", ["event", "ring", "graph", "unknown"])
def test_require_waveform_extractor_type_rejects_non_waveform_extractors(
    extractor_type: str,
) -> None:
    with pytest.raises(ValueError, match="extractor_type"):
        require_waveform_extractor_type(extractor_type, field="extractor_type")


def test_modbus_adapter_host_validation_is_schema_backed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scpn_phase_orchestrator.adapters.hardware_io as hardware_io

    monkeypatch.setattr(hardware_io, "HAS_MODBUS", True)
    monkeypatch.setattr(
        hardware_io,
        "ModbusTcpClient",
        lambda *args, **kwargs: MagicMock(),
    )

    with pytest.raises(ValueError, match="Modbus host"):
        hardware_io.ModbusAdapter("   ", port=502)


def test_modbus_adapter_port_validation_is_schema_backed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scpn_phase_orchestrator.adapters.hardware_io as hardware_io

    monkeypatch.setattr(hardware_io, "HAS_MODBUS", True)
    monkeypatch.setattr(
        hardware_io,
        "ModbusTcpClient",
        lambda *args, **kwargs: MagicMock(),
    )

    with pytest.raises(ValueError, match="Modbus port"):
        hardware_io.ModbusAdapter("plc.local", port=70000)


def test_redis_store_schema_rejects_invalid_host() -> None:
    from scpn_phase_orchestrator.adapters.redis_store import RedisStateStore

    with pytest.raises(ValueError, match="Redis host"):
        RedisStateStore(host="", client=MagicMock())


def test_redis_store_schema_rejects_invalid_port() -> None:
    from scpn_phase_orchestrator.adapters.redis_store import RedisStateStore

    with pytest.raises(ValueError, match="Redis port"):
        RedisStateStore(port=0, client=MagicMock())


def test_redis_store_schema_rejects_negative_db() -> None:
    from scpn_phase_orchestrator.adapters.redis_store import RedisStateStore

    with pytest.raises(ValueError, match="Redis db"):
        RedisStateStore(db=-1, client=MagicMock())


def test_secure_modbus_adapter_fuzzes_invalid_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import scpn_phase_orchestrator.adapters.modbus_tls as modbus_tls

    cert = tmp_path / "client.pem"
    key = tmp_path / "client.key"
    cert.write_text("dummy")
    key.write_text("dummy")

    # Keep no real network/TLS side effects here.
    monkeypatch.setattr(modbus_tls, "ModbusTlsClient", MagicMock())
    with pytest.raises(ValueError, match="Modbus host"):
        modbus_tls.SecureModbusAdapter("   ", 802, cert, key)
