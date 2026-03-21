# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Modbus TLS adapter tests

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestSecureModbusAdapterTLS:
    def test_missing_cert_raises(self, tmp_path):
        from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

        key = tmp_path / "client.key"
        key.write_text("dummy-key")
        with pytest.raises(ConnectionError, match="certificate not found"):
            SecureModbusAdapter("localhost", 802, tmp_path / "no.pem", key)

    def test_missing_key_raises(self, tmp_path):
        from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

        cert = tmp_path / "client.pem"
        cert.write_text("dummy-cert")
        with pytest.raises(ConnectionError, match="key not found"):
            SecureModbusAdapter("localhost", 802, cert, tmp_path / "no.key")

    def test_pymodbus_missing_raises(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        with patch(
            "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient", None
        ), patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl:
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            with pytest.raises(ConnectionError, match="pymodbus"):
                SecureModbusAdapter("localhost", 802, cert, key)

    def test_read_register_success(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_result.registers = [42]
        mock_client.read_holding_registers.return_value = mock_result
        mock_client.connect.return_value = True

        with patch(
            "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
            return_value=mock_client,
        ), patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl:
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            adapter = SecureModbusAdapter("localhost", 802, cert, key)
            assert adapter.read_register(100) == 42

    def test_write_register_error(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_write_result = MagicMock()
        mock_write_result.isError.return_value = True
        mock_client.write_register.return_value = mock_write_result
        mock_client.connect.return_value = True

        with patch(
            "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
            return_value=mock_client,
        ), patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl:
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            adapter = SecureModbusAdapter("localhost", 802, cert, key)
            with pytest.raises(ConnectionError, match="write error"):
                adapter.write_register(100, 99)

    def test_validate_connection_true(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_client.connect.return_value = True
        mock_client.connected = True

        with patch(
            "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
            return_value=mock_client,
        ), patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl:
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            adapter = SecureModbusAdapter("localhost", 802, cert, key)
            assert adapter.validate_connection() is True

    def test_validate_connection_exception(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_client.connect.return_value = True
        type(mock_client).connected = property(lambda self: (_ for _ in ()).throw(OSError("boom")))

        with patch(
            "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
            return_value=mock_client,
        ), patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl:
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            adapter = SecureModbusAdapter("localhost", 802, cert, key)
            assert adapter.validate_connection() is False

    def test_connection_failure_raises(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_client.connect.return_value = False

        with patch(
            "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
            return_value=mock_client,
        ), patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl:
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            with pytest.raises(ConnectionError, match="connection failed"):
                SecureModbusAdapter("localhost", 802, cert, key)
