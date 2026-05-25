# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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

    def test_missing_ca_bundle_raises_without_full_path(self, tmp_path):
        from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        ca = tmp_path / "private" / "root-ca.pem"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        with pytest.raises(ConnectionError) as excinfo:
            SecureModbusAdapter("localhost", 802, cert, key, ca)

        msg = str(excinfo.value)
        assert "root-ca.pem" in msg
        assert "private" not in msg
        assert str(tmp_path) not in msg

    def test_default_tls_context_uses_system_trust_and_requires_server_auth(
        self,
        tmp_path,
    ):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_client.connect.return_value = True

        with (
            patch(
                "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
                return_value=mock_client,
            ),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.Purpose.SERVER_AUTH = object()
            mock_ssl.CERT_REQUIRED = object()
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            SecureModbusAdapter("localhost", 802, cert, key)

            mock_ctx.load_default_certs.assert_called_once_with(
                mock_ssl.Purpose.SERVER_AUTH
            )
            assert mock_ctx.check_hostname is True
            assert mock_ctx.verify_mode is mock_ssl.CERT_REQUIRED

    def test_tls_context_uses_explicit_ca_bundle_when_configured(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        ca = tmp_path / "ca.pem"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")
        ca.write_text("dummy-ca")

        mock_client = MagicMock()
        mock_client.connect.return_value = True

        with (
            patch(
                "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
                return_value=mock_client,
            ),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.CERT_REQUIRED = object()
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            SecureModbusAdapter("localhost", 802, cert, key, ca)

            mock_ctx.load_verify_locations.assert_called_once_with(cafile=str(ca))
            mock_ctx.load_default_certs.assert_not_called()
            assert mock_ctx.check_hostname is True
            assert mock_ctx.verify_mode is mock_ssl.CERT_REQUIRED

    def test_pymodbus_missing_raises(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        with (
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient", None),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
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

        with (
            patch(
                "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
                return_value=mock_client,
            ),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            adapter = SecureModbusAdapter("localhost", 802, cert, key)
            assert adapter.read_register(100) == 42

    @pytest.mark.parametrize("address", [-1, True, 1.5])
    def test_read_register_rejects_invalid_address(self, tmp_path, address):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_client.connect.return_value = True

        with (
            patch(
                "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
                return_value=mock_client,
            ),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            adapter = SecureModbusAdapter("localhost", 802, cert, key)
            with pytest.raises(ValueError, match="address"):
                adapter.read_register(address)  # type: ignore[arg-type]
            mock_client.read_holding_registers.assert_not_called()

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

        with (
            patch(
                "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
                return_value=mock_client,
            ),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            adapter = SecureModbusAdapter("localhost", 802, cert, key)
            with pytest.raises(ConnectionError, match="write error"):
                adapter.write_register(100, 99)

    @pytest.mark.parametrize(
        ("address", "value"),
        [(-1, 1), (True, 1), (1.5, 1), (0, True), (0, 1.5)],
    )
    def test_write_register_rejects_invalid_payload(self, tmp_path, address, value):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_client.connect.return_value = True

        with (
            patch(
                "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
                return_value=mock_client,
            ),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            adapter = SecureModbusAdapter("localhost", 802, cert, key)
            with pytest.raises(ValueError, match="address|value"):
                adapter.write_register(address, value)  # type: ignore[arg-type]
            mock_client.write_register.assert_not_called()

    def test_validate_connection_true(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_client.connect.return_value = True
        mock_client.connected = True

        with (
            patch(
                "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
                return_value=mock_client,
            ),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
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
        type(mock_client).connected = property(
            lambda self: (_ for _ in ()).throw(OSError("boom"))
        )

        with (
            patch(
                "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
                return_value=mock_client,
            ),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            adapter = SecureModbusAdapter("localhost", 802, cert, key)
            assert adapter.validate_connection() is False

    def test_validate_connection_unexpected_exception_propagates(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_client.connect.return_value = True
        type(mock_client).connected = property(
            lambda self: (_ for _ in ()).throw(ValueError("bad state"))
        )

        with (
            patch(
                "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
                return_value=mock_client,
            ),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            adapter = SecureModbusAdapter("localhost", 802, cert, key)
            with pytest.raises(ValueError, match="bad state"):
                adapter.validate_connection()

    def test_context_manager_closes_client(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_client.connect.return_value = True

        with (
            patch(
                "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
                return_value=mock_client,
            ),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            with SecureModbusAdapter("localhost", 802, cert, key):
                pass

            mock_client.close.assert_called_once_with()

    def test_connection_failure_raises(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_client.connect.return_value = False

        with (
            patch(
                "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
                return_value=mock_client,
            ),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            with pytest.raises(ConnectionError, match="connection failed"):
                SecureModbusAdapter("localhost", 802, cert, key)

    def test_connection_failure_does_not_leak_endpoint(self, tmp_path):
        cert = tmp_path / "client.pem"
        key = tmp_path / "client.key"
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        mock_client = MagicMock()
        mock_client.connect.return_value = False

        with (
            patch(
                "scpn_phase_orchestrator.adapters.modbus_tls.ModbusTlsClient",
                return_value=mock_client,
            ),
            patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl,
        ):
            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            with pytest.raises(ConnectionError) as excinfo:
                SecureModbusAdapter("plc.internal.example", 802, cert, key)
            msg = str(excinfo.value)
            assert "connection failed" in msg
            assert "plc.internal.example" not in msg
            assert "802" not in msg

    def test_tls_context_error_does_not_leak_backend_detail(self, tmp_path):
        cert = tmp_path / "secret" / "client.pem"
        key = tmp_path / "secret" / "client.key"
        cert.parent.mkdir()
        cert.write_text("dummy-cert")
        key.write_text("dummy-key")

        with patch("scpn_phase_orchestrator.adapters.modbus_tls.ssl") as mock_ssl:
            mock_ctx = MagicMock()
            mock_ctx.load_cert_chain.side_effect = Exception(
                f"bad certificate path {cert} key {key}"
            )
            mock_ssl.SSLContext.return_value = mock_ctx
            mock_ssl.PROTOCOL_TLS_CLIENT = 2
            mock_ssl.SSLError = Exception

            from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

            with pytest.raises(ConnectionError) as excinfo:
                SecureModbusAdapter("localhost", 802, cert, key)
            msg = str(excinfo.value)
            assert msg == "TLS context creation failed"
            assert "secret" not in msg
            assert "client.pem" not in msg
            assert "client.key" not in msg
            assert str(tmp_path) not in msg

    def test_missing_cert_error_does_not_leak_full_path(self, tmp_path):
        from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

        key = tmp_path / "client.key"
        key.write_text("dummy-key")
        missing_cert = tmp_path / "subdir" / "deeply" / "nested" / "no.pem"
        with pytest.raises(ConnectionError) as excinfo:
            SecureModbusAdapter("localhost", 802, missing_cert, key)
        msg = str(excinfo.value)
        assert "no.pem" in msg
        assert str(tmp_path) not in msg
        assert "subdir" not in msg

    def test_missing_key_error_does_not_leak_full_path(self, tmp_path):
        from scpn_phase_orchestrator.adapters.modbus_tls import SecureModbusAdapter

        cert = tmp_path / "client.pem"
        cert.write_text("dummy-cert")
        missing_key = tmp_path / "vault" / "secrets" / "private.key"
        with pytest.raises(ConnectionError) as excinfo:
            SecureModbusAdapter("localhost", 802, cert, missing_key)
        msg = str(excinfo.value)
        assert "private.key" in msg
        assert "vault" not in msg
        assert "secrets" not in msg
        assert str(tmp_path) not in msg


# Pipeline wiring: SecureModbusAdapter is the output actuator — tests above
# verify read/write/validate/error paths that connect SPO actions to SCADA,
# including that secret file locations are not leaked through ConnectionError.
