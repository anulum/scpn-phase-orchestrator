# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Modbus adapter with TLS authentication

"""Secure Modbus TCP adapter with mutual TLS.

Wraps pymodbus with an ``ssl.SSLContext`` for certificate-authenticated
connections to SCADA endpoints per IEC 62443 zone/conduit requirements.
"""

from __future__ import annotations

import ssl
from pathlib import Path

__all__ = ["SecureModbusAdapter", "HAS_PYMODBUS"]

try:
    from pymodbus.client import ModbusTlsClient

    HAS_PYMODBUS = True
except ImportError:
    ModbusTlsClient = None
    HAS_PYMODBUS = False


class SecureModbusAdapter:
    """Modbus TCP client with TLS mutual authentication.

    Parameters
    ----------
    host : str
        Target Modbus device hostname or IP.
    port : int
        TCP port (default Modbus/TLS: 802).
    tls_cert_path : str | Path
        Path to client certificate (PEM).
    tls_key_path : str | Path
        Path to client private key (PEM).
    """

    def __init__(
        self,
        host: str,
        port: int,
        tls_cert_path: str | Path,
        tls_key_path: str | Path,
    ) -> None:
        self._host = host
        self._port = port
        self._cert = Path(tls_cert_path)
        self._key = Path(tls_key_path)
        self._ctx = self._build_tls_context()
        self._client = self._connect()

    def _build_tls_context(self) -> ssl.SSLContext:
        if not self._cert.exists():
            raise ConnectionError(f"TLS certificate not found: {self._cert}")
        if not self._key.exists():
            raise ConnectionError(f"TLS key not found: {self._key}")
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.load_cert_chain(certfile=str(self._cert), keyfile=str(self._key))
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        except ssl.SSLError as exc:
            raise ConnectionError(f"TLS context creation failed: {exc}") from exc

    def _connect(self) -> object:
        if ModbusTlsClient is None:
            raise ConnectionError(
                "pymodbus is required for Modbus TLS. Install: pip install pymodbus"
            )
        client = ModbusTlsClient(
            host=self._host,
            port=self._port,
            sslctx=self._ctx,
        )
        if not client.connect():
            raise ConnectionError(
                f"Modbus TLS connection failed: {self._host}:{self._port}"
            )
        return client

    def read_register(self, address: int) -> int:
        """Read a single holding register.

        Raises ConnectionError if the read fails or returns an error frame.
        """
        result = self._client.read_holding_registers(  # type: ignore[attr-defined]
            address, count=1
        )
        if result.isError():
            raise ConnectionError(f"Modbus read error at address {address}: {result}")
        return int(result.registers[0])

    def write_register(self, address: int, value: int) -> None:
        """Write a single holding register.

        Raises ConnectionError if the write fails.
        """
        result = self._client.write_register(address, value)  # type: ignore[attr-defined]
        if result.isError():
            raise ConnectionError(f"Modbus write error at address {address}: {result}")

    def validate_connection(self) -> bool:
        """Return True if the TLS-wrapped Modbus connection is active."""
        try:
            return bool(self._client.connected)  # type: ignore[attr-defined]
        except Exception:
            return False
