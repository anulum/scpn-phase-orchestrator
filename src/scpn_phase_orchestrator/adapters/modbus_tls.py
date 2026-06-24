# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Modbus adapter with TLS authentication

"""Secure Modbus TCP adapter with mutual TLS.

Wraps pymodbus with an ``ssl.SSLContext`` for certificate-authenticated
connections to SCADA endpoints per IEC 62443 zone/conduit requirements.
Server certificate verification is always enabled: pass ``ca_cert_path`` for a
deployment CA bundle or rely on the operating-system trust store.
"""

from __future__ import annotations

import ssl
from numbers import Integral
from pathlib import Path

from scpn_phase_orchestrator.adapters._schema import (
    require_non_empty_str,
    require_tcp_port,
)

__all__ = ["SecureModbusAdapter", "HAS_PYMODBUS"]

try:
    # type ignore: pymodbus is optional and has incomplete typing on supported releases.
    from pymodbus.client import ModbusTlsClient  # type: ignore[import-not-found]

    HAS_PYMODBUS = True
except ImportError:
    # type ignore: None sentinel mirrors the optional pymodbus import boundary.
    ModbusTlsClient = None  # type: ignore[assignment,misc]
    HAS_PYMODBUS = False


def _non_negative_int(value: object, *, field: str) -> int:
    """Return ``value`` as a non-negative integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _int_value(value: object, *, field: str) -> int:
    """Return a named integer field from a mapping, else raise."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    return int(value)


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
    ca_cert_path : str | Path | None
        Optional CA bundle (PEM) for server verification. When omitted, the
        operating-system trust store is used. Server verification is never
        disabled by this adapter.
    """

    def __init__(
        self,
        host: str,
        port: int,
        tls_cert_path: str | Path,
        tls_key_path: str | Path,
        ca_cert_path: str | Path | None = None,
    ) -> None:
        self._host = require_non_empty_str(host, field="Modbus host")
        self._port = require_tcp_port(port, field="Modbus port")
        self._cert = Path(tls_cert_path)
        self._key = Path(tls_key_path)
        self._ca = Path(ca_cert_path) if ca_cert_path is not None else None
        self._ctx = self._build_tls_context()
        self._client = self._connect()

    def _build_tls_context(self) -> ssl.SSLContext:
        # Filename only in the raised message — full paths to key material
        # must never surface in error strings that may reach logs or clients.
        """Build the mutual-TLS SSL context for the Modbus connection."""
        if not self._cert.exists():
            raise ConnectionError(f"TLS certificate not found: {self._cert.name}")
        if not self._key.exists():
            raise ConnectionError(f"TLS key not found: {self._key.name}")
        if self._ca is not None and not self._ca.exists():
            raise ConnectionError(f"TLS CA bundle not found: {self._ca.name}")
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.load_cert_chain(certfile=str(self._cert), keyfile=str(self._key))
            if self._ca is not None:
                ctx.load_verify_locations(cafile=str(self._ca))
            else:
                ctx.load_default_certs(ssl.Purpose.SERVER_AUTH)
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            return ctx
        except ssl.SSLError:
            raise ConnectionError("TLS context creation failed") from None

    def _connect(self) -> object:
        """Open the mutual-TLS Modbus TCP connection."""
        if ModbusTlsClient is None:
            raise ConnectionError(
                "pymodbus is required for Modbus TLS. Install: pip install pymodbus"
            )
        client = ModbusTlsClient(
            host=self._host,
            port=self._port,
            sslctx=self._ctx,
        )
        if not client.connect():  # type: ignore[no-untyped-call]  # pymodbus client.connect is untyped
            raise ConnectionError("Modbus TLS connection failed")
        return client

    def read_register(self, address: int) -> int:
        """Read a single holding register.

        Raises ConnectionError if the read fails or returns an error frame.

        Parameters
        ----------
        address : int
            Modbus register address.

        Returns
        -------
        int
            The holding-register value.

        Raises
        ------
        ConnectionError
            If the TLS Modbus connection is not active.
        """
        address = _non_negative_int(address, field="address")
        # type ignore: optional pymodbus client is stored as object after runtime guard.
        result = self._client.read_holding_registers(  # type: ignore[attr-defined]
            address, count=1
        )
        if result.isError():
            raise ConnectionError(f"Modbus read error at address {address}: {result}")
        return int(result.registers[0])

    def write_register(self, address: int, value: int) -> None:
        """Write a single holding register.

        Raises ConnectionError if the write fails.

        Parameters
        ----------
        address : int
            Modbus register address.
        value : int
            Register value to write.

        Raises
        ------
        ConnectionError
            If the TLS Modbus connection is not active.
        """
        address = _non_negative_int(address, field="address")
        value = _int_value(value, field="value")
        # type ignore: optional pymodbus client is stored as object after runtime guard.
        result = self._client.write_register(address, value)  # type: ignore[attr-defined]
        if result.isError():
            raise ConnectionError(f"Modbus write error at address {address}: {result}")

    def validate_connection(self) -> bool:
        """Return True if the TLS-wrapped Modbus connection is active.

        Returns
        -------
        bool
            Return True if the TLS-wrapped Modbus connection is active.
        """
        try:
            # type ignore: optional pymodbus client is stored as object.
            return bool(self._client.connected)  # type: ignore[attr-defined]
        except (AttributeError, OSError, RuntimeError):
            return False

    def close(self) -> None:
        """Close the Modbus/TLS client when the pymodbus client exposes close()."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> SecureModbusAdapter:
        """Return self for context-manager use."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the client on context-manager exit."""
        self.close()
