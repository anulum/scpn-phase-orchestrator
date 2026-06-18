# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Redis persistent state store

"""Redis-backed JSON state persistence adapter with explicit dependency checks.

``RedisStateStore`` validates host, port, database, and key parameters before
using an injected client or constructing a Redis client when the optional
package is installed. Stored payloads must be JSON objects, and missing keys
return ``None``. The adapter persists caller-provided state only; it does not
manage simulation lifecycle or background synchronization.
"""

from __future__ import annotations

import json
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

from scpn_phase_orchestrator.adapters._schema import (
    require_non_empty_str,
    require_non_negative_int,
    require_tcp_port,
)

__all__ = ["RedisStateStore"]

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

try:
    import redis as _redis_mod

    _HAS_REDIS = True
except ModuleNotFoundError:  # pragma: no cover
    # type ignore: optional redis dependency uses a None module sentinel.
    _redis_mod = None  # type: ignore[assignment]
    _HAS_REDIS = False


class RedisStateStore:
    """Persist simulation state in Redis for survival across restarts.

    When redis is not installed, all operations raise RuntimeError.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        key: str = "spo:sim_state",
        client: Any = None,
        password: str | None = None,
        ssl: bool = True,
        ssl_ca_certs: str | Path | None = None,
        ssl_certfile: str | Path | None = None,
        ssl_keyfile: str | Path | None = None,
    ) -> None:
        self._host = require_non_empty_str(host, field="Redis host")
        self._port = require_tcp_port(port, field="Redis port")
        self._db = require_non_negative_int(db, field="Redis db")
        self._key = require_non_empty_str(key, field="Redis key")
        if not isinstance(ssl, bool):
            raise ValueError("Redis ssl must be a bool")
        if password is not None:
            password = require_non_empty_str(password, field="Redis password")
        self._ssl = ssl
        self._password = password
        self._ssl_ca_certs = _optional_path(ssl_ca_certs, "Redis TLS CA bundle")
        self._ssl_certfile = _optional_path(ssl_certfile, "Redis TLS certificate")
        self._ssl_keyfile = _optional_path(ssl_keyfile, "Redis TLS key")
        if (self._ssl_certfile is None) != (self._ssl_keyfile is None):
            raise ValueError("Redis TLS certificate and key must be provided together")
        if not self._ssl and self._host not in _LOOPBACK_HOSTS:
            raise ValueError(
                "plaintext Redis connections are allowed only for loopback hosts"
            )
        if self._host not in _LOOPBACK_HOSTS and self._password is None:
            raise ValueError("remote Redis connections require password authentication")
        if (
            self._host not in _LOOPBACK_HOSTS
            and self._ssl
            and self._ssl_ca_certs is None
        ):
            raise ValueError("remote Redis TLS connections require a CA bundle")
        if client is not None:
            self._client = client
        elif not _HAS_REDIS:
            raise RuntimeError("redis package not installed — pip install redis")
        else:
            client_kwargs: dict[str, object] = {
                "host": self._host,
                "port": self._port,
                "db": self._db,
                "ssl": self._ssl,
            }
            if self._password is not None:
                client_kwargs["password"] = self._password
            if self._ssl:
                client_kwargs["ssl_cert_reqs"] = "required"
                if self._ssl_ca_certs is not None:
                    client_kwargs["ssl_ca_certs"] = self._ssl_ca_certs
                if self._ssl_certfile is not None:
                    client_kwargs["ssl_certfile"] = self._ssl_certfile
                if self._ssl_keyfile is not None:
                    client_kwargs["ssl_keyfile"] = self._ssl_keyfile
            self._client = _redis_mod.Redis(
                **client_kwargs,
            )

    def save_state(self, sim_state: dict[str, Any]) -> None:
        """Serialise state dict to JSON and store in Redis."""
        if not isinstance(sim_state, dict):
            raise ValueError("sim_state must be a JSON-serializable dict")
        try:
            payload = json.dumps(sim_state, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("sim_state must be JSON serializable") from exc
        self._client.set(self._key, payload)

    def load_state(self) -> dict[str, Any] | None:
        """Load state from Redis. Returns None if key does not exist."""
        raw = self._client.get(self._key)
        if raw is None:
            return None
        try:
            result = json.loads(raw, parse_constant=_reject_json_constant)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Redis payload must be a JSON object") from exc
        if not isinstance(result, dict):
            raise ValueError("Redis payload must be a JSON object")
        _require_finite_json_numbers(result)
        return result

    def delete_state(self) -> None:
        """Remove the stored state key."""
        self._client.delete(self._key)

    @property
    def key(self) -> str:
        """Redis key used for state storage."""
        return self._key


def _optional_path(value: str | Path | None, field: str) -> str | None:
    if value is None:
        return None
    return require_non_empty_str(str(value), field=field)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _require_finite_json_numbers(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_json_numbers(item)
        return
    if isinstance(value, list):
        for item in value:
            _require_finite_json_numbers(item)
        return
    if isinstance(value, bool):
        return
    if isinstance(value, Real) and not isfinite(float(value)):
        raise ValueError("Redis payload must contain only finite JSON numbers")
