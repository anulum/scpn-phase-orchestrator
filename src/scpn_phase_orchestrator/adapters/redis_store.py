# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Redis persistent state store

from __future__ import annotations

import json
from typing import Any

__all__ = ["RedisStateStore"]

try:
    import redis as _redis_mod

    _HAS_REDIS = True
except ModuleNotFoundError:  # pragma: no cover
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
    ) -> None:
        self._key = key
        if client is not None:
            self._client = client
        elif not _HAS_REDIS:
            raise RuntimeError(
                "redis package not installed — pip install redis"
            )
        else:
            self._client = _redis_mod.Redis(host=host, port=port, db=db)

    def save_state(self, sim_state: dict) -> None:
        """Serialise state dict to JSON and store in Redis."""
        self._client.set(self._key, json.dumps(sim_state))

    def load_state(self) -> dict | None:
        """Load state from Redis. Returns None if key does not exist."""
        raw = self._client.get(self._key)
        if raw is None:
            return None
        return json.loads(raw)

    def delete_state(self) -> None:
        """Remove the stored state key."""
        self._client.delete(self._key)

    @property
    def key(self) -> str:
        return self._key
