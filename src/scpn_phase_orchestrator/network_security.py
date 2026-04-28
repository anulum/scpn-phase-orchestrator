# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Network service security helpers

from __future__ import annotations

import os
import threading
import time

__all__ = ["FixedWindowRateLimiter", "env_int", "is_production_mode"]


def is_production_mode(prefix: str) -> bool:
    """Return True when a service-specific or generic env profile is production."""
    for key in (f"{prefix}_ENV", f"{prefix}_PROFILE", "SPO_ENV", "SPO_PROFILE"):
        if os.environ.get(key, "").strip().lower() == "production":
            return True
    return False


def env_int(name: str, default: int) -> int:
    """Read a non-negative integer from the environment."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


class FixedWindowRateLimiter:
    """Thread-safe per-identity fixed-window rate limiter."""

    def __init__(self, limit_per_minute: int) -> None:
        if limit_per_minute < 1:
            raise ValueError("limit_per_minute must be >= 1")
        self._limit = limit_per_minute
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[int, int]] = {}

    def allow(self, identity: str, now: float | None = None) -> bool:
        """Return True if *identity* has capacity in the current minute."""
        timestamp = time.time() if now is None else now
        window = int(timestamp // 60)
        key = identity or "anonymous"
        with self._lock:
            current_window, count = self._windows.get(key, (window, 0))
            if current_window != window:
                current_window, count = window, 0
            if count >= self._limit:
                self._windows[key] = (current_window, count)
                return False
            self._windows[key] = (current_window, count + 1)
            return True
