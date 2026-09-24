# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Network service security helpers

"""Small network-service security helpers shared by optional HTTP surfaces.

The module provides production-mode environment detection, validated
non-negative integer environment parsing, and a thread-safe per-identity
token-bucket rate limiter. Helpers are intentionally local and dependency-free:
they do not configure servers, store credentials, or perform authentication by
themselves.
"""

from __future__ import annotations

import os
import threading
import time
from math import ceil, isfinite
from numbers import Real

__all__ = [
    "FixedWindowRateLimiter",
    "TokenBucketRateLimiter",
    "env_int",
    "is_production_mode",
]


def _validated_identifier(value: str, label: str) -> str:
    """Return the validated network identifier, else raise."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def is_production_mode(prefix: str) -> bool:
    """Return True when a service-specific or generic env profile is production.

    Parameters
    ----------
    prefix : str
        Service-specific environment-variable prefix.

    Returns
    -------
    bool
        ``True`` when the environment profile is production.
    """
    prefix = _validated_identifier(prefix, "prefix")
    for key in (f"{prefix}_ENV", f"{prefix}_PROFILE", "SPO_ENV", "SPO_PROFILE"):
        if os.environ.get(key, "").strip().lower() == "production":
            return True
    return False


def env_int(name: str, default: int) -> int:
    """Read a non-negative integer from the environment.

    Parameters
    ----------
    name : str
        The span or resource name.
    default : int
        Default value when the variable is unset.

    Returns
    -------
    int
        The non-negative integer read from the environment.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    name = _validated_identifier(name, "name")
    if not isinstance(default, int) or isinstance(default, bool) or default < 0:
        raise ValueError(f"{name} default must be a non-negative integer")
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


#: Tracked identities above which buckets that have refilled completely are
#: dropped; a full bucket is indistinguishable from an identity never seen.
_MAX_TRACKED_IDENTITIES = 10_000


class TokenBucketRateLimiter:
    """Thread-safe per-identity token-bucket rate limiter.

    Buckets are kept per identity; once more than ``_MAX_TRACKED_IDENTITIES``
    are tracked, the ones that have refilled to capacity are dropped, so a
    stream of distinct identities cannot grow the table without bound.
    """

    def __init__(
        self,
        limit_per_minute: int,
        *,
        burst_capacity: int | None = None,
    ) -> None:
        if not isinstance(limit_per_minute, int) or isinstance(limit_per_minute, bool):
            raise ValueError("limit_per_minute must be an integer >= 1")
        if limit_per_minute < 1:
            raise ValueError("limit_per_minute must be >= 1")
        if burst_capacity is None:
            burst = max(1, min(limit_per_minute, ceil(limit_per_minute / 10)))
        else:
            if not isinstance(burst_capacity, int) or isinstance(burst_capacity, bool):
                raise ValueError("burst_capacity must be an integer >= 1")
            if burst_capacity < 1:
                raise ValueError("burst_capacity must be >= 1")
            if burst_capacity > limit_per_minute:
                raise ValueError("burst_capacity must be <= limit_per_minute")
            burst = burst_capacity
        self._limit = limit_per_minute
        self._capacity = burst
        self._refill_per_second = limit_per_minute / 60.0
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, identity: str, now: float | None = None) -> bool:
        """Return True if *identity* has at least one available token.

        Parameters
        ----------
        identity : str
            Caller identity for rate limiting.
        now : float | None
            Current time in seconds, or ``None``.

        Returns
        -------
        bool
            ``True`` when the identity has an available token.

        Raises
        ------
        ValueError
            If the inputs are invalid or inconsistent.
        """
        key = _validated_identifier(identity, "identity")
        if now is None:
            timestamp = time.time()
        else:
            if (
                not isinstance(now, Real)
                or isinstance(now, bool)
                or not isfinite(float(now))
            ):
                raise ValueError("now must be a finite real timestamp")
            timestamp = float(now)
        with self._lock:
            tokens, updated_at = self._buckets.get(
                key, (float(self._capacity), timestamp)
            )
            elapsed = max(0.0, timestamp - updated_at)
            tokens = min(
                float(self._capacity), tokens + elapsed * self._refill_per_second
            )
            if tokens < 1.0:
                self._buckets[key] = (tokens, timestamp)
                return False
            self._buckets[key] = (tokens - 1.0, timestamp)
            if len(self._buckets) > _MAX_TRACKED_IDENTITIES:
                self._drop_refilled_buckets(timestamp)
            return True

    def _drop_refilled_buckets(self, timestamp: float) -> None:
        """Drop buckets that have refilled to capacity by ``timestamp``.

        Called with the lock held. A dropped identity starts again from a full
        bucket, which is exactly the state it had refilled to.
        """
        capacity = float(self._capacity)
        self._buckets = {
            identity: (tokens, updated_at)
            for identity, (tokens, updated_at) in self._buckets.items()
            if tokens + max(0.0, timestamp - updated_at) * self._refill_per_second
            < capacity
        }


class FixedWindowRateLimiter(TokenBucketRateLimiter):
    """Backward-compatible name for the production token-bucket limiter."""
