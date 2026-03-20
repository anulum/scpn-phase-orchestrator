# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves Prometheus collector

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["MetricBuffer", "PrometheusCollector"]

logger = logging.getLogger(__name__)

try:
    from httpx import HTTPError as _HTTPError
except ImportError:  # pragma: no cover
    _HTTPError = OSError  # type: ignore[assignment,misc]

_SCRAPE_ERRORS: tuple[type[BaseException], ...] = (OSError, RuntimeError, _HTTPError)


class MetricBuffer:
    """Fixed-size ring buffer of (timestamp, value) pairs for one service."""

    def __init__(self, maxlen: int = 64):
        self._maxlen = maxlen
        self._buf: deque[tuple[float, float]] = deque(maxlen=maxlen)

    def push(self, timestamp: float, value: float) -> None:
        self._buf.append((timestamp, value))

    @property
    def ready(self) -> bool:
        return len(self._buf) >= 4

    @property
    def full(self) -> bool:
        return len(self._buf) >= self._maxlen

    def values_array(self) -> NDArray:
        return np.array([v for _, v in self._buf], dtype=np.float64)

    def __len__(self) -> int:
        return len(self._buf)


class PrometheusCollector:
    """Scrapes Prometheus instant query API for configured services.

    Requires httpx (installed via pip install scpn-phase-orchestrator[queuewaves]).
    """

    def __init__(
        self,
        prometheus_url: str,
        queries: dict[str, str],
        buffer_length: int = 64,
    ):
        self._base_url = prometheus_url.rstrip("/")
        self._queries = queries
        self._buffers: dict[str, MetricBuffer] = {
            name: MetricBuffer(maxlen=buffer_length) for name in queries
        }
        self._client: Any = None

    @property
    def buffers(self) -> dict[str, MetricBuffer]:
        return self._buffers

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def scrape(self) -> dict[str, MetricBuffer]:
        """Fire one PromQL instant query per service, push results into buffers."""
        client = await self._get_client()
        for name, promql in self._queries.items():
            try:
                resp = await client.get(
                    f"{self._base_url}/api/v1/query",
                    params={"query": promql},
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("data", {}).get("result", [])
                if results:
                    ts, val = results[0]["value"]
                    self._buffers[name].push(float(ts), float(val))
            except _SCRAPE_ERRORS:
                logger.warning("scrape failed for %s", name, exc_info=True)
        return self._buffers

    def scrape_sync(
        self,
        values: dict[str, tuple[float, float]],
    ) -> dict[str, MetricBuffer]:
        """Synchronous push for testing: values = {name: (timestamp, value)}."""
        for name, (ts, val) in values.items():
            if name in self._buffers:
                self._buffers[name].push(ts, val)
        return self._buffers

    def get_signal_arrays(self) -> dict[str, NDArray]:
        """Return value arrays for all ready buffers."""
        return {
            name: buf.values_array() for name, buf in self._buffers.items() if buf.ready
        }
