# SCPN Phase Orchestrator — QueueWaves Prometheus collector
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import logging
from collections import deque

import numpy as np
from numpy.typing import NDArray

__all__ = ["MetricBuffer", "PrometheusCollector"]

logger = logging.getLogger(__name__)


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

    @property
    def buffers(self) -> dict[str, MetricBuffer]:
        return self._buffers

    async def scrape(self) -> dict[str, MetricBuffer]:
        """Fire one PromQL instant query per service, push results into buffers."""
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
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
                except Exception:
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
