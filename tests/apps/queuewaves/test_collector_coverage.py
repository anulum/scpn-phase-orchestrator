# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves collector coverage tests

"""Coverage tests for collector.py — async scrape, successful response path,
sync push, and buffer operations."""

from __future__ import annotations

import asyncio
import importlib.util

import numpy as np
import pytest

from scpn_phase_orchestrator.apps.queuewaves.collector import (
    MetricBuffer,
    PrometheusCollector,
)

_HAS_HTTPX = importlib.util.find_spec("httpx") is not None


@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
def test_scrape_unreachable_prometheus():
    """Scrape against unreachable URL should log a warning but not raise."""

    async def _run():
        collector = PrometheusCollector(
            "http://127.0.0.1:1",
            {"svc": "up"},
            buffer_length=4,
        )
        buffers = await collector.scrape()
        assert "svc" in buffers
        assert len(buffers["svc"]) == 0
        await collector.close()

    asyncio.run(_run())


def test_scrape_sync_push():
    """scrape_sync pushes values into named buffers."""
    collector = PrometheusCollector("http://unused:9090", {"a": "up", "b": "down"}, 8)
    collector.scrape_sync({"a": (1.0, 42.0), "b": (2.0, 99.0)})
    assert len(collector.buffers["a"]) == 1
    assert len(collector.buffers["b"]) == 1
    arr = collector.buffers["a"].values_array()
    np.testing.assert_allclose(arr, [42.0])


def test_scrape_sync_unknown_key():
    """Unknown keys in scrape_sync are silently ignored."""
    collector = PrometheusCollector("http://unused:9090", {"a": "up"}, 8)
    collector.scrape_sync({"unknown": (1.0, 0.0)})
    assert len(collector.buffers["a"]) == 0


def test_get_signal_arrays_requires_min_4():
    """get_signal_arrays only returns buffers with >= 4 samples."""
    collector = PrometheusCollector("http://unused:9090", {"a": "up"}, 8)
    for i in range(3):
        collector.scrape_sync({"a": (float(i), float(i))})
    assert collector.get_signal_arrays() == {}
    collector.scrape_sync({"a": (3.0, 3.0)})
    arrays = collector.get_signal_arrays()
    assert "a" in arrays
    assert len(arrays["a"]) == 4


def test_metric_buffer_full():
    buf = MetricBuffer(maxlen=3)
    assert not buf.full
    for i in range(3):
        buf.push(float(i), float(i))
    assert buf.full
    assert len(buf) == 3


@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
def test_scrape_successful_response():
    """Mock httpx response to exercise the successful scrape code path."""

    async def _run():
        import httpx

        prom_response = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{"metric": {"__name__": "up"}, "value": [1000.0, "1.5"]}],
            },
        }

        mock_response = httpx.Response(
            200,
            json=prom_response,
            request=httpx.Request("GET", "http://fake:9090/api/v1/query"),
        )

        collector = PrometheusCollector(
            "http://fake:9090",
            {"svc": "up"},
            buffer_length=8,
        )

        # Replace the client with a mock that returns our response
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        collector._client = mock_client

        buffers = await collector.scrape()
        assert len(buffers["svc"]) == 1
        arr = buffers["svc"].values_array()
        np.testing.assert_allclose(arr, [1.5])

        await collector.close()

    asyncio.run(_run())
