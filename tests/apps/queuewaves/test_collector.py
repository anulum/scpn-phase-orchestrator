# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves collector tests

from __future__ import annotations

import asyncio
from typing import get_type_hints

import numpy as np
import pytest

from tests.typing_contracts import assert_precise_ndarray_hint

try:
    import httpx  # noqa: F401

    _HAS_HTTPX = True
except ModuleNotFoundError:
    _HAS_HTTPX = False

from scpn_phase_orchestrator.apps.queuewaves.collector import (
    MetricBuffer,
    PrometheusCollector,
)


def test_metric_buffer_push_and_ready() -> None:
    buf = MetricBuffer(maxlen=8)
    assert not buf.ready
    for i in range(4):
        buf.push(float(i), float(i) * 10.0)
    assert buf.ready
    assert len(buf) == 4


def test_metric_buffer_ring_overflow() -> None:
    buf = MetricBuffer(maxlen=4)
    for i in range(10):
        buf.push(float(i), float(i))
    assert len(buf) == 4
    arr = buf.values_array()
    np.testing.assert_array_equal(arr, [6.0, 7.0, 8.0, 9.0])


def test_metric_buffer_full() -> None:
    buf = MetricBuffer(maxlen=4)
    assert not buf.full
    for i in range(4):
        buf.push(float(i), float(i))
    assert buf.full


def test_collector_sync_push() -> None:
    queries = {"svc-a": "rate(x[1m])", "svc-b": "rate(y[1m])"}
    collector = PrometheusCollector("http://localhost:9090", queries, buffer_length=8)
    for i in range(8):
        collector.scrape_sync(
            {
                "svc-a": (float(i), np.sin(i * 0.5)),
                "svc-b": (float(i), np.cos(i * 0.5)),
            }
        )
    signals = collector.get_signal_arrays()
    assert "svc-a" in signals
    assert "svc-b" in signals
    assert len(signals["svc-a"]) == 8


def test_collector_missing_service_ignored() -> None:
    collector = PrometheusCollector(
        "http://localhost:9090", {"s": "up"}, buffer_length=4
    )
    collector.scrape_sync({"unknown": (0.0, 1.0)})
    assert len(collector.buffers["s"]) == 0


def test_collector_get_signal_arrays_skips_not_ready() -> None:
    collector = PrometheusCollector(
        "http://localhost:9090", {"s": "up"}, buffer_length=8
    )
    collector.scrape_sync({"s": (0.0, 1.0)})
    assert "s" not in collector.get_signal_arrays()


@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
def test_collector_client_lifecycle() -> None:
    async def _run() -> None:
        collector = PrometheusCollector(
            "http://localhost:9090", {"s": "up"}, buffer_length=4
        )
        assert collector._client is None
        client = await collector._get_client()
        assert client is not None
        same = await collector._get_client()
        assert same is client
        await collector.close()
        assert collector._client is None

    asyncio.run(_run())


def test_collector_array_annotations_use_float64_ndarray() -> None:
    values_hints = get_type_hints(MetricBuffer.values_array)
    arrays_hints = get_type_hints(PrometheusCollector.get_signal_arrays)
    assert_precise_ndarray_hint(values_hints["return"])
    assert "numpy.float64" in str(values_hints["return"])
    assert_precise_ndarray_hint(arrays_hints["return"])
    assert "numpy.float64" in str(arrays_hints["return"])


# Salvaged module-specific behavioural contracts from deleted bucket files.
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


# Salvaged module-specific behavioural contracts from deleted broad tests.
class TestMetricBufferValidation:
    def test_rejects_zero_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            MetricBuffer(maxlen=0)

    def test_rejects_negative_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            MetricBuffer(maxlen=-5)
