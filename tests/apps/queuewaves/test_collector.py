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
    assert "numpy.ndarray" in str(values_hints["return"])
    assert "numpy.float64" in str(values_hints["return"])
    assert "numpy.ndarray" in str(arrays_hints["return"])
    assert "numpy.float64" in str(arrays_hints["return"])
