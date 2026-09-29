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

from scpn_phase_orchestrator.apps.queuewaves.collector import (
    MetricBuffer,
    PrometheusCollector,
)
from tests.prometheus_range_server import prometheus_range_server
from tests.typing_contracts import assert_precise_ndarray_hint


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


async def test_collector_client_lifecycle() -> None:
    """Closing twice permits another public scrape with the same buffers."""
    with prometheus_range_server() as url:
        collector = PrometheusCollector(url, {"s": "up"}, buffer_length=4)
        try:
            await collector.scrape()
            await collector.close()
            await collector.close()
            await collector.scrape()
            np.testing.assert_allclose(
                collector.buffers["s"].values_array(), [1.1, 1.2]
            )
        finally:
            await collector.close()


def test_collector_array_annotations_use_float64_ndarray() -> None:
    values_hints = get_type_hints(MetricBuffer.values_array)
    arrays_hints = get_type_hints(PrometheusCollector.get_signal_arrays)
    assert_precise_ndarray_hint(values_hints["return"])
    assert "numpy.float64" in str(values_hints["return"])
    assert_precise_ndarray_hint(arrays_hints["return"])
    assert "numpy.float64" in str(arrays_hints["return"])


# Salvaged module-specific behavioural contracts from deleted bucket files.
def test_scrape_unreachable_prometheus() -> None:
    """Scrape against unreachable URL should log a warning but not raise."""

    async def _run() -> None:
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


def test_scrape_sync_push() -> None:
    """scrape_sync pushes values into named buffers."""
    collector = PrometheusCollector("http://unused:9090", {"a": "up", "b": "down"}, 8)
    collector.scrape_sync({"a": (1.0, 42.0), "b": (2.0, 99.0)})
    assert len(collector.buffers["a"]) == 1
    assert len(collector.buffers["b"]) == 1
    arr = collector.buffers["a"].values_array()
    np.testing.assert_allclose(arr, [42.0])


def test_scrape_sync_unknown_key() -> None:
    """Unknown keys in scrape_sync are silently ignored."""
    collector = PrometheusCollector("http://unused:9090", {"a": "up"}, 8)
    collector.scrape_sync({"unknown": (1.0, 0.0)})
    assert len(collector.buffers["a"]) == 0


def test_get_signal_arrays_requires_min_4() -> None:
    """get_signal_arrays only returns buffers with >= 4 samples."""
    collector = PrometheusCollector("http://unused:9090", {"a": "up"}, 8)
    for i in range(3):
        collector.scrape_sync({"a": (float(i), float(i))})
    assert collector.get_signal_arrays() == {}
    collector.scrape_sync({"a": (3.0, 3.0)})
    arrays = collector.get_signal_arrays()
    assert "a" in arrays
    assert len(arrays["a"]) == 4


async def test_scrape_successful_response() -> None:
    """An actual instant query appends the returned numeric sample."""
    requests: list[tuple[str, dict[str, str]]] = []
    with prometheus_range_server(requests=requests) as url:
        collector = PrometheusCollector(url + "/", {"svc": "rate(x[1m])"}, 8)
        try:
            buffers = await collector.scrape()
            np.testing.assert_allclose(buffers["svc"].values_array(), [1.1])
            assert requests == [("/api/v1/query", {"query": "rate(x[1m])"})]
        finally:
            await collector.close()


async def test_scrape_isolates_malformed_response_per_service(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed HTTP200 data preserves the bad buffer and permits recovery."""
    responses: dict[str, dict[str, object]] = {
        "up": {"status": "success", "data": {"result": [{"metric": {}}]}}
    }
    with prometheus_range_server(responses=responses) as url:
        collector = PrometheusCollector(url, {"bad": "up", "good": "up2"}, 8)
        collector.scrape_sync({"bad": (999.0, 7.0)})
        try:
            buffers = await collector.scrape()
            np.testing.assert_array_equal(buffers["bad"].values_array(), [7.0])
            np.testing.assert_allclose(buffers["good"].values_array(), [1.1])
            assert "malformed Prometheus response for bad" in caplog.text
            responses.clear()
            await collector.scrape()
            assert len(buffers["bad"]) == 2
            assert len(buffers["good"]) == 2
        finally:
            await collector.close()


@pytest.mark.parametrize(
    ("samples", "step_s", "message"),
    [
        (0, 1.0, "samples must be at least 1"),
        (-1, 1.0, "samples must be at least 1"),
        (4, 0.0, "step_s must be positive"),
        (4, -0.5, "step_s must be positive"),
        (4, float("nan"), "step_s must be positive"),
    ],
)
async def test_backfill_rejects_invalid_history_before_http(
    samples: int, step_s: float, message: str
) -> None:
    """Invalid history leaves existing samples and the HTTP endpoint untouched."""
    requests: list[tuple[str, dict[str, str]]] = []
    with prometheus_range_server(requests=requests) as url:
        collector = PrometheusCollector(url, {"service": "up"}, 4)
        collector.scrape_sync({"service": (9.0, 7.0)})
        try:
            with pytest.raises(ValueError, match=message):
                await collector.backfill(end=10.0, samples=samples, step_s=step_s)
            assert requests == []
            np.testing.assert_array_equal(
                collector.buffers["service"].values_array(), [7.0]
            )
            await collector.backfill(end=10.0, samples=4, step_s=0.5)
            assert collector.buffers["service"].full
            assert "service" in collector.get_signal_arrays()
        finally:
            await collector.close()


@pytest.mark.parametrize(
    "result",
    [
        {"metric": {}},
        {"values": [[10.0, "not-numeric"]]},
        {"values": [["not-a-timestamp", "1.0"]]},
        {"values": [[10.0, "1.0", "extra"]]},
        {"values": None},
    ],
)
async def test_backfill_isolates_malformed_service_and_recovers(
    result: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    """A malformed range does not block another service or a later valid range."""
    responses: dict[str, dict[str, object]] = {
        "bad_query": {"status": "success", "data": {"result": [result]}},
        "good_query": {
            "status": "success",
            "data": {
                "result": [
                    {
                        "values": [
                            [10.0, "2.5"],
                            [10.5, "3.5"],
                            [11.0, "4.5"],
                            [11.5, "5.5"],
                        ]
                    }
                ]
            },
        },
    }
    requests: list[tuple[str, dict[str, str]]] = []
    with prometheus_range_server(responses=responses, requests=requests) as url:
        collector = PrometheusCollector(
            url, {"bad": "bad_query", "good": "good_query"}, 4
        )
        collector.scrape_sync({"bad": (9.0, 7.0)})
        try:
            buffers = await collector.backfill(end=11.5, samples=4, step_s=0.5)
            np.testing.assert_array_equal(buffers["bad"].values_array(), [7.0])
            np.testing.assert_array_equal(
                collector.get_signal_arrays()["good"], [2.5, 3.5, 4.5, 5.5]
            )
            assert "bad" not in collector.get_signal_arrays()
            assert "malformed Prometheus range response for bad" in caplog.text
            assert requests == [
                (
                    "/api/v1/query_range",
                    {"query": query, "start": "10.0", "end": "11.5", "step": "0.5"},
                )
                for query in ("bad_query", "good_query")
            ]
            responses["bad_query"] = responses["good_query"]
            await collector.backfill(end=11.5, samples=4, step_s=0.5)
            np.testing.assert_array_equal(
                collector.get_signal_arrays()["bad"], [2.5, 3.5, 4.5, 5.5]
            )
        finally:
            await collector.close()


# Salvaged module-specific behavioural contracts from deleted broad tests.
class TestMetricBufferValidation:
    def test_rejects_zero_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            MetricBuffer(maxlen=0)

    def test_rejects_negative_maxlen(self) -> None:
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            MetricBuffer(maxlen=-5)
