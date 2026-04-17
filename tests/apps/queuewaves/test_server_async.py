# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves async server tests

"""Async tests using httpx.AsyncClient for proper ASGI lifecycle testing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from scpn_phase_orchestrator.apps.queuewaves.config import QueueWavesConfig

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pytest_asyncio")

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from scpn_phase_orchestrator.apps.queuewaves.server import create_app

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def async_client(minimal_config: QueueWavesConfig) -> AsyncClient:
    app = create_app(minimal_config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_health_async(async_client: AsyncClient) -> None:
    r = await async_client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_state_503_async(async_client: AsyncClient) -> None:
    r = await async_client.get("/api/v1/state")
    assert r.status_code == 503


async def test_concurrent_requests(async_client: AsyncClient) -> None:
    """Multiple endpoints hit concurrently return correct responses."""
    import asyncio

    results = await asyncio.gather(
        async_client.get("/api/v1/health"),
        async_client.get("/api/v1/anomalies"),
        async_client.get("/api/v1/services"),
        async_client.get("/api/v1/plv"),
    )
    assert results[0].status_code == 200
    assert results[1].json() == []
    assert results[2].json() == []
    assert results[3].json() == {"matrix": []}


async def test_check_with_mocked_data_async(
    minimal_config: QueueWavesConfig,
) -> None:
    """Async check endpoint with mocked collector signals."""
    rng = np.random.default_rng(0)
    signals = {svc.name: rng.standard_normal(16) for svc in minimal_config.services}

    with patch(
        "scpn_phase_orchestrator.apps.queuewaves.server.PrometheusCollector"
    ) as MockCollector:
        mock_instance = MockCollector.return_value
        mock_instance.scrape = AsyncMock()
        mock_instance.get_signal_arrays = MagicMock(return_value=signals)
        mock_instance.close = AsyncMock()

        app = create_app(minimal_config)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post("/api/v1/check")
            assert r.status_code == 200
            data = r.json()
            assert "r_good" in data
            assert "regime" in data


async def test_prom_metrics_content_type_async(async_client: AsyncClient) -> None:
    r = await async_client.get("/api/v1/metrics/prometheus")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")


async def test_state_history_empty_async(async_client: AsyncClient) -> None:
    r = await async_client.get("/api/v1/state/history?n=5")
    assert r.status_code == 200
    assert r.json() == []
