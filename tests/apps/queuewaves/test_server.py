# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves server tests

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scpn_phase_orchestrator.apps.queuewaves.config import (
    QueueWavesConfig,
    SecurityConfig,
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("starlette")

import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from scpn_phase_orchestrator.apps.queuewaves.server import create_app


@pytest.fixture()
def client(minimal_config: QueueWavesConfig) -> TestClient:
    app = create_app(minimal_config)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_payload_is_deterministic(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "tick": 0}


def test_state_503_before_data(client: TestClient) -> None:
    r = client.get("/api/v1/state")
    assert r.status_code == 503


def test_state_payload_is_deterministic(client: TestClient) -> None:
    r = client.get("/api/v1/state")
    assert r.status_code == 503
    assert r.json() == {"error": "no data yet"}


def test_anomalies_empty(client: TestClient) -> None:
    r = client.get("/api/v1/anomalies")
    assert r.status_code == 200
    assert r.json() == []


def test_services_empty(client: TestClient) -> None:
    r = client.get("/api/v1/services")
    assert r.status_code == 200
    assert r.json() == []


def test_plv_empty(client: TestClient) -> None:
    r = client.get("/api/v1/plv")
    assert r.status_code == 200
    assert r.json() == {"matrix": []}


def test_prom_metrics(client: TestClient) -> None:
    r = client.get("/api/v1/metrics/prometheus")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")


def test_prom_metrics_without_data_has_deterministic_payload(
    client: TestClient,
) -> None:
    r = client.get("/api/v1/metrics/prometheus")
    assert r.status_code == 200
    assert r.text == "\n"


def test_root_returns_html_or_text(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200


def test_check_503_without_data(client: TestClient) -> None:
    r = client.post("/api/v1/check")
    assert r.status_code == 503


def test_check_503_payload_is_deterministic(minimal_config: QueueWavesConfig) -> None:
    with patch(
        "scpn_phase_orchestrator.apps.queuewaves.server.PrometheusCollector"
    ) as MockCollector:
        mock_instance = MockCollector.return_value
        mock_instance.scrape = AsyncMock()
        mock_instance.get_signal_arrays = MagicMock(return_value={})
        mock_instance.close = AsyncMock()

        app = create_app(minimal_config)
        client = TestClient(app)
        r = client.post("/api/v1/check")

    assert r.status_code == 503
    assert r.json() == {"error": "not enough data"}


def test_check_has_stable_response_fields(minimal_config: QueueWavesConfig) -> None:
    with patch(
        "scpn_phase_orchestrator.apps.queuewaves.server.PrometheusCollector"
    ) as MockCollector:
        mock_instance = MockCollector.return_value
        mock_instance.scrape = AsyncMock()
        rng = np.random.default_rng(0)
        signals = {svc.name: rng.standard_normal(16) for svc in minimal_config.services}
        mock_instance.get_signal_arrays = MagicMock(return_value=signals)
        mock_instance.close = AsyncMock()

        app = create_app(minimal_config)
        client = TestClient(app)
        r = client.post("/api/v1/check")

    assert r.status_code == 200
    assert set(r.json()) == {"r_good", "r_bad", "regime", "anomalies"}


def test_health_tick_reflects_pipeline_advancement(
    minimal_config: QueueWavesConfig,
) -> None:
    with patch(
        "scpn_phase_orchestrator.apps.queuewaves.server.PrometheusCollector"
    ) as MockCollector:
        mock_instance = MockCollector.return_value
        mock_instance.scrape = AsyncMock()
        rng = np.random.default_rng(1)
        signals = {svc.name: rng.standard_normal(16) for svc in minimal_config.services}
        mock_instance.get_signal_arrays = MagicMock(return_value=signals)
        mock_instance.close = AsyncMock()

        app = create_app(minimal_config)
        client = TestClient(app)
        initial = client.get("/api/v1/health")
        check = client.post("/api/v1/check")
        after = client.get("/api/v1/health")

    assert initial.status_code == 200
    assert initial.json() == {"status": "ok", "tick": 0}
    assert check.status_code == 200
    assert after.status_code == 200
    assert after.json() == {"status": "ok", "tick": 1}


def test_websocket_closes_on_too_large_message(
    minimal_config: QueueWavesConfig,
) -> None:
    app = create_app(minimal_config)
    client = TestClient(app)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text("x" * 1025)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_text()
    assert exc_info.value.code == 1009


def test_state_history_empty(client: TestClient) -> None:
    r = client.get("/api/v1/state/history?n=10")
    assert r.status_code == 200
    assert r.json() == []


def test_production_requires_api_key_env(
    minimal_config: QueueWavesConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = QueueWavesConfig(
        prometheus_url=minimal_config.prometheus_url,
        services=minimal_config.services,
        scrape_interval_s=minimal_config.scrape_interval_s,
        buffer_length=minimal_config.buffer_length,
        thresholds=minimal_config.thresholds,
        coupling=minimal_config.coupling,
        alert_sinks=minimal_config.alert_sinks,
        server=minimal_config.server,
        security=SecurityConfig(mode="production"),
    )
    monkeypatch.delenv("QUEUEWAVES_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="QUEUEWAVES_API_KEY"):
        create_app(cfg)


def test_production_requires_request_api_key(
    minimal_config: QueueWavesConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = QueueWavesConfig(
        prometheus_url=minimal_config.prometheus_url,
        services=minimal_config.services,
        scrape_interval_s=minimal_config.scrape_interval_s,
        buffer_length=minimal_config.buffer_length,
        thresholds=minimal_config.thresholds,
        coupling=minimal_config.coupling,
        alert_sinks=minimal_config.alert_sinks,
        server=minimal_config.server,
        security=SecurityConfig(mode="production"),
    )
    monkeypatch.setenv("QUEUEWAVES_API_KEY", "test-key")
    client = TestClient(create_app(cfg))

    missing = client.get("/api/v1/health")
    present = client.get("/api/v1/health", headers={"X-API-Key": "test-key"})

    assert missing.status_code == 401
    assert present.status_code == 200


def test_production_rate_limits_requests(
    minimal_config: QueueWavesConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = QueueWavesConfig(
        prometheus_url=minimal_config.prometheus_url,
        services=minimal_config.services,
        scrape_interval_s=minimal_config.scrape_interval_s,
        buffer_length=minimal_config.buffer_length,
        thresholds=minimal_config.thresholds,
        coupling=minimal_config.coupling,
        alert_sinks=minimal_config.alert_sinks,
        server=minimal_config.server,
        security=SecurityConfig(mode="production", rate_limit_per_minute=1),
    )
    monkeypatch.setenv("QUEUEWAVES_API_KEY", "test-key")
    client = TestClient(create_app(cfg))
    headers = {"X-API-Key": "test-key"}

    first = client.get("/api/v1/health", headers=headers)
    second = client.get("/api/v1/health", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 429


def test_production_template_server_requires_request_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scpn_phase_orchestrator.apps.queuewaves.config import load_config

    cfg = load_config(Path("domainpacks/queuewaves/queuewaves.production.yaml"))
    monkeypatch.setenv("QUEUEWAVES_API_KEY", "template-key")
    client = TestClient(create_app(cfg))

    missing = client.get("/api/v1/health")
    present = client.get("/api/v1/health", headers={"X-API-Key": "template-key"})

    assert missing.status_code == 401
    assert present.status_code == 200
