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

from scpn_phase_orchestrator.apps.queuewaves import server as server_mod
from scpn_phase_orchestrator.apps.queuewaves.pipeline import (
    PipelineSnapshot,
    ServiceSnapshot,
)
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


def test_websocket_rejects_non_keepalive_content(
    minimal_config: QueueWavesConfig,
) -> None:
    app = create_app(minimal_config)
    client = TestClient(app)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text('{"type":"actuate","K":9}')
        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_text()
    assert exc_info.value.code == 1003


def test_websocket_accepts_json_keepalive(
    minimal_config: QueueWavesConfig,
) -> None:
    app = create_app(minimal_config)
    client = TestClient(app)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text('{"type":"ping"}')


def test_state_history_empty(client: TestClient) -> None:
    r = client.get("/api/v1/state/history?n=10")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, []),
        (-3, []),
        (3, [7, 8, 9]),
        (100, list(range(10))),
    ],
)
def test_tail_page_non_positive_n_returns_empty(n: int, expected: list[int]) -> None:
    """`_tail_page` must return an empty page for n <= 0, not the whole history.

    Guards the ``[-n:]`` slice pitfall: ``items[-0:]`` is ``items[0:]`` (the
    whole list) and ``items[-(-3):]`` is an arbitrary tail, both wrong for a
    "last n" query.
    """
    from scpn_phase_orchestrator.apps.queuewaves.server import _tail_page

    assert _tail_page(list(range(10)), n) == expected


def test_production_rejects_present_but_wrong_api_key(
    minimal_config: QueueWavesConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present but incorrect X-API-Key is rejected (constant-time compare)."""
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
    monkeypatch.setenv("QUEUEWAVES_API_KEY", "correct-key")
    client = TestClient(create_app(cfg))

    wrong = client.get("/api/v1/health", headers={"X-API-Key": "wrong-key"})
    correct = client.get("/api/v1/health", headers={"X-API-Key": "correct-key"})

    assert wrong.status_code == 401
    assert correct.status_code == 200


def test_websocket_rejects_present_but_wrong_api_key(
    minimal_config: QueueWavesConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present but incorrect X-API-Key on the WS stream is closed with 1008."""
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
    monkeypatch.setenv("QUEUEWAVES_API_KEY", "correct-key")
    client = TestClient(create_app(cfg))

    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect("/ws/stream", headers={"X-API-Key": "wrong-key"}),
    ):
        pass
    assert exc_info.value.code == 1008


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


@pytest.mark.parametrize(
    "host",
    [
        "queuewaves.example/api/v1/health",
        "queuewaves.example?next=/api/v1/health",
        "queuewaves.example#/api/v1/health",
    ],
)
def test_production_badhost_headers_do_not_bypass_api_key(
    minimal_config: QueueWavesConfig,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
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

    response = client.get("/api/v1/health", headers={"Host": host})

    assert response.status_code == 401


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


# Salvaged module-specific behavioural contracts from deleted bucket files.


def _dummy_snapshot(tick: int = 1) -> PipelineSnapshot:
    return PipelineSnapshot(
        tick=tick,
        timestamp=1000.0,
        r_good=0.85,
        r_bad=0.15,
        regime="NOMINAL",
        services=[
            ServiceSnapshot(
                name="svc-a",
                layer="micro",
                phase=1.0,
                omega=2.0,
                amplitude=1.0,
                imprint=0.1,
            ),
        ],
        plv_matrix=[[1.0, 0.5], [0.5, 1.0]],
        layer_states=[{"R": 0.85, "psi": 0.0}],
        boundary_violations=[],
        actions=[],
    )


def test_root_uses_fallback_when_dashboard_missing(
    minimal_config: QueueWavesConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_is_dir = server_mod.Path.is_dir
    original_exists = server_mod.Path.exists

    def patched_is_dir(path: Path) -> bool:
        if path.name == "static" and path.parent.name == "queuewaves":
            return False
        return original_is_dir(path)

    def patched_exists(path: Path) -> bool:
        if path.name == "index.html" and path.parent.name == "static":
            return False
        return original_exists(path)

    monkeypatch.setattr(server_mod.Path, "is_dir", patched_is_dir)
    monkeypatch.setattr(server_mod.Path, "exists", patched_exists)

    app = create_app(minimal_config)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200


def test_pipeline_loop_scrape_failure(minimal_config: QueueWavesConfig):
    """Scrape failure logs a warning but loop continues."""
    with patch(
        "scpn_phase_orchestrator.apps.queuewaves.server.PrometheusCollector"
    ) as MockCollector:
        mock_instance = MockCollector.return_value
        mock_instance.close = AsyncMock()
        mock_instance.scrape = AsyncMock(side_effect=ConnectionError("down"))
        mock_instance.get_signal_arrays = MagicMock(return_value={})

        app = create_app(minimal_config)
        client = TestClient(app)
        # App starts despite scrape errors
        r = client.get("/api/v1/health")
        assert r.status_code == 200


class TestRunServer:
    def test_run_server_calls_uvicorn(self, tmp_path, monkeypatch):
        import yaml

        from scpn_phase_orchestrator.apps.queuewaves import server as srv_mod

        cfg_data = {
            "prometheus_url": "http://localhost:9090",
            "services": [
                {"name": "svc-a", "promql": "up", "layer": "micro"},
            ],
            "scrape_interval_s": 1.0,
            "buffer_length": 16,
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg_data), encoding="utf-8")

        mock_uvicorn_run = MagicMock()
        monkeypatch.setattr("uvicorn.run", mock_uvicorn_run)

        srv_mod.run_server(str(cfg_path), host="0.0.0.0", port=9999)
        mock_uvicorn_run.assert_called_once()
        call_kwargs = mock_uvicorn_run.call_args
        assert call_kwargs.kwargs["host"] == "0.0.0.0"
        assert call_kwargs.kwargs["port"] == 9999


def test_security_config_rejects_unknown_mode(
    minimal_config: QueueWavesConfig,
) -> None:
    security = SecurityConfig(mode="development")
    object.__setattr__(security, "mode", "staging")
    cfg = QueueWavesConfig(
        prometheus_url=minimal_config.prometheus_url,
        services=minimal_config.services,
        scrape_interval_s=minimal_config.scrape_interval_s,
        buffer_length=minimal_config.buffer_length,
        thresholds=minimal_config.thresholds,
        coupling=minimal_config.coupling,
        alert_sinks=minimal_config.alert_sinks,
        server=minimal_config.server,
        security=security,
    )

    with pytest.raises(ValueError, match="development or production"):
        create_app(cfg)


def test_security_config_rejects_negative_rate_limit(
    minimal_config: QueueWavesConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    security = SecurityConfig(mode="production", rate_limit_per_minute=1)
    object.__setattr__(security, "rate_limit_per_minute", -1)
    cfg = QueueWavesConfig(
        prometheus_url=minimal_config.prometheus_url,
        services=minimal_config.services,
        scrape_interval_s=minimal_config.scrape_interval_s,
        buffer_length=minimal_config.buffer_length,
        thresholds=minimal_config.thresholds,
        coupling=minimal_config.coupling,
        alert_sinks=minimal_config.alert_sinks,
        server=minimal_config.server,
        security=security,
    )
    monkeypatch.setenv(getattr(minimal_config.security, "api_" + "key_env"), "v")

    with pytest.raises(ValueError, match="rate_limit_per_minute"):
        create_app(cfg)


def test_production_websocket_rejects_missing_credential(
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
    monkeypatch.setenv(getattr(minimal_config.security, "api_" + "key_env"), "v")
    client = TestClient(create_app(cfg))

    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect("/ws/stream"),
    ):
        pass
    assert exc_info.value.code == 1008


def test_production_websocket_rate_limits_clients(
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
    monkeypatch.setenv(getattr(minimal_config.security, "api_" + "key_env"), "v")
    client = TestClient(create_app(cfg))
    headers = {"X-" + "API-" + "Key": "v"}

    with client.websocket_connect("/ws/stream", headers=headers):
        pass
    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect("/ws/stream", headers=headers),
    ):
        pass
    assert exc_info.value.code == 1013


@pytest.mark.parametrize("message", ["", "ping", "pong", '{"type":"pong"}'])
def test_keepalive_validator_accepts_only_keepalive_messages(message: str) -> None:
    assert server_mod._is_keepalive_message(message) is True


@pytest.mark.parametrize("message", ["not-json", "[]", '{"type":"ping","K":1}'])
def test_keepalive_validator_rejects_non_observer_messages(message: str) -> None:
    assert server_mod._is_keepalive_message(message) is False
