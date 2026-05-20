# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves server coverage tests

"""Coverage tests for QueueWaves server — exercises routes, WebSocket, and
run_server lifecycle."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from scpn_phase_orchestrator.apps.queuewaves.config import (
    QueueWavesConfig,
    SecurityConfig,
)
from scpn_phase_orchestrator.apps.queuewaves.pipeline import (
    PipelineSnapshot,
    ServiceSnapshot,
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from scpn_phase_orchestrator.apps.queuewaves import server as server_mod
from scpn_phase_orchestrator.apps.queuewaves.server import create_app


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


class TestServerNoData:
    def test_state_returns_503_without_data(self, minimal_config: QueueWavesConfig):
        app = create_app(minimal_config)
        client = TestClient(app)
        r = client.get("/api/v1/state")
        assert r.status_code == 503

    def test_prom_metrics_empty(self, minimal_config: QueueWavesConfig):
        app = create_app(minimal_config)
        client = TestClient(app)
        r = client.get("/api/v1/metrics/prometheus")
        assert r.status_code == 200

    def test_root_without_static(self, minimal_config: QueueWavesConfig):
        app = create_app(minimal_config)
        client = TestClient(app)
        r = client.get("/")
        assert r.status_code == 200

    def test_root_uses_fallback_when_dashboard_missing(
        self, minimal_config: QueueWavesConfig, monkeypatch: pytest.MonkeyPatch
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
        assert r.text == "QueueWaves is running. No dashboard found."

    def test_websocket_connect_disconnect(self, minimal_config: QueueWavesConfig):
        app = create_app(minimal_config)
        client = TestClient(app)
        with client.websocket_connect("/ws/stream"):
            pass

    def test_check_no_data(self, minimal_config: QueueWavesConfig):
        app = create_app(minimal_config)
        client = TestClient(app)
        r = client.post("/api/v1/check")
        assert r.status_code == 503

    def test_services_empty_before_data(self, minimal_config: QueueWavesConfig):
        app = create_app(minimal_config)
        client = TestClient(app)
        r = client.get("/api/v1/services")
        assert r.status_code == 200
        assert r.json() == []

    def test_plv_empty(self, minimal_config: QueueWavesConfig):
        app = create_app(minimal_config)
        client = TestClient(app)
        r = client.get("/api/v1/plv")
        assert r.status_code == 200
        assert r.json() == {"matrix": []}

    def test_anomalies_empty(self, minimal_config: QueueWavesConfig):
        app = create_app(minimal_config)
        client = TestClient(app)
        r = client.get("/api/v1/anomalies")
        assert r.status_code == 200
        assert r.json() == []

    def test_health(self, minimal_config: QueueWavesConfig):
        app = create_app(minimal_config)
        client = TestClient(app)
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_state_history_empty(self, minimal_config: QueueWavesConfig):
        app = create_app(minimal_config)
        client = TestClient(app)
        r = client.get("/api/v1/state/history")
        assert r.status_code == 200
        assert r.json() == []


class TestCheckWithData:
    def test_check_with_mocked_collector(self, minimal_config: QueueWavesConfig):
        """Mock the collector to return signal data, verify check response."""
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
            client = TestClient(app)
            r = client.post("/api/v1/check")
            assert r.status_code == 200
            data = r.json()
            assert "r_good" in data
            assert "regime" in data
            assert "anomalies" in data


class TestBroadcastAndLoop:
    """Test the _broadcast and _pipeline_loop code paths via the server's
    lifespan and WebSocket."""

    def test_pipeline_loop_scrape_failure(self, minimal_config: QueueWavesConfig):
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

    def test_websocket_rejects_missing_api_key_in_production(
        self, minimal_config: QueueWavesConfig, monkeypatch: pytest.MonkeyPatch
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
        app = create_app(cfg)
        client = TestClient(app)

        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect("/ws/stream"),
        ):
            pass
        assert exc_info.value.code == 1008

    def test_websocket_rate_limit_enforced_when_exceeded(
        self, minimal_config: QueueWavesConfig, monkeypatch: pytest.MonkeyPatch
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
        app = create_app(cfg)
        client = TestClient(app)
        headers = {"x-api-key": "test-key"}

        with (
            client.websocket_connect("/ws/stream", headers=headers),
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect("/ws/stream", headers=headers),
        ):
            pass
        assert exc_info.value.code == 1013

    def test_create_app_rejects_unknown_security_mode(
        self, minimal_config: QueueWavesConfig
    ) -> None:
        object.__setattr__(minimal_config.security, "mode", "staging")
        with pytest.raises(ValueError, match="QueueWaves security.mode"):
            create_app(minimal_config)


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
