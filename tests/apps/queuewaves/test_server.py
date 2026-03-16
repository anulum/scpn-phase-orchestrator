# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves server tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.apps.queuewaves.config import QueueWavesConfig

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from scpn_phase_orchestrator.apps.queuewaves.server import create_app


@pytest.fixture()
def client(minimal_config: QueueWavesConfig) -> TestClient:
    app = create_app(minimal_config)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_state_503_before_data(client: TestClient) -> None:
    r = client.get("/api/v1/state")
    assert r.status_code == 503


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


def test_root_returns_html_or_text(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200


def test_check_503_without_data(client: TestClient) -> None:
    r = client.post("/api/v1/check")
    assert r.status_code == 503


def test_state_history_empty(client: TestClient) -> None:
    r = client.get("/api/v1/state/history?n=10")
    assert r.status_code == 200
    assert r.json() == []
