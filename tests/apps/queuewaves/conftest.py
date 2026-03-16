# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves test fixtures

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.apps.queuewaves.config import (
    CouplingConfig,
    QueueWavesConfig,
    ServerConfig,
    ServiceDef,
    ThresholdConfig,
)


@pytest.fixture()
def minimal_services() -> list[ServiceDef]:
    return [
        ServiceDef(name="svc-a", promql='rate(http_total{s="a"}[1m])', layer="micro"),
        ServiceDef(name="svc-b", promql='rate(http_total{s="b"}[1m])', layer="micro"),
        ServiceDef(name="throughput", promql="rate(http_total[5m])", layer="macro"),
    ]


@pytest.fixture()
def minimal_config(minimal_services: list[ServiceDef]) -> QueueWavesConfig:
    return QueueWavesConfig(
        prometheus_url="http://localhost:9090",
        services=minimal_services,
        scrape_interval_s=1.0,
        buffer_length=16,
        thresholds=ThresholdConfig(),
        coupling=CouplingConfig(),
        alert_sinks=[],
        server=ServerConfig(port=0),
    )
