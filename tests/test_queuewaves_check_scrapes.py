# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — queuewaves check analyses real Prometheus data

"""``spo queuewaves check`` must analyse the services, not synthetic noise.

The command documented as "scrapes once, runs the pipeline … use in CI or
cron" never contacted Prometheus: it fed seeded random noise to every
service and reported "No anomalies detected." with exit 0, even when the
configured Prometheus did not exist.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml
from click.testing import CliRunner

from scpn_phase_orchestrator.apps.queuewaves.collector import PrometheusCollector
from scpn_phase_orchestrator.runtime.cli import main
from tests.prometheus_range_server import prometheus_range_server

_PERMISSIVE = {
    "r_bad_warn": 99.0,
    "r_bad_critical": 99.0,
    "plv_cascade": 99.0,
    "imprint_chronic": 99.0,
}


def _config(tmp_path: Path, url: str, **extra: object) -> Path:
    cfg = {
        "prometheus_url": url,
        "services": [
            {"name": "svc-a", "promql": "rate_a", "layer": "micro"},
            {"name": "svc-b", "promql": "rate_b", "layer": "macro"},
        ],
        "scrape_interval_s": 1.0,
        "buffer_length": 16,
        **extra,
    }
    path = tmp_path / "qw.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


def _check(path: Path) -> tuple[int, str]:
    result = CliRunner().invoke(main, ["queuewaves", "check", "--config", str(path)])
    return result.exit_code, result.output


def test_check_analyses_data_served_by_prometheus(tmp_path: Path) -> None:
    with prometheus_range_server() as url:
        code, output = _check(_config(tmp_path, url, thresholds=_PERMISSIVE))
    assert code == 0, output
    assert "R_good=" in output
    assert "No anomalies detected." in output


def test_unreachable_prometheus_is_unknown_not_healthy(tmp_path: Path) -> None:
    with prometheus_range_server() as url:
        pass  # server is shut down: the port now refuses connections
    code, output = _check(_config(tmp_path, url, thresholds=_PERMISSIVE))
    assert code == 2
    assert "UNKNOWN: not enough Prometheus data" in output
    assert "No anomalies detected" not in output


def test_service_without_data_makes_the_check_unknown(tmp_path: Path) -> None:
    with prometheus_range_server(empty_queries=frozenset({"rate_b"})) as url:
        code, output = _check(_config(tmp_path, url, thresholds=_PERMISSIVE))
    assert code == 2
    assert "svc-b" in output
    assert "No anomalies detected" not in output


def test_backfill_fills_each_buffer_with_the_requested_history() -> None:
    with prometheus_range_server() as url:
        collector = PrometheusCollector(url, {"a": "rate_a", "b": "rate_b"}, 16)

        async def run() -> None:
            try:
                await collector.backfill(end=1_000_000.0, samples=16, step_s=1.0)
            finally:
                await collector.close()

        asyncio.run(run())
    arrays = collector.get_signal_arrays()
    assert sorted(arrays) == ["a", "b"]
    assert all(len(values) == 16 for values in arrays.values())
