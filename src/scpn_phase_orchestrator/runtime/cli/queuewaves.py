# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI queuewaves serving commands

"""Command-line entry point for validation, replay, export, and review workflows.

The CLI wraps public SPO APIs behind explicit commands for binding validation,
inspection, auto-binding proposals, coupling estimation, formal export, replay,
plugin catalogs, scaffolding, and selected runtime utilities. Commands validate
local inputs and emit text or JSON review artifacts; they do not push commits,
start network services, or perform live actuation unless an explicit subcommand
is invoked for that runtime path.
"""

from __future__ import annotations

import click

from scpn_phase_orchestrator.runtime.cli._app import (
    main,
)


@main.group()
def queuewaves() -> None:
    """QueueWaves — real-time cascade failure detector."""


main.add_command(queuewaves)


@queuewaves.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8080, type=int)
def serve(config_path: str, host: str, port: int) -> None:
    """Start QueueWaves server.

    Parameters
    ----------
    config_path : str
        Filesystem path to the configuration file.
    host : str
        Host interface to bind.
    port : int
        Port to bind.
    """
    from scpn_phase_orchestrator.apps.queuewaves.server import run_server

    run_server(config_path, host=host, port=port)


@queuewaves.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def check(config_path: str) -> None:
    """One-shot: query Prometheus, analyse, exit 0 (ok), 1 (anomalies), 2 (no data).

    Each service's recent history (``buffer_length`` points spaced
    ``scrape_interval_s`` apart) is fetched with one PromQL range query and run
    through the phase pipeline. A service with too little data makes the
    result unknown: the command exits 2 and never reports "No anomalies".

    Parameters
    ----------
    config_path : str
        Filesystem path to the configuration file.

    Raises
    ------
    SystemExit
        With code 1 when anomalies are detected, 2 when a service returned too
        little data to analyse.
    ClickException
        If the optional ``httpx`` dependency is not installed.
    """
    import asyncio
    import time
    from pathlib import Path as _Path

    from scpn_phase_orchestrator.apps.queuewaves.collector import PrometheusCollector
    from scpn_phase_orchestrator.apps.queuewaves.config import load_config
    from scpn_phase_orchestrator.apps.queuewaves.detector import AnomalyDetector
    from scpn_phase_orchestrator.apps.queuewaves.pipeline import PhaseComputePipeline

    cfg = load_config(_Path(config_path))
    collector = PrometheusCollector(
        cfg.prometheus_url,
        {svc.name: svc.promql for svc in cfg.services},
        cfg.buffer_length,
    )

    async def collect() -> None:
        try:
            await collector.backfill(
                end=time.time(),
                samples=cfg.buffer_length,
                step_s=cfg.scrape_interval_s,
            )
        finally:
            await collector.close()

    try:
        asyncio.run(collect())
    except ImportError as exc:
        raise click.ClickException(
            "queuewaves check needs httpx: pip install scpn-phase-orchestrator"
            "[queuewaves]"
        ) from exc
    signals = collector.get_signal_arrays()
    missing = [svc.name for svc in cfg.services if svc.name not in signals]
    if missing:
        click.echo(
            "UNKNOWN: not enough Prometheus data for "
            f"{', '.join(missing)} at {cfg.prometheus_url}; nothing was analysed",
            err=True,
        )
        raise SystemExit(2)
    pipeline = PhaseComputePipeline(cfg)
    snap = pipeline.tick(signals)
    detector = AnomalyDetector(cfg.thresholds)
    anomalies = detector.detect(snap)
    click.echo(
        f"R_good={snap.r_good:.4f}  R_bad={snap.r_bad:.4f}  regime={snap.regime}"
    )
    if anomalies:
        for a in anomalies:
            click.echo(f"  [{a.severity}] {a.message}")
        raise SystemExit(1)
    click.echo("No anomalies detected.")
