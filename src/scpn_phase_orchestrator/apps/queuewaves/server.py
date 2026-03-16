# SCPN Phase Orchestrator — QueueWaves FastAPI server
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from scpn_phase_orchestrator.apps.queuewaves.alerter import WebhookAlerter
from scpn_phase_orchestrator.apps.queuewaves.collector import PrometheusCollector
from scpn_phase_orchestrator.apps.queuewaves.config import QueueWavesConfig, load_config
from scpn_phase_orchestrator.apps.queuewaves.detector import AnomalyDetector
from scpn_phase_orchestrator.apps.queuewaves.pipeline import (
    PhaseComputePipeline,
    PipelineSnapshot,
)

__all__ = ["create_app"]

logger = logging.getLogger(__name__)

_MAX_HISTORY = 500


def create_app(cfg: QueueWavesConfig) -> Any:
    """Build a FastAPI application wired to the given config."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: F811
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles

    queries = {svc.name: svc.promql for svc in cfg.services}
    collector = PrometheusCollector(cfg.prometheus_url, queries, cfg.buffer_length)
    pipeline = PhaseComputePipeline(cfg)
    detector = AnomalyDetector(cfg.thresholds)
    alerter = WebhookAlerter(cfg.alert_sinks, cfg.thresholds.cooldown_seconds)

    history: deque[PipelineSnapshot] = deque(maxlen=_MAX_HISTORY)
    active_anomalies: list[Any] = []
    ws_clients: set[WebSocket] = set()

    async def _broadcast(msg: dict[str, Any]) -> None:  # pragma: no cover
        payload = json.dumps(msg)
        dead: list[WebSocket] = []
        for ws in ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients.discard(ws)

    async def _pipeline_loop() -> None:  # pragma: no cover
        nonlocal active_anomalies
        while True:
            try:
                await collector.scrape()
            except Exception:
                logger.warning("scrape cycle failed", exc_info=True)

            signals = collector.get_signal_arrays()
            if not signals:
                await asyncio.sleep(cfg.scrape_interval_s)
                continue

            snap = pipeline.tick(signals)
            history.append(snap)

            anomalies = detector.detect(snap)
            active_anomalies = anomalies

            if anomalies:
                try:
                    await alerter.send(anomalies)
                except Exception:
                    logger.warning("alert send failed", exc_info=True)

            tick_msg = {"type": "tick", "data": snap.to_dict()}
            await _broadcast(tick_msg)

            for a in anomalies:
                await _broadcast(
                    {
                        "type": "anomaly",
                        "data": {
                            "type": a.type,
                            "severity": a.severity,
                            "service": a.service,
                            "message": a.message,
                            "value": a.value,
                            "tick": a.tick,
                        },
                    }
                )

            await asyncio.sleep(cfg.scrape_interval_s)

    @asynccontextmanager
    async def _lifespan(  # pragma: no cover
        _app: FastAPI,
    ) -> AsyncIterator[None]:
        task = asyncio.create_task(_pipeline_loop())
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await collector.close()

    app = FastAPI(title="QueueWaves", version="0.1.0", lifespan=_lifespan)

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(static_dir), html=True),
            name="static",
        )

    # --- REST endpoints ---

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "tick": pipeline.tick_count}

    @app.get("/api/v1/state")
    async def state() -> Any:
        if not history:
            return JSONResponse({"error": "no data yet"}, status_code=503)
        return history[-1].to_dict()  # pragma: no cover — data branch, ASGI thread

    @app.get("/api/v1/state/history")
    async def state_history(n: int = 100) -> list[dict[str, Any]]:
        sliced = list(history)[-n:]
        return [s.to_dict() for s in sliced]

    @app.get("/api/v1/anomalies")
    async def anomalies() -> list[dict[str, Any]]:
        return [
            {
                "type": a.type,
                "severity": a.severity,
                "service": a.service,
                "value": a.value,
                "message": a.message,
                "tick": a.tick,
            }
            for a in active_anomalies
        ]

    @app.get("/api/v1/services")
    async def services() -> list[dict[str, Any]]:
        if not history:
            return []
        snap = history[-1]  # pragma: no cover — data branch, ASGI thread
        return [  # pragma: no cover
            {
                "name": s.name,
                "layer": s.layer,
                "phase": s.phase,
                "omega": s.omega,
                "imprint": s.imprint,
            }
            for s in snap.services
        ]

    @app.get("/api/v1/plv")
    async def plv() -> dict[str, Any]:
        if not history:
            return {"matrix": []}
        return {"matrix": history[-1].plv_matrix}  # pragma: no cover

    @app.get("/api/v1/metrics/prometheus")
    async def prom_metrics() -> PlainTextResponse:
        lines: list[str] = []
        if history:  # pragma: no cover — data branch, ASGI thread
            snap = history[-1]
            lines.append(f"queuewaves_r_good {snap.r_good:.6f}")
            lines.append(f"queuewaves_r_bad {snap.r_bad:.6f}")
            lines.append(f'queuewaves_regime{{name="{snap.regime}"}} 1')
            lines.append(f"queuewaves_tick {snap.tick}")
            for svc in snap.services:
                lines.append(
                    f'queuewaves_phase{{service="{svc.name}"}} {svc.phase:.6f}'
                )
                lines.append(
                    f'queuewaves_imprint{{service="{svc.name}"}} {svc.imprint:.6f}'
                )
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")

    @app.post("/api/v1/check")
    async def check() -> Any:
        """One-shot: scrape, analyze, return result."""
        await collector.scrape()
        signals = collector.get_signal_arrays()
        if not signals:
            return JSONResponse({"error": "not enough data"}, status_code=503)
        snap = pipeline.tick(signals)
        anoms = detector.detect(snap)
        return {
            "r_good": snap.r_good,
            "r_bad": snap.r_bad,
            "regime": snap.regime,
            "anomalies": [
                {"type": a.type, "severity": a.severity, "message": a.message}
                for a in anoms
            ],
        }

    @app.get("/")
    async def root() -> Any:
        index = static_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        msg = "QueueWaves is running. No dashboard found."
        return PlainTextResponse(msg)  # pragma: no cover

    # --- WebSocket ---

    @app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        ws_clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            ws_clients.discard(websocket)

    return app


def run_server(config_path: str, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Entry point for CLI: load config, create app, run uvicorn."""
    import uvicorn

    cfg = load_config(Path(config_path))
    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port, log_level="info")
