# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — QueueWaves FastAPI server

import asyncio
import contextlib
import json
import logging
import os
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

try:
    from httpx import HTTPError as _HTTPError
except ImportError:  # pragma: no cover
    # type ignore: optional httpx dependency falls back to a base I/O error.
    _HTTPError = OSError  # type: ignore[assignment,misc]

from scpn_phase_orchestrator.apps.queuewaves.alerter import WebhookAlerter
from scpn_phase_orchestrator.apps.queuewaves.collector import PrometheusCollector
from scpn_phase_orchestrator.apps.queuewaves.config import QueueWavesConfig, load_config
from scpn_phase_orchestrator.apps.queuewaves.detector import AnomalyDetector
from scpn_phase_orchestrator.apps.queuewaves.pipeline import (
    PhaseComputePipeline,
    PipelineSnapshot,
)
from scpn_phase_orchestrator.network_security import FixedWindowRateLimiter

_IO_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    RuntimeError,
    OSError,
    _HTTPError,
)

__all__ = ["create_app"]

logger = logging.getLogger(__name__)

_MAX_HISTORY = 500


def create_app(cfg: QueueWavesConfig) -> object:
    """Build a FastAPI application wired to the given config."""
    from fastapi import (
        Depends,
        FastAPI,
        Header,
        HTTPException,
        Request,
        WebSocket,
        WebSocketDisconnect,
    )
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
    mode = cfg.security.mode.strip().lower()
    if mode not in ("development", "production"):
        raise ValueError("QueueWaves security.mode must be development or production")
    api_key = os.environ.get(cfg.security.api_key_env)
    if mode == "production" and not api_key:
        raise RuntimeError(
            f"{cfg.security.api_key_env} is required when QueueWaves runs in production"
        )
    rate_limit = cfg.security.rate_limit_per_minute if mode == "production" else 0
    if rate_limit < 0:
        raise ValueError("QueueWaves rate_limit_per_minute must be non-negative")
    limiter = FixedWindowRateLimiter(rate_limit) if rate_limit > 0 else None

    async def _require_network_access(
        request: Request,
        x_api_key: str | None = Header(None),
    ) -> None:
        if api_key is None:
            identity = request.client.host if request.client is not None else "local"
        elif x_api_key != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
        else:
            identity = x_api_key
        if limiter is not None and not limiter.allow(identity):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    async def _broadcast(msg: dict[str, Any]) -> None:  # pragma: no cover
        payload = json.dumps(msg)
        dead: list[WebSocket] = []
        for ws in ws_clients:
            try:
                await ws.send_text(payload)
            except _IO_ERRORS:
                dead.append(ws)
        for ws in dead:
            ws_clients.discard(ws)

    async def _pipeline_loop() -> None:  # pragma: no cover
        nonlocal active_anomalies
        while True:
            try:
                await collector.scrape()
            except _IO_ERRORS:
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
                except _IO_ERRORS:
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

    @app.get("/api/v1/health", dependencies=[Depends(_require_network_access)])
    async def health() -> dict[str, Any]:
        """Handle GET /api/v1/health — liveness check with tick counter."""
        return {"status": "ok", "tick": pipeline.tick_count}

    @app.get("/api/v1/state", dependencies=[Depends(_require_network_access)])
    async def state() -> Any:
        """Handle GET /api/v1/state — return latest pipeline snapshot."""
        if not history:
            return JSONResponse({"error": "no data yet"}, status_code=503)
        return history[-1].to_dict()  # pragma: no cover — data branch, ASGI thread

    @app.get("/api/v1/state/history", dependencies=[Depends(_require_network_access)])
    async def state_history(n: int = 100) -> list[dict[str, Any]]:
        """Handle GET /api/v1/state/history — return last n snapshots."""
        sliced = list(history)[-n:]
        return [s.to_dict() for s in sliced]

    @app.get("/api/v1/anomalies", dependencies=[Depends(_require_network_access)])
    async def anomalies() -> list[dict[str, Any]]:
        """Handle GET /api/v1/anomalies — return active anomaly list."""
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

    @app.get("/api/v1/services", dependencies=[Depends(_require_network_access)])
    async def services() -> list[dict[str, Any]]:
        """Handle GET /api/v1/services — return per-service phase state."""
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

    @app.get("/api/v1/plv", dependencies=[Depends(_require_network_access)])
    async def plv() -> dict[str, Any]:
        """Handle GET /api/v1/plv — return cross-layer PLV matrix."""
        if not history:
            return {"matrix": []}
        return {"matrix": history[-1].plv_matrix}  # pragma: no cover

    @app.get(
        "/api/v1/metrics/prometheus", dependencies=[Depends(_require_network_access)]
    )
    async def prom_metrics() -> PlainTextResponse:
        """Handle GET /api/v1/metrics/prometheus — export Prometheus text metrics."""
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

    @app.post("/api/v1/check", dependencies=[Depends(_require_network_access)])
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
        """Handle GET / — serve dashboard index or fallback text."""
        index = static_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        msg = "QueueWaves is running. No dashboard found."  # pragma: no cover
        return PlainTextResponse(msg)  # pragma: no cover

    # --- WebSocket ---

    @app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket) -> None:
        """Read-only observer stream. Incoming messages are ignored (keepalive only)."""
        if api_key is not None and websocket.headers.get("x-api-key") != api_key:
            await websocket.close(code=1008, reason="Invalid or missing X-API-Key")
            return
        if limiter is not None:
            identity = websocket.headers.get("x-api-key") or "websocket"
            if not limiter.allow(identity):
                await websocket.close(code=1013, reason="Rate limit exceeded")
                return
        await websocket.accept()
        ws_clients.add(websocket)
        try:
            while True:
                msg = await websocket.receive_text()
                if len(msg) > 1024:
                    await websocket.close(code=1009, reason="Message too large")
                    break
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
    # type ignore: uvicorn's app parameter typing is narrower than FastAPI instances.
    uvicorn.run(app, host=host, port=port, log_level="info")  # type: ignore[arg-type]
