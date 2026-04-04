# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - WebXR Manifold Streamer

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import numpy as np

try:
    import websockets

    HAS_WEBSOCKETS = True
except ImportError:
    websockets = None  # type: ignore[assignment]
    HAS_WEBSOCKETS = False

__all__ = ["VisualizerStreamer"]


class VisualizerStreamer:
    """Real-time Manifold Streamer for WebXR / Three.js visualization.

    Broadcasts phase states, metric tensors (h_munu), and topological
    metrics via WebSockets at high frequency (60Hz target).
    Enables holographic 3D visualization of the synchronization manifold.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any = None
        self._clients: set[Any] = set()
        self._thread = threading.Thread(target=self._run_server, daemon=True)

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if not HAS_WEBSOCKETS:
            msg = "websockets required: pip install scpn-phase-orchestrator[queuewaves]"
            raise RuntimeError(msg)
        self._thread.start()

    def stop(self) -> None:
        """Stop the server."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    async def _handler(self, websocket: Any) -> None:
        self._clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self._clients.remove(websocket)

    def _run_server(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        start_server = websockets.serve(self._handler, self.host, self.port)
        self._server = self._loop.run_until_complete(start_server)
        self._loop.run_forever()

    def broadcast(self, data: dict[str, Any]) -> None:
        """Broadcast simulation state to all connected visualization clients."""
        if not self._clients or not self._loop:
            return

        # Convert numpy arrays to lists for JSON serialization
        serializable_data = self._json_safe(data)
        message = json.dumps(serializable_data)

        for client in self._clients:
            asyncio.run_coroutine_threadsafe(client.send(message), self._loop)

    def _json_safe(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._json_safe(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._json_safe(v) for v in data]
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, (np.float64, np.float32)):
            return float(data)
        if isinstance(data, (np.int64, np.int32)):
            return int(data)
        return data
