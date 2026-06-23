# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — WebXR Manifold Streamer

"""Optional WebSocket streamer for visualization-only manifold broadcasts.

``VisualizerStreamer`` owns its event loop, background thread, client set, and
host/port validation for WebXR or Three.js consumers. Broadcast payloads are
converted into JSON-safe Python values before being sent to connected clients.
The streamer is an outbound visualization adapter only; it does not ingest
commands, mutate simulation state, or provide a control-plane API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import websockets

    HAS_WEBSOCKETS = True
except ImportError:
    # type ignore: optional websockets dependency uses a None sentinel fallback.
    websockets = None  # type: ignore[assignment]
    HAS_WEBSOCKETS = False

__all__ = ["VisualizerStreamer"]
logger = logging.getLogger(__name__)


def _validate_host(host: str) -> str:
    """Return the host as a validated non-empty string, else raise."""
    if not isinstance(host, str) or not host:
        raise ValueError("host must be a non-empty string")
    if any(ord(char) < 32 for char in host):
        raise ValueError("host must not contain control characters")
    return host


def _validate_port(port: int) -> int:
    """Return the port as a validated TCP port integer, else raise."""
    if not isinstance(port, int) or isinstance(port, bool):
        raise ValueError("port must be an integer in [1, 65535]")
    if port < 1 or port > 65535:
        raise ValueError("port must be an integer in [1, 65535]")
    return port


class VisualizerStreamer:
    """Real-time Manifold Streamer for WebXR / Three.js visualization.

    Broadcasts phase states, metric tensors (h_munu), and topological
    metrics via WebSockets at high frequency (60Hz target).
    Enables holographic 3D visualization of the synchronization manifold.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = _validate_host(host)
        self.port = _validate_port(port)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any = None
        self._clients: set[Any] = set()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the WebSocket server in a background thread.

        Raises
        ------
        RuntimeError
            If the operation fails.
        """
        if not HAS_WEBSOCKETS:
            msg = "websockets required: pip install scpn-phase-orchestrator[queuewaves]"
            raise RuntimeError(msg)
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the server."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    async def _handler(self, websocket: Any) -> None:
        """Handle one WebSocket client connection for manifold broadcasts."""
        self._clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)

    def _run_server(self) -> None:
        """Run the WebSocket broadcast server until cancelled."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        start_server = websockets.serve(self._handler, self.host, self.port)
        self._server = self._loop.run_until_complete(start_server)
        self._loop.run_forever()

    def broadcast(self, data: dict[str, Any]) -> None:
        """Broadcast simulation state to all connected visualization clients.

        Parameters
        ----------
        data : dict[str, Any]
            Arbitrary JSON-safe payload.
        """
        if not self._clients or not self._loop:
            return

        # Convert numpy arrays to lists for JSON serialization.
        try:
            serializable_data = self._json_safe(data)
            message = json.dumps(serializable_data)
        except (TypeError, ValueError):
            return

        for client in tuple(self._clients):
            send_coro = client.send(message)
            try:
                send_task = asyncio.run_coroutine_threadsafe(send_coro, self._loop)
            except (RuntimeError, TypeError) as exc:
                logger.warning(
                    "visualizer.broadcast_submission_failed: %s",
                    type(exc).__name__,
                )
                if hasattr(send_coro, "close"):
                    send_coro.close()
                self._clients.discard(client)
                continue
            add_done_callback = getattr(send_task, "add_done_callback", None)
            if callable(add_done_callback):
                add_done_callback(self._make_send_cleanup_callback(client))

    def _make_send_cleanup_callback(self, client: Any) -> Callable[[Any], None]:
        """Return a callback that cleans up a completed send task."""

        def _cleanup_send_result(send_result: Any) -> None:
            """Discard a completed send task and surface any error."""
            try:
                cancelled = bool(send_result.cancelled())
            except (AttributeError, RuntimeError, TypeError):
                self._clients.discard(client)
                return
            if cancelled:
                self._clients.discard(client)
                return
            try:
                failed = send_result.exception() is not None
            except (RuntimeError, TypeError):
                self._clients.discard(client)
                return
            if failed:
                self._clients.discard(client)

        return _cleanup_send_result

    def _json_safe(self, data: Any) -> Any:
        """Return ``value`` as a JSON-safe payload, else raise."""
        if isinstance(data, dict):
            return {k: self._json_safe(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._json_safe(v) for v in data]
        if isinstance(data, tuple):
            return [self._json_safe(v) for v in data]
        if isinstance(data, np.ndarray):
            if np.issubdtype(data.dtype, np.floating) and not np.all(np.isfinite(data)):
                raise ValueError("array values must be finite")
            return data.tolist()
        if isinstance(data, np.generic):
            return self._json_safe(data.item())
        if isinstance(data, float) and not math.isfinite(data):
            raise ValueError("float values must be finite")
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data
        raise TypeError(f"Object of type {type(data)!r} is not JSON serializable")
