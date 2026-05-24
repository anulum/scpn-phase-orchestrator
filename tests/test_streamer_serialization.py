# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Visualizer streamer serialization contracts

"""Serialization and lifecycle contracts for VisualizerStreamer websocket broadcasts."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from scpn_phase_orchestrator.visualization import streamer


def test_visualizer_streamer_serializes_numpy_and_broadcasts_to_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[str] = []

    class Client:
        async def send(self, message: str) -> None:
            sent.append(message)

    class Future:
        pass

    loop = asyncio.new_event_loop()

    def fake_submit(coro: object, loop_arg: object) -> Future:
        assert loop_arg is loop
        asyncio.run(coro)
        return Future()

    monkeypatch.setattr(streamer.asyncio, "run_coroutine_threadsafe", fake_submit)
    visualizer = streamer.VisualizerStreamer()
    visualizer._loop = loop
    visualizer._clients.add(Client())

    visualizer.broadcast(
        {
            "phase": np.array([0.1, 0.2], dtype=np.float64),
            "count": np.int64(3),
            "nested": [np.float32(0.5), {"matrix": np.eye(2, dtype=np.float64)}],
        }
    )

    assert len(sent) == 1
    assert sent[0] == (
        '{"phase": [0.1, 0.2], "count": 3, '
        '"nested": [0.5, {"matrix": [[1.0, 0.0], [0.0, 1.0]]}]}'
    )
    loop.close()


def test_visualizer_streamer_lifecycle_error_and_handler_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streamer, "HAS_WEBSOCKETS", False)
    with pytest.raises(RuntimeError, match="websockets required"):
        streamer.VisualizerStreamer().start()

    class WebSocket:
        def __init__(self) -> None:
            self.closed = False

        async def wait_closed(self) -> None:
            self.closed = True

    async def exercise_handler() -> None:
        visualizer = streamer.VisualizerStreamer()
        ws = WebSocket()
        await visualizer._handler(ws)
