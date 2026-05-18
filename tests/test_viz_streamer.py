# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Visualizer Streamer tests

import numpy as np
import pytest

pytest.importorskip("websockets", reason="websockets not installed")
from scpn_phase_orchestrator.visualization import streamer as streamer_module

VisualizerStreamer = streamer_module.VisualizerStreamer


def test_viz_streamer_initialization():
    streamer = VisualizerStreamer(host="127.0.0.1", port=9999)
    assert streamer.host == "127.0.0.1"
    assert streamer.port == 9999


def test_json_safe_conversion():
    streamer = VisualizerStreamer()
    data = {
        "array": np.array([1.0, 2.0, 3.0]),
        "scalar": np.float64(4.2),
        "int": np.int64(7),
        "nested": {"val": np.array([0.0])},
    }
    safe = streamer._json_safe(data)
    assert isinstance(safe["array"], list)
    assert isinstance(safe["scalar"], float)
    assert isinstance(safe["int"], int)
    assert isinstance(safe["nested"]["val"], list)


def test_default_host_and_port():
    """Default constructor produces a loopback endpoint on 8765."""
    streamer = VisualizerStreamer()
    assert streamer.host == "127.0.0.1"
    assert streamer.port == 8765


@pytest.mark.parametrize("host", ["", "bad\nhost", object()])
def test_constructor_rejects_invalid_host(host):
    """Host validation must fail before an invalid bind reaches a server thread."""
    with pytest.raises(ValueError, match="host"):
        VisualizerStreamer(host=host)


@pytest.mark.parametrize("port", [0, -1, 65536, True, 1.5, "8765"])
def test_constructor_rejects_invalid_port(port):
    """Port validation must reject malformed values before WebSocket binding."""
    with pytest.raises(ValueError, match="port"):
        VisualizerStreamer(port=port)


def test_json_safe_passes_through_primitives():
    """Native Python primitives must round-trip through _json_safe unchanged."""
    streamer = VisualizerStreamer()
    assert streamer._json_safe(42) == 42
    assert streamer._json_safe(3.14) == 3.14
    assert streamer._json_safe("hello") == "hello"
    assert streamer._json_safe(None) is None
    assert streamer._json_safe(True) is True


def test_json_safe_handles_empty_containers():
    """Empty dict/list survive conversion with identical empty shape."""
    streamer = VisualizerStreamer()
    assert streamer._json_safe({}) == {}
    assert streamer._json_safe([]) == []
    assert streamer._json_safe({"x": []}) == {"x": []}


def test_json_safe_handles_multidim_array():
    """A 2-D numpy array flattens into a nested list (row-major)."""
    streamer = VisualizerStreamer()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = streamer._json_safe(arr)
    assert result == [[1.0, 2.0], [3.0, 4.0]]
    assert isinstance(result, list)
    assert isinstance(result[0], list)


def test_json_safe_handles_list_of_arrays():
    """List elements that are numpy arrays are individually converted."""
    streamer = VisualizerStreamer()
    data = [np.array([1, 2]), np.array([3, 4])]
    result = streamer._json_safe(data)
    assert result == [[1, 2], [3, 4]]


def test_json_safe_handles_numpy_scalars_of_every_dtype():
    """float32, float64, int32, int64 all collapse to native Python types."""
    streamer = VisualizerStreamer()
    for dtype, expected_type in (
        (np.float32, float),
        (np.float64, float),
        (np.int32, int),
        (np.int64, int),
    ):
        value = streamer._json_safe(dtype(5))
        assert isinstance(value, expected_type), (
            f"{dtype.__name__} did not collapse to {expected_type.__name__}"
        )


def test_broadcast_without_clients_is_noop():
    """Broadcast with no connected clients must not crash and must skip
    JSON conversion (cheap fast path)."""
    streamer = VisualizerStreamer()
    streamer.broadcast({"R": 0.5, "phases": np.array([0.1, 0.2])})


def test_broadcast_before_start_does_not_raise():
    """Calling broadcast() before start() must be a safe no-op."""
    streamer = VisualizerStreamer()
    assert streamer._loop is None
    assert streamer._clients == set()
    streamer.broadcast({"any": "data"})  # should not throw


def test_start_delegates_to_background_thread():
    """start() must delegate socket ownership to the configured daemon thread."""

    class ThreadProbe:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

    thread = ThreadProbe()
    streamer = VisualizerStreamer()
    streamer._thread = thread

    streamer.start()

    assert thread.started is True


def test_stop_signals_existing_event_loop_threadsafe():
    """stop() must signal the running event loop without touching clients."""

    calls = []

    class LoopProbe:
        def stop(self):
            calls.append("stop")

        def call_soon_threadsafe(self, callback):
            calls.append(callback.__name__)
            callback()

    streamer = VisualizerStreamer()
    streamer._loop = LoopProbe()

    streamer.stop()

    assert calls == ["stop", "stop"]


def test_run_server_builds_websocket_server_without_network(monkeypatch):
    """_run_server wires a new event loop into websockets.serve deterministically."""

    serve_calls = []
    set_loop_calls = []

    class LoopProbe:
        def __init__(self):
            self.run_until_complete_arg = None
            self.forever_called = False

        def run_until_complete(self, awaitable):
            self.run_until_complete_arg = awaitable
            return "server-object"

        def run_forever(self):
            self.forever_called = True

    loop = LoopProbe()

    def fake_set_event_loop(loop_arg):
        set_loop_calls.append(loop_arg)

    class WebsocketsProbe:
        @staticmethod
        def serve(handler, host, port):
            serve_calls.append((handler, host, port))
            return "serve-awaitable"

    monkeypatch.setattr(streamer_module.asyncio, "new_event_loop", lambda: loop)
    monkeypatch.setattr(streamer_module.asyncio, "set_event_loop", fake_set_event_loop)
    monkeypatch.setattr(streamer_module, "websockets", WebsocketsProbe)

    streamer = VisualizerStreamer(host="127.0.0.1", port=9988)

    streamer._run_server()

    assert streamer._loop is loop
    assert set_loop_calls == [loop]
    assert serve_calls == [(streamer._handler, "127.0.0.1", 9988)]
    assert loop.run_until_complete_arg == "serve-awaitable"
    assert streamer._server == "server-object"
    assert loop.forever_called is True


def test_json_safe_deep_nesting():
    """Nested dict/list combinations convert recursively without flattening."""
    streamer = VisualizerStreamer()
    data = {
        "layers": [
            {"R": np.float64(0.5), "psi": np.float64(1.0)},
            {"R": np.float64(0.8), "psi": np.float64(2.0)},
        ],
        "meta": {"step": np.int64(42), "regime": "DEGRADED"},
    }
    safe = streamer._json_safe(data)
    assert safe["layers"][0]["R"] == 0.5
    assert isinstance(safe["layers"][0]["R"], float)
    assert safe["meta"]["step"] == 42
    assert isinstance(safe["meta"]["step"], int)
    assert safe["meta"]["regime"] == "DEGRADED"


# Pipeline wiring: _json_safe is the hot path for every broadcast to the
# WebXR frontend. Unsafe numpy types leak through json.dumps and crash
# the socket, so every numpy dtype on the API boundary must round-trip
# cleanly. The tests above exercise scalars, 1-D, 2-D, empty containers,
# deep nesting and the no-client fast path.
