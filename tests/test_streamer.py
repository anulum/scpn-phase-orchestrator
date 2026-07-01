# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Visualization streamer tests

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scpn_phase_orchestrator.visualization import streamer as streamer_module
from scpn_phase_orchestrator.visualization.streamer import (
    VisualizerStreamer,
    _validate_host,
    _validate_port,
)


def test_validate_host_rejects_empty_and_control_characters() -> None:
    with pytest.raises(ValueError, match="host must be a non-empty string"):
        _validate_host("")
    with pytest.raises(ValueError, match="host must not contain control characters"):
        _validate_host("127.0.0.1\n")


def test_validate_port_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="port must be an integer in \\[1, 65535\\]"):
        _validate_port(0)
    with pytest.raises(ValueError, match="port must be an integer in \\[1, 65535\\]"):
        _validate_port(70000)
    with pytest.raises(ValueError, match="port must be an integer in \\[1, 65535\\]"):
        _validate_port(True)


def test_json_safe_converts_numpy_values_and_tuples() -> None:
    stream = VisualizerStreamer()
    payload = {
        "values": np.array([0.0, 1.0]),
        "shape": (1, 2),
    }
    output = stream._json_safe(payload)
    assert output["values"] == [0.0, 1.0]
    assert output["shape"] == [1, 2]


def test_json_safe_rejects_non_finite_numbers() -> None:
    stream = VisualizerStreamer()
    with pytest.raises(ValueError, match="float values must be finite"):
        stream._json_safe({"value": float("nan")})
    with pytest.raises(ValueError, match="array values must be finite"):
        stream._json_safe(np.array([0.0, float("inf")]))


def test_json_safe_rejects_non_jsonable_objects() -> None:
    stream = VisualizerStreamer()
    with pytest.raises(TypeError, match="Object of type"):
        stream._json_safe({"value": SimpleNamespace(x=1)})


def test_start_requires_websocket_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streamer_module, "HAS_WEBSOCKETS", False)
    monkeypatch.setattr(streamer_module, "websockets", None)

    with pytest.raises(RuntimeError, match="websockets required"):
        VisualizerStreamer().start()


def test_broadcast_is_noop_when_loop_not_running() -> None:
    stream = VisualizerStreamer()
    stream._clients.add(object())
    stream.broadcast({})


def test_send_cleanup_discards_client_when_status_probe_raises() -> None:
    stream = VisualizerStreamer()
    client = object()
    stream._clients.add(client)
    callback = stream._make_send_cleanup_callback(client)

    # SimpleNamespace has no ``cancelled`` attribute, so the probe raises
    # AttributeError and the client is discarded defensively.
    callback(SimpleNamespace())

    assert client not in stream._clients


def test_module_marks_websockets_absent_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = streamer_module.HAS_WEBSOCKETS
    monkeypatch.setitem(sys.modules, "websockets", None)

    reloaded = importlib.reload(streamer_module)
    try:
        assert reloaded.HAS_WEBSOCKETS is False
        assert reloaded.websockets is None
    finally:
        monkeypatch.undo()
        importlib.reload(streamer_module)

    assert streamer_module.HAS_WEBSOCKETS is original
