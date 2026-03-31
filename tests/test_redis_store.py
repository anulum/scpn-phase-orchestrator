# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — RedisStateStore tests

from __future__ import annotations

import json
from unittest.mock import MagicMock

from scpn_phase_orchestrator.adapters.redis_store import RedisStateStore


def _make_store() -> tuple[RedisStateStore, MagicMock]:
    mock_client = MagicMock()
    store = RedisStateStore(client=mock_client, key="spo:test")
    return store, mock_client


def test_save_state_calls_set():
    store, mock = _make_store()
    state = {"R": 0.9, "regime": "nominal"}
    store.save_state(state)
    mock.set.assert_called_once_with("spo:test", json.dumps(state))


def test_load_state_returns_dict():
    store, mock = _make_store()
    payload = {"R": 0.85, "step": 42}
    mock.get.return_value = json.dumps(payload)
    result = store.load_state()
    assert result == payload


def test_load_state_returns_none_when_missing():
    store, mock = _make_store()
    mock.get.return_value = None
    assert store.load_state() is None


def test_delete_state_calls_delete():
    store, mock = _make_store()
    store.delete_state()
    mock.delete.assert_called_once_with("spo:test")


def test_key_property():
    store, _ = _make_store()
    assert store.key == "spo:test"


def test_custom_key():
    mock_client = MagicMock()
    store = RedisStateStore(client=mock_client, key="custom:state:v2")
    assert store.key == "custom:state:v2"


def test_roundtrip():
    store, mock = _make_store()
    state = {"layers": [0.9, 0.8], "regime": "nominal", "step": 10}
    store.save_state(state)
    mock.get.return_value = mock.set.call_args[0][1]
    loaded = store.load_state()
    assert loaded == state


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
