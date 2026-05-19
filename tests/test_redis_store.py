# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — RedisStateStore tests

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

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


def test_save_state_rejects_non_dict_payload():
    store, mock = _make_store()
    with pytest.raises(ValueError, match="sim_state"):
        store.save_state(["not", "a", "dict"])  # type: ignore[arg-type]
    mock.set.assert_not_called()


def test_save_state_rejects_non_json_payload():
    store, mock = _make_store()
    with pytest.raises(ValueError, match="JSON"):
        store.save_state({"bad": object()})
    mock.set.assert_not_called()


@pytest.mark.parametrize("payload", ["[1, 2, 3]", '"string"', "42"])
def test_load_state_rejects_non_object_json_payload(payload: str):
    store, mock = _make_store()
    mock.get.return_value = payload
    with pytest.raises(ValueError, match="Redis payload"):
        store.load_state()


def test_load_state_rejects_malformed_json_payload():
    store, mock = _make_store()
    mock.get.return_value = "{not-json"
    with pytest.raises(ValueError, match="Redis payload"):
        store.load_state()


def test_constructed_client_uses_tls_auth_and_certificates(monkeypatch):
    captured: dict[str, object] = {}
    test_credential = "".join(("test", "-redis-auth-token"))

    class FakeRedis:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.SimpleNamespace(Redis=FakeRedis)
    monkeypatch.setitem(sys.modules, "redis", fake_module)
    monkeypatch.setattr(
        "scpn_phase_orchestrator.adapters.redis_store._redis_mod",
        fake_module,
    )
    monkeypatch.setattr("scpn_phase_orchestrator.adapters.redis_store._HAS_REDIS", True)

    RedisStateStore(
        host="redis.internal",
        port=6380,
        db=2,
        password=test_credential,
        ssl_ca_certs="/etc/redis/ca.pem",
        ssl_certfile="/etc/redis/client.pem",
        ssl_keyfile="/etc/redis/client.key",
    )

    assert captured == {
        "host": "redis.internal",
        "port": 6380,
        "db": 2,
        "ssl": True,
        "password": test_credential,
        "ssl_cert_reqs": "required",
        "ssl_ca_certs": "/etc/redis/ca.pem",
        "ssl_certfile": "/etc/redis/client.pem",
        "ssl_keyfile": "/etc/redis/client.key",
    }


def test_plaintext_redis_is_rejected_for_non_loopback_hosts():
    with pytest.raises(ValueError, match="loopback"):
        RedisStateStore(host="redis.internal", ssl=False)


def test_plaintext_redis_is_allowed_for_explicit_loopback_development_client():
    mock_client = MagicMock()
    store = RedisStateStore(host="127.0.0.1", ssl=False, client=mock_client)
    assert store.key == "spo:sim_state"


class TestRedisStorePipelineWiring:
    """Pipeline: engine state → save → load → verify."""

    def test_engine_state_save_load_roundtrip(self):
        """Save engine-produced state → load → verify preserved."""
        import numpy as np
        import pytest

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, np.zeros((n, n)))
        r, _ = compute_order_parameter(phases)

        state = {"R": float(r), "phases": phases.tolist(), "step": 100}
        store, mock = _make_store()
        store.save_state(state)
        mock.get.return_value = mock.set.call_args[0][1]
        loaded = store.load_state()
        assert loaded["R"] == pytest.approx(r)
        assert loaded["step"] == 100
