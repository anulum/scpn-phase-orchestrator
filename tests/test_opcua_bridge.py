# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — OPC-UA SCADA tag bridge tests

"""Tests for the OPC-UA phase bridge.

The configuration and phase-extraction surface is exercised without any server
or ``asyncua`` dependency; the live read path is exercised against an in-process
``asyncua`` server when the optional dependency is installed.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters import opcua_bridge
from scpn_phase_orchestrator.adapters.opcua_bridge import (
    HAS_ASYNCUA,
    OpcUaBridgeConfig,
    OpcUaPhaseBridge,
    OpcUaTag,
)

ENDPOINT = "opc.tcp://127.0.0.1:48499/spo"


# ---------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------


def test_tag_round_trips_audit_record() -> None:
    tag = OpcUaTag(node_id="ns=2;i=4", name="temp", scale=2.0, offset=1.0)
    record = tag.to_audit_record()
    assert record["node_id"] == "ns=2;i=4"
    assert record["channel"] == "P"
    assert json.loads(json.dumps(record)) == record


def test_tag_rejects_empty_node_id() -> None:
    with pytest.raises(ValueError, match="node_id"):
        OpcUaTag(node_id="  ", name="temp")


def test_tag_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        OpcUaTag(node_id="ns=2;i=4", name="")


def test_tag_rejects_unknown_channel() -> None:
    with pytest.raises(ValueError, match="channel"):
        OpcUaTag(node_id="ns=2;i=4", name="temp", channel="Z")


@pytest.mark.parametrize("scale", [np.inf, np.nan, True])
def test_tag_rejects_non_finite_scale(scale: object) -> None:
    with pytest.raises(ValueError, match="scale"):
        OpcUaTag(node_id="ns=2;i=4", name="temp", scale=scale)  # type: ignore[arg-type]


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_tag_rejects_non_positive_sample_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        OpcUaTag(node_id="ns=2;i=4", name="temp", sample_rate_hz=rate)


def test_config_round_trips_audit_record() -> None:
    config = OpcUaBridgeConfig(
        endpoint_url=ENDPOINT,
        tags=(OpcUaTag(node_id="ns=2;i=4", name="temp"),),
    )
    record = config.to_audit_record()
    assert record["endpoint_url"] == ENDPOINT
    assert json.loads(json.dumps(record)) == record


def test_config_rejects_non_opcua_scheme() -> None:
    with pytest.raises(ValueError, match="opc.tcp"):
        OpcUaBridgeConfig(
            endpoint_url="http://host:4840",
            tags=(OpcUaTag(node_id="ns=2;i=4", name="temp"),),
        )


def test_config_rejects_no_tags() -> None:
    with pytest.raises(ValueError, match="at least one tag"):
        OpcUaBridgeConfig(endpoint_url=ENDPOINT, tags=())


def test_config_rejects_duplicate_node_ids() -> None:
    with pytest.raises(ValueError, match="node_ids must be unique"):
        OpcUaBridgeConfig(
            endpoint_url=ENDPOINT,
            tags=(
                OpcUaTag(node_id="ns=2;i=4", name="a"),
                OpcUaTag(node_id="ns=2;i=4", name="b"),
            ),
        )


def test_config_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="names must be unique"):
        OpcUaBridgeConfig(
            endpoint_url=ENDPOINT,
            tags=(
                OpcUaTag(node_id="ns=2;i=4", name="temp"),
                OpcUaTag(node_id="ns=2;i=5", name="temp"),
            ),
        )


def test_config_rejects_unknown_security_policy() -> None:
    with pytest.raises(ValueError, match="security_policy"):
        OpcUaBridgeConfig(
            endpoint_url=ENDPOINT,
            tags=(OpcUaTag(node_id="ns=2;i=4", name="temp"),),
            security_policy="WeakPolicy",
        )


def test_config_rejects_unknown_security_mode() -> None:
    with pytest.raises(ValueError, match="security_mode"):
        OpcUaBridgeConfig(
            endpoint_url=ENDPOINT,
            tags=(OpcUaTag(node_id="ns=2;i=4", name="temp"),),
            security_mode="Scramble",
        )


def test_config_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="request_timeout_s"):
        OpcUaBridgeConfig(
            endpoint_url=ENDPOINT,
            tags=(OpcUaTag(node_id="ns=2;i=4", name="temp"),),
            request_timeout_s=0.0,
        )


# ---------------------------------------------------------------------
# Phase extraction (dependency-free)
# ---------------------------------------------------------------------


def _bridge() -> OpcUaPhaseBridge:
    return OpcUaPhaseBridge.from_tags(
        ENDPOINT,
        [
            OpcUaTag(node_id="ns=2;i=4", name="temp", sample_rate_hz=64.0),
            OpcUaTag(node_id="ns=2;i=5", name="flow", scale=2.0, offset=1.0),
        ],
    )


def test_extract_phases_returns_state_per_tag() -> None:
    bridge = _bridge()
    sinusoid = (0.5 + 0.1 * np.sin(2.0 * np.pi * 2.0 * np.arange(128) / 64.0)).tolist()
    phases = bridge.extract_phases({"temp": sinusoid, "flow": [0.3, 0.4, 0.5, 0.6]})
    assert set(phases) == {"temp", "flow"}
    assert phases["temp"].channel == "P"
    assert phases["temp"].node_id == "temp"
    assert np.isfinite(phases["temp"].theta)
    assert 0.0 <= phases["temp"].quality <= 1.0


def test_extract_phases_applies_scale_and_offset() -> None:
    bridge = OpcUaPhaseBridge.from_tags(
        ENDPOINT,
        [OpcUaTag(node_id="ns=2;i=4", name="temp", scale=10.0, offset=-1.0)],
    )
    base = bridge.extract_phases({"temp": [0.1, 0.2, 0.3, 0.4]})
    # Calibrated series differs from the raw one, so amplitude scales with it.
    raw = OpcUaPhaseBridge.from_tags(
        ENDPOINT, [OpcUaTag(node_id="ns=2;i=4", name="temp")]
    ).extract_phases({"temp": [0.1, 0.2, 0.3, 0.4]})
    assert base["temp"].amplitude != pytest.approx(raw["temp"].amplitude)


def test_extract_phases_missing_tag_rejected() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="missing samples for tag 'flow'"):
        bridge.extract_phases({"temp": [0.1, 0.2]})


def test_extract_phases_empty_series_rejected() -> None:
    bridge = OpcUaPhaseBridge.from_tags(
        ENDPOINT, [OpcUaTag(node_id="ns=2;i=4", name="temp")]
    )
    with pytest.raises(ValueError, match="has no samples"):
        bridge.extract_phases({"temp": []})


def test_extract_phases_non_finite_sample_rejected() -> None:
    bridge = OpcUaPhaseBridge.from_tags(
        ENDPOINT, [OpcUaTag(node_id="ns=2;i=4", name="temp")]
    )
    with pytest.raises(ValueError, match="sample"):
        bridge.extract_phases({"temp": [0.1, float("inf")]})


# ---------------------------------------------------------------------
# Injected synchronous reader
# ---------------------------------------------------------------------


def test_collect_samples_polls_each_tag() -> None:
    bridge = _bridge()
    values = {"ns=2;i=4": 1.5, "ns=2;i=5": 2.5}
    samples = bridge.collect_samples(lambda nid: values[nid], samples_per_tag=3)
    assert samples == {"temp": [1.5, 1.5, 1.5], "flow": [2.5, 2.5, 2.5]}


def test_collect_samples_rejects_bad_count() -> None:
    bridge = _bridge()
    with pytest.raises(ValueError, match="samples_per_tag"):
        bridge.collect_samples(lambda nid: 1.0, samples_per_tag=0)


def test_collect_samples_rejects_non_finite_reading() -> None:
    bridge = OpcUaPhaseBridge.from_tags(
        ENDPOINT, [OpcUaTag(node_id="ns=2;i=4", name="temp")]
    )
    with pytest.raises(ValueError, match="reading"):
        bridge.collect_samples(lambda nid: float("nan"), samples_per_tag=1)


def test_bridge_audit_record() -> None:
    bridge = _bridge()
    record = bridge.to_audit_record()
    assert record["asyncua_available"] == HAS_ASYNCUA
    assert json.loads(json.dumps(record)) == record


# ---------------------------------------------------------------------
# Live-read validation (no server required)
# ---------------------------------------------------------------------


class _FakeNode:
    def __init__(self, value: float) -> None:
        self._value = value

    async def read_value(self) -> float:
        return self._value


class _FakeClient:
    def __init__(self, values: dict[str, float]) -> None:
        self._values = values

    def get_node(self, node_id: str) -> _FakeNode:
        return _FakeNode(self._values[node_id])


async def test_read_live_with_injected_client() -> None:
    bridge = _bridge()
    client = _FakeClient({"ns=2;i=4": 1.0, "ns=2;i=5": 2.0})
    samples = await bridge.read_live(client, samples_per_tag=2)
    assert samples == {"temp": [1.0, 1.0], "flow": [2.0, 2.0]}


async def test_read_live_rejects_bad_count() -> None:
    bridge = _bridge()
    client = _FakeClient({"ns=2;i=4": 1.0, "ns=2;i=5": 2.0})
    with pytest.raises(ValueError, match="samples_per_tag"):
        await bridge.read_live(client, samples_per_tag=0)


async def test_read_live_rejects_negative_interval() -> None:
    bridge = _bridge()
    client = _FakeClient({"ns=2;i=4": 1.0, "ns=2;i=5": 2.0})
    with pytest.raises(ValueError, match="interval_s"):
        await bridge.read_live(client, samples_per_tag=1, interval_s=-1.0)


async def test_read_live_honours_interval() -> None:
    bridge = OpcUaPhaseBridge.from_tags(
        ENDPOINT, [OpcUaTag(node_id="ns=2;i=4", name="temp")]
    )
    client = _FakeClient({"ns=2;i=4": 1.0})
    samples = await bridge.read_live(client, samples_per_tag=2, interval_s=0.001)
    assert samples == {"temp": [1.0, 1.0]}


def test_connect_without_asyncua_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(opcua_bridge, "HAS_ASYNCUA", False)
    bridge = _bridge()
    with pytest.raises(RuntimeError, match="asyncua is not installed"):
        bridge.connect()


def test_connect_uses_asyncua_client_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Client:
        def __init__(self, *, url: str, timeout: float) -> None:
            self.url = url
            self.timeout = timeout

    module = types.ModuleType("asyncua")
    module.__dict__["Client"] = _Client
    monkeypatch.setitem(sys.modules, "asyncua", module)
    monkeypatch.setattr(opcua_bridge, "HAS_ASYNCUA", True)

    bridge = _bridge()
    client = bridge.connect()

    assert isinstance(client, _Client)
    assert client.url == ENDPOINT
    assert client.timeout == pytest.approx(4.0)


@pytest.mark.skipif(not HAS_ASYNCUA, reason="asyncua extra not installed")
def test_connect_constructs_client_without_connecting() -> None:
    bridge = _bridge()
    client = bridge.connect()
    # Construction only — no network round-trip is performed by connect().
    assert client is not None


class _FakeAsyncClient:
    def __init__(self, values: dict[str, float]) -> None:
        self._values = values

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def get_node(self, node_id: str) -> _FakeNode:
        return _FakeNode(self._values[node_id])


async def test_collect_live_with_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = OpcUaPhaseBridge.from_tags(
        ENDPOINT, [OpcUaTag(node_id="ns=2;i=4", name="temp")]
    )
    monkeypatch.setattr(bridge, "connect", lambda: _FakeAsyncClient({"ns=2;i=4": 2.0}))
    samples = await bridge.collect_live(samples_per_tag=2)
    assert samples == {"temp": [2.0, 2.0]}


# ---------------------------------------------------------------------
# Live read against an in-process asyncua server
# ---------------------------------------------------------------------


@asynccontextmanager
async def _opcua_server(port: int, value: float) -> AsyncIterator[tuple[str, str]]:
    from asyncua import Server

    server = Server()
    await server.init()
    url = f"opc.tcp://127.0.0.1:{port}/spo/test"
    server.set_endpoint(url)
    namespace = await server.register_namespace("spo-opcua-test")
    objects = server.nodes.objects
    reactor = await objects.add_object(namespace, "Reactor")
    variable = await reactor.add_variable(namespace, "Temp", value)
    await variable.set_writable()
    async with server:
        yield url, variable.nodeid.to_string()


@pytest.mark.skipif(not HAS_ASYNCUA, reason="asyncua extra not installed")
async def test_live_read_against_in_process_server(unused_tcp_port: int) -> None:
    async with _opcua_server(unused_tcp_port, 3.5) as (url, node_id):
        bridge = OpcUaPhaseBridge.from_tags(
            url, [OpcUaTag(node_id=node_id, name="temp", sample_rate_hz=10.0)]
        )
        client = bridge.connect()
        async with client:
            samples = await bridge.read_live(client, samples_per_tag=4)
    assert samples == {"temp": [3.5, 3.5, 3.5, 3.5]}


@pytest.mark.skipif(not HAS_ASYNCUA, reason="asyncua extra not installed")
async def test_collect_live_connects_and_reads(unused_tcp_port: int) -> None:
    async with _opcua_server(unused_tcp_port, 7.0) as (url, node_id):
        bridge = OpcUaPhaseBridge.from_tags(
            url, [OpcUaTag(node_id=node_id, name="pressure", sample_rate_hz=5.0)]
        )
        samples = await bridge.collect_live(samples_per_tag=3)
        phases = bridge.extract_phases(samples)
    assert samples == {"pressure": [7.0, 7.0, 7.0]}
    assert np.isfinite(phases["pressure"].theta)
