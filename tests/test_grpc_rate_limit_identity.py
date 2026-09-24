# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC authorisation and rate-limit identity

"""Prove the gRPC rate limit cannot be dodged and the key check is constant-time.

Without a configured API key the servicer keyed its rate limiter on the
``x-api-key`` metadata the client sends, so each request with a new value got
a fresh bucket. It now keys an unauthenticated caller by its peer address. The
limiter also dropped no bucket, so distinct identities grew its table without
bound. Calls go through a real gRPC server and channel on localhost.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent import futures
from pathlib import Path

import pytest

grpc = pytest.importorskip("grpc")

from scpn_phase_orchestrator.binding import load_binding_spec  # noqa: E402
from scpn_phase_orchestrator.runtime.grpc_gen import (  # noqa: E402
    USING_GENERATED_GRPC,
    StateRequest,
    add_PhaseOrchestratorServicer_to_server,
)
from scpn_phase_orchestrator.runtime.network_security import (  # noqa: E402
    TokenBucketRateLimiter,
)
from scpn_phase_orchestrator.runtime.server import SimulationState  # noqa: E402
from scpn_phase_orchestrator.runtime.server_grpc import (  # noqa: E402
    PhaseStreamServicer,
)

if not USING_GENERATED_GRPC:  # pragma: no cover - the generated stubs ship in-tree
    pytest.skip("generated gRPC stubs are unavailable", allow_module_level=True)

from scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2_grpc import (  # noqa: E402
    PhaseOrchestratorStub,
)

SPEC = Path(__file__).resolve().parents[1] / "domainpacks" / "minimal_domain"


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "SPO_GRPC_API_KEY",
        "SPO_API_KEY",
        "SPO_GRPC_ENV",
        "SPO_GRPC_PROFILE",
        "SPO_ENV",
        "SPO_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def stub_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[object]:
    servers: list[object] = []
    channels: list[object] = []

    def _start(**env: str) -> object:
        _clear_env(monkeypatch)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        sim = SimulationState(load_binding_spec(SPEC / "binding_spec.yaml"))
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        add_PhaseOrchestratorServicer_to_server(PhaseStreamServicer(sim), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        servers.append(server)
        channels.append(channel)
        return PhaseOrchestratorStub(channel)

    yield _start
    for channel in channels:
        channel.close()
    for server in servers:
        server.stop(None)


def _code(stub: object, key: str | None) -> object:
    metadata = () if key is None else (("x-api-key", key),)
    try:
        stub.GetState(StateRequest(), metadata=metadata, timeout=30)
    except grpc.RpcError as error:
        return error.code()
    return grpc.StatusCode.OK


def test_changing_the_key_metadata_does_not_reset_the_limit(stub_factory) -> None:
    """Without a server key, a new x-api-key value used to buy a new bucket."""
    stub = stub_factory(SPO_GRPC_RATE_LIMIT_PER_MINUTE="1")

    assert _code(stub, "first") == grpc.StatusCode.OK
    assert _code(stub, "second") == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert _code(stub, None) == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_configured_key_is_required_and_matched(stub_factory) -> None:
    stub = stub_factory(SPO_GRPC_API_KEY="secret-key")

    assert _code(stub, None) == grpc.StatusCode.UNAUTHENTICATED
    assert _code(stub, "secret-kez") == grpc.StatusCode.UNAUTHENTICATED
    assert _code(stub, "secret-key") == grpc.StatusCode.OK


def test_authenticated_callers_are_limited_per_key(stub_factory) -> None:
    stub = stub_factory(
        SPO_GRPC_API_KEY="secret-key", SPO_GRPC_RATE_LIMIT_PER_MINUTE="1"
    )

    assert _code(stub, "secret-key") == grpc.StatusCode.OK
    assert _code(stub, "secret-key") == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_refilled_buckets_are_dropped_once_many_identities_are_tracked() -> None:
    limiter = TokenBucketRateLimiter(60)
    for index in range(10_001):
        assert limiter.allow(f"caller-{index}", now=0.0)
    assert len(limiter._buckets) == 10_001  # none refilled yet: nothing dropped

    assert limiter.allow("late-caller", now=3_600.0)
    assert set(limiter._buckets) == {"late-caller"}


def test_a_dropped_identity_starts_from_a_full_bucket() -> None:
    limiter = TokenBucketRateLimiter(60, burst_capacity=1)
    assert limiter.allow("caller", now=0.0)
    assert not limiter.allow("caller", now=0.0)
    for index in range(10_001):
        limiter.allow(f"other-{index}", now=10.0)
    assert limiter.allow("caller", now=10.0)
    assert not limiter.allow("caller", now=10.0)
