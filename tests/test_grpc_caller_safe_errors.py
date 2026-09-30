# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC caller-safe error details

"""Prove that no exception text reaches a gRPC caller.

grpcio answers an exception escaping a servicer method with ``UNKNOWN`` and the
detail ``"Exception calling application: <exception text>"``. Two paths did
that: binary (``-bin``) request metadata broke the UTF-8 decode in the
authorisation step for any unauthenticated caller, and any unexpected fault in
a call body surfaced its interpreter text. Authored request refusals must still
arrive verbatim as ``INVALID_ARGUMENT``. Calls go through a real gRPC server and
channel on localhost.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent import futures
from pathlib import Path
from typing import Any

import pytest

grpc = pytest.importorskip("grpc")

from scpn_phase_orchestrator.binding import load_binding_spec  # noqa: E402
from scpn_phase_orchestrator.exceptions import SPOError  # noqa: E402
from scpn_phase_orchestrator.runtime import server_grpc  # noqa: E402
from scpn_phase_orchestrator.runtime.grpc_gen import (  # noqa: E402
    USING_GENERATED_GRPC,
    ConfigRequest,
    ResetRequest,
    StateRequest,
    StepRequest,
    StreamRequest,
    add_PhaseOrchestratorServicer_to_server,
)
from scpn_phase_orchestrator.runtime.server import SimulationState  # noqa: E402
from scpn_phase_orchestrator.runtime.server_grpc import (  # noqa: E402
    GrpcRequestRefusalError,
    PhaseStreamServicer,
)

if not USING_GENERATED_GRPC:  # pragma: no cover - the generated stubs ship in-tree
    pytest.skip("generated gRPC stubs are unavailable", allow_module_level=True)

from scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2_grpc import (  # noqa: E402
    PhaseOrchestratorStub,
)

SPEC = Path(__file__).resolve().parents[1] / "domainpacks" / "minimal_domain"
ENGINE_DETAIL = "engine-internal-detail-xyz"
INTERPRETER_PHRASES = (
    "Exception calling application",
    "codec can't decode",
    "Traceback",
    ENGINE_DETAIL,
)


class _FailingSimulation(SimulationState):
    """A real simulation whose engine calls fail on demand.

    No production input makes the engine raise by design, so the unexpected-
    failure guard is exercised with this subclass. It changes only the three
    engine entry points; authorisation, validation and transport stay real.
    """

    def __init__(self, fail_on: str) -> None:
        super().__init__(load_binding_spec(SPEC / "binding_spec.yaml"))
        self._fail_on = fail_on

    def step(self) -> dict[str, Any]:
        if self._fail_on == "step":
            raise RuntimeError(ENGINE_DETAIL)
        return super().step()

    def reset(self) -> dict[str, Any]:
        if self._fail_on == "reset":
            raise KeyError(ENGINE_DETAIL)
        return super().reset()

    def snapshot(self) -> dict[str, Any]:
        if self._fail_on == "snapshot":
            raise ValueError(f"could not convert string to float: '{ENGINE_DETAIL}'")
        return super().snapshot()


@pytest.fixture
def stub_for(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[SimulationState], object]]:
    for name in (
        "SPO_GRPC_API_KEY",
        "SPO_API_KEY",
        "SPO_GRPC_ENV",
        "SPO_GRPC_PROFILE",
        "SPO_ENV",
        "SPO_PROFILE",
        "SPO_GRPC_RATE_LIMIT_PER_MINUTE",
    ):
        monkeypatch.delenv(name, raising=False)
    servers: list[Any] = []
    channels: list[Any] = []

    def _start(sim: SimulationState) -> object:
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


def _real_simulation() -> SimulationState:
    return SimulationState(load_binding_spec(SPEC / "binding_spec.yaml"))


def _failure(call: Callable[[], object]) -> tuple[Any, str]:
    with pytest.raises(grpc.RpcError) as caught:
        call()
    return caught.value.code(), caught.value.details()


def _assert_caller_safe(details: str) -> None:
    for phrase in INTERPRETER_PHRASES:
        assert phrase not in details


def test_binary_metadata_no_longer_breaks_authorisation(stub_for) -> None:
    stub = stub_for(_real_simulation())
    metadata = (("x-trace-bin", b"\xff\xfe"),)

    response = stub.GetState(StateRequest(), metadata=metadata, timeout=30)

    assert response.step >= 0


def test_binary_metadata_does_not_bypass_a_configured_key(
    stub_for, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPO_GRPC_API_KEY", "secret-key")
    stub = stub_for(_real_simulation())
    metadata = (("x-api-key-bin", b"secret-key"), ("x-trace-bin", b"\xff"))

    code, details = _failure(
        lambda: stub.GetState(StateRequest(), metadata=metadata, timeout=30)
    )

    assert code == grpc.StatusCode.UNAUTHENTICATED
    assert details == "Invalid or missing x-api-key"


@pytest.mark.parametrize(
    ("call", "sentence"),
    [
        pytest.param(
            lambda stub: stub.Step(StepRequest(n_steps=-3), timeout=30),
            "n_steps must be a positive integer",
            id="step-negative",
        ),
        pytest.param(
            lambda stub: list(
                stub.StreamPhases(
                    StreamRequest(max_steps=1, interval_s=float("nan")), timeout=30
                )
            ),
            "interval_s must be a non-negative real",
            id="stream-nan-interval",
        ),
        pytest.param(
            lambda stub: list(
                stub.StreamPhases(StreamRequest(max_steps=-1), timeout=30)
            ),
            "max_steps must be a positive integer",
            id="stream-negative-max-steps",
        ),
    ],
)
def test_authored_refusals_arrive_verbatim_as_invalid_argument(
    stub_for, call: Callable[[object], object], sentence: str
) -> None:
    stub = stub_for(_real_simulation())

    code, details = _failure(lambda: call(stub))

    assert code == grpc.StatusCode.INVALID_ARGUMENT
    assert details == sentence


@pytest.mark.parametrize(
    ("fail_on", "call"),
    [
        pytest.param(
            "step",
            lambda stub: stub.Step(StepRequest(n_steps=1), timeout=30),
            id="step",
        ),
        pytest.param(
            "reset", lambda stub: stub.Reset(ResetRequest(), timeout=30), id="reset"
        ),
        pytest.param(
            "snapshot",
            lambda stub: stub.GetState(StateRequest(), timeout=30),
            id="get-state",
        ),
        pytest.param(
            "snapshot",
            lambda stub: list(
                stub.StreamPhases(
                    StreamRequest(max_steps=2, interval_s=0.0), timeout=30
                )
            ),
            id="stream",
        ),
    ],
)
def test_unexpected_failures_answer_internal_with_a_fixed_detail(
    stub_for, fail_on: str, call: Callable[[object], object]
) -> None:
    stub = stub_for(_FailingSimulation(fail_on))

    code, details = _failure(lambda: call(stub))

    assert code == grpc.StatusCode.INTERNAL
    assert details == "internal error"
    _assert_caller_safe(details)


def test_config_is_unaffected_by_the_guard(stub_for) -> None:
    stub = stub_for(_real_simulation())

    response = stub.GetConfig(ConfigRequest(), timeout=30)

    assert response.n_oscillators > 0


def test_refusal_type_is_the_only_text_the_servicer_repeats() -> None:
    assert issubclass(GrpcRequestRefusalError, ValueError)
    assert issubclass(GrpcRequestRefusalError, SPOError)
    with pytest.raises(GrpcRequestRefusalError, match="n_steps"):
        server_grpc._validate_positive_int(0, "n_steps")
    with pytest.raises(GrpcRequestRefusalError, match="interval_s"):
        server_grpc._validate_non_negative_real(float("inf"), "interval_s")


def test_undecodable_metadata_pairs_are_skipped() -> None:
    metadata = server_grpc._normalise_metadata(
        [("x-trace-bin", b"\xff"), (b"\xfe", "v"), ("x-api-key", b"key")]
    )

    assert metadata == {"x-api-key": "key"}


def test_abort_without_a_context_still_stops_the_call() -> None:
    servicer = PhaseStreamServicer(_real_simulation())

    with pytest.raises(PermissionError, match="^denied$"):
        servicer._abort(None, None, "denied")
