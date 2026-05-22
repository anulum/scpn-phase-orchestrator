# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC compatibility alias tests

from __future__ import annotations

import importlib
import runpy
import sys
from types import ModuleType, SimpleNamespace
from typing import Any


def test_top_level_grpc_aliases_resolve_to_runtime_modules() -> None:
    fake_grpc = ModuleType("grpc")
    grpc_attrs: Any = fake_grpc
    grpc_attrs.__version__ = "1.78.0"
    grpc_attrs.__path__ = []
    fake_utilities = ModuleType("grpc._utilities")
    utilities_attrs: Any = fake_utilities
    utilities_attrs.first_version_is_lower = lambda *_args: False
    previous_grpc = sys.modules.get("grpc")
    previous_utilities = sys.modules.get("grpc._utilities")
    sys.modules["grpc"] = fake_grpc
    sys.modules["grpc._utilities"] = fake_utilities
    alias_pairs = {
        "scpn_phase_orchestrator.grpc_gen._spo_pb2_fallback": (
            "scpn_phase_orchestrator.runtime.grpc_gen._spo_pb2_fallback"
        ),
        "scpn_phase_orchestrator.grpc_gen._spo_pb2_grpc_fallback": (
            "scpn_phase_orchestrator.runtime.grpc_gen._spo_pb2_grpc_fallback"
        ),
        "scpn_phase_orchestrator.grpc_gen.spo_pb2": (
            "scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2"
        ),
        "scpn_phase_orchestrator.grpc_gen.spo_pb2_grpc": (
            "scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2_grpc"
        ),
    }

    try:
        for alias_name, runtime_name in alias_pairs.items():
            alias_module = importlib.import_module(alias_name)
            runtime_module = importlib.import_module(runtime_name)

            assert alias_module.__file__ == runtime_module.__file__
            assert set(getattr(alias_module, "__all__", ())) == set(
                getattr(runtime_module, "__all__", ())
            )
    finally:
        if previous_grpc is None:
            sys.modules.pop("grpc", None)
        else:
            sys.modules["grpc"] = previous_grpc
        if previous_utilities is None:
            sys.modules.pop("grpc._utilities", None)
        else:
            sys.modules["grpc._utilities"] = previous_utilities


def test_fallback_message_alias_preserves_dataclass_contract() -> None:
    sys.modules.pop("scpn_phase_orchestrator.grpc_gen._spo_pb2_fallback", None)
    fallback = importlib.import_module(
        "scpn_phase_orchestrator.grpc_gen._spo_pb2_fallback"
    )

    response = fallback.StateResponse(
        step=7,
        R_global=0.75,
        regime="locked",
        layers=[fallback.LayerState(name="P", R=0.8, psi=0.2)],
    )

    assert response.step == 7
    assert response.layers[0].name == "P"
    assert response.layers[0].R == 0.8


def test_fallback_alias_module_script_executes_without_error() -> None:
    module_path = "src/scpn_phase_orchestrator/grpc_gen/_spo_pb2_fallback.py"
    result = runpy.run_path(module_path, run_name="__main__")
    assert "_module" in result


def test_fallback_grpc_alias_module_script_executes_without_error() -> None:
    module_path = "src/scpn_phase_orchestrator/grpc_gen/_spo_pb2_grpc_fallback.py"
    result = runpy.run_path(module_path, run_name="__main__")
    assert "_module" in result


def test_spo_pb2_alias_module_script_executes_without_error() -> None:
    module_path = "src/scpn_phase_orchestrator/grpc_gen/spo_pb2.py"
    result = runpy.run_path(module_path, run_name="__main__")
    assert "_module" in result


def test_spo_pb2_grpc_alias_module_script_executes_without_error() -> None:
    module_path = "src/scpn_phase_orchestrator/grpc_gen/spo_pb2_grpc.py"
    result = runpy.run_path(module_path, run_name="__main__")
    assert "_module" in result


def test_fallback_servicer_alias_registers_handlers() -> None:
    fallback_grpc = importlib.import_module(
        "scpn_phase_orchestrator.grpc_gen._spo_pb2_grpc_fallback"
    )
    fallback_pb2 = importlib.import_module(
        "scpn_phase_orchestrator.grpc_gen._spo_pb2_fallback"
    )

    class Servicer(fallback_grpc.PhaseOrchestratorServicer):
        def GetState(self, request, context):
            return fallback_pb2.StateResponse()

        def Step(self, request, context):
            return fallback_pb2.StateResponse(step=request.n_steps)

        def Reset(self, request, context):
            return fallback_pb2.StateResponse()

        def StreamPhases(self, request, context):
            yield fallback_pb2.StateResponse(step=request.max_steps)

        def GetConfig(self, request, context):
            return fallback_pb2.ConfigResponse(name="demo")

    class Server:
        def __init__(self) -> None:
            self.handlers = ()

        def add_generic_rpc_handlers(self, handlers) -> None:
            self.handlers = tuple(handlers)

    fake_grpc = SimpleNamespace(
        method_service_handler=lambda service, handlers: (service, handlers),
        unary_stream_rpc_method_handler=lambda method: ("stream", method),
        unary_unary_rpc_method_handler=lambda method: ("unary", method),
    )
    server = Server()
    previous_grpc = sys.modules.get("grpc")
    sys.modules["grpc"] = fake_grpc
    try:
        fallback_grpc.add_PhaseOrchestratorServicer_to_server(Servicer(), server)
    finally:
        if previous_grpc is None:
            sys.modules.pop("grpc", None)
        else:
            sys.modules["grpc"] = previous_grpc

    assert len(server.handlers) == 1
    service_name, method_handlers = server.handlers[0]
    assert service_name == "spo.PhaseOrchestrator"
    assert sorted(method_handlers) == [
        "GetConfig",
        "GetState",
        "Reset",
        "Step",
        "StreamPhases",
    ]
