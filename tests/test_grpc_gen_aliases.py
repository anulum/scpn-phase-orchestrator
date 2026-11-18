# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gRPC compatibility alias tests

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import scpn_phase_orchestrator

_GenericHandler = tuple[str, dict[str, tuple[str, object]]]


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


@pytest.mark.parametrize(
    "alias_file",
    [
        "_spo_pb2_fallback",
        "_spo_pb2_grpc_fallback",
        "spo_pb2",
        "spo_pb2_grpc",
    ],
)
def test_alias_module_file_rebinds_itself_to_the_runtime_module(
    alias_file: str,
) -> None:
    """Each installed alias file replaces itself with the runtime module.

    The package-level alias redirects normal imports to the runtime package,
    so each alias file is executed here directly from the installed package
    under its own public module name. Running the files as ``__main__``
    through ``runpy`` executed the code but left it outside package-based
    coverage, which matches measured files by module name.
    """
    package_dir = Path(scpn_phase_orchestrator.__file__).resolve().parent
    alias_path = package_dir / "grpc_gen" / f"{alias_file}.py"
    module_name = f"scpn_phase_orchestrator.grpc_gen.{alias_file}"
    runtime = importlib.import_module(
        f"scpn_phase_orchestrator.runtime.grpc_gen.{alias_file}"
    )
    spec = importlib.util.spec_from_file_location(module_name, alias_path)
    assert spec is not None and spec.loader is not None
    alias = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = alias
    try:
        spec.loader.exec_module(alias)
        assert vars(alias)["_module"] is runtime
        assert sys.modules[module_name] is runtime
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def test_fallback_servicer_alias_registers_handlers() -> None:
    from scpn_phase_orchestrator.runtime.grpc_gen import (
        _spo_pb2_fallback as runtime_pb2,
    )
    from scpn_phase_orchestrator.runtime.grpc_gen import (
        _spo_pb2_grpc_fallback as runtime_grpc,
    )

    fallback_grpc = importlib.import_module(
        "scpn_phase_orchestrator.grpc_gen._spo_pb2_grpc_fallback"
    )
    fallback_pb2 = importlib.import_module(
        "scpn_phase_orchestrator.grpc_gen._spo_pb2_fallback"
    )
    # The alias package re-binds to the runtime package, so these imports load
    # a second copy of the runtime files under the alias names.
    assert fallback_grpc.__file__ == runtime_grpc.__file__
    assert fallback_pb2.__file__ == runtime_pb2.__file__
    assert (
        fallback_grpc.PhaseOrchestratorServicer.__abstractmethods__
        == runtime_grpc.PhaseOrchestratorServicer.__abstractmethods__
    )

    class Servicer(runtime_grpc.PhaseOrchestratorServicer):
        def GetState(
            self, request: runtime_pb2.StateRequest, context: object
        ) -> runtime_pb2.StateResponse:
            return runtime_pb2.StateResponse()

        def Step(
            self, request: runtime_pb2.StepRequest, context: object
        ) -> runtime_pb2.StateResponse:
            return runtime_pb2.StateResponse(step=request.n_steps)

        def Reset(self, request: object, context: object) -> runtime_pb2.StateResponse:
            return runtime_pb2.StateResponse()

        def StreamPhases(
            self, request: runtime_pb2.StreamRequest, context: object
        ) -> Iterator[runtime_pb2.StateResponse]:
            yield runtime_pb2.StateResponse(step=request.max_steps)

        def GetConfig(
            self, request: runtime_pb2.ConfigRequest, context: object
        ) -> runtime_pb2.ConfigResponse:
            return runtime_pb2.ConfigResponse(name="demo")

    class Server:
        def __init__(self) -> None:
            self.handlers: tuple[_GenericHandler, ...] = ()

        def add_generic_rpc_handlers(self, handlers: Iterable[_GenericHandler]) -> None:
            self.handlers = tuple(handlers)

    fake_grpc = ModuleType("grpc")
    fake_grpc_attrs: Any = fake_grpc
    fake_grpc_attrs.method_service_handler = lambda service, handlers: (
        service,
        handlers,
    )
    fake_grpc_attrs.unary_stream_rpc_method_handler = lambda method: (
        "stream",
        method,
    )
    fake_grpc_attrs.unary_unary_rpc_method_handler = lambda method: (
        "unary",
        method,
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
