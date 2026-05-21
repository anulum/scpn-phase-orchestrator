# gRPC generated fallback aliases

## `scpn_phase_orchestrator.grpc_gen._spo_pb2_fallback`

Purpose:
- Preserve the public historical import path for fallback protobuf message
  classes when generated gRPC stubs are not available.

Canonical runtime source:
- `scpn_phase_orchestrator.runtime.grpc_gen._spo_pb2_fallback`

Compatibility guarantees:
1. Import path stability for existing clients.
2. Runtime and public modules expose equivalent symbols.
3. Alias module delegates by replacing itself in `sys.modules` with the runtime
   module object.

Verification:
- `tests/test_grpc_gen_aliases.py::test_top_level_grpc_aliases_resolve_to_runtime_modules`
- `tests/test_grpc_gen_aliases.py::test_fallback_message_alias_preserves_dataclass_contract`

## `scpn_phase_orchestrator.grpc_gen._spo_pb2_grpc_fallback`

Purpose:
- Preserve the public historical import path for fallback gRPC service stubs.

Canonical runtime source:
- `scpn_phase_orchestrator.runtime.grpc_gen._spo_pb2_grpc_fallback`

Compatibility guarantees:
1. Import path stability for fallback service wiring.
2. Public and runtime modules preserve equivalent gRPC service symbols.
3. Alias module delegates through `sys.modules` replacement.

Verification:
- `tests/test_grpc_gen_aliases.py::test_top_level_grpc_aliases_resolve_to_runtime_modules`
- `tests/test_grpc_gen_aliases.py::test_fallback_servicer_alias_registers_handlers`
