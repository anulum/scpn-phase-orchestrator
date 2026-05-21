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
