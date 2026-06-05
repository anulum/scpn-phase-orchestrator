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

## `scpn_phase_orchestrator.grpc_gen.spo_pb2_grpc`

Purpose:
- Preserve the public generated gRPC service import path for SPO service
  stubs and servicer wiring.

Canonical runtime source:
- `scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2_grpc`

Compatibility guarantees:
1. Public and runtime gRPC symbols are equivalent by contract.
2. Service-method naming and route shape are not mutated in alias layer.
3. Public import path remains stable across refactors.

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

## `scpn_phase_orchestrator.grpc_gen.spo_pb2`

Purpose:
- Preserve the public protobuf message import path for generated SPO gRPC
  bindings while runtime modules remain canonical.

Canonical runtime source:
- `scpn_phase_orchestrator.runtime.grpc_gen.spo_pb2`

Compatibility guarantees:
1. Public path stability for generated protobuf message symbols.
2. Runtime and public modules expose equivalent generated message contracts.
3. Alias indirection is explicit and import-safe.

Verification:
- `tests/test_grpc_gen_aliases.py::test_top_level_grpc_aliases_resolve_to_runtime_modules`

## Why compatibility aliases are kept

Client tooling often depends on historical import paths. These aliases preserve
that surface while still centralizing runtime behavior in one location.

- Backward compatibility is preserved for environments that import through top-level
  `scpn_phase_orchestrator.grpc_gen` paths.
- Runtime tests prove that aliases do not change observable service symbols, so
  adapter consumers keep stable behavior across refactors.
- The alias indirection pattern makes runtime module upgrades safer because public
  import paths remain unchanged.

## Operational role

These aliases are release-time compatibility shims. In production rollouts, teams
often pin downstream integrations (dashboards, operator tools, orchestration
scripts) to historical module paths. The fallback alias layer allows internal runtime
module replacements without forcing simultaneous consumer refactors.

The practical effect is simpler rollout sequencing:

- Internal implementations can move between generated and fallback modules.
- Consumers continue importing stable top-level paths.
- CI and parity validation can target one canonical runtime contract while preserving
  public surface stability.

## Risk posture and governance

The module surface is intentionally narrow:

- Aliases never expand the public protocol schema.
- Alias failures are test-gated so contract mismatches are caught before release.
- Runtime module replacement is evaluated through existing CI checks on import/route
  parity, not through silent import-path drift.

This keeps compatibility work auditable: what changed is the module owner, not
the operator contract.
