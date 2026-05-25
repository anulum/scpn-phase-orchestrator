# Python Facade

High-level local simulation facade for application code that should not invoke
the `spo` command-line interface directly.

## Use cases

- Load a reviewed `binding_spec.yaml` and run a deterministic local Kuramoto
  simulation from Python.
- Embed SPO in notebooks, services, and downstream validation scripts without
  shelling out to Click commands.
- Inspect final phase state, coupling matrices, and order parameters while
  keeping hardware actuation and network side effects disabled.

## Contract

`Orchestrator` is intentionally narrower than the CLI. It accepts validated
research-tier Kuramoto binding specs, executes local numerical dynamics only,
and returns an immutable `OrchestratorState` record.

::: scpn_phase_orchestrator.api
