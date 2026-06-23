# Subsystem: `plugins` / `scaffold` / `domainpacks` — extensibility & domain catalogue

How third parties extend SPO and how domains are described. `plugins` 11 files
(~3.5k LOC, incl. a 10-module `registry/` package), `scaffold` 2, plus 36
domainpacks under `domainpacks/`.

## Plugins

A manifest-registry mechanism over the `scpn_phase_orchestrator.plugins`
entry-point group. Five plugin kinds: `domainpack`, `extractor`, `monitor`,
`actuator`, `bridge`. `PluginManifest` / `PluginCapability` declare capabilities;
`discover_plugin_manifests` enumerates entry points; `load_plugin_capability`
loads under an explicit `PluginRuntimeLoadPolicy` (loading disabled by default).

- The former `registry.py` god-file has been split into a 10-module package.
- The discovery mechanism is wired but **unpopulated** in the base project — no
  plugins are declared in the base `pyproject.toml`; it activates when external
  packages register entry points.

## Scaffold

`propose_domainpack_from_description` generates a reviewable domainpack proposal
from a natural-language description via an LLM provider, with prompt-injection
guards; emits YAML + an audit record, never auto-activated. Reached through
`spo scaffold`.

## Domainpacks

36 packs, each a `binding_spec.yaml` + `README.md`. They are the topology-vs-
physics catalogue. Examples spanning the target domains: `power_grid`,
`rotating_machinery`, `plasma_control`, `fusion_equilibrium`,
`quantum_simulation`, `neuroscience_eeg`, `brain_connectome`, `cardiac_rhythm`,
`swarm_robotics`, `satellite_constellation`, `laser_array`, `epidemic_sir`,
`financial_markets`, `traffic_flow`, `queuewaves`, `minimal_domain`, and more.

### Schema (the extension contract)

`HierarchyLayer`, `OscillatorFamily` (channel, extractor_type, config),
`CouplingSpec`, `DriverSpec`, `BoundaryDef`, `ActuatorMapping`, `AmplitudeSpec`,
`ImprintSpec`, validated by `binding/types.py` + `binding/loader.py`. Valid
safety tiers: research / clinical / production.

## Cross-repo

Optional extras map packs to siblings: `quantum → scpn-quantum-control`,
`plasma → scpn-control`, `fusion → scpn-fusion-core (≥3.9.0)`,
`studio → scpn-studio`. Domainpacks are file-based; there is no programmatic
catalogue loader yet.
