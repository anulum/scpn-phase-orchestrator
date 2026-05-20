# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin API reference

# Plugins

The plugin interface defines metadata and compatibility checks for extension
packages that provide domainpacks, phase extractors, monitors, actuators, or
bridges.

Plugins expose a manifest through the `scpn_phase_orchestrator.plugins` Python
entry-point group. The manifest is validated before a marketplace, CI job, or
deployment imports domain-specific implementation code.

```python
from scpn_phase_orchestrator.plugins import (
    PluginCapability,
    PluginManifest,
    validate_plugin_manifest,
)

manifest = PluginManifest(
    name="grid_pack",
    version="0.1.0",
    package="grid_pack",
    capabilities=(
        PluginCapability(
            kind="extractor",
            name="pmu",
            target="grid_pack.extractors:PMUExtractor",
            channels=("P",),
        ),
    ),
)

validate_plugin_manifest(manifest)
```

Supported capability kinds are `domainpack`, `extractor`, `monitor`,
`actuator`, and `bridge`. Monitor capabilities must declare at least one
channel so registry, marketplace, and Rust-dispatch metadata can expose the
observed signal surface explicitly.

Marketplace and CI tooling can package discovered manifests into a
deterministic metadata catalogue without importing plugin implementation
targets:

```python
from scpn_phase_orchestrator.plugins import build_plugin_marketplace_catalog

catalogue = build_plugin_marketplace_catalog((manifest,))
```

The catalogue includes the package manifest, compatibility result, reason list,
SPO version, schema version, and capability counts for every supported
capability kind. By default incompatible manifests are counted but omitted from
the published `plugins` list; pass `include_incompatible=True` when a review job
needs the full rejection report.

Rust-side dispatchers can consume a flattened metadata registry without
importing plugin implementation targets:

```python
from scpn_phase_orchestrator.plugins import build_rust_plugin_registry

registry = build_rust_plugin_registry((manifest,))
```

The Rust registry uses schema `scpn_rust_plugin_registry_v1` and flattens every
compatible capability into records containing plugin name, package, kind,
target, channels, knobs, and compatibility status. This is the metadata bridge
for Rust loaders that need stable JSON before any Python callback is invoked.

Rust runtime handoff review jobs can consume a stricter no-load handoff:

```python
from scpn_phase_orchestrator.plugins import build_rust_plugin_runtime_handoff

handoff = build_rust_plugin_runtime_handoff((manifest,))
```

The handoff uses schema `scpn_rust_plugin_runtime_handoff_v1`, groups compatible
capabilities by kind for dispatch review, records deterministic target hashes,
keeps incompatible capabilities in a blocked review list when requested, and
sets `loading_permitted` to `false`. It is a runtime preflight contract for
native dispatchers, not a dynamic plugin loader.

Python-owned runtime loading is a separate explicit gate. It is disabled by
default, validates the manifest before import, only resolves a declared
capability by kind and name, keeps implementation targets inside the manifest
package by default, and returns deterministic audit metadata beside the loaded
callable:

```python
from scpn_phase_orchestrator.plugins import (
    PluginRuntimeExecutionPolicy,
    PluginRuntimeLoadPolicy,
    execute_plugin_capability,
    load_plugin_capability,
)

loaded = load_plugin_capability(
    manifest,
    "extractor",
    "pmu",
    policy=PluginRuntimeLoadPolicy(loading_permitted=True),
)

extractor_cls = loaded.target_object
audit_record = loaded.audit_record
```

Runtime loading is intended for approved deployment boundaries after manifest
review. Do not use it as a generic import helper: incompatible manifests,
undeclared capabilities, non-callable targets, domainpack metadata records, and
targets outside the plugin package fail closed.

Approved runtime invocation is a second gate. It is also disabled by default and
records invocation shape without serialising argument values:

```python
executed = execute_plugin_capability(
    manifest,
    "monitor",
    "frequency_drift",
    args=(phase_window,),
    kwargs={"sample_rate_hz": 100.0},
    policy=PluginRuntimeExecutionPolicy(
        loading_permitted=True,
        execution_permitted=True,
    ),
)

result = executed.result
execution_audit = executed.audit_record
```

The execution record uses schema `scpn_plugin_runtime_execute_v1`, links back to
the load hash and target hash, records argument count, keyword names, result
type, and a deterministic execution hash. Argument payloads are intentionally
not copied into the audit record.

Deployments can also require a pre-approved target hash before import:

```python
policy = PluginRuntimeExecutionPolicy(
    loading_permitted=True,
    execution_permitted=True,
    approved_target_hashes=("64_hex_characters_from_reviewed_load_audit",),
    require_target_hash_approval=True,
)
```

When `require_target_hash_approval` is true, the loader computes the declared
target hash from manifest metadata and policy shape before import. Targets that
do not match the approved set fail closed without loading the implementation
module.

A runnable metadata-only example is available at
`examples/plugin_marketplace_catalog.py`. It builds a validated
extractor/actuator manifest and prints the resulting catalogue JSON without
loading the implementation targets declared in the manifest.

The same catalogue is available from the command line:

```bash
spo plugins catalog
spo plugins catalog --include-incompatible
spo plugins catalog --rust-registry
spo plugins catalog --rust-runtime-handoff
```

The default output omits incompatible entries from `plugins` while still
counting them. Use `--include-incompatible` in CI or review jobs that need the
full rejection reason list. Use `--rust-registry` when a Rust-side dispatcher
needs the flattened capability registry instead of the marketplace catalogue.
Use `--rust-runtime-handoff` when a runtime review job needs grouped dispatch
records, target hashes, and explicit no-load policy. `--rust-registry` and
`--rust-runtime-handoff` are mutually exclusive output modes.

::: scpn_phase_orchestrator.plugins.registry
