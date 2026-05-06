# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin API reference

# Plugins

The plugin interface defines metadata and compatibility checks for extension
packages that provide domainpacks, phase extractors, actuators, or bridges.

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

Marketplace and CI tooling can package discovered manifests into a
deterministic metadata catalogue without importing plugin implementation
targets:

```python
from scpn_phase_orchestrator.plugins import build_plugin_marketplace_catalog

catalogue = build_plugin_marketplace_catalog((manifest,))
```

The catalogue includes the package manifest, compatibility result, reason list,
SPO version, schema version, and capability counts. By default incompatible
manifests are counted but omitted from the published `plugins` list; pass
`include_incompatible=True` when a review job needs the full rejection report.

::: scpn_phase_orchestrator.plugins.registry
