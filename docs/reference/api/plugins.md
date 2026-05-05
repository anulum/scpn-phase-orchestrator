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

::: scpn_phase_orchestrator.plugins.registry
