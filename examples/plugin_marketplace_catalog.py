#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Example plugin marketplace catalogue

"""Build a metadata-only plugin marketplace catalogue example."""

from __future__ import annotations

import json
from typing import Any

from scpn_phase_orchestrator.plugins import (
    PluginCapability,
    PluginManifest,
    build_plugin_marketplace_catalog,
    validate_plugin_manifest,
)


def build_example_catalogue() -> dict[str, object]:
    """Return a validated catalogue for one example plugin package."""
    manifest = PluginManifest(
        name="grid_controls_pack",
        version="0.1.0",
        package="grid_controls_pack",
        capabilities=(
            PluginCapability(
                kind="extractor",
                name="pmu_phase",
                target="grid_controls_pack.extractors:PMUPhaseExtractor",
                channels=("P",),
            ),
            PluginCapability(
                kind="actuator",
                name="breaker_mapper",
                target="grid_controls_pack.actuators:BreakerMapper",
                knobs=("K", "zeta"),
            ),
        ),
        min_spo_version="0.1.0",
    )
    validate_plugin_manifest(manifest)
    return build_plugin_marketplace_catalog((manifest,))


def main() -> None:
    """Print the example catalogue as formatted JSON."""
    catalogue: dict[str, Any] = build_example_catalogue()
    print(json.dumps(catalogue, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
