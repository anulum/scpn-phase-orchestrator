# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for plugin manifest registry

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, get_type_hints

import pytest

from scpn_phase_orchestrator.plugins import (
    PluginCapability,
    PluginCompatibilityReport,
    PluginManifest,
    compatibility_report,
    discover_plugin_manifests,
    validate_plugin_manifest,
)


def _manifest() -> PluginManifest:
    return PluginManifest(
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
            PluginCapability(
                kind="actuator",
                name="breaker",
                target="grid_pack.actuators:BreakerMapper",
                knobs=("K", "zeta"),
            ),
        ),
        min_spo_version="0.1.0",
    )


class TestPluginManifestContracts:
    def test_public_contracts_are_typed(self) -> None:
        hints = get_type_hints(compatibility_report)

        assert hints["manifest"] is PluginManifest
        assert hints["return"] is PluginCompatibilityReport

    def test_manifest_from_mapping_round_trips_to_audit_record(self) -> None:
        manifest = PluginManifest.from_mapping(_manifest().to_audit_record())

        record = manifest.to_audit_record()

        assert record["name"] == "grid_pack"
        assert record["capabilities"][0]["kind"] == "extractor"
        assert record["capabilities"][1]["knobs"] == ["K", "zeta"]

    def test_invalid_capability_kind_is_rejected(self) -> None:
        invalid_kind: Any = "invalid"

        with pytest.raises(ValueError, match="unsupported"):
            PluginCapability(kind=invalid_kind, name="x", target="pkg:x")

    def test_bad_version_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
            PluginManifest(
                name="bad",
                version="1",
                package="bad",
                capabilities=(PluginCapability("domainpack", "pack", "bad:pack"),),
            )


class TestPluginCompatibility:
    def test_valid_manifest_passes_compatibility(self) -> None:
        manifest = _manifest()

        report = compatibility_report(manifest)

        assert report.compatible
        assert report.reasons == ()
        assert validate_plugin_manifest(manifest) is manifest

    def test_extractor_must_declare_channels(self) -> None:
        manifest = PluginManifest(
            name="bad_extractor",
            version="0.1.0",
            package="bad_extractor",
            capabilities=(
                PluginCapability(
                    kind="extractor",
                    name="empty",
                    target="bad.extractors:Empty",
                ),
            ),
        )

        report = compatibility_report(manifest)

        assert not report.compatible
        assert "must declare channels" in report.reasons[0]
        with pytest.raises(ValueError, match="must declare channels"):
            validate_plugin_manifest(manifest)

    def test_actuator_must_declare_knobs(self) -> None:
        manifest = PluginManifest(
            name="bad_actuator",
            version="0.1.0",
            package="bad_actuator",
            capabilities=(
                PluginCapability(
                    kind="actuator",
                    name="empty",
                    target="bad.actuators:Empty",
                ),
            ),
        )

        report = compatibility_report(manifest)

        assert not report.compatible
        assert "must declare knobs" in report.reasons[0]

    def test_duplicate_capabilities_are_reported(self) -> None:
        capability = PluginCapability(
            kind="domainpack",
            name="pack",
            target="grid_pack.domainpacks:PACK_DIR",
        )
        manifest = PluginManifest(
            name="dup",
            version="0.1.0",
            package="dup",
            capabilities=(capability, capability),
        )

        report = compatibility_report(manifest)

        assert not report.compatible
        assert "duplicate capability" in report.reasons[0]


class TestPluginDiscovery:
    def test_discovers_entry_point_manifests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = _manifest()

        @dataclass(frozen=True)
        class FakeEntryPoint:
            def load(self) -> Any:
                return lambda: manifest.to_audit_record()

        class FakeEntryPoints(tuple):
            def select(self, *, group: str) -> tuple[FakeEntryPoint, ...]:
                if group == "scpn_phase_orchestrator.plugins":
                    return (FakeEntryPoint(),)
                return ()

        import scpn_phase_orchestrator.plugins.registry as registry

        monkeypatch.setattr(
            registry.metadata,
            "entry_points",
            lambda: FakeEntryPoints(),
        )

        discovered = discover_plugin_manifests()

        assert discovered == (manifest,)
