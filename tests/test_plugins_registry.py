# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for plugin manifest registry

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from runpy import run_path
from typing import Any, cast, get_type_hints

import pytest

from scpn_phase_orchestrator.plugins import (
    PluginCapability,
    PluginCompatibilityReport,
    PluginManifest,
    build_plugin_marketplace_catalog,
    build_rust_plugin_registry,
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

    def test_mapping_without_capabilities_is_rejected(self) -> None:
        payload: dict[str, Any] = {
            "name": "empty_pack",
            "version": "0.1.0",
            "package": "empty_pack",
        }

        with pytest.raises(ValueError, match="requires at least one capability"):
            PluginManifest.from_mapping(payload)

    @pytest.mark.parametrize(
        ("payload", "diagnostic"),
        (
            (
                {
                    "kind": "bridge",
                    "name": "bci bridge",
                    "target": "bridge_pack.bridges:BCIBridge",
                },
                "capability name must not contain whitespace",
            ),
            (
                {
                    "kind": "extractor",
                    "name": "pmu",
                    "target": "",
                    "channels": ("P",),
                },
                "capability target must be a non-empty string",
            ),
            (
                {
                    "kind": "extractor",
                    "name": "pmu",
                    "target": "grid_pack.extractors:PMUExtractor",
                    "channels": ("voltage phase",),
                },
                "capability channel must not contain whitespace",
            ),
            (
                {
                    "kind": "actuator",
                    "name": "breaker",
                    "target": "grid_pack.actuators:BreakerMapper",
                    "knobs": ("gain limit",),
                },
                "capability knob must not contain whitespace",
            ),
        ),
    )
    def test_capability_fields_fail_with_specific_diagnostics(
        self,
        payload: dict[str, Any],
        diagnostic: str,
    ) -> None:
        with pytest.raises(ValueError, match=diagnostic):
            PluginCapability(**payload)

    def test_manifest_identity_fields_fail_with_specific_diagnostics(self) -> None:
        capability = PluginCapability(
            kind="domainpack",
            name="pack",
            target="grid_pack.domainpacks:PACK_DIR",
        )

        with pytest.raises(ValueError, match="plugin name must not contain whitespace"):
            PluginManifest(
                name="grid pack",
                version="0.1.0",
                package="grid_pack",
                capabilities=(capability,),
            )

        with pytest.raises(ValueError, match="plugin package must be a non-empty"):
            PluginManifest(
                name="grid_pack",
                version="0.1.0",
                package="",
                capabilities=(capability,),
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

    def test_future_spo_version_fails_closed_with_actionable_reason(self) -> None:
        import scpn_phase_orchestrator.plugins.registry as registry

        manifest = PluginManifest(
            name="future_pack",
            version="0.1.0",
            package="future_pack",
            capabilities=(
                PluginCapability(
                    kind="bridge",
                    name="dispatch",
                    target="future_pack.bridges:DispatchBridge",
                ),
            ),
            min_spo_version="999.0.0",
        )

        report = compatibility_report(manifest)

        assert not report.compatible
        assert report.reasons == (
            f"requires SPO >= 999.0.0, current {registry.__version__}",
        )
        with pytest.raises(ValueError, match="requires SPO >= 999.0.0"):
            validate_plugin_manifest(manifest)

    def test_malformed_runtime_version_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scpn_phase_orchestrator.plugins.registry as registry

        monkeypatch.setattr(registry, "__version__", "release-candidate")
        manifest = PluginManifest(
            name="strict_pack",
            version="0.1.0",
            package="strict_pack",
            capabilities=(
                PluginCapability(
                    kind="bridge",
                    name="dispatch",
                    target="strict_pack.bridges:DispatchBridge",
                ),
            ),
            min_spo_version="0.1.0",
        )

        report = registry.compatibility_report(manifest)

        assert not report.compatible
        assert report.reasons == (
            "requires SPO >= 0.1.0, current release-candidate",
        )


class TestPluginMarketplaceCatalog:
    def test_catalog_contract_is_typed(self) -> None:
        hints = get_type_hints(build_plugin_marketplace_catalog)

        assert "PluginManifest" in str(hints["manifests"])
        assert hints["include_incompatible"] is bool
        assert "dict" in str(hints["return"])

    def test_catalog_packages_compatible_manifests_deterministically(self) -> None:
        actuator = PluginManifest(
            name="actuator_pack",
            version="0.2.0",
            package="actuator_pack",
            capabilities=(
                PluginCapability(
                    kind="actuator",
                    name="valve",
                    target="actuator_pack.actuators:ValveMapper",
                    knobs=("K",),
                ),
            ),
        )
        catalog = build_plugin_marketplace_catalog((_manifest(), actuator))

        assert catalog["schema_version"] == "1.0.0"
        assert catalog["plugin_count"] == 2
        assert catalog["compatible_count"] == 2
        assert catalog["incompatible_count"] == 0
        assert catalog["capability_counts"] == {
            "actuator": 2,
            "bridge": 0,
            "domainpack": 0,
            "extractor": 1,
        }
        plugin_records = cast("list[dict[str, Any]]", catalog["plugins"])
        assert plugin_records[0]["manifest"]["name"] == "actuator_pack"
        assert plugin_records[1]["manifest"]["name"] == "grid_pack"

    def test_catalog_can_include_incompatible_reports(self) -> None:
        invalid = PluginManifest(
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

        default_catalog = build_plugin_marketplace_catalog((_manifest(), invalid))
        full_catalog = build_plugin_marketplace_catalog(
            (_manifest(), invalid),
            include_incompatible=True,
        )

        assert default_catalog["plugin_count"] == 1
        assert default_catalog["compatible_count"] == 1
        assert default_catalog["incompatible_count"] == 1
        assert full_catalog["plugin_count"] == 2
        plugin_records = cast("list[dict[str, Any]]", full_catalog["plugins"])
        assert plugin_records[0]["compatible"] is False
        assert "must declare channels" in plugin_records[0]["reasons"][0]

    def test_rust_plugin_registry_flattens_capabilities(self) -> None:
        registry = build_rust_plugin_registry((_manifest(),))

        assert registry["schema"] == "scpn_rust_plugin_registry_v1"
        assert registry["capability_count"] == 2
        capability_records = cast("list[dict[str, Any]]", registry["capabilities"])
        assert capability_records == [
            {
                "plugin": "grid_pack",
                "plugin_version": "0.1.0",
                "package": "grid_pack",
                "kind": "actuator",
                "name": "breaker",
                "target": "grid_pack.actuators:BreakerMapper",
                "version": "0.1.0",
                "channels": [],
                "knobs": ["K", "zeta"],
                "compatible": True,
            },
            {
                "plugin": "grid_pack",
                "plugin_version": "0.1.0",
                "package": "grid_pack",
                "kind": "extractor",
                "name": "pmu",
                "target": "grid_pack.extractors:PMUExtractor",
                "version": "0.1.0",
                "channels": ["P"],
                "knobs": [],
                "compatible": True,
            },
        ]

    def test_rust_plugin_registry_can_carry_incompatible_entries(self) -> None:
        invalid = PluginManifest(
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

        default_registry = build_rust_plugin_registry((_manifest(), invalid))
        full_registry = build_rust_plugin_registry(
            (_manifest(), invalid),
            include_incompatible=True,
        )

        assert default_registry["capability_count"] == 2
        assert full_registry["capability_count"] == 3
        capability_records = cast("list[dict[str, Any]]", full_registry["capabilities"])
        assert capability_records[0]["plugin"] == "bad_extractor"
        assert capability_records[0]["compatible"] is False

    @pytest.mark.parametrize(
        ("plugins_payload", "diagnostic"),
        (
            ({"not": "a list"}, "plugin catalogue payload is malformed"),
            ([{"manifest": "bad", "compatible": True}], "plugin manifest"),
            (
                [
                    {
                        "manifest": {
                            "name": "bad",
                            "version": "0.1.0",
                            "package": "bad",
                            "capabilities": "bad",
                        },
                        "compatible": True,
                    }
                ],
                "plugin capabilities",
            ),
            (
                [
                    {
                        "manifest": {
                            "name": "bad",
                            "version": "0.1.0",
                            "package": "bad",
                            "capabilities": ["bad"],
                        },
                        "compatible": True,
                    }
                ],
                "plugin capability payload",
            ),
        ),
    )
    def test_rust_plugin_registry_rejects_malformed_catalogue_payloads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        plugins_payload: object,
        diagnostic: str,
    ) -> None:
        import scpn_phase_orchestrator.plugins.registry as registry

        def malformed_catalog(
            manifests: tuple[PluginManifest, ...],
            *,
            include_incompatible: bool = False,
        ) -> dict[str, object]:
            assert manifests == (_manifest(),)
            assert include_incompatible is False
            return {
                "spo_version": "0.1.0",
                "plugins": plugins_payload,
                "capability_counts": {},
            }

        monkeypatch.setattr(
            registry,
            "build_plugin_marketplace_catalog",
            malformed_catalog,
        )

        with pytest.raises(TypeError, match=diagnostic):
            registry.build_rust_plugin_registry((_manifest(),))

    def test_marketplace_catalog_example_is_valid(self) -> None:
        example_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "plugin_marketplace_catalog.py"
        )
        namespace = run_path(str(example_path))
        build_example_catalogue = namespace["build_example_catalogue"]
        assert callable(build_example_catalogue)

        catalog = cast("dict[str, Any]", build_example_catalogue())
        plugin_records = cast("list[dict[str, Any]]", catalog["plugins"])

        assert catalog["schema_version"] == "1.0.0"
        assert catalog["plugin_count"] == 1
        assert catalog["compatible_count"] == 1
        assert catalog["capability_counts"] == {
            "actuator": 1,
            "bridge": 0,
            "domainpack": 0,
            "extractor": 1,
        }
        assert plugin_records[0]["compatible"] is True
        assert plugin_records[0]["manifest"]["name"] == "grid_controls_pack"


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
