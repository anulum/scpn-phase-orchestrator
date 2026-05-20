# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for plugin manifest registry

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from runpy import run_path
from typing import Any, cast, get_type_hints

import pytest

from scpn_phase_orchestrator.plugins import (
    PluginCapability,
    PluginCompatibilityReport,
    PluginExecutionApproval,
    PluginExecutionPlan,
    PluginExecutionRequest,
    PluginExecutionRequestStorageManifest,
    PluginManifest,
    PluginRuntimeExecutionPolicy,
    PluginRuntimeLoadPolicy,
    build_plugin_execution_approval,
    build_plugin_execution_plan,
    build_plugin_execution_request,
    build_plugin_execution_request_storage_bundle,
    build_plugin_execution_request_storage_manifest,
    build_plugin_marketplace_catalog,
    build_rust_plugin_registry,
    build_rust_plugin_runtime_handoff,
    compatibility_report,
    discover_plugin_manifests,
    execute_plugin_capability,
    execute_plugin_execution_request,
    load_plugin_capability,
    validate_plugin_execution_request,
    validate_plugin_execution_request_storage_bundle,
    validate_plugin_execution_request_storage_manifest,
    validate_plugin_manifest,
    write_plugin_execution_request_storage_bundle,
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
                kind="monitor",
                name="frequency_drift",
                target="grid_pack.monitors:FrequencyDriftMonitor",
                channels=("frequency",),
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
        assert record["capabilities"][1]["kind"] == "monitor"
        assert record["capabilities"][2]["knobs"] == ["K", "zeta"]

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

    def test_mapping_with_missing_package_is_rejected(self) -> None:
        payload: dict[str, Any] = {
            "name": "bad_pack",
            "version": "0.1.0",
            "capabilities": (
                {
                    "kind": "bridge",
                    "name": "bridge",
                    "target": "bad_pack.bridges:Bridge",
                },
            ),
        }

        with pytest.raises(KeyError, match="package"):
            PluginManifest.from_mapping(payload)

    def test_mapping_with_non_mapping_capability_is_rejected(self) -> None:
        payload: dict[str, Any] = {
            "name": "bad_pack",
            "version": "0.1.0",
            "package": "bad_pack",
            "capabilities": ("bridge",),
        }

        with pytest.raises(TypeError, match="string indices"):
            PluginManifest.from_mapping(payload)

    def test_mapping_with_invalid_capability_keys_is_rejected(self) -> None:
        payload: dict[str, Any] = {
            "name": "bad_pack",
            "version": "0.1.0",
            "package": "bad_pack",
            "capabilities": ({"name": "bridge", "target": "bad_pack.bridges:Bridge"},),
        }

        with pytest.raises(KeyError, match="kind"):
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

    def test_monitor_must_declare_channels(self) -> None:
        manifest = PluginManifest(
            name="bad_monitor",
            version="0.1.0",
            package="bad_monitor",
            capabilities=(
                PluginCapability(
                    kind="monitor",
                    name="empty",
                    target="bad.monitors:Empty",
                ),
            ),
        )

        report = compatibility_report(manifest)

        assert not report.compatible
        assert "monitor empty must declare channels" in report.reasons[0]
        with pytest.raises(ValueError, match="monitor empty must declare channels"):
            validate_plugin_manifest(manifest)

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

    def test_compatibility_aggregates_multiple_incompatible_reasons(self) -> None:
        manifest = PluginManifest(
            name="bad_pack",
            version="0.1.0",
            package="bad_pack",
            capabilities=(
                PluginCapability(
                    kind="extractor",
                    name="sensor",
                    target="bad_pack.extractors:SensorExtractor",
                ),
                PluginCapability(
                    kind="extractor",
                    name="sensor",
                    target="bad_pack.extractors:SensorExtractorV2",
                ),
                PluginCapability(
                    kind="actuator",
                    name="output",
                    target="bad_pack.actuators:OutputActuator",
                ),
            ),
        )

        report = compatibility_report(manifest)

        assert not report.compatible
        assert "extractor sensor must declare channels" in report.reasons
        assert "duplicate capability extractor:sensor" in report.reasons
        assert "actuator output must declare knobs" in report.reasons

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
        assert report.reasons == ("requires SPO >= 0.1.0, current release-candidate",)


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
            "monitor": 1,
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

    def test_catalog_sorting_is_deterministic(self) -> None:
        reversed_order = build_plugin_marketplace_catalog((_manifest(),))  # baseline
        reverse_alpha = PluginManifest(
            name="alpha_pack",
            version="0.1.0",
            package="alpha_pack",
            capabilities=(
                PluginCapability(
                    kind="domainpack",
                    name="core",
                    target="alpha_pack.domainpacks:Core",
                ),
            ),
        )
        reverse_zeta = PluginManifest(
            name="zeta_pack",
            version="0.1.0",
            package="zeta_pack",
            capabilities=(
                PluginCapability(
                    kind="bridge",
                    name="link",
                    target="zeta_pack.bridges:Link",
                    knobs=("K",),
                ),
            ),
        )
        catalog = build_plugin_marketplace_catalog((reverse_zeta, reverse_alpha))

        plugin_records = cast("list[dict[str, Any]]", catalog["plugins"])
        assert [record["manifest"]["name"] for record in plugin_records] == [
            "alpha_pack",
            "zeta_pack",
        ]
        assert catalog["plugin_count"] == 2
        assert catalog["capability_counts"] == {
            "actuator": 0,
            "bridge": 1,
            "domainpack": 1,
            "extractor": 0,
            "monitor": 0,
        }
        assert catalog["schema_version"] == reversed_order["schema_version"]

    def test_rust_plugin_registry_flattens_capabilities(self) -> None:
        registry = build_rust_plugin_registry((_manifest(),))

        assert registry["schema"] == "scpn_rust_plugin_registry_v1"
        assert registry["capability_count"] == 3
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
            {
                "plugin": "grid_pack",
                "plugin_version": "0.1.0",
                "package": "grid_pack",
                "kind": "monitor",
                "name": "frequency_drift",
                "target": "grid_pack.monitors:FrequencyDriftMonitor",
                "version": "0.1.0",
                "channels": ["frequency"],
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

        assert default_registry["capability_count"] == 3
        assert full_registry["capability_count"] == 4
        capability_records = cast("list[dict[str, Any]]", full_registry["capabilities"])
        assert capability_records[0]["plugin"] == "bad_extractor"
        assert capability_records[0]["compatible"] is False

    def test_rust_registry_sorts_capabilities_deterministically(self) -> None:
        first = PluginManifest(
            name="zeta_pack",
            version="0.1.0",
            package="zeta_pack",
            capabilities=(
                PluginCapability(
                    kind="bridge",
                    name="bridge_alpha",
                    target="zeta_pack.bridges:BridgeAlpha",
                ),
                PluginCapability(
                    kind="actuator",
                    name="actuator_beta",
                    target="zeta_pack.actuators:ActuatorBeta",
                    knobs=("K",),
                ),
            ),
        )
        second = PluginManifest(
            name="alpha_pack",
            version="0.2.0",
            package="alpha_pack",
            capabilities=(
                PluginCapability(
                    kind="extractor",
                    name="extractor_gamma",
                    target="alpha_pack.extractors:ExtractorGamma",
                    channels=("I",),
                ),
                PluginCapability(
                    kind="actuator",
                    name="actuator_alpha",
                    target="alpha_pack.actuators:ActuatorAlpha",
                    knobs=("K",),
                ),
                PluginCapability(
                    kind="bridge",
                    name="bridge_beta",
                    target="alpha_pack.bridges:BridgeBeta",
                ),
            ),
        )
        registry = build_rust_plugin_registry((first, second))

        capability_records = cast("list[dict[str, Any]]", registry["capabilities"])
        assert [item["name"] for item in capability_records] == [
            "actuator_alpha",
            "bridge_beta",
            "extractor_gamma",
            "actuator_beta",
            "bridge_alpha",
        ]
        assert [item["plugin"] for item in capability_records] == [
            "alpha_pack",
            "alpha_pack",
            "alpha_pack",
            "zeta_pack",
            "zeta_pack",
        ]
        assert [item["kind"] for item in capability_records] == [
            "actuator",
            "bridge",
            "extractor",
            "actuator",
            "bridge",
        ]

    def test_rust_runtime_handoff_groups_compatible_capabilities(self) -> None:
        handoff = build_rust_plugin_runtime_handoff((_manifest(),))
        repeated = build_rust_plugin_runtime_handoff((_manifest(),))
        dispatch_groups = cast(
            "dict[str, list[dict[str, Any]]]",
            handoff["dispatch_groups"],
        )
        target_hashes = cast("dict[str, str]", handoff["target_hashes"])

        assert handoff["schema"] == "scpn_rust_plugin_runtime_handoff_v1"
        assert handoff["registry_schema"] == "scpn_rust_plugin_registry_v1"
        assert handoff["loading_permitted"] is False
        assert handoff["load_policy"] == "metadata_only_review"
        assert handoff["compatible_capability_count"] == 3
        assert handoff["blocked_capability_count"] == 0
        assert handoff["blocked_capabilities"] == []
        assert set(dispatch_groups) == {
            "actuator",
            "bridge",
            "domainpack",
            "extractor",
            "monitor",
        }
        assert [record["name"] for record in dispatch_groups["actuator"]] == ["breaker"]
        assert [record["name"] for record in dispatch_groups["extractor"]] == ["pmu"]
        assert [record["name"] for record in dispatch_groups["monitor"]] == [
            "frequency_drift"
        ]
        assert dispatch_groups["bridge"] == []
        assert all(
            record["loading_permitted"] is False
            for records in dispatch_groups.values()
            for record in records
        )
        assert sorted(target_hashes) == [
            "grid_pack:actuator:breaker:0.1.0",
            "grid_pack:extractor:pmu:0.1.0",
            "grid_pack:monitor:frequency_drift:0.1.0",
        ]
        assert all(len(value) == 64 for value in target_hashes.values())
        assert len(str(handoff["handoff_hash"])) == 64
        assert handoff["handoff_hash"] == repeated["handoff_hash"]

    def test_rust_runtime_handoff_keeps_incompatible_capabilities_blocked(
        self,
    ) -> None:
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

        handoff = build_rust_plugin_runtime_handoff(
            (_manifest(), invalid),
            include_incompatible=True,
        )
        blocked = cast("list[dict[str, Any]]", handoff["blocked_capabilities"])

        assert handoff["compatible_capability_count"] == 3
        assert handoff["blocked_capability_count"] == 1
        assert blocked[0]["plugin"] == "bad_extractor"
        assert blocked[0]["compatible"] is False
        assert blocked[0]["loading_permitted"] is False
        assert blocked[0]["blocked_reason"] == "incompatible_manifest"

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

    def test_runtime_handoff_rejects_malformed_registry_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scpn_phase_orchestrator.plugins.registry as registry

        def malformed_registry(
            manifests: tuple[PluginManifest, ...],
            *,
            include_incompatible: bool = False,
        ) -> dict[str, object]:
            assert manifests == (_manifest(),)
            assert include_incompatible is False
            return {
                "schema": "scpn_rust_plugin_registry_v1",
                "capabilities": (),
            }

        monkeypatch.setattr(
            registry,
            "build_rust_plugin_registry",
            malformed_registry,
        )

        with pytest.raises(
            TypeError, match="rust plugin registry payload is malformed"
        ):
            registry.build_rust_plugin_runtime_handoff((_manifest(),))

    def test_runtime_handoff_keeps_blocked_policies_and_compatible_counts(self) -> None:
        invalid = PluginManifest(
            name="bad_monitor",
            version="0.1.0",
            package="bad_monitor",
            capabilities=(
                PluginCapability(
                    kind="monitor",
                    name="empty",
                    target="bad.monitors:Empty",
                ),
            ),
        )

        handoff = build_rust_plugin_runtime_handoff(
            (invalid,), include_incompatible=True
        )

        assert handoff["loading_permitted"] is False
        assert handoff["load_policy"] == "metadata_only_review"
        assert handoff["compatible_capability_count"] == 0
        assert handoff["blocked_capability_count"] == 1
        blocked = cast("list[dict[str, Any]]", handoff["blocked_capabilities"])
        assert blocked[0]["compatible"] is False
        assert blocked[0]["loading_permitted"] is False
        assert blocked[0]["load_policy"] == "metadata_only_review"
        assert blocked[0]["blocked_reason"] == "incompatible_manifest"

        manifest_only = build_rust_plugin_runtime_handoff((_manifest(), invalid))
        assert manifest_only["compatible_capability_count"] == 3
        assert manifest_only["blocked_capability_count"] == 0
        assert manifest_only["blocked_capabilities"] == []

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
            "monitor": 0,
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


class TestPluginRuntimeLoading:
    def test_runtime_loading_is_disabled_by_default_without_importing_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scpn_phase_orchestrator.plugins.registry as registry

        def fail_import(_module: str) -> object:
            raise AssertionError("disabled runtime loading must not import targets")

        monkeypatch.setattr(registry.importlib, "import_module", fail_import)

        with pytest.raises(PermissionError, match="disabled"):
            load_plugin_capability(_manifest(), "extractor", "pmu")

    def test_runtime_loading_resolves_declared_callable_with_audit_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import types

        import scpn_phase_orchestrator.plugins.registry as registry

        class PMUExtractor:
            pass

        module = types.SimpleNamespace(PMUExtractor=PMUExtractor)

        def fake_import(module_name: str) -> object:
            assert module_name == "grid_pack.extractors"
            return module

        monkeypatch.setattr(registry.importlib, "import_module", fake_import)

        loaded = load_plugin_capability(
            _manifest(),
            "extractor",
            "pmu",
            policy=PluginRuntimeLoadPolicy(loading_permitted=True),
        )
        repeated = load_plugin_capability(
            _manifest(),
            "extractor",
            "pmu",
            policy=PluginRuntimeLoadPolicy(loading_permitted=True),
        )

        assert loaded.target_object is PMUExtractor
        assert loaded.capability.name == "pmu"
        assert loaded.audit_record["schema"] == "scpn_plugin_runtime_load_v1"
        assert loaded.audit_record["loading_permitted"] is True
        assert loaded.audit_record["target"] == "grid_pack.extractors:PMUExtractor"
        assert loaded.audit_record["callable"] is True
        assert len(str(loaded.audit_record["target_hash"])) == 64
        assert loaded.audit_record["load_hash"] == repeated.audit_record["load_hash"]

    def test_runtime_loading_rejects_targets_outside_manifest_package(self) -> None:
        manifest = PluginManifest(
            name="escape_pack",
            version="0.1.0",
            package="escape_pack",
            capabilities=(
                PluginCapability(
                    kind="monitor",
                    name="escape",
                    target="other_package.monitors:Monitor",
                    channels=("P",),
                ),
            ),
        )

        with pytest.raises(ValueError, match="outside plugin package"):
            load_plugin_capability(
                manifest,
                "monitor",
                "escape",
                policy=PluginRuntimeLoadPolicy(loading_permitted=True),
            )

    def test_runtime_loading_rejects_non_runtime_domainpack_capability(self) -> None:
        manifest = PluginManifest(
            name="domain_pack",
            version="0.1.0",
            package="domain_pack",
            capabilities=(
                PluginCapability(
                    kind="domainpack",
                    name="core",
                    target="domain_pack.domainpacks:PACK_DIR",
                ),
            ),
        )

        with pytest.raises(ValueError, match="not permitted by runtime load policy"):
            load_plugin_capability(
                manifest,
                "domainpack",
                "core",
                policy=PluginRuntimeLoadPolicy(loading_permitted=True),
            )


class TestPluginRuntimeExecution:
    def test_runtime_execution_plan_rejects_disabled_execution_policy_without_import(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scpn_phase_orchestrator.plugins.registry as registry

        def fail_import(_module: str) -> object:
            raise AssertionError(
                "disabled execution should be rejected before importing target"
            )

        monkeypatch.setattr(registry.importlib, "import_module", fail_import)

        with pytest.raises(PermissionError, match="execution is disabled"):
            build_plugin_execution_plan(_manifest(), "monitor", "frequency_drift")

    def test_runtime_execution_plan_is_deterministic_without_importing_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scpn_phase_orchestrator.plugins.registry as registry

        def fail_import(_module: str) -> object:
            raise AssertionError("execution plan build must not import targets")

        monkeypatch.setattr(registry.importlib, "import_module", fail_import)

        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            args=("value",),
            kwargs={"scale": 3.5},
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )

        assert isinstance(plan, PluginExecutionPlan)
        assert plan.argument_count == 1
        assert plan.keyword_names == ("scale",)
        assert len(plan.target_hash) == 64
        assert len(plan.plan_hash) == 64
        assert plan.audit_record["schema"] == "scpn_plugin_runtime_execution_plan_v1"

    def test_runtime_execution_plan_can_require_preapproved_target_hash(
        self,
    ) -> None:
        seed_plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        approved_hash = seed_plan.target_hash
        approved_plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                approved_target_hashes=(approved_hash,),
                require_target_hash_approval=True,
            ),
        )

        assert approved_plan.target_hash == approved_hash
        assert approved_plan.audit_record["target_hash_approved"] is True
        assert approved_plan.audit_record["approved_target_hashes"] == [approved_hash]

    def test_runtime_execution_plan_does_not_leak_argument_values(
        self,
    ) -> None:
        private_argument = "PRIVATE_ARGUMENT_XYZ_123"
        private_keyword_value = "PRIVATE_KEYWORD_VALUE_456"
        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            args=(private_argument,),
            kwargs={"multiplier": private_keyword_value},
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )

        assert plan.keyword_names == ("multiplier",)
        assert plan.argument_count == 1
        assert private_argument not in str(plan.audit_record)
        assert private_keyword_value not in str(plan.audit_record)
        assert "argument_count" in plan.audit_record
        assert "keyword_names" in plan.audit_record
        assert "argument_values" not in plan.audit_record

    def test_runtime_execution_approval_is_deterministic_and_metadata_only(
        self,
    ) -> None:
        operator = "operator_alpha"
        reference = "CHG-2026-05-19-OP-1"
        reason = "operator approved for production"
        private_argument = "PRIVATE_ARGUMENT_XYZ_123"
        private_keyword_value = "PRIVATE_KEYWORD_VALUE_456"

        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            args=(private_argument,),
            kwargs={"multiplier": private_keyword_value},
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )

        first = build_plugin_execution_approval(
            plan,
            operator_identity=operator,
            approval_reference=reference,
            approval_reason=reason,
        )
        second = build_plugin_execution_approval(
            plan,
            operator_identity=operator,
            approval_reference=reference,
            approval_reason=reason,
        )

        assert first == second
        assert isinstance(first, PluginExecutionApproval)
        assert first.approved is True
        assert first.schema == "scpn_plugin_execution_approval_v1"
        assert first.version == "1.0.0"
        assert first.plan_hash == plan.plan_hash
        assert first.target_hash == plan.target_hash
        assert first.plugin == "grid_pack"
        assert first.kind == "monitor"
        assert first.name == "frequency_drift"
        assert first.operator_identity == operator
        assert first.approval_reference == reference
        assert first.approval_reason == reason
        assert first.execution_permitted is True
        assert len(first.approval_hash) == 64
        assert "argument_values" not in first.audit_record
        assert private_argument not in str(first.audit_record)
        assert private_keyword_value not in str(first.audit_record)

    def test_runtime_execution_request_is_deterministic_and_policy_convertible(
        self,
    ) -> None:
        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-01",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)
        second = build_plugin_execution_request(plan, approval)

        assert request == second
        assert isinstance(request, PluginExecutionRequest)
        assert request.schema == "scpn_plugin_runtime_execution_request_v1"
        assert request.version == "1.0.0"
        assert request.plan_hash == plan.plan_hash
        assert request.approval_hash == approval.approval_hash
        assert request.target_hash == plan.target_hash
        assert request.loading_permitted is True
        assert request.execution_permitted is True
        assert request.require_target_hash_approval is True
        assert request.approved_target_hashes == (plan.target_hash,)
        assert request.audit_record["approved_target_hashes"] == [plan.target_hash]
        assert len(request.audit_record["request_hash"]) == 64

        runtime_policy = request.to_execution_policy()
        assert runtime_policy.loading_permitted is True
        assert runtime_policy.execution_permitted is True
        assert runtime_policy.require_target_hash_approval is True
        assert runtime_policy.approved_target_hashes == (plan.target_hash,)
        assert runtime_policy == PluginRuntimeExecutionPolicy(
            loading_permitted=True,
            execution_permitted=True,
            approved_target_hashes=(plan.target_hash,),
            require_target_hash_approval=True,
        )
        assert validate_plugin_execution_request(request) is request

    def test_runtime_execution_request_validation_rejects_tampering(
        self,
    ) -> None:
        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-10",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)
        tampered = replace(
            request,
            audit_record={**request.audit_record, "approval_reference": "changed"},
        )

        with pytest.raises(ValueError, match="audit record mismatch"):
            validate_plugin_execution_request(tampered)

    def test_runtime_execution_request_validation_rejects_revocation(
        self,
    ) -> None:
        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-11",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)

        with pytest.raises(PermissionError, match="revoked"):
            validate_plugin_execution_request(
                request,
                revoked_request_hashes=(
                    str(request.audit_record["request_hash"]),
                ),
            )

    def test_execution_request_storage_manifest_is_deterministic(
        self,
    ) -> None:
        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-13",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)
        manifest = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri="file:///var/lib/spo/plugin-requests/grid_pack.json",
            storage_backend="local_file",
            retention_policy="retain_until_revoked",
            created_by="deployment_gate",
        )
        repeated = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri="file:///var/lib/spo/plugin-requests/grid_pack.json",
            storage_backend="local_file",
            retention_policy="retain_until_revoked",
            created_by="deployment_gate",
        )

        assert manifest == repeated
        assert isinstance(manifest, PluginExecutionRequestStorageManifest)
        assert manifest.schema == "scpn_plugin_execution_request_storage_manifest_v1"
        assert manifest.request_hash == request.audit_record["request_hash"]
        assert manifest.plan_hash == request.plan_hash
        assert manifest.approval_hash == request.approval_hash
        assert manifest.target_hash == request.target_hash
        assert manifest.revoked_request_hashes == ()
        assert len(manifest.revocation_hash) == 64
        assert len(manifest.manifest_hash) == 64
        assert validate_plugin_execution_request_storage_manifest(
            request,
            manifest,
        ) is manifest

    def test_execution_request_storage_manifest_rejects_tampering(
        self,
    ) -> None:
        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-14",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)
        manifest = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri="file:///var/lib/spo/plugin-requests/grid_pack.json",
            storage_backend="local_file",
            retention_policy="retain_until_revoked",
            created_by="deployment_gate",
        )
        tampered = replace(
            manifest,
            audit_record={**manifest.audit_record, "storage_backend": "other"},
        )

        with pytest.raises(ValueError, match="audit record mismatch"):
            validate_plugin_execution_request_storage_manifest(request, tampered)

    def test_execution_request_storage_manifest_rejects_revoked_request(
        self,
    ) -> None:
        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-15",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)

        with pytest.raises(PermissionError, match="revoked"):
            build_plugin_execution_request_storage_manifest(
                request,
                storage_uri="file:///var/lib/spo/plugin-requests/grid_pack.json",
                storage_backend="local_file",
                retention_policy="retain_until_revoked",
                created_by="deployment_gate",
                revoked_request_hashes=(str(request.audit_record["request_hash"]),),
            )

    def test_execution_request_storage_bundle_is_deterministic(
        self,
    ) -> None:
        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-16",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)
        manifest = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri="file:///var/lib/spo/plugin-requests/grid_pack.json",
            storage_backend="local_file",
            retention_policy="retain_until_revoked",
            created_by="deployment_gate",
        )

        bundle = build_plugin_execution_request_storage_bundle(request, manifest)
        repeated = build_plugin_execution_request_storage_bundle(request, manifest)

        assert bundle == repeated
        assert bundle["schema"] == "scpn_plugin_execution_request_storage_bundle_v1"
        assert bundle["request"] == request.audit_record
        assert bundle["storage_manifest"] == manifest.audit_record
        assert len(str(bundle["bundle_hash"])) == 64
        assert validate_plugin_execution_request_storage_bundle(bundle) is bundle

    def test_execution_request_storage_bundle_rejects_tampering(
        self,
    ) -> None:
        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-17",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)
        manifest = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri="file:///var/lib/spo/plugin-requests/grid_pack.json",
            storage_backend="local_file",
            retention_policy="retain_until_revoked",
            created_by="deployment_gate",
        )
        bundle = build_plugin_execution_request_storage_bundle(request, manifest)
        tampered = {
            **bundle,
            "storage_manifest": {
                **manifest.audit_record,
                "storage_backend": "other",
            },
        }

        with pytest.raises(ValueError, match="storage bundle hash mismatch"):
            validate_plugin_execution_request_storage_bundle(tampered)

    def test_execution_request_storage_bundle_writes_atomically(
        self,
        tmp_path: Path,
    ) -> None:
        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-18",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)
        manifest = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri=f"file://{tmp_path / 'request.json'}",
            storage_backend="local_file",
            retention_policy="retain_until_revoked",
            created_by="deployment_gate",
        )
        output_path = tmp_path / "request.json"

        bundle = write_plugin_execution_request_storage_bundle(
            request,
            manifest,
            output_path,
        )

        assert output_path.exists()
        assert validate_plugin_execution_request_storage_bundle(
            json.loads(output_path.read_text(encoding="utf-8")),
        ) == bundle
        with pytest.raises(FileExistsError):
            write_plugin_execution_request_storage_bundle(
                request,
                manifest,
                output_path,
            )

    def test_runtime_execution_request_rejects_plan_hash_mismatch(
        self,
    ) -> None:
        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-02",
            approval_reason="operator approved",
        )
        bad = replace(approval, plan_hash="0" * 64)

        with pytest.raises(ValueError, match="plan hash mismatch"):
            build_plugin_execution_request(plan, bad)

    def test_runtime_execution_request_rejects_target_hash_mismatch(
        self,
    ) -> None:
        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-03",
            approval_reason="operator approved",
        )
        bad = replace(approval, target_hash="1" * 64)

        with pytest.raises(ValueError, match="target hash mismatch"):
            build_plugin_execution_request(plan, bad)

    def test_runtime_execution_request_rejects_plugin_identity_mismatch(
        self,
    ) -> None:
        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-04",
            approval_reason="operator approved",
        )
        bad = replace(approval, plugin="other_pack")

        with pytest.raises(ValueError, match="plugin name mismatch"):
            build_plugin_execution_request(plan, bad)

    def test_runtime_execution_request_rejects_rejected_approval(
        self,
    ) -> None:
        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-05",
            approval_reason="operator approved",
        )
        rejected = replace(approval, approved=False)

        with pytest.raises(PermissionError, match="must be granted"):
            build_plugin_execution_request(plan, rejected)

    def test_runtime_execution_request_rejects_disabled_plan(
        self,
    ) -> None:
        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-06",
            approval_reason="operator approved",
        )
        disabled = replace(
            plan,
            audit_record={**plan.audit_record, "execution_permitted": False},
        )

        with pytest.raises(
            PermissionError,
            match="execution must be permitted",
        ):
            build_plugin_execution_request(disabled, approval)

    def test_runtime_execution_approval_rejects_unapproved_target_hash(
        self,
    ) -> None:
        approved = build_plugin_execution_plan(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        unapproved = replace(
            approved,
            audit_record={
                **approved.audit_record,
                "require_target_hash_approval": True,
                "target_hash_approved": False,
            },
        )

        with pytest.raises(PermissionError, match="is not approved"):
            build_plugin_execution_approval(
                unapproved,
                operator_identity="operator_alpha",
                approval_reference="RFC-2026-05-20-01",
                approval_reason="recheck required",
            )

    def test_runtime_execution_approval_rejects_disabled_plan(self) -> None:
        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        disabled_plan = replace(
            plan,
            audit_record={**plan.audit_record, "execution_permitted": False},
        )

        with pytest.raises(
            PermissionError,
            match="execution must be permitted",
        ):
            build_plugin_execution_approval(
                disabled_plan,
                operator_identity="operator_alpha",
                approval_reference="RFC-2026-05-20-02",
                approval_reason="plan disabled",
            )

    def test_runtime_execution_approval_rejects_argument_value_leaks(self) -> None:
        private_argument = "PRIVATE_ARGUMENT_XYZ_123"
        private_keyword_value = "PRIVATE_KEYWORD_VALUE_456"

        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            args=(private_argument,),
            kwargs={"multiplier": private_keyword_value},
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="RFC-2026-05-20-03",
            approval_reason="operator approved",
        )

        assert private_argument not in str(approval)
        assert private_keyword_value not in str(approval.audit_record)
        assert private_keyword_value not in str(approval)

    def test_runtime_execution_is_disabled_by_default_without_importing_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scpn_phase_orchestrator.plugins.registry as registry

        def fail_import(_module: str) -> object:
            raise AssertionError("disabled runtime execution must not import targets")

        monkeypatch.setattr(registry.importlib, "import_module", fail_import)

        with pytest.raises(PermissionError, match="execution is disabled"):
            execute_plugin_capability(_manifest(), "monitor", "frequency_drift")

    def test_runtime_execution_invokes_declared_callable_with_audit_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import types

        import scpn_phase_orchestrator.plugins.registry as registry

        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def frequency_drift(value: float, *, scale: float) -> float:
            calls.append(((value,), {"scale": scale}))
            return value * scale

        module = types.SimpleNamespace(FrequencyDriftMonitor=frequency_drift)

        def fake_import(module_name: str) -> object:
            assert module_name == "grid_pack.monitors"
            return module

        monkeypatch.setattr(registry.importlib, "import_module", fake_import)

        executed = execute_plugin_capability(
            _manifest(),
            "monitor",
            "frequency_drift",
            args=(2.0,),
            kwargs={"scale": 3.5},
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        repeated = execute_plugin_capability(
            _manifest(),
            "monitor",
            "frequency_drift",
            args=(2.0,),
            kwargs={"scale": 3.5},
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )

        assert executed.result == 7.0
        assert calls == [((2.0,), {"scale": 3.5}), ((2.0,), {"scale": 3.5})]
        assert executed.audit_record["schema"] == "scpn_plugin_runtime_execute_v1"
        assert executed.audit_record["execution_permitted"] is True
        assert executed.audit_record["argument_count"] == 1
        assert executed.audit_record["keyword_names"] == ["scale"]
        assert executed.audit_record["result_type"] == "float"
        assert len(str(executed.audit_record["execution_hash"])) == 64
        assert (
            executed.audit_record["execution_hash"]
            == repeated.audit_record["execution_hash"]
        )

    def test_runtime_execution_blocks_when_loading_allowed_but_execution_denied(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scpn_phase_orchestrator.plugins.registry as registry

        def fail_import(_module: str) -> object:
            raise AssertionError("execution denial must occur before import")

        monkeypatch.setattr(registry.importlib, "import_module", fail_import)

        with pytest.raises(PermissionError, match="execution is disabled"):
            execute_plugin_capability(
                _manifest(),
                "actuator",
                "breaker",
                policy=PluginRuntimeExecutionPolicy(
                    loading_permitted=True,
                    execution_permitted=False,
                ),
            )

    def test_runtime_execution_can_require_preapproved_target_hash(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import types

        import scpn_phase_orchestrator.plugins.registry as registry

        def breaker() -> str:
            return "armed"

        module = types.SimpleNamespace(BreakerMapper=breaker)
        monkeypatch.setattr(
            registry.importlib,
            "import_module",
            lambda module_name: module,
        )
        initial = execute_plugin_capability(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        approved_hash = str(initial.audit_record["target_hash"])

        approved = execute_plugin_capability(
            _manifest(),
            "actuator",
            "breaker",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                approved_target_hashes=(approved_hash,),
                require_target_hash_approval=True,
            ),
        )

        assert approved.result == "armed"
        assert approved.audit_record["target_hash_approved"] is True
        assert approved.audit_record["approved_target_hashes"] == [approved_hash]

    def test_runtime_execution_rejects_unapproved_target_hash_before_import(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scpn_phase_orchestrator.plugins.registry as registry

        def fail_import(_module: str) -> object:
            raise AssertionError("target-hash rejection must occur before import")

        monkeypatch.setattr(registry.importlib, "import_module", fail_import)

        with pytest.raises(PermissionError, match="not approved"):
            execute_plugin_capability(
                _manifest(),
                "actuator",
                "breaker",
                policy=PluginRuntimeExecutionPolicy(
                    loading_permitted=True,
                    execution_permitted=True,
                    approved_target_hashes=("0" * 64,),
                    require_target_hash_approval=True,
                ),
            )

    def test_request_bound_runtime_execution_requires_exact_plan_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import types

        import scpn_phase_orchestrator.plugins.registry as registry

        def frequency_drift(value: float, *, scale: float) -> float:
            return value * scale

        module = types.SimpleNamespace(FrequencyDriftMonitor=frequency_drift)
        monkeypatch.setattr(
            registry.importlib,
            "import_module",
            lambda module_name: module,
        )
        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            args=(2.0,),
            kwargs={"scale": 3.5},
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            args=(2.0,),
            kwargs={"scale": 3.5},
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-07",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)

        executed = execute_plugin_execution_request(
            _manifest(),
            request,
            args=(2.0,),
            kwargs={"scale": 3.5},
        )

        assert executed.result == 7.0
        assert executed.audit_record["schema"] == "scpn_plugin_runtime_execute_v1"
        assert executed.audit_record["plan_hash"] == request.plan_hash
        assert executed.audit_record["request_hash"] == request.audit_record[
            "request_hash"
        ]
        assert executed.audit_record["approval_hash"] == request.approval_hash
        assert executed.audit_record["target_hash_approved"] is True
        assert executed.audit_record["operator_identity"] == "operator_alpha"

    def test_request_bound_runtime_execution_rejects_shape_before_import(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scpn_phase_orchestrator.plugins.registry as registry

        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            args=(2.0,),
            kwargs={"scale": 3.5},
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            args=(2.0,),
            kwargs={"scale": 3.5},
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-08",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)

        def fail_import(_module: str) -> object:
            raise AssertionError("shape mismatch must fail before import")

        monkeypatch.setattr(registry.importlib, "import_module", fail_import)

        with pytest.raises(PermissionError, match="plan hash mismatch"):
            execute_plugin_execution_request(
                _manifest(),
                request,
                args=(2.0,),
                kwargs={"different": 3.5},
            )

    def test_request_bound_runtime_execution_rejects_revocation_before_import(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import scpn_phase_orchestrator.plugins.registry as registry

        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-12",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)

        def fail_import(_module: str) -> object:
            raise AssertionError("revoked request must fail before import")

        monkeypatch.setattr(registry.importlib, "import_module", fail_import)

        with pytest.raises(PermissionError, match="revoked"):
            execute_plugin_execution_request(
                _manifest(),
                request,
                revoked_request_hashes=(str(request.audit_record["request_hash"]),),
            )

    def test_request_bound_runtime_execution_rejects_manifest_mismatch(
        self,
    ) -> None:
        draft_plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
            ),
        )
        plan = build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                require_target_hash_approval=True,
                approved_target_hashes=(draft_plan.target_hash,),
            ),
        )
        approval = build_plugin_execution_approval(
            plan,
            operator_identity="operator_alpha",
            approval_reference="REQ-2026-05-20-09",
            approval_reason="operator approved",
        )
        request = build_plugin_execution_request(plan, approval)
        other_manifest = PluginManifest(
            name="other_pack",
            version="0.1.0",
            package="other_pack",
            capabilities=(
                PluginCapability(
                    kind="monitor",
                    name="frequency_drift",
                    target="other_pack.monitors:FrequencyDriftMonitor",
                    channels=("frequency",),
                ),
            ),
        )

        with pytest.raises(ValueError, match="does not match manifest"):
            execute_plugin_execution_request(other_manifest, request)

    def test_discovery_rejects_invalid_entry_point_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        @dataclass(frozen=True)
        class InvalidEntryPoint:
            def load(self) -> Any:
                return {
                    "name": "empty_pack",
                    "version": "0.1.0",
                    "package": "empty_pack",
                }

        class FakeEntryPoints(tuple):
            def select(self, *, group: str) -> tuple[InvalidEntryPoint, ...]:
                if group == "scpn_phase_orchestrator.plugins":
                    return (InvalidEntryPoint(),)
                return ()

        import scpn_phase_orchestrator.plugins.registry as registry

        monkeypatch.setattr(
            registry.metadata,
            "entry_points",
            lambda: FakeEntryPoints(),
        )

        with pytest.raises(ValueError, match="requires at least one capability"):
            discover_plugin_manifests()

    def test_discovery_can_return_no_manifests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class EmptyEntryPoints(tuple):
            def select(self, *, group: str) -> tuple[tuple[()], ...]:
                return ()

        import scpn_phase_orchestrator.plugins.registry as registry

        monkeypatch.setattr(
            registry.metadata,
            "entry_points",
            lambda: EmptyEntryPoints(),
        )

        assert discover_plugin_manifests() == ()

    def test_discovery_accepts_manifest_objects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = _manifest()

        @dataclass(frozen=True)
        class ObjectEntryPoint:
            def load(self) -> Any:
                return manifest

        class FakeEntryPoints(tuple):
            def select(self, *, group: str) -> tuple[ObjectEntryPoint, ...]:
                if group == "scpn_phase_orchestrator.plugins":
                    return (ObjectEntryPoint(),)
                return ()

        import scpn_phase_orchestrator.plugins.registry as registry

        monkeypatch.setattr(
            registry.metadata,
            "entry_points",
            lambda: FakeEntryPoints(),
        )

        discovered = discover_plugin_manifests()

        assert discovered == (manifest,)
