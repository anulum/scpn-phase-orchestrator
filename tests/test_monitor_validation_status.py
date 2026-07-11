# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Monitor validation-status registry tests

"""Tests for the monitor external-validation posture registry.

These cover the record validation, the fail-closed registry builder, the query
helpers, and — most importantly — the drift guard that forces every newly added
public monitor module to be either classified or explicitly excluded, so the
honest validation posture cannot silently rot.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from scpn_phase_orchestrator.monitor import validation_status as vs
from scpn_phase_orchestrator.monitor.validation_status import (
    MONITOR_VALIDATION,
    NON_MONITOR_MODULES,
    MonitorValidationRecord,
    MonitorValidationStatus,
    _build_registry,
    monitors_by_status,
    validation_record,
    validation_summary,
)

_MONITOR_PACKAGE = Path(vs.__file__).parent


def _public_monitor_modules() -> set[str]:
    """Return the stems of the public monitor modules on disk.

    Public means a top-level ``.py`` module under ``monitor/`` that is neither
    ``__init__`` nor a private (underscore-prefixed) backend or helper.
    """
    return {
        path.stem
        for path in _MONITOR_PACKAGE.glob("*.py")
        if path.stem != "__init__" and not path.stem.startswith("_")
    }


class TestMonitorValidationStatus:
    """The enum carries stable machine-readable tokens for every tier."""

    def test_tier_tokens_are_stable(self) -> None:
        assert MonitorValidationStatus.EXTERNALLY_VALIDATED.value == "external"
        assert MonitorValidationStatus.SYNTHETIC_ONLY.value == "synthetic-only"
        assert MonitorValidationStatus.RESEARCH.value == "research"

    def test_exactly_three_tiers(self) -> None:
        assert len(MonitorValidationStatus) == 3


class TestMonitorValidationRecord:
    """The record fails closed on an underspecified or mistyped declaration."""

    def test_valid_record_round_trips(self) -> None:
        record = MonitorValidationRecord(
            monitor="grid_modal_growth",
            display_name="Grid modal growth",
            status=MonitorValidationStatus.EXTERNALLY_VALIDATED,
            basis="leads real transitions above chance",
            evidence="docs/studies/early_warning_matched_false_alarm.md §3.5",
        )
        assert record.monitor == "grid_modal_growth"
        assert record.status is MonitorValidationStatus.EXTERNALLY_VALIDATED

    def test_frozen_record_rejects_mutation(self) -> None:
        record = next(iter(MONITOR_VALIDATION.values()))
        with pytest.raises((AttributeError, TypeError)):
            record.monitor = "mutated"  # type: ignore[misc]

    def test_non_status_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="MonitorValidationStatus"):
            MonitorValidationRecord(
                monitor="m",
                display_name="M",
                status="external",  # type: ignore[arg-type]
                basis="b",
                evidence="",
            )

    @pytest.mark.parametrize("field_name", ["monitor", "display_name", "basis"])
    def test_empty_string_field_raises_value_error(self, field_name: str) -> None:
        kwargs = {
            "monitor": "m",
            "display_name": "M",
            "status": MonitorValidationStatus.RESEARCH,
            "basis": "b",
            "evidence": "",
        }
        kwargs[field_name] = ""
        with pytest.raises(ValueError, match=field_name):
            MonitorValidationRecord(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("field_name", ["monitor", "display_name", "basis"])
    def test_blank_string_field_raises_value_error(self, field_name: str) -> None:
        kwargs = {
            "monitor": "m",
            "display_name": "M",
            "status": MonitorValidationStatus.RESEARCH,
            "basis": "b",
            "evidence": "",
        }
        kwargs[field_name] = "   "
        with pytest.raises(ValueError, match=field_name):
            MonitorValidationRecord(**kwargs)  # type: ignore[arg-type]

    def test_empty_evidence_is_permitted(self) -> None:
        record = MonitorValidationRecord(
            monitor="boundaries",
            display_name="Boundary observer",
            status=MonitorValidationStatus.RESEARCH,
            basis="structural safety observer, no empirical validation record",
            evidence="",
        )
        assert record.evidence == ""


class TestBuildRegistry:
    """The registry builder is fail-closed on duplicates and missing tiers."""

    @staticmethod
    def _record(
        monitor: str, status: MonitorValidationStatus
    ) -> MonitorValidationRecord:
        return MonitorValidationRecord(
            monitor=monitor,
            display_name=monitor,
            status=status,
            basis="basis",
            evidence="",
        )

    def test_duplicate_monitor_raises(self) -> None:
        records = (
            self._record("a", MonitorValidationStatus.EXTERNALLY_VALIDATED),
            self._record("a", MonitorValidationStatus.SYNTHETIC_ONLY),
            self._record("b", MonitorValidationStatus.RESEARCH),
        )
        with pytest.raises(ValueError, match="duplicate monitor validation record"):
            _build_registry(records)

    def test_missing_tier_raises(self) -> None:
        records = (
            self._record("a", MonitorValidationStatus.EXTERNALLY_VALIDATED),
            self._record("b", MonitorValidationStatus.RESEARCH),
        )
        with pytest.raises(ValueError, match="missing tier"):
            _build_registry(records)

    def test_all_tiers_present_builds_read_only_mapping(self) -> None:
        records = (
            self._record("a", MonitorValidationStatus.EXTERNALLY_VALIDATED),
            self._record("b", MonitorValidationStatus.SYNTHETIC_ONLY),
            self._record("c", MonitorValidationStatus.RESEARCH),
        )
        registry = _build_registry(records)
        assert isinstance(registry, MappingProxyType)
        assert list(registry) == ["a", "b", "c"]


class TestRegistry:
    """The shipped registry is internally consistent and read-only."""

    def test_registry_is_read_only(self) -> None:
        assert isinstance(MONITOR_VALIDATION, MappingProxyType)
        with pytest.raises(TypeError):
            MONITOR_VALIDATION["x"] = None  # type: ignore[index]

    def test_keys_match_record_monitor(self) -> None:
        for key, record in MONITOR_VALIDATION.items():
            assert key == record.monitor

    def test_all_three_tiers_represented(self) -> None:
        tiers = {record.status for record in MONITOR_VALIDATION.values()}
        assert tiers == set(MonitorValidationStatus)

    def test_only_grid_modal_is_externally_validated(self) -> None:
        external = {
            record.monitor
            for record in monitors_by_status(
                MonitorValidationStatus.EXTERNALLY_VALIDATED
            )
        }
        assert external == {"grid_modal_growth", "grid_modal_stream"}

    def test_promoted_records_cite_evidence(self) -> None:
        for record in MONITOR_VALIDATION.values():
            if record.status is not MonitorValidationStatus.RESEARCH:
                assert record.evidence, record.monitor
                assert "early_warning_matched_false_alarm" in record.evidence


class TestDriftGuard:
    """Every public monitor module must be classified or explicitly excluded."""

    def test_registry_covers_every_public_monitor(self) -> None:
        classified_or_excluded = set(MONITOR_VALIDATION) | NON_MONITOR_MODULES
        assert _public_monitor_modules() == classified_or_excluded

    def test_no_overlap_between_registry_and_exclusions(self) -> None:
        assert set(MONITOR_VALIDATION).isdisjoint(NON_MONITOR_MODULES)

    def test_exclusions_are_real_modules(self) -> None:
        on_disk = _public_monitor_modules()
        assert on_disk >= NON_MONITOR_MODULES


class TestValidationRecord:
    """The lookup helper returns records and fails closed on unknown names."""

    def test_known_monitor_returns_record(self) -> None:
        record = validation_record("chimera")
        assert record.monitor == "chimera"
        assert record.status is MonitorValidationStatus.RESEARCH

    def test_unknown_monitor_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="unknown monitor 'nope'"):
            validation_record("nope")


class TestMonitorsByStatus:
    """The filter is type-guarded and returns a name-sorted tuple."""

    def test_returns_sorted_tuple(self) -> None:
        records = monitors_by_status(MonitorValidationStatus.SYNTHETIC_ONLY)
        monitors = [record.monitor for record in records]
        assert monitors == sorted(monitors)
        assert all(
            record.status is MonitorValidationStatus.SYNTHETIC_ONLY
            for record in records
        )

    def test_non_status_argument_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="MonitorValidationStatus"):
            monitors_by_status("research")  # type: ignore[arg-type]


class TestValidationSummary:
    """The summary counts every tier and totals the registry size."""

    def test_summary_totals_the_registry(self) -> None:
        summary = validation_summary()
        assert sum(summary.values()) == len(MONITOR_VALIDATION)

    def test_summary_covers_every_tier(self) -> None:
        summary = validation_summary()
        assert set(summary) == set(MonitorValidationStatus)

    def test_summary_is_read_only(self) -> None:
        summary = validation_summary()
        assert isinstance(summary, MappingProxyType)
        with pytest.raises(TypeError):
            summary[MonitorValidationStatus.RESEARCH] = 0  # type: ignore[index]

    def test_summary_matches_by_status(self) -> None:
        summary = validation_summary()
        for status in MonitorValidationStatus:
            assert summary[status] == len(monitors_by_status(status))
