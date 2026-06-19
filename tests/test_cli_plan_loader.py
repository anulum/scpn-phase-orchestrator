# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plan-payload loader validation guards

from __future__ import annotations

from typing import Any

import click
import pytest

from scpn_phase_orchestrator.plugins.registry import PluginCapability, PluginManifest
from scpn_phase_orchestrator.runtime.cli import (
    _build_plan_payload_for_hash,
    _load_plan_from_payload,
    _record_hash,
)

_TARGET_HASH = "b" * 64


def _manifest() -> PluginManifest:
    return PluginManifest(
        name="grid_pack",
        version="0.1.0",
        package="grid_pack",
        capabilities=(
            PluginCapability(
                kind="monitor",
                name="frequency_drift",
                target="grid_pack.monitors:FrequencyDriftMonitor",
                channels=("frequency",),
            ),
        ),
        min_spo_version="0.1.0",
    )


def _plan(*, recompute: bool = True, **overrides: Any) -> dict[str, Any]:
    """Return a valid CLI plan payload with a recomputed hash by default."""
    payload: dict[str, Any] = {
        "schema": "scpn_plugin_runtime_execution_plan_v1",
        "manifest": _manifest().to_audit_record(),
        "capability": {"kind": "monitor", "name": "frequency_drift"},
        "argument_count": 0,
        "keyword_names": [],
        "target_hash": _TARGET_HASH,
        "execution_permitted": True,
        "plan_hash": "0" * 64,
    }
    payload.update(overrides)
    if recompute:
        payload["plan_hash"] = _record_hash(_build_plan_payload_for_hash(payload))
    return payload


def test_valid_plan_round_trips() -> None:
    plan, record = _load_plan_from_payload(_plan())
    assert plan.target_hash == _TARGET_HASH
    assert record["target_hash"] == _TARGET_HASH


class TestPlanLoaderGuards:
    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="plan schema mismatch"):
            _load_plan_from_payload(_plan(schema="wrong"))

    def test_rejects_missing_manifest_object(self) -> None:
        with pytest.raises(click.ClickException, match="missing manifest object"):
            _load_plan_from_payload(_plan(manifest=5))

    def test_rejects_missing_capability_object(self) -> None:
        with pytest.raises(click.ClickException, match="missing capability object"):
            _load_plan_from_payload(_plan(capability=5))

    def test_rejects_malformed_manifest(self) -> None:
        with pytest.raises(click.ClickException, match="manifest schema mismatch"):
            _load_plan_from_payload(_plan(manifest={"name": "x"}))

    def test_rejects_non_string_capability_fields(self) -> None:
        with pytest.raises(click.ClickException, match="capability schema mismatch"):
            _load_plan_from_payload(_plan(capability={"kind": 1, "name": 2}))

    def test_rejects_unknown_capability(self) -> None:
        with pytest.raises(click.ClickException, match="capability schema mismatch"):
            _load_plan_from_payload(
                _plan(capability={"kind": "monitor", "name": "absent"})
            )

    def test_rejects_negative_argument_count(self) -> None:
        with pytest.raises(click.ClickException, match="argument_count must be"):
            _load_plan_from_payload(_plan(argument_count=-1))

    def test_rejects_non_list_keyword_names(self) -> None:
        with pytest.raises(click.ClickException, match="keyword_names must be a list"):
            _load_plan_from_payload(_plan(keyword_names="abc"))

    def test_rejects_non_string_keyword_names(self) -> None:
        with pytest.raises(click.ClickException, match="keyword_names must contain"):
            _load_plan_from_payload(_plan(keyword_names=[1, 2]))

    def test_rejects_hash_mismatch(self) -> None:
        # Build a valid payload, then tamper a hashed field without recomputing.
        payload = _plan()
        payload["argument_count"] = 5
        with pytest.raises(click.ClickException, match="plan hash mismatch"):
            _load_plan_from_payload(payload)

    def test_rejects_non_boolean_execution_permitted(self) -> None:
        with pytest.raises(click.ClickException, match="execution_permitted must be"):
            _load_plan_from_payload(_plan(execution_permitted="yes"))

    def test_rejects_unpermitted_execution(self) -> None:
        with pytest.raises(
            click.ClickException, match="must be permitted for approval"
        ):
            _load_plan_from_payload(_plan(execution_permitted=False))

    def test_rejects_unapproved_target_hash(self) -> None:
        with pytest.raises(click.ClickException, match="is not approved"):
            _load_plan_from_payload(_plan(require_target_hash_approval=True))
