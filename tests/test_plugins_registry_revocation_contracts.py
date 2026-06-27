# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin revocation contract tests

"""Focused plugin execution-request revocation validation contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from scpn_phase_orchestrator.plugins.registry import (
    PluginCapability,
    PluginExecutionRequest,
    PluginExecutionRequestRevocation,
    PluginExecutionRequestRevocationList,
    PluginManifest,
    PluginRuntimeExecutionPolicy,
    _record_hash,
    build_plugin_execution_approval,
    build_plugin_execution_plan,
    build_plugin_execution_request,
    build_plugin_execution_request_revocation,
    build_plugin_execution_request_revocation_list,
    validate_plugin_execution_request_revocation_list,
)


def _manifest(name: str = "grid_pack") -> PluginManifest:
    """Build a minimal plugin manifest with a callable actuator target."""
    return PluginManifest(
        name=name,
        version="0.1.0",
        package=name,
        capabilities=(
            PluginCapability(
                kind="actuator",
                name="breaker",
                target=f"{name}.actuators:BreakerMapper",
                knobs=("K",),
            ),
        ),
        min_spo_version="0.1.0",
    )


def _request(name: str = "grid_pack") -> PluginExecutionRequest:
    """Build a real approved plugin execution request."""
    draft_plan = build_plugin_execution_plan(
        _manifest(name),
        "actuator",
        "breaker",
        policy=PluginRuntimeExecutionPolicy(
            loading_permitted=True,
            execution_permitted=True,
        ),
    )
    plan = build_plugin_execution_plan(
        _manifest(name),
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
        approval_reference=f"REQ-2026-06-27-{name.upper()}",
        approval_reason="operator approved",
    )
    return build_plugin_execution_request(plan, approval)


def _revocation(name: str = "grid_pack") -> PluginExecutionRequestRevocation:
    """Build a real request revocation artefact."""
    return build_plugin_execution_request_revocation(
        _request(name),
        revoked_by="deployment_gate",
        revocation_reference=f"REV-2026-06-27-{name.upper()}",
        revocation_reason="operator rotation",
    )


def _revocation_list() -> PluginExecutionRequestRevocationList:
    """Build a deterministic aggregate revocation list."""
    return build_plugin_execution_request_revocation_list(
        (_revocation("grid_pack"), _revocation("edge_pack")),
        created_by="deployment_gate",
    )


def _replace_list_record(
    revocation_list: PluginExecutionRequestRevocationList,
    *,
    record: dict[str, object],
) -> PluginExecutionRequestRevocationList:
    """Replace the persisted audit record while keeping dataclass fields stable."""
    return replace(revocation_list, audit_record=record)


def _unsafe_count_mutated_list(
    revocation_list: PluginExecutionRequestRevocationList,
    *,
    revocation_count: int,
) -> PluginExecutionRequestRevocationList:
    """Reconstruct a corrupted stored list object without dataclass validation."""
    corrupted = object.__new__(PluginExecutionRequestRevocationList)
    object.__setattr__(corrupted, "schema", revocation_list.schema)
    object.__setattr__(corrupted, "version", revocation_list.version)
    object.__setattr__(corrupted, "request_hashes", revocation_list.request_hashes)
    object.__setattr__(
        corrupted,
        "revocation_hashes",
        revocation_list.revocation_hashes,
    )
    object.__setattr__(corrupted, "revocation_count", revocation_count)
    object.__setattr__(corrupted, "created_by", revocation_list.created_by)
    object.__setattr__(
        corrupted,
        "revocation_list_hash",
        revocation_list.revocation_list_hash,
    )
    object.__setattr__(corrupted, "audit_record", revocation_list.audit_record)
    return corrupted


def _rehashed_list_record(
    revocation_list: PluginExecutionRequestRevocationList,
    **overrides: object,
) -> dict[str, object]:
    """Return a revocation-list audit record with a refreshed outer hash."""
    record = dict(revocation_list.audit_record)
    record.update(overrides)
    record.pop("revocation_list_hash", None)
    record["revocation_list_hash"] = _record_hash(record)
    return record


class TestPluginExecutionRequestRevocationGuards:
    """Validate request-level revocation artefact fail-closed contracts."""

    @pytest.mark.parametrize(
        ("overrides", "match"),
        (
            ({"schema": "unexpected"}, "revocation schema must be"),
            ({"version": "2.0.0"}, "revocation version must be 1.0.0"),
            ({"kind": "unsupported"}, "unsupported plugin capability kind"),
        ),
    )
    def test_rejects_identity_and_kind_mutations(
        self,
        overrides: dict[str, object],
        match: str,
    ) -> None:
        """Revocation artefacts validate their schema, version, and kind."""
        with pytest.raises(ValueError, match=match):
            replace(_revocation(), **overrides)

    def test_rejects_non_revoked_artefact(self) -> None:
        """Revocation artefacts must explicitly mark the request as revoked."""
        with pytest.raises(PermissionError, match="must mark request revoked"):
            replace(_revocation(), revoked=False)

    def test_rejects_missing_audit_record(self) -> None:
        """Revocation artefacts require a persisted audit payload."""
        with pytest.raises(ValueError, match="audit_record must be provided"):
            replace(_revocation(), audit_record=cast(dict[str, object], None))


class TestPluginExecutionRequestRevocationListGuards:
    """Validate aggregate revocation-list construction and persistence guards."""

    @pytest.mark.parametrize(
        ("overrides", "match"),
        (
            ({"schema": "unexpected"}, "revocation list schema must be"),
            ({"version": "2.0.0"}, "revocation list version must be 1.0.0"),
            ({"revocation_count": 3}, "must match request hash count"),
            ({"revocation_hashes": ("a" * 64,)}, "must match revocation hash count"),
        ),
    )
    def test_rejects_list_constructor_mutations(
        self,
        overrides: dict[str, object],
        match: str,
    ) -> None:
        """Revocation-list dataclass fields fail closed before persistence."""
        with pytest.raises(ValueError, match=match):
            replace(_revocation_list(), **overrides)

    def test_rejects_missing_list_audit_record(self) -> None:
        """Revocation lists require a persisted audit payload."""
        with pytest.raises(ValueError, match="audit_record must be provided"):
            replace(_revocation_list(), audit_record=cast(dict[str, object], None))

    def test_builder_rejects_empty_revocation_sequence(self) -> None:
        """Deployment lists must contain at least one revocation artefact."""
        with pytest.raises(ValueError, match="requires at least one revocation"):
            build_plugin_execution_request_revocation_list(
                (),
                created_by="deployment_gate",
            )

    def test_builder_rejects_tampered_revocation_record(self) -> None:
        """The builder detects revocation records changed after hashing."""
        revocation = _revocation()
        tampered_record = dict(revocation.audit_record)
        tampered_record["revocation_reason"] = "post-approval mutation"

        with pytest.raises(ValueError, match="revocation audit record mismatch"):
            build_plugin_execution_request_revocation_list(
                (replace(revocation, audit_record=tampered_record),),
                created_by="deployment_gate",
            )

    def test_builder_rejects_duplicate_request_hashes(self) -> None:
        """Deployment lists reject duplicate revoked request hashes."""
        revocation = _revocation()

        with pytest.raises(ValueError, match="duplicate request hashes"):
            build_plugin_execution_request_revocation_list(
                (revocation, revocation),
                created_by="deployment_gate",
            )


class TestStoredPluginExecutionRequestRevocationListValidation:
    """Validate fail-closed checks for stored aggregate revocation-list records."""

    def test_validates_real_list(self) -> None:
        """A real builder-produced revocation list validates without mutation."""
        revocation_list = _revocation_list()

        assert validate_plugin_execution_request_revocation_list(revocation_list) is (
            revocation_list
        )
        assert len(revocation_list.as_revoked_request_hashes()) == 2

    @pytest.mark.parametrize(
        ("overrides", "match"),
        (
            ({"schema": "unexpected"}, "revocation list schema mismatch"),
            ({"version": "2.0.0"}, "revocation list version must be 1.0.0"),
            (
                {"request_hashes": "not-a-list"},
                "revocation list request_hashes must be a string list",
            ),
            (
                {"revocation_hashes": "not-a-list"},
                "revocation list revocation_hashes must be a string list",
            ),
            (
                {"revocations": ["not-an-object"]},
                "revocation list revocations must be object records",
            ),
        ),
    )
    def test_rejects_rehashed_schema_and_shape_mutations(
        self,
        overrides: dict[str, object],
        match: str,
    ) -> None:
        """Stored records fail closed when schema or sequence shapes drift."""
        revocation_list = _revocation_list()
        record = _rehashed_list_record(revocation_list, **overrides)

        with pytest.raises(ValueError, match=match):
            validate_plugin_execution_request_revocation_list(
                _replace_list_record(revocation_list, record=record),
            )

    def test_rejects_missing_outer_hash(self) -> None:
        """Stored records must carry their deterministic list hash."""
        revocation_list = _revocation_list()
        record = dict(revocation_list.audit_record)
        record.pop("revocation_list_hash")

        with pytest.raises(ValueError, match="missing revocation_list_hash"):
            validate_plugin_execution_request_revocation_list(
                _replace_list_record(revocation_list, record=record),
            )

    def test_rejects_outer_hash_mismatch(self) -> None:
        """Stored records reject stale outer hashes after payload mutation."""
        revocation_list = _revocation_list()
        record = dict(revocation_list.audit_record)
        record["created_by"] = "tampered_gate"

        with pytest.raises(ValueError, match="revocation list hash mismatch"):
            validate_plugin_execution_request_revocation_list(
                _replace_list_record(revocation_list, record=record),
            )

    def test_rejects_duplicate_request_hashes_after_rehash(self) -> None:
        """Duplicate request hashes remain invalid even with a fresh outer hash."""
        revocation_list = _revocation_list()
        first_hash = revocation_list.request_hashes[0]
        record = _rehashed_list_record(
            revocation_list,
            request_hashes=[first_hash, first_hash],
        )

        with pytest.raises(ValueError, match="duplicate request hashes"):
            validate_plugin_execution_request_revocation_list(
                _replace_list_record(revocation_list, record=record),
            )

    def test_rejects_request_hash_field_mismatch(self) -> None:
        """Stored records must match the dataclass request-hash field."""
        revocation_list = _revocation_list()
        record = _rehashed_list_record(
            revocation_list,
            request_hashes=list(reversed(revocation_list.request_hashes)),
        )

        with pytest.raises(ValueError, match="request hash field mismatch"):
            validate_plugin_execution_request_revocation_list(
                _replace_list_record(revocation_list, record=record),
            )

    def test_rejects_revocation_hash_field_mismatch(self) -> None:
        """Stored records must match the dataclass revocation-hash field."""
        revocation_list = _revocation_list()
        record = _rehashed_list_record(
            revocation_list,
            revocation_hashes=list(reversed(revocation_list.revocation_hashes)),
        )

        with pytest.raises(ValueError, match="revocation hash field mismatch"):
            validate_plugin_execution_request_revocation_list(
                _replace_list_record(revocation_list, record=record),
            )

    def test_rejects_count_mismatch(self) -> None:
        """Stored records must match the aggregate dataclass count."""
        revocation_list = _revocation_list()
        corrupted = _unsafe_count_mutated_list(revocation_list, revocation_count=1)

        with pytest.raises(ValueError, match="revocation list count mismatch"):
            validate_plugin_execution_request_revocation_list(corrupted)

    def test_rejects_nested_revocation_without_hash(self) -> None:
        """Each nested revocation record must carry its own hash."""
        revocation_list = _revocation_list()
        nested = cast(
            list[dict[str, object]],
            revocation_list.audit_record["revocations"],
        )
        revocation = dict(nested[0])
        revocation.pop("revocation_hash")
        record = _rehashed_list_record(
            revocation_list,
            revocations=[revocation],
            request_hashes=[revocation_list.request_hashes[0]],
            revocation_hashes=[revocation_list.revocation_hashes[0]],
            revocation_count=1,
        )
        narrowed = replace(
            revocation_list,
            request_hashes=(revocation_list.request_hashes[0],),
            revocation_hashes=(revocation_list.revocation_hashes[0],),
            revocation_count=1,
            audit_record=record,
        )

        with pytest.raises(ValueError, match="missing revocation_hash"):
            validate_plugin_execution_request_revocation_list(narrowed)

    def test_rejects_nested_revocation_hash_mismatch(self) -> None:
        """Nested revocation records must match their embedded hash."""
        revocation_list = _revocation_list()
        nested = cast(
            list[dict[str, object]],
            revocation_list.audit_record["revocations"],
        )
        revocation = dict(nested[0])
        revocation["revocation_reason"] = "tampered after hashing"
        record = _rehashed_list_record(
            revocation_list,
            revocations=[revocation],
            request_hashes=[revocation_list.request_hashes[0]],
            revocation_hashes=[revocation_list.revocation_hashes[0]],
            revocation_count=1,
        )
        narrowed = replace(
            revocation_list,
            request_hashes=(revocation_list.request_hashes[0],),
            revocation_hashes=(revocation_list.revocation_hashes[0],),
            revocation_count=1,
            audit_record=record,
        )

        with pytest.raises(ValueError, match="revocation audit record mismatch"):
            validate_plugin_execution_request_revocation_list(narrowed)
