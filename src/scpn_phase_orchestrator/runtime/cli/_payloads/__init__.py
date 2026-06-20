# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI payload and plan loaders

"""JSON payload loaders and hash helpers shared by the plugins CLI commands.

These functions parse and validate the on-disk JSON artifacts the ``spo plugins``
commands consume — execution plans and approvals, execution requests, revocations
and revocation lists, storage and storage-adapter manifests, and the lifecycle,
remediation, and scheduler records — reconstructing the corresponding plugin
domain objects and raising :class:`click.ClickException` on malformed input. The
loaders are split into per-domain modules (shared, plan, revocation, storage,
lifecycle, remediation, scheduler) behind a stable re-export surface; they also
provide the canonical record hashing (:func:`_record_hash`), SHA-256 digest
validation, plan-payload canonicalisation, and plugin/capability lookups the
command layer reuses. ``__all__`` lists the internal loader surface so re-export
stays explicit. Pure functions with no Click command surface of their own.
"""

from __future__ import annotations

from ._shared import (
    _PLUGIN_KIND_OPTIONS,
    _build_plan_payload_for_hash,
    _find_capability,
    _find_discovered_plugin,
    _load_json_file,
    _normalize_approved_target_hashes,
    _record_hash,
    _require_sha256,
)
from .lifecycle import (
    _load_lifecycle_from_payload,
    _load_lifecycle_multistore_drilldown_payload,
    _load_lifecycle_policy_report_payload,
    _load_lifecycle_summary_from_payload,
)
from .plan import (
    _load_approval_from_payload,
    _load_plan_from_payload,
    _load_request_from_payload,
)
from .remediation import (
    _load_lifecycle_remediation_action_status_payload,
    _load_lifecycle_remediation_deployment_handoff_payload,
    _load_lifecycle_remediation_execution_dashboard_payload,
    _load_lifecycle_remediation_plan_payload,
)
from .revocation import (
    _load_revocation_from_payload,
    _load_revocation_list_from_payload,
)
from .scheduler import (
    _load_lifecycle_remediation_scheduler_acknowledgement_payload,
    _load_lifecycle_remediation_scheduler_adapter_handoff_payload,
    _load_lifecycle_remediation_scheduler_queue_payload,
    _load_lifecycle_remediation_scheduler_telemetry_payload,
)
from .storage import (
    _load_storage_adapter_from_payload,
    _load_storage_manifest_from_payload,
)

__all__ = [
    "_find_discovered_plugin",
    "_find_capability",
    "_normalize_approved_target_hashes",
    "_require_sha256",
    "_load_json_file",
    "_record_hash",
    "_build_plan_payload_for_hash",
    "_PLUGIN_KIND_OPTIONS",
    "_load_plan_from_payload",
    "_load_approval_from_payload",
    "_load_request_from_payload",
    "_load_revocation_from_payload",
    "_load_revocation_list_from_payload",
    "_load_storage_manifest_from_payload",
    "_load_storage_adapter_from_payload",
    "_load_lifecycle_from_payload",
    "_load_lifecycle_summary_from_payload",
    "_load_lifecycle_policy_report_payload",
    "_load_lifecycle_multistore_drilldown_payload",
    "_load_lifecycle_remediation_plan_payload",
    "_load_lifecycle_remediation_action_status_payload",
    "_load_lifecycle_remediation_execution_dashboard_payload",
    "_load_lifecycle_remediation_deployment_handoff_payload",
    "_load_lifecycle_remediation_scheduler_queue_payload",
    "_load_lifecycle_remediation_scheduler_telemetry_payload",
    "_load_lifecycle_remediation_scheduler_adapter_handoff_payload",
    "_load_lifecycle_remediation_scheduler_acknowledgement_payload",
]
