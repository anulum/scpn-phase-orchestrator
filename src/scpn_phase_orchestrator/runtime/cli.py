# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI entry point

"""Command-line entry point for validation, replay, export, and review workflows.

The CLI wraps public SPO APIs behind explicit commands for binding validation,
inspection, auto-binding proposals, coupling estimation, formal export, replay,
plugin catalogs, scaffolding, and selected runtime utilities. Commands validate
local inputs and emit text or JSON review artifacts; they do not push commits,
start network services, or perform live actuation unless an explicit subcommand
is invoked for that runtime path.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal, TypeAlias, cast

import click
import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator import plugins as plugin_api
from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.autotune.binding_proposal import (
    propose_binding_from_event_log,
    propose_binding_from_graph,
    propose_binding_from_time_series_csv,
)
from scpn_phase_orchestrator.binding import (
    ChannelRuntimeExecutor,
    compile_symbolic_binding,
    format_resolved_binding_config,
    load_binding_spec,
    resolved_binding_config,
    validate_binding_spec,
)
from scpn_phase_orchestrator.binding.types import ProtocolNetSpec
from scpn_phase_orchestrator.coupling.geometry_constraints import (
    GeometryConstraint,
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
)
from scpn_phase_orchestrator.coupling.infer import auto_coupling_estimation
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder, CouplingState
from scpn_phase_orchestrator.drivers.psi_informational import InformationalDriver
from scpn_phase_orchestrator.drivers.psi_physical import PhysicalDriver
from scpn_phase_orchestrator.drivers.psi_symbolic import SymbolicDriver
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.meta import CrossDomainMetaTransfer
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.plugins import (
    PluginCapability,
    PluginExecutionApproval,
    PluginExecutionPlan,
    PluginExecutionRequest,
    PluginExecutionRequestLifecycleRecord,
    PluginExecutionRequestLifecycleSummary,
    PluginExecutionRequestRevocation,
    PluginExecutionRequestRevocationList,
    PluginExecutionRequestStorageAdapterManifest,
    PluginExecutionRequestStorageManifest,
    PluginManifest,
    PluginRuntimeExecutionPolicy,
    build_plugin_execution_approval,
    build_plugin_execution_plan,
    build_plugin_execution_request_lifecycle_policy_report,
    build_plugin_execution_request_lifecycle_record,
    build_plugin_execution_request_lifecycle_summary,
    build_plugin_execution_request_revocation,
    build_plugin_execution_request_revocation_list,
    build_plugin_execution_request_storage_adapter_manifest,
    build_plugin_execution_request_storage_manifest,
    build_plugin_marketplace_catalog,
    build_rust_plugin_registry,
    build_rust_plugin_runtime_handoff,
    compatibility_report,
    discover_plugin_manifests,
    validate_plugin_execution_request_revocation_list,
    write_plugin_execution_request_storage_bundle,
)
from scpn_phase_orchestrator.plugins import registry as plugin_registry
from scpn_phase_orchestrator.reporting.summary import build_audit_report_summary
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.runtime.audit_stream import (
    AuditStreamEvent,
    iter_event_stream,
    read_event_stream,
    verify_event_stream_integrity,
)
from scpn_phase_orchestrator.runtime.replay import ReplayEngine
from scpn_phase_orchestrator.scaffold.llm import (
    LLMScaffoldProvider,
    StaticJSONScaffoldProvider,
    configured_llm_scaffold_provider,
    propose_domainpack_from_description,
)
from scpn_phase_orchestrator.supervisor.events import EventBus
from scpn_phase_orchestrator.supervisor.formal_export import (
    FormalSafetyProperty,
    audit_formal_checker_availability,
    build_formal_verification_package,
    export_petri_net_prism,
    export_petri_net_tla,
    export_policy_rules_prism,
    export_policy_rules_tla,
    export_stl_specs_prism,
)
from scpn_phase_orchestrator.supervisor.petri_adapter import PetriNetAdapter
from scpn_phase_orchestrator.supervisor.petri_net import (
    Arc,
    Marking,
    PetriNet,
    Place,
    Transition,
    parse_guard,
)
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.policy_diagnostics import (
    PolicyDryRunReport,
    dry_run_policy_rules,
)
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyEngine,
    load_policy_rules,
    load_policy_stl_specs,
)
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import (
    compute_order_parameter,
    compute_plv,
)
from scpn_phase_orchestrator.upde.pac import modulation_index
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

FloatArray: TypeAlias = NDArray[np.float64]


@click.group()
def main() -> None:
    """SCPN Phase Orchestrator CLI."""


@main.command()
@click.argument("binding_spec", type=click.Path(exists=True))
def validate(binding_spec: str) -> None:
    """Validate a binding specification file."""
    spec = load_binding_spec(Path(binding_spec))
    errors = validate_binding_spec(spec)
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)
    click.echo("Valid")
    summary = resolved_binding_config(spec)
    for line in format_resolved_binding_config(summary):
        click.echo(line)


@main.command("inspect")
@click.argument("binding_spec", type=click.Path(exists=True))
@click.option("--json-out", is_flag=True, help="Output resolved summary as JSON")
def inspect_binding(binding_spec: str, json_out: bool) -> None:
    """Inspect resolved runtime choices for a binding spec."""
    spec = load_binding_spec(Path(binding_spec))
    errors = validate_binding_spec(spec)
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)

    summary = resolved_binding_config(spec)
    if json_out:
        click.echo(json.dumps(summary, indent=2, sort_keys=True))
        return

    for line in format_resolved_binding_config(summary):
        click.echo(line)


@main.command("auto-bind")
@click.argument(
    "source_kind",
    type=click.Choice(["time-series-csv", "event-log-json", "graph-json"]),
)
@click.argument("source_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--project-name",
    required=True,
    help="Name to embed in the review-only proposed binding spec.",
)
@click.option(
    "--sample-rate-hz",
    type=float,
    default=None,
    help="Sampling rate for time-series CSV sources.",
)
@click.option("--json-out", is_flag=True, help="Output proposal audit record as JSON")
def auto_bind(
    source_kind: str,
    source_path: str,
    project_name: str,
    sample_rate_hz: float | None,
    json_out: bool,
) -> None:
    """Propose a review-only binding spec from raw local source data."""
    try:
        source_text = Path(source_path).read_text(encoding="utf-8")
        if source_kind == "time-series-csv":
            proposal = propose_binding_from_time_series_csv(
                source_text,
                sample_rate_hz=sample_rate_hz,
                project_name=project_name,
            )
        elif source_kind == "event-log-json":
            proposal = propose_binding_from_event_log(
                source_text,
                project_name=project_name,
            )
        else:
            proposal = propose_binding_from_graph(
                source_text,
                project_name=project_name,
            )
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1) from exc
    except OSError as exc:
        click.echo(f"ERROR: could not read source file: {exc.strerror}", err=True)
        raise SystemExit(1) from exc

    if json_out:
        click.echo(json.dumps(proposal.to_audit_record(), indent=2, sort_keys=True))
        return
    click.echo(proposal.binding.yaml_text, nl=False)


def _load_phase_series_table(source_path: Path) -> FloatArray:
    try:
        if source_path.suffix.lower() == ".npy":
            values = np.load(source_path, allow_pickle=False)
        else:
            values = np.loadtxt(source_path, delimiter=",")
    except (OSError, ValueError) as exc:
        raise click.ClickException(
            f"could not read numeric phase-series data: {exc}"
        ) from exc
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 2:
        raise click.ClickException(
            f"phase-series source must be a 2-D table, got shape {series.shape}"
        )
    return np.ascontiguousarray(series, dtype=np.float64)


@main.command("auto-coupling-estimation")
@click.argument("source_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--orientation",
    type=click.Choice(["oscillator-by-time", "time-by-oscillator"]),
    default="oscillator-by-time",
    show_default=True,
    help="Input table orientation.",
)
@click.option("--n-bins", type=int, default=8, show_default=True)
@click.option("--threshold-quantile", type=float, default=0.75, show_default=True)
@click.option("--threshold-absolute", type=float, default=None)
@click.option(
    "--normalisation",
    type=click.Choice(["max", "none"]),
    default="max",
    show_default=True,
)
@click.option("--json-out", is_flag=True, help="Output JSON audit record")
def auto_coupling_estimation_command(
    source_path: str,
    orientation: str,
    n_bins: int,
    threshold_quantile: float,
    threshold_absolute: float | None,
    normalisation: str,
    json_out: bool,
) -> None:
    """Infer directed K_nm from phase time-series data."""

    series = _load_phase_series_table(Path(source_path))
    if orientation == "time-by-oscillator":
        series = np.ascontiguousarray(series.T, dtype=np.float64)
    try:
        result = auto_coupling_estimation(
            series,
            n_bins=n_bins,
            threshold_quantile=threshold_quantile,
            threshold_absolute=threshold_absolute,
            normalisation=cast(Literal["max", "none"], normalisation),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    record = result.to_audit_record()
    if json_out:
        click.echo(json.dumps(record, indent=2, sort_keys=True))
        return

    click.echo(
        "auto-coupling-estimation "
        f"method={record['method']} orientation={record['orientation']} "
        f"shape={record['shape']} edges={result.edge_count} "
        f"density={result.density:.6g}"
    )
    for row in result.knm:
        click.echo(",".join(f"{value:.17g}" for value in row))


def _petri_net_from_protocol(protocol: ProtocolNetSpec) -> tuple[PetriNet, Marking]:
    places = [Place(name) for name in protocol.places]
    transitions = []
    for ts in protocol.transitions:
        guard = parse_guard(ts.guard) if ts.guard else None
        transitions.append(
            Transition(
                name=ts.name,
                inputs=[Arc(a["place"], a.get("weight", 1)) for a in ts.inputs],
                outputs=[Arc(a["place"], a.get("weight", 1)) for a in ts.outputs],
                guard=guard,
            )
        )
    return PetriNet(places, transitions), Marking(tokens=dict(protocol.initial))


_PLUGIN_KIND_OPTIONS: tuple[str, ...] = (
    "actuator",
    "bridge",
    "domainpack",
    "extractor",
    "monitor",
)


def _find_discovered_plugin(
    manifests: tuple[PluginManifest, ...],
    plugin_name: str,
) -> PluginManifest:
    matches = tuple(manifest for manifest in manifests if manifest.name == plugin_name)
    if not matches:
        raise click.ClickException(f"plugin {plugin_name!r} is not discovered")
    if len(matches) > 1:
        raise click.ClickException(
            f"multiple discovered plugin manifests matched {plugin_name!r}; "
            "selection is ambiguous"
        )
    return matches[0]


def _find_capability(
    manifest: PluginManifest,
    kind: str,
    capability_name: str,
) -> PluginCapability:
    matches = tuple(
        capability
        for capability in manifest.capabilities
        if capability.kind == kind and capability.name == capability_name
    )
    if not matches:
        raise click.ClickException(
            f"plugin {manifest.name!r} does not expose {kind}:{capability_name!r}"
        )
    if len(matches) > 1:
        raise click.ClickException(
            f"plugin {manifest.name!r} declares {kind}:{capability_name!r} "
            "more than once"
        )
    return matches[0]


def _normalize_approved_target_hashes(
    approved_target_hashes: tuple[str, ...],
) -> tuple[str, ...]:
    normalized: list[str] = []
    for approved_hash in approved_target_hashes:
        if re.fullmatch(r"[0-9a-fA-F]{64}", approved_hash) is None:
            raise click.ClickException(
                f"approved target hash {approved_hash!r} is not a valid SHA-256 digest"
            )
        normalized.append(approved_hash.lower())
    return tuple(dict.fromkeys(normalized))


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise click.ClickException(
            f"{field_name} must be a 64-character SHA-256 digest"
        )
    if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise click.ClickException(
            f"{field_name} {value!r} is not a valid SHA-256 digest"
        )
    return value.lower()


def _load_json_file(path: Path, *, artifact: str = "plan") -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(f"cannot read plan file {path!s}: {exc}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"malformed {artifact} JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise click.ClickException(f"{artifact} payload must be a JSON object")
    return payload


def _record_hash(record: dict[str, object]) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_plan_payload_for_hash(plan_payload: dict[str, object]) -> dict[str, object]:
    if "plan_hash" not in plan_payload:
        raise click.ClickException("plan payload is missing required field plan_hash")
    if "target_hash" not in plan_payload:
        raise click.ClickException("plan payload is missing required field target_hash")
    payload_without_plan_hash = dict(plan_payload)
    payload_without_plan_hash.pop("plan_hash", None)
    payload_without_plan_hash.pop("manifest", None)
    payload_without_plan_hash.pop("capability", None)
    payload_without_plan_hash.pop("compatible", None)
    payload_without_plan_hash.pop("compatibility_reasons", None)
    return payload_without_plan_hash


def _load_plan_from_payload(
    plan_payload: dict[str, object],
) -> tuple[PluginExecutionPlan, dict[str, object]]:
    if plan_payload.get("schema") != "scpn_plugin_runtime_execution_plan_v1":
        raise click.ClickException(
            "plan schema mismatch: expected scpn_plugin_runtime_execution_plan_v1"
        )

    manifest_payload = plan_payload.get("manifest")
    capability_payload = plan_payload.get("capability")
    if not isinstance(manifest_payload, dict):
        raise click.ClickException("plan payload is missing manifest object")
    if not isinstance(capability_payload, dict):
        raise click.ClickException("plan payload is missing capability object")

    try:
        manifest = PluginManifest.from_mapping(manifest_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(f"manifest schema mismatch: {exc}") from exc

    kind = capability_payload.get("kind")
    name = capability_payload.get("name")
    if not isinstance(kind, str) or not isinstance(name, str):
        raise click.ClickException(
            "capability schema mismatch: kind and name are required"
        )

    try:
        capability = _find_capability(manifest, kind, name)
    except click.ClickException as exc:
        raise click.ClickException(f"capability schema mismatch: {exc}") from exc

    raw_argument_count = plan_payload.get("argument_count")
    if not isinstance(raw_argument_count, int) or raw_argument_count < 0:
        raise click.ClickException(
            "plan schema mismatch: argument_count must be a non-negative integer"
        )
    raw_keyword_names = plan_payload.get("keyword_names")
    if not isinstance(raw_keyword_names, list):
        raise click.ClickException("plan schema mismatch: keyword_names must be a list")
    if not all(isinstance(name, str) for name in raw_keyword_names):
        raise click.ClickException(
            "plan schema mismatch: keyword_names must contain strings"
        )

    expected_plan_hash = _record_hash(
        _build_plan_payload_for_hash(plan_payload),
    )
    plan_hash = _require_sha256(plan_payload.get("plan_hash"), "plan_hash")
    if expected_plan_hash != plan_hash:
        raise click.ClickException("plan hash mismatch")

    target_hash = _require_sha256(plan_payload.get("target_hash"), "target_hash")

    audit_record = dict(plan_payload)
    audit_record["target_hash"] = target_hash
    execution_permitted = plan_payload.get("execution_permitted")
    if not isinstance(execution_permitted, bool):
        raise click.ClickException(
            "plan schema mismatch: execution_permitted must be a boolean"
        )
    if not execution_permitted:
        raise click.ClickException(
            "plugin runtime execution must be permitted for approval"
        )

    if plan_payload.get("require_target_hash_approval") is True:
        target_hash_approved = plan_payload.get("target_hash_approved")
        if target_hash_approved is not True:
            raise click.ClickException(
                f"plugin runtime target hash {target_hash} is not approved"
            )

    return PluginExecutionPlan(
        manifest=manifest,
        capability=capability,
        argument_count=raw_argument_count,
        keyword_names=tuple(raw_keyword_names),
        target_hash=target_hash,
        plan_hash=plan_hash,
        audit_record=audit_record,
    ), audit_record


def _load_approval_from_payload(
    approval_payload: dict[str, object],
) -> PluginExecutionApproval:
    if approval_payload.get("schema") != "scpn_plugin_execution_approval_v1":
        raise click.ClickException(
            "approval schema mismatch: expected scpn_plugin_execution_approval_v1"
        )

    plan_hash = _require_sha256(approval_payload.get("plan_hash"), "plan_hash")
    target_hash = _require_sha256(approval_payload.get("target_hash"), "target_hash")
    approval_hash = _require_sha256(
        approval_payload.get("approval_hash"), "approval_hash"
    )
    plugin = approval_payload.get("plugin")
    kind = approval_payload.get("kind")
    name = approval_payload.get("name")
    operator_identity = approval_payload.get("operator_identity")
    approval_reference = approval_payload.get("approval_reference")
    approval_reason = approval_payload.get("approval_reason")
    approved = approval_payload.get("approved")
    execution_permitted = approval_payload.get("execution_permitted")
    version = approval_payload.get("version")

    for field_name, value in (
        ("plugin", plugin),
        ("kind", kind),
        ("name", name),
        ("operator_identity", operator_identity),
        ("approval_reference", approval_reference),
        ("approval_reason", approval_reason),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"approval schema mismatch: {field_name} must be a non-empty string"
            )

    if not isinstance(approved, bool):
        raise click.ClickException(
            "approval schema mismatch: approved must be a boolean"
        )
    if not isinstance(execution_permitted, bool):
        raise click.ClickException(
            "approval schema mismatch: execution_permitted must be a boolean"
        )
    if kind not in _PLUGIN_KIND_OPTIONS:
        raise click.ClickException(
            f"approval schema mismatch: unsupported kind {kind!r}"
        )

    return PluginExecutionApproval(
        schema="scpn_plugin_execution_approval_v1",
        version=str(version),
        plan_hash=plan_hash,
        target_hash=target_hash,
        plugin=str(plugin),
        kind=cast(
            Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
            str(kind),
        ),
        name=str(name),
        operator_identity=str(operator_identity),
        approval_reference=str(approval_reference),
        approval_reason=str(approval_reason),
        approved=bool(approved),
        execution_permitted=bool(execution_permitted),
        approval_hash=approval_hash,
        audit_record=approval_payload,
    )


def _load_request_from_payload(
    request_payload: dict[str, object],
) -> PluginExecutionRequest:
    if request_payload.get("schema") != "scpn_plugin_runtime_execution_request_v1":
        raise click.ClickException(
            "request schema mismatch: expected scpn_plugin_runtime_execution_request_v1"
        )

    plan_hash = _require_sha256(request_payload.get("plan_hash"), "plan_hash")
    target_hash = _require_sha256(request_payload.get("target_hash"), "target_hash")
    approval_hash = _require_sha256(
        request_payload.get("approval_hash"), "approval_hash"
    )
    plugin = request_payload.get("plugin")
    kind = request_payload.get("kind")
    name = request_payload.get("name")
    operator_identity = request_payload.get("operator_identity")
    approval_reference = request_payload.get("approval_reference")
    loading_permitted = request_payload.get("loading_permitted")
    execution_permitted = request_payload.get("execution_permitted")
    require_target_hash_approval = request_payload.get("require_target_hash_approval")
    require_package_target = request_payload.get("require_package_target")
    approved_target_hashes = request_payload.get("approved_target_hashes")
    allowed_kinds = request_payload.get("allowed_kinds")
    version = request_payload.get("version")

    for field_name, value in (
        ("plugin", plugin),
        ("kind", kind),
        ("name", name),
        ("operator_identity", operator_identity),
        ("approval_reference", approval_reference),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"request schema mismatch: {field_name} must be a non-empty string"
            )
    if kind not in _PLUGIN_KIND_OPTIONS:
        raise click.ClickException(
            f"request schema mismatch: unsupported kind {kind!r}"
        )
    for field_name, value in (
        ("loading_permitted", loading_permitted),
        ("execution_permitted", execution_permitted),
        ("require_target_hash_approval", require_target_hash_approval),
        ("require_package_target", require_package_target),
    ):
        if not isinstance(value, bool):
            raise click.ClickException(
                f"request schema mismatch: {field_name} must be a boolean"
            )
    if not isinstance(approved_target_hashes, list) or not all(
        isinstance(item, str) for item in approved_target_hashes
    ):
        raise click.ClickException(
            "request schema mismatch: approved_target_hashes must be a string list"
        )
    if not isinstance(allowed_kinds, list) or not all(
        isinstance(item, str) and item in _PLUGIN_KIND_OPTIONS for item in allowed_kinds
    ):
        raise click.ClickException(
            "request schema mismatch: allowed_kinds must be valid kind strings"
        )
    normalized_target_hashes = tuple(
        _require_sha256(item, "approved_target_hash") for item in approved_target_hashes
    )

    return PluginExecutionRequest(
        schema="scpn_plugin_runtime_execution_request_v1",
        version=str(version),
        plan_hash=plan_hash,
        approval_hash=approval_hash,
        target_hash=target_hash,
        plugin=str(plugin),
        kind=cast(
            Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
            str(kind),
        ),
        name=str(name),
        operator_identity=str(operator_identity),
        approval_reference=str(approval_reference),
        loading_permitted=bool(loading_permitted),
        execution_permitted=bool(execution_permitted),
        require_target_hash_approval=bool(require_target_hash_approval),
        approved_target_hashes=normalized_target_hashes,
        allowed_kinds=cast(
            tuple[
                Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
                ...,
            ],
            tuple(allowed_kinds),
        ),
        require_package_target=bool(require_package_target),
        audit_record=request_payload,
    )


def _load_revocation_from_payload(
    revocation_payload: dict[str, object],
) -> PluginExecutionRequestRevocation:
    if (
        revocation_payload.get("schema")
        != "scpn_plugin_execution_request_revocation_v1"
    ):
        raise click.ClickException(
            "revocation schema mismatch: expected "
            "scpn_plugin_execution_request_revocation_v1"
        )
    request_hash = _require_sha256(
        revocation_payload.get("request_hash"), "request_hash"
    )
    plan_hash = _require_sha256(revocation_payload.get("plan_hash"), "plan_hash")
    approval_hash = _require_sha256(
        revocation_payload.get("approval_hash"), "approval_hash"
    )
    target_hash = _require_sha256(revocation_payload.get("target_hash"), "target_hash")
    revocation_hash = _require_sha256(
        revocation_payload.get("revocation_hash"), "revocation_hash"
    )
    plugin = revocation_payload.get("plugin")
    kind = revocation_payload.get("kind")
    name = revocation_payload.get("name")
    operator_identity = revocation_payload.get("operator_identity")
    approval_reference = revocation_payload.get("approval_reference")
    revoked_by = revocation_payload.get("revoked_by")
    revocation_reference = revocation_payload.get("revocation_reference")
    revocation_reason = revocation_payload.get("revocation_reason")
    revoked = revocation_payload.get("revoked")
    version = revocation_payload.get("version")

    for field_name, value in (
        ("plugin", plugin),
        ("kind", kind),
        ("name", name),
        ("operator_identity", operator_identity),
        ("approval_reference", approval_reference),
        ("revoked_by", revoked_by),
        ("revocation_reference", revocation_reference),
        ("revocation_reason", revocation_reason),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"revocation schema mismatch: {field_name} must be non-empty"
            )
    if kind not in _PLUGIN_KIND_OPTIONS:
        raise click.ClickException(
            f"revocation schema mismatch: unsupported kind {kind!r}"
        )
    if revoked is not True:
        raise click.ClickException("revocation schema mismatch: revoked must be true")

    return PluginExecutionRequestRevocation(
        schema="scpn_plugin_execution_request_revocation_v1",
        version=str(version),
        request_hash=request_hash,
        plan_hash=plan_hash,
        approval_hash=approval_hash,
        target_hash=target_hash,
        plugin=str(plugin),
        kind=cast(
            Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
            str(kind),
        ),
        name=str(name),
        operator_identity=str(operator_identity),
        approval_reference=str(approval_reference),
        revoked_by=str(revoked_by),
        revocation_reference=str(revocation_reference),
        revocation_reason=str(revocation_reason),
        revoked=True,
        revocation_hash=revocation_hash,
        audit_record=revocation_payload,
    )


def _load_revocation_list_from_payload(
    revocation_list_payload: dict[str, object],
) -> PluginExecutionRequestRevocationList:
    if (
        revocation_list_payload.get("schema")
        != "scpn_plugin_execution_request_revocation_list_v1"
    ):
        raise click.ClickException(
            "revocation list schema mismatch: expected "
            "scpn_plugin_execution_request_revocation_list_v1"
        )
    request_hashes = revocation_list_payload.get("request_hashes")
    revocation_hashes = revocation_list_payload.get("revocation_hashes")
    revocation_count = revocation_list_payload.get("revocation_count")
    created_by = revocation_list_payload.get("created_by")
    revocation_list_hash = _require_sha256(
        revocation_list_payload.get("revocation_list_hash"),
        "revocation_list_hash",
    )
    version = revocation_list_payload.get("version")
    if not isinstance(version, str) or not version:
        raise click.ClickException("revocation list version must be non-empty")
    if not isinstance(created_by, str) or not created_by:
        raise click.ClickException("revocation list created_by must be non-empty")
    if not isinstance(revocation_count, int) or revocation_count < 1:
        raise click.ClickException(
            "revocation list revocation_count must be a positive integer"
        )
    if not isinstance(request_hashes, list) or not all(
        isinstance(item, str) for item in request_hashes
    ):
        raise click.ClickException(
            "revocation list request_hashes must be a string list"
        )
    if not isinstance(revocation_hashes, list) or not all(
        isinstance(item, str) for item in revocation_hashes
    ):
        raise click.ClickException(
            "revocation list revocation_hashes must be a string list"
        )
    normalized_request_hashes = tuple(
        _require_sha256(item, "revoked request hash") for item in request_hashes
    )
    normalized_revocation_hashes = tuple(
        _require_sha256(item, "revocation hash") for item in revocation_hashes
    )
    return validate_plugin_execution_request_revocation_list(
        PluginExecutionRequestRevocationList(
            schema="scpn_plugin_execution_request_revocation_list_v1",
            version=version,
            request_hashes=normalized_request_hashes,
            revocation_hashes=normalized_revocation_hashes,
            revocation_count=revocation_count,
            created_by=created_by,
            revocation_list_hash=revocation_list_hash,
            audit_record=revocation_list_payload,
        )
    )


def _load_storage_manifest_from_payload(
    manifest_payload: dict[str, object],
) -> PluginExecutionRequestStorageManifest:
    if (
        manifest_payload.get("schema")
        != "scpn_plugin_execution_request_storage_manifest_v1"
    ):
        raise click.ClickException(
            "storage manifest schema mismatch: expected "
            "scpn_plugin_execution_request_storage_manifest_v1"
        )
    request_hash = _require_sha256(manifest_payload.get("request_hash"), "request_hash")
    plan_hash = _require_sha256(manifest_payload.get("plan_hash"), "plan_hash")
    approval_hash = _require_sha256(
        manifest_payload.get("approval_hash"), "approval_hash"
    )
    target_hash = _require_sha256(manifest_payload.get("target_hash"), "target_hash")
    revocation_hash = _require_sha256(
        manifest_payload.get("revocation_hash"), "revocation_hash"
    )
    manifest_hash = _require_sha256(
        manifest_payload.get("manifest_hash"), "manifest_hash"
    )
    plugin = manifest_payload.get("plugin")
    kind = manifest_payload.get("kind")
    name = manifest_payload.get("name")
    operator_identity = manifest_payload.get("operator_identity")
    approval_reference = manifest_payload.get("approval_reference")
    storage_uri = manifest_payload.get("storage_uri")
    storage_backend = manifest_payload.get("storage_backend")
    retention_policy = manifest_payload.get("retention_policy")
    created_by = manifest_payload.get("created_by")
    revoked_request_hashes = manifest_payload.get("revoked_request_hashes")
    version = manifest_payload.get("version")
    for field_name, value in (
        ("plugin", plugin),
        ("kind", kind),
        ("name", name),
        ("operator_identity", operator_identity),
        ("approval_reference", approval_reference),
        ("storage_uri", storage_uri),
        ("storage_backend", storage_backend),
        ("retention_policy", retention_policy),
        ("created_by", created_by),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"storage manifest schema mismatch: {field_name} must be non-empty"
            )
    if kind not in _PLUGIN_KIND_OPTIONS:
        raise click.ClickException(
            f"storage manifest schema mismatch: unsupported kind {kind!r}"
        )
    if not isinstance(revoked_request_hashes, list) or not all(
        isinstance(item, str) for item in revoked_request_hashes
    ):
        raise click.ClickException(
            "storage manifest revoked_request_hashes must be a string list"
        )
    normalized_revocations = tuple(
        _require_sha256(item, "revoked request hash") for item in revoked_request_hashes
    )
    return PluginExecutionRequestStorageManifest(
        schema="scpn_plugin_execution_request_storage_manifest_v1",
        version=str(version),
        request_hash=request_hash,
        plan_hash=plan_hash,
        approval_hash=approval_hash,
        target_hash=target_hash,
        plugin=str(plugin),
        kind=cast(
            Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
            str(kind),
        ),
        name=str(name),
        operator_identity=str(operator_identity),
        approval_reference=str(approval_reference),
        storage_uri=str(storage_uri),
        storage_backend=str(storage_backend),
        retention_policy=str(retention_policy),
        created_by=str(created_by),
        revoked_request_hashes=normalized_revocations,
        revocation_hash=revocation_hash,
        manifest_hash=manifest_hash,
        audit_record=manifest_payload,
    )


def _load_lifecycle_from_payload(
    lifecycle_payload: dict[str, object],
) -> PluginExecutionRequestLifecycleRecord:
    if lifecycle_payload.get("schema") != "scpn_plugin_execution_request_lifecycle_v1":
        raise click.ClickException(
            "lifecycle schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_v1"
        )
    request_hash = _require_sha256(
        lifecycle_payload.get("request_hash"), "request_hash"
    )
    lifecycle_hash = _require_sha256(
        lifecycle_payload.get("lifecycle_hash"), "lifecycle_hash"
    )
    status = lifecycle_payload.get("status")
    plugin = lifecycle_payload.get("plugin")
    kind = lifecycle_payload.get("kind")
    name = lifecycle_payload.get("name")
    operator_identity = lifecycle_payload.get("operator_identity")
    approval_reference = lifecycle_payload.get("approval_reference")
    storage_manifest_hash = lifecycle_payload.get("storage_manifest_hash")
    storage_backend = lifecycle_payload.get("storage_backend")
    storage_uri = lifecycle_payload.get("storage_uri")
    revoked = lifecycle_payload.get("revoked")
    revocation_list_hash = lifecycle_payload.get("revocation_list_hash")
    revocation_hash = lifecycle_payload.get("revocation_hash")
    revoked_by = lifecycle_payload.get("revoked_by")
    revocation_reference = lifecycle_payload.get("revocation_reference")
    created_by = lifecycle_payload.get("created_by")
    version = lifecycle_payload.get("version")
    for field_name, value in (
        ("plugin", plugin),
        ("kind", kind),
        ("name", name),
        ("operator_identity", operator_identity),
        ("approval_reference", approval_reference),
        ("created_by", created_by),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"lifecycle schema mismatch: {field_name} must be non-empty"
            )
    if status not in {"approved", "stored", "revoked"}:
        raise click.ClickException("lifecycle schema mismatch: unsupported status")
    if kind not in _PLUGIN_KIND_OPTIONS:
        raise click.ClickException(
            f"lifecycle schema mismatch: unsupported kind {kind!r}"
        )
    if not isinstance(revoked, bool):
        raise click.ClickException("lifecycle schema mismatch: revoked must be boolean")

    def optional_hash(value: object, field_name: str) -> str | None:
        if value is None:
            return None
        return _require_sha256(value, field_name)

    return PluginExecutionRequestLifecycleRecord(
        schema="scpn_plugin_execution_request_lifecycle_v1",
        version=str(version),
        request_hash=request_hash,
        status=cast(Literal["approved", "stored", "revoked"], str(status)),
        plugin=str(plugin),
        kind=cast(
            Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
            str(kind),
        ),
        name=str(name),
        operator_identity=str(operator_identity),
        approval_reference=str(approval_reference),
        storage_manifest_hash=optional_hash(
            storage_manifest_hash, "storage_manifest_hash"
        ),
        storage_backend=str(storage_backend) if storage_backend is not None else None,
        storage_uri=str(storage_uri) if storage_uri is not None else None,
        revoked=revoked,
        revocation_list_hash=optional_hash(
            revocation_list_hash, "revocation_list_hash"
        ),
        revocation_hash=optional_hash(revocation_hash, "revocation_hash"),
        revoked_by=str(revoked_by) if revoked_by is not None else None,
        revocation_reference=(
            str(revocation_reference) if revocation_reference is not None else None
        ),
        created_by=str(created_by),
        lifecycle_hash=lifecycle_hash,
        audit_record=lifecycle_payload,
    )


def _load_lifecycle_summary_from_payload(
    summary_payload: dict[str, object],
) -> PluginExecutionRequestLifecycleSummary:
    if (
        summary_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_summary_v1"
    ):
        raise click.ClickException(
            "lifecycle summary schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_summary_v1"
        )
    summary_hash = _require_sha256(summary_payload.get("summary_hash"), "summary_hash")
    request_count = summary_payload.get("request_count")
    status_counts = summary_payload.get("status_counts")
    lifecycle_hashes = summary_payload.get("lifecycle_hashes")
    approved_request_hashes = summary_payload.get("approved_request_hashes")
    stored_request_hashes = summary_payload.get("stored_request_hashes")
    revoked_request_hashes = summary_payload.get("revoked_request_hashes")
    storage_missing_request_hashes = summary_payload.get(
        "storage_missing_request_hashes"
    )
    renewal_required_request_hashes = summary_payload.get(
        "renewal_required_request_hashes"
    )
    created_by = summary_payload.get("created_by")
    version = summary_payload.get("version")
    if not isinstance(version, str) or not version:
        raise click.ClickException("lifecycle summary version must be non-empty")
    if not isinstance(created_by, str) or not created_by:
        raise click.ClickException("lifecycle summary created_by must be non-empty")
    if not isinstance(request_count, int) or request_count < 1:
        raise click.ClickException(
            "lifecycle summary request_count must be a positive integer"
        )
    if not isinstance(status_counts, dict) or not all(
        isinstance(key, str) and isinstance(value, int)
        for key, value in status_counts.items()
    ):
        raise click.ClickException("lifecycle summary status_counts is malformed")

    def hash_list(value: object, field_name: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise click.ClickException(
                f"lifecycle summary {field_name} must be a string list"
            )
        return tuple(_require_sha256(item, field_name) for item in value)

    return PluginExecutionRequestLifecycleSummary(
        schema="scpn_plugin_execution_request_lifecycle_summary_v1",
        version=version,
        request_count=request_count,
        status_counts=cast(dict[str, int], status_counts),
        lifecycle_hashes=hash_list(lifecycle_hashes, "lifecycle_hashes"),
        approved_request_hashes=hash_list(
            approved_request_hashes, "approved_request_hashes"
        ),
        stored_request_hashes=hash_list(stored_request_hashes, "stored_request_hashes"),
        revoked_request_hashes=hash_list(
            revoked_request_hashes, "revoked_request_hashes"
        ),
        storage_missing_request_hashes=hash_list(
            storage_missing_request_hashes,
            "storage_missing_request_hashes",
        ),
        renewal_required_request_hashes=hash_list(
            renewal_required_request_hashes,
            "renewal_required_request_hashes",
        ),
        created_by=created_by,
        summary_hash=summary_hash,
        audit_record=summary_payload,
    )


def _load_storage_adapter_from_payload(
    adapter_payload: dict[str, object],
) -> PluginExecutionRequestStorageAdapterManifest:
    if (
        adapter_payload.get("schema")
        != "scpn_plugin_execution_request_storage_adapter_v1"
    ):
        raise click.ClickException(
            "storage adapter schema mismatch: expected "
            "scpn_plugin_execution_request_storage_adapter_v1"
        )
    request_hash = _require_sha256(adapter_payload.get("request_hash"), "request_hash")
    storage_manifest_hash = _require_sha256(
        adapter_payload.get("storage_manifest_hash"),
        "storage_manifest_hash",
    )
    bundle_hash = _require_sha256(adapter_payload.get("bundle_hash"), "bundle_hash")
    adapter_hash = _require_sha256(adapter_payload.get("adapter_hash"), "adapter_hash")
    storage_backend = adapter_payload.get("storage_backend")
    storage_uri = adapter_payload.get("storage_uri")
    storage_scheme = adapter_payload.get("storage_scheme")
    adapter_mode = adapter_payload.get("adapter_mode")
    write_performed = adapter_payload.get("write_performed")
    created_by = adapter_payload.get("created_by")
    version = adapter_payload.get("version")
    for field_name, value in (
        ("storage_backend", storage_backend),
        ("storage_uri", storage_uri),
        ("storage_scheme", storage_scheme),
        ("adapter_mode", adapter_mode),
        ("created_by", created_by),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"storage adapter schema mismatch: {field_name} must be non-empty"
            )
    if not isinstance(write_performed, bool):
        raise click.ClickException(
            "storage adapter schema mismatch: write_performed must be boolean"
        )
    return PluginExecutionRequestStorageAdapterManifest(
        schema="scpn_plugin_execution_request_storage_adapter_v1",
        version=str(version),
        request_hash=request_hash,
        storage_manifest_hash=storage_manifest_hash,
        storage_backend=str(storage_backend),
        storage_uri=str(storage_uri),
        storage_scheme=str(storage_scheme),
        adapter_mode=str(adapter_mode),
        bundle_hash=bundle_hash,
        write_performed=write_performed,
        created_by=str(created_by),
        adapter_hash=adapter_hash,
        audit_record=adapter_payload,
    )


def _load_lifecycle_policy_report_payload(
    policy_payload: dict[str, object],
) -> dict[str, object]:
    if (
        policy_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_policy_v1"
    ):
        raise click.ClickException(
            "lifecycle policy schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_policy_v1"
        )
    _require_sha256(policy_payload.get("summary_hash"), "summary_hash")
    _require_sha256(policy_payload.get("policy_hash"), "policy_hash")
    request_count = policy_payload.get("request_count")
    if not isinstance(request_count, int) or request_count < 1:
        raise click.ClickException(
            "lifecycle policy schema mismatch: request_count must be a positive integer"
        )
    for field_name in (
        "renewal_required_request_hashes",
        "missing_adapter_request_hashes",
        "external_write_followup_request_hashes",
    ):
        value = policy_payload.get(field_name)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise click.ClickException(
                f"lifecycle policy schema mismatch: {field_name} must be a string list"
            )
        for item in value:
            _require_sha256(item, field_name)
    return policy_payload


def _load_lifecycle_multistore_drilldown_payload(
    drilldown_payload: dict[str, object],
) -> dict[str, object]:
    if (
        drilldown_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_multistore_drilldown_v1"
    ):
        raise click.ClickException(
            "multi-store drilldown schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_multistore_drilldown_v1"
        )
    _require_sha256(drilldown_payload.get("drilldown_hash"), "drilldown_hash")
    policy_count = drilldown_payload.get("policy_count")
    stores = drilldown_payload.get("stores")
    global_flagged = drilldown_payload.get("global_flagged_request_hashes")
    if not isinstance(policy_count, int) or policy_count < 1:
        raise click.ClickException(
            "multi-store drilldown schema mismatch: policy_count must be positive"
        )
    if not isinstance(stores, list) or not stores:
        raise click.ClickException(
            "multi-store drilldown schema mismatch: stores must be non-empty list"
        )
    if not isinstance(global_flagged, list) or not all(
        isinstance(item, str) for item in global_flagged
    ):
        raise click.ClickException(
            "multi-store drilldown schema mismatch: global_flagged_request_hashes "
            "must be a string list"
        )
    for item in global_flagged:
        _require_sha256(item, "global_flagged_request_hash")
    for store in stores:
        if not isinstance(store, dict):
            raise click.ClickException(
                "multi-store drilldown schema mismatch: store record must be object"
            )
        _require_sha256(store.get("store_hash"), "store_hash")
        _require_sha256(store.get("policy_hash"), "policy_hash")
        _require_sha256(store.get("summary_hash"), "summary_hash")
        for field_name in (
            "renewal_required_request_hashes",
            "storage_missing_request_hashes",
            "missing_adapter_request_hashes",
            "external_write_followup_request_hashes",
        ):
            value = store.get(field_name)
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise click.ClickException(
                    f"multi-store drilldown schema mismatch: {field_name} "
                    "must be a string list"
                )
            for item in value:
                _require_sha256(item, field_name)
    return drilldown_payload


def _load_lifecycle_remediation_plan_payload(
    plan_payload: dict[str, object],
) -> dict[str, object]:
    if (
        plan_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_plan_v1"
    ):
        raise click.ClickException(
            "remediation plan schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_plan_v1"
        )
    _require_sha256(plan_payload.get("plan_hash"), "plan_hash")
    _require_sha256(plan_payload.get("drilldown_hash"), "drilldown_hash")
    action_count = plan_payload.get("action_count")
    actions = plan_payload.get("actions")
    if not isinstance(action_count, int) or action_count < 0:
        raise click.ClickException(
            "remediation plan schema mismatch: action_count must be non-negative"
        )
    if not isinstance(actions, list):
        raise click.ClickException(
            "remediation plan schema mismatch: actions must be a list"
        )
    if action_count != len(actions):
        raise click.ClickException(
            "remediation plan schema mismatch: action_count does not match actions"
        )
    for action in actions:
        if not isinstance(action, dict):
            raise click.ClickException(
                "remediation plan schema mismatch: action must be an object"
            )
        _require_sha256(action.get("action_hash"), "action_hash")
        _require_sha256(action.get("request_hash"), "request_hash")
        _require_sha256(action.get("store_hash"), "store_hash")
        _require_sha256(action.get("policy_hash"), "policy_hash")
        _require_sha256(action.get("summary_hash"), "summary_hash")
        action_type = action.get("action_type")
        if action_type not in {
            "renew_approval",
            "persist_request",
            "register_storage_adapter",
            "confirm_external_write",
        }:
            raise click.ClickException(
                "remediation plan schema mismatch: unsupported action_type"
            )
        priority = action.get("priority")
        if not isinstance(priority, int) or priority < 1:
            raise click.ClickException(
                "remediation plan schema mismatch: priority must be a positive integer"
            )
    return plan_payload


def _load_lifecycle_remediation_action_status_payload(
    status_payload: dict[str, object],
) -> dict[str, object]:
    if (
        status_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_action_status_v1"
    ):
        raise click.ClickException(
            "remediation action status schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_action_status_v1"
        )
    _require_sha256(status_payload.get("status_hash"), "status_hash")
    _require_sha256(status_payload.get("action_hash"), "action_hash")
    _require_sha256(status_payload.get("plan_hash"), "plan_hash")
    state = status_payload.get("state")
    if state not in {"pending", "in_progress", "completed", "blocked"}:
        raise click.ClickException(
            "remediation action status schema mismatch: unsupported state"
        )
    return status_payload


def _load_lifecycle_remediation_execution_dashboard_payload(
    dashboard_payload: dict[str, object],
) -> dict[str, object]:
    if (
        dashboard_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_execution_dashboard_v1"
    ):
        raise click.ClickException(
            "remediation execution dashboard schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_execution_dashboard_v1"
        )
    _require_sha256(dashboard_payload.get("execution_hash"), "execution_hash")
    _require_sha256(dashboard_payload.get("plan_hash"), "plan_hash")
    action_count = dashboard_payload.get("action_count")
    rows = dashboard_payload.get("rows")
    if not isinstance(action_count, int) or action_count < 0:
        raise click.ClickException(
            "remediation execution dashboard schema mismatch: "
            "action_count must be non-negative"
        )
    if not isinstance(rows, list):
        raise click.ClickException(
            "remediation execution dashboard schema mismatch: rows must be a list"
        )
    if action_count != len(rows):
        raise click.ClickException(
            "remediation execution dashboard schema mismatch: "
            "action_count does not match rows"
        )
    for row in rows:
        if not isinstance(row, dict):
            raise click.ClickException(
                "remediation execution dashboard schema mismatch: row must be object"
            )
        _require_sha256(row.get("action_hash"), "action_hash")
        _require_sha256(row.get("request_hash"), "request_hash")
        state = row.get("state")
        if state not in {"pending", "in_progress", "completed", "blocked"}:
            raise click.ClickException(
                "remediation execution dashboard schema mismatch: unsupported state"
            )
        action_type = row.get("action_type")
        if action_type not in {
            "renew_approval",
            "persist_request",
            "register_storage_adapter",
            "confirm_external_write",
        }:
            raise click.ClickException(
                "remediation execution dashboard schema mismatch: "
                "unsupported action_type"
            )
        priority = row.get("priority")
        if not isinstance(priority, int) or priority < 1:
            raise click.ClickException(
                "remediation execution dashboard schema mismatch: "
                "priority must be a positive integer"
            )
    return dashboard_payload


def _load_lifecycle_remediation_deployment_handoff_payload(
    handoff_payload: dict[str, object],
) -> dict[str, object]:
    if (
        handoff_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_deployment_handoff_v1"
    ):
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_deployment_handoff_v1"
        )
    _require_sha256(handoff_payload.get("handoff_hash"), "handoff_hash")
    _require_sha256(handoff_payload.get("plan_hash"), "plan_hash")
    _require_sha256(handoff_payload.get("execution_hash"), "execution_hash")
    unresolved_count = handoff_payload.get("unresolved_action_count")
    handoff_actions = handoff_payload.get("handoff_actions")
    if not isinstance(unresolved_count, int) or unresolved_count < 0:
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: "
            "unresolved_action_count must be non-negative"
        )
    if not isinstance(handoff_actions, list):
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: "
            "handoff_actions must be a list"
        )
    if unresolved_count != len(handoff_actions):
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: "
            "unresolved_action_count does not match handoff_actions"
        )
    for action in handoff_actions:
        if not isinstance(action, dict):
            raise click.ClickException(
                "remediation deployment handoff schema mismatch: action must be object"
            )
        _require_sha256(action.get("handoff_action_hash"), "handoff_action_hash")
        _require_sha256(action.get("action_hash"), "action_hash")
        _require_sha256(action.get("request_hash"), "request_hash")
        action_type = action.get("action_type")
        if action_type not in {
            "renew_approval",
            "persist_request",
            "register_storage_adapter",
            "confirm_external_write",
        }:
            raise click.ClickException(
                "remediation deployment handoff schema mismatch: unsupported action_type"
            )
        priority = action.get("priority")
        if not isinstance(priority, int) or priority < 1:
            raise click.ClickException(
                "remediation deployment handoff schema mismatch: "
                "priority must be a positive integer"
            )
        template = action.get("deployment_command_template")
        if not isinstance(template, str) or not template:
            raise click.ClickException(
                "remediation deployment handoff schema mismatch: "
                "deployment_command_template must be non-empty"
            )
    return handoff_payload


def _load_lifecycle_remediation_scheduler_queue_payload(
    queue_payload: dict[str, object],
) -> dict[str, object]:
    if (
        queue_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_scheduler_queue_v1"
    ):
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_queue_v1"
        )
    _require_sha256(queue_payload.get("scheduler_hash"), "scheduler_hash")
    _require_sha256(queue_payload.get("plan_hash"), "plan_hash")
    _require_sha256(queue_payload.get("execution_hash"), "execution_hash")
    _require_sha256(queue_payload.get("handoff_hash"), "handoff_hash")
    queue_entry_count = queue_payload.get("queue_entry_count")
    queue_entries = queue_payload.get("queue_entries")
    if not isinstance(queue_entry_count, int) or queue_entry_count < 0:
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: "
            "queue_entry_count must be non-negative"
        )
    if not isinstance(queue_entries, list):
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: queue_entries must be a list"
        )
    if queue_entry_count != len(queue_entries):
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: "
            "queue_entry_count does not match queue_entries"
        )
    for entry in queue_entries:
        if not isinstance(entry, dict):
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: entry must be object"
            )
        _require_sha256(entry.get("entry_hash"), "entry_hash")
        _require_sha256(entry.get("handoff_action_hash"), "handoff_action_hash")
        _require_sha256(entry.get("action_hash"), "action_hash")
        _require_sha256(entry.get("request_hash"), "request_hash")
        action_type = entry.get("action_type")
        if action_type not in {
            "renew_approval",
            "persist_request",
            "register_storage_adapter",
            "confirm_external_write",
        }:
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: unsupported action_type"
            )
        priority = entry.get("priority")
        if not isinstance(priority, int) or priority < 1:
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: "
                "priority must be a positive integer"
            )
        schedule_epoch = entry.get("schedule_epoch")
        if not isinstance(schedule_epoch, int) or schedule_epoch < 0:
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: "
                "schedule_epoch must be non-negative integer"
            )
        template = entry.get("scheduler_command_template")
        if not isinstance(template, str) or not template:
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: "
                "scheduler_command_template must be non-empty"
            )
    return queue_payload


def _load_lifecycle_remediation_scheduler_telemetry_payload(
    telemetry_payload: dict[str, object],
) -> dict[str, object]:
    if (
        telemetry_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_scheduler_telemetry_v1"
    ):
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_telemetry_v1"
        )
    _require_sha256(telemetry_payload.get("telemetry_hash"), "telemetry_hash")
    _require_sha256(telemetry_payload.get("plan_hash"), "plan_hash")
    _require_sha256(telemetry_payload.get("execution_hash"), "execution_hash")
    _require_sha256(telemetry_payload.get("handoff_hash"), "handoff_hash")
    _require_sha256(telemetry_payload.get("scheduler_hash"), "scheduler_hash")
    queue_entry_count = telemetry_payload.get("queue_entry_count")
    rows = telemetry_payload.get("rows")
    if not isinstance(queue_entry_count, int) or queue_entry_count < 0:
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: "
            "queue_entry_count must be non-negative"
        )
    if not isinstance(rows, list):
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: rows must be a list"
        )
    if queue_entry_count != len(rows):
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: "
            "queue_entry_count does not match rows"
        )
    return telemetry_payload


def _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
    handoff_payload: dict[str, object],
) -> dict[str, object]:
    if (
        handoff_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_scheduler_adapter_handoff_v1"
    ):
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_adapter_handoff_v1"
        )
    _require_sha256(handoff_payload.get("adapter_handoff_hash"), "adapter_handoff_hash")
    _require_sha256(handoff_payload.get("telemetry_hash"), "telemetry_hash")
    _require_sha256(handoff_payload.get("plan_hash"), "plan_hash")
    _require_sha256(handoff_payload.get("execution_hash"), "execution_hash")
    entries = handoff_payload.get("entries")
    entry_count = handoff_payload.get("entry_count")
    if not isinstance(entry_count, int) or entry_count < 0:
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: "
            "entry_count must be non-negative"
        )
    if not isinstance(entries, list):
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: entries must be list"
        )
    if entry_count != len(entries):
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: "
            "entry_count does not match entries"
        )
    for entry in entries:
        if not isinstance(entry, dict):
            raise click.ClickException(
                "remediation scheduler adapter handoff schema mismatch: entry must be object"
            )
        _require_sha256(entry.get("adapter_entry_hash"), "adapter_entry_hash")
        _require_sha256(entry.get("entry_hash"), "entry_hash")
        _require_sha256(entry.get("action_hash"), "action_hash")
        _require_sha256(entry.get("request_hash"), "request_hash")
    return handoff_payload


def _build_plugin_execution_request(
    plan: PluginExecutionPlan,
    approval: PluginExecutionApproval,
) -> object:
    builder_candidates = (
        "build_plugin_execution_request",
        "build_plugin_execution_request_from_approval",
        "build_plugin_execution_request_from_plan_and_approval",
    )
    for name in builder_candidates:
        for module in (plugin_registry, plugin_api):
            candidate = getattr(module, name, None)
            if not callable(candidate):
                continue
            try:
                return candidate(plan, approval)
            except TypeError:
                pass
            try:
                return candidate(plan=plan, approval=approval)
            except TypeError:
                pass
            try:
                return candidate(plan=plan, approved_execution=approval)
            except TypeError:
                pass
            try:
                return candidate(plan=plan, approval_record=approval)
            except TypeError:
                pass

    raise click.ClickException(
        "registry request builder not available: expected "
        "build_plugin_execution_request"
    )


@main.group("plugins")
def plugins_group() -> None:
    """Inspect extension plugin manifests."""


@plugins_group.command("catalog")
@click.option(
    "--include-incompatible",
    is_flag=True,
    help="Include incompatible manifests and rejection reasons in the output",
)
@click.option(
    "--rust-registry",
    is_flag=True,
    help="Emit flattened Rust-facing capability registry JSON",
)
@click.option(
    "--rust-runtime-handoff",
    is_flag=True,
    help="Emit guarded Rust runtime handoff JSON with loading disabled",
)
def plugins_catalog(
    include_incompatible: bool,
    rust_registry: bool,
    rust_runtime_handoff: bool,
) -> None:
    """Print the discovered plugin marketplace catalogue as JSON."""
    if rust_registry and rust_runtime_handoff:
        raise click.ClickException(
            "--rust-registry and --rust-runtime-handoff are mutually exclusive"
        )
    manifests = discover_plugin_manifests()
    if rust_runtime_handoff:
        builder = build_rust_plugin_runtime_handoff
    elif rust_registry:
        builder = build_rust_plugin_registry
    else:
        builder = build_plugin_marketplace_catalog
    catalog = builder(manifests, include_incompatible=include_incompatible)
    click.echo(json.dumps(catalog, indent=2, sort_keys=True))


@plugins_group.command("plan-execution")
@click.argument("plugin_name")
@click.argument("kind", type=click.Choice(_PLUGIN_KIND_OPTIONS))
@click.argument("capability_name")
@click.option(
    "--approved-target-hash",
    "approved_target_hashes",
    multiple=True,
    help="Approved runtime target hash(es) for this execution planning decision.",
)
@click.option(
    "--require-target-hash-approval",
    is_flag=True,
    help="Fail unless the discovered capability target hash is approved.",
)
def plugins_plan_execution(
    plugin_name: str,
    kind: str,
    capability_name: str,
    approved_target_hashes: tuple[str, ...],
    require_target_hash_approval: bool,
) -> None:
    """Emit a non-executing plan for a discovered plugin capability."""
    manifests = discover_plugin_manifests()
    manifest = _find_discovered_plugin(manifests, plugin_name)
    compatibility = compatibility_report(manifest)
    capability = _find_capability(manifest, kind, capability_name)
    normalized_hashes = _normalize_approved_target_hashes(approved_target_hashes)

    try:
        plan = build_plugin_execution_plan(
            manifest,
            capability.kind,
            capability_name,
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                approved_target_hashes=normalized_hashes,
                require_target_hash_approval=require_target_hash_approval,
            ),
        )
    except (LookupError, PermissionError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    payload = {
        **plan.audit_record,
        "manifest": manifest.to_audit_record(),
        "capability": {
            "kind": capability.kind,
            "name": capability.name,
            "target": capability.target,
            "version": capability.version,
            "channels": list(capability.channels),
            "knobs": list(capability.knobs),
        },
        "compatible": compatibility.compatible,
        "compatibility_reasons": list(compatibility.reasons),
    }
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("approve-execution-plan")
@click.argument(
    "plan_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--operator-id",
    required=True,
    type=str,
    help="Operator identity approving the plan",
)
@click.option(
    "--approval-reference",
    required=True,
    type=str,
    help="Reference for the approval decision",
)
@click.option(
    "--approval-reason",
    required=True,
    type=str,
    help="Human reason for this approval",
)
def plugins_approve_execution_plan(
    plan_json: Path,
    operator_id: str,
    approval_reference: str,
    approval_reason: str,
) -> None:
    """Emit a deterministic operator approval artefact for a stored execution plan."""
    if not operator_id:
        raise click.ClickException("operator identity is required")
    if not approval_reference:
        raise click.ClickException("approval reference is required")
    if not approval_reason:
        raise click.ClickException("approval reason is required")

    plan_payload = _load_json_file(plan_json)
    plan, _audit_record = _load_plan_from_payload(plan_payload)
    try:
        approval = build_plugin_execution_approval(
            plan,
            operator_identity=operator_id,
            approval_reference=approval_reference,
            approval_reason=approval_reason,
        )
    except (LookupError, PermissionError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(approval.audit_record, indent=2, sort_keys=True))


@plugins_group.command("request-execution")
@click.argument(
    "plan_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "approval_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def plugins_request_execution(plan_json: Path, approval_json: Path) -> None:
    """Emit a deterministic execution request from a stored plan and approval."""
    plan_payload = _load_json_file(plan_json, artifact="plan")
    plan, _ = _load_plan_from_payload(plan_payload)
    approval_payload = _load_json_file(approval_json, artifact="approval")
    approval = _load_approval_from_payload(approval_payload)

    if plan.plan_hash != approval.plan_hash:
        raise click.ClickException("plan hash mismatch")
    if plan.target_hash != approval.target_hash:
        raise click.ClickException("target hash mismatch")
    if approval.plugin != plan.manifest.name:
        raise click.ClickException("plugin mismatch between plan and approval")
    if approval.kind != plan.capability.kind:
        raise click.ClickException("kind mismatch between plan and approval")
    if approval.name != plan.capability.name:
        raise click.ClickException("name mismatch between plan and approval")
    if not approval.approved:
        raise click.ClickException("approval is not approved")
    if approval.approved is not True or approval.execution_permitted is not True:
        raise click.ClickException("approval does not permit execution")

    try:
        request = _build_plugin_execution_request(plan, approval)
    except (PermissionError, TypeError, ValueError, KeyError, LookupError) as exc:
        raise click.ClickException(str(exc)) from exc

    if isinstance(request, PluginExecutionApproval):
        payload = request.audit_record
    else:
        payload = cast(dict[str, object], getattr(request, "audit_record", request))
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("persist-execution-request")
@click.argument(
    "request_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--storage-uri",
    required=True,
    help="Deployment-owned URI for the persisted request bundle.",
)
@click.option(
    "--storage-backend",
    default="local_file",
    show_default=True,
    help="Storage backend identifier; local writes require local_file.",
)
@click.option(
    "--retention-policy",
    default="retain_until_revoked",
    show_default=True,
    help="Retention policy identifier for the request bundle.",
)
@click.option(
    "--created-by",
    required=True,
    help="Deployment component creating the request bundle.",
)
@click.option(
    "--revoked-request-hash",
    "revoked_request_hashes",
    multiple=True,
    help="Revoked request hash to bind into the storage manifest.",
)
@click.option(
    "--revocation-list",
    "revocation_list_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Aggregate revocation-list JSON to bind into the storage manifest.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Allow replacing an existing local request bundle.",
)
def plugins_persist_execution_request(
    request_json: Path,
    output_path: Path,
    storage_uri: str,
    storage_backend: str,
    retention_policy: str,
    created_by: str,
    revoked_request_hashes: tuple[str, ...],
    revocation_list_path: Path | None,
    overwrite: bool,
) -> None:
    """Persist a validated execution request as a local storage bundle."""
    request_payload = _load_json_file(request_json, artifact="request")
    request = _load_request_from_payload(request_payload)
    direct_revocations = _normalize_approved_target_hashes(revoked_request_hashes)
    revocation_list_hashes: tuple[str, ...] = ()

    try:
        if revocation_list_path is not None:
            revocation_list = _load_revocation_list_from_payload(
                _load_json_file(revocation_list_path, artifact="revocation list")
            )
            revocation_list_hashes = revocation_list.as_revoked_request_hashes()
        normalized_revocations = tuple(
            dict.fromkeys((*direct_revocations, *revocation_list_hashes))
        )
        storage_manifest = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri=storage_uri,
            storage_backend=storage_backend,
            retention_policy=retention_policy,
            created_by=created_by,
            revoked_request_hashes=normalized_revocations,
        )
        bundle = write_plugin_execution_request_storage_bundle(
            request,
            storage_manifest,
            output_path,
            overwrite=overwrite,
        )
    except (OSError, PermissionError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(bundle, indent=2, sort_keys=True))


@plugins_group.command("storage-adapter-manifest")
@click.argument(
    "request_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--storage-uri",
    required=True,
    help="Deployment-owned URI for the request storage target.",
)
@click.option(
    "--storage-backend",
    required=True,
    help=(
        "Storage backend identifier: local_file, s3_object, gcs_object, "
        "azure_blob, oci_object, or https_api."
    ),
)
@click.option(
    "--retention-policy",
    default="retain_until_revoked",
    show_default=True,
    help="Retention policy identifier for the request bundle.",
)
@click.option(
    "--created-by",
    required=True,
    help="Deployment component creating the adapter manifest.",
)
@click.option(
    "--revoked-request-hash",
    "revoked_request_hashes",
    multiple=True,
    help="Revoked request hash to bind into the storage manifest.",
)
@click.option(
    "--revocation-list",
    "revocation_list_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Aggregate revocation-list JSON to bind into the storage manifest.",
)
def plugins_storage_adapter_manifest(
    request_json: Path,
    storage_uri: str,
    storage_backend: str,
    retention_policy: str,
    created_by: str,
    revoked_request_hashes: tuple[str, ...],
    revocation_list_path: Path | None,
) -> None:
    """Emit a deterministic storage-adapter handoff manifest without writing."""
    request_payload = _load_json_file(request_json, artifact="request")
    request = _load_request_from_payload(request_payload)
    direct_revocations = _normalize_approved_target_hashes(revoked_request_hashes)
    revocation_list_hashes: tuple[str, ...] = ()

    try:
        if revocation_list_path is not None:
            revocation_list = _load_revocation_list_from_payload(
                _load_json_file(revocation_list_path, artifact="revocation list")
            )
            revocation_list_hashes = revocation_list.as_revoked_request_hashes()
        normalized_revocations = tuple(
            dict.fromkeys((*direct_revocations, *revocation_list_hashes))
        )
        storage_manifest = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri=storage_uri,
            storage_backend=storage_backend,
            retention_policy=retention_policy,
            created_by=created_by,
            revoked_request_hashes=normalized_revocations,
        )
        adapter_manifest = build_plugin_execution_request_storage_adapter_manifest(
            request,
            storage_manifest,
        )
    except (PermissionError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(adapter_manifest.audit_record, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-status")
@click.argument(
    "request_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--storage-bundle",
    "storage_bundle_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Persisted request bundle JSON to include storage status.",
)
@click.option(
    "--revocation-list",
    "revocation_list_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Aggregate revocation-list JSON to include lifecycle status.",
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the lifecycle status.",
)
def plugins_lifecycle_status(
    request_json: Path,
    storage_bundle_path: Path | None,
    revocation_list_path: Path | None,
    created_by: str,
) -> None:
    """Emit an operator lifecycle status record for an execution request."""
    request_payload = _load_json_file(request_json, artifact="request")
    request = _load_request_from_payload(request_payload)
    storage_manifest: PluginExecutionRequestStorageManifest | None = None
    revocation_list: PluginExecutionRequestRevocationList | None = None

    try:
        if storage_bundle_path is not None:
            bundle = _load_json_file(storage_bundle_path, artifact="storage bundle")
            manifest_payload = bundle.get("storage_manifest")
            if not isinstance(manifest_payload, dict):
                raise click.ClickException(
                    "storage bundle storage_manifest must be an object"
                )
            storage_manifest = _load_storage_manifest_from_payload(
                cast(dict[str, object], manifest_payload)
            )
        if revocation_list_path is not None:
            revocation_list = _load_revocation_list_from_payload(
                _load_json_file(revocation_list_path, artifact="revocation list")
            )
        lifecycle_record = build_plugin_execution_request_lifecycle_record(
            request,
            created_by=created_by,
            storage_manifest=storage_manifest,
            revocation_list=revocation_list,
        )
    except (PermissionError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(lifecycle_record.audit_record, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-summary")
@click.argument(
    "lifecycle_json",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the lifecycle summary.",
)
def plugins_lifecycle_summary(
    lifecycle_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit a deterministic batch summary for lifecycle-status records."""
    lifecycle_records = tuple(
        _load_lifecycle_from_payload(_load_json_file(path, artifact="lifecycle"))
        for path in lifecycle_json
    )
    try:
        summary = build_plugin_execution_request_lifecycle_summary(
            lifecycle_records,
            created_by=created_by,
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(summary.audit_record, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-policy-report")
@click.argument(
    "summary_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--storage-adapter",
    "storage_adapter_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Storage-adapter manifest JSON to include in the policy report.",
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the policy report.",
)
def plugins_lifecycle_policy_report(
    summary_json: Path,
    storage_adapter_paths: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit a deterministic lifecycle policy report for operator dashboards."""
    summary = _load_lifecycle_summary_from_payload(
        _load_json_file(summary_json, artifact="lifecycle summary")
    )
    storage_adapters = tuple(
        _load_storage_adapter_from_payload(
            _load_json_file(path, artifact="storage adapter")
        )
        for path in storage_adapter_paths
    )
    try:
        report = build_plugin_execution_request_lifecycle_policy_report(
            summary,
            storage_adapters=storage_adapters,
            created_by=created_by,
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(report.audit_record, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-renewal-queue")
@click.argument(
    "summary_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--policy-report",
    "policy_report_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Lifecycle policy report JSON to add adapter/write follow-up queues.",
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the renewal queue.",
)
def plugins_lifecycle_renewal_queue(
    summary_json: Path,
    policy_report_path: Path | None,
    created_by: str,
) -> None:
    """Emit a deterministic renewal/follow-up queue for lifecycle operations."""
    summary = _load_lifecycle_summary_from_payload(
        _load_json_file(summary_json, artifact="lifecycle summary")
    )
    policy_payload: dict[str, object] | None = None
    if policy_report_path is not None:
        policy_payload = _load_lifecycle_policy_report_payload(
            _load_json_file(policy_report_path, artifact="lifecycle policy")
        )
        if policy_payload["summary_hash"] != summary.summary_hash:
            raise click.ClickException(
                "lifecycle policy summary_hash does not match lifecycle summary"
            )
        if policy_payload["request_count"] != summary.request_count:
            raise click.ClickException(
                "lifecycle policy request_count does not match lifecycle summary"
            )

    renewal_hashes = tuple(sorted(summary.renewal_required_request_hashes))
    storage_missing_hashes = tuple(sorted(summary.storage_missing_request_hashes))
    missing_adapter_hashes: tuple[str, ...] = ()
    external_followup_hashes: tuple[str, ...] = ()
    if policy_payload is not None:
        missing_adapter_hashes = tuple(
            sorted(
                cast(
                    list[str],
                    policy_payload["missing_adapter_request_hashes"],
                )
            )
        )
        external_followup_hashes = tuple(
            sorted(
                cast(
                    list[str],
                    policy_payload["external_write_followup_request_hashes"],
                )
            )
        )

    queue_payload: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_lifecycle_renewal_queue_v1",
        "version": "1.0.0",
        "summary_hash": summary.summary_hash,
        "request_count": summary.request_count,
        "renewal_required_request_hashes": list(renewal_hashes),
        "storage_missing_request_hashes": list(storage_missing_hashes),
        "missing_adapter_request_hashes": list(missing_adapter_hashes),
        "external_write_followup_request_hashes": list(external_followup_hashes),
        "created_by": created_by,
    }
    queue_payload["queue_hash"] = _record_hash(queue_payload)
    click.echo(json.dumps(queue_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-multistore-dashboard")
@click.argument(
    "policy_json",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the multi-store dashboard.",
)
def plugins_lifecycle_multistore_dashboard(
    policy_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit a deterministic aggregate dashboard across policy reports."""
    if not created_by:
        raise click.ClickException(
            "multi-store dashboard schema mismatch: created_by must be non-empty"
        )
    policies = tuple(
        _load_lifecycle_policy_report_payload(
            _load_json_file(path, artifact="lifecycle policy")
        )
        for path in policy_json
    )
    policy_hashes = tuple(
        sorted(
            _require_sha256(policy["policy_hash"], "policy_hash")
            for policy in policies
        )
    )
    if len(set(policy_hashes)) != len(policy_hashes):
        raise click.ClickException("duplicate lifecycle policy hash")
    summary_hashes = tuple(
        sorted(
            _require_sha256(policy["summary_hash"], "summary_hash")
            for policy in policies
        )
    )
    unique_requests: set[str] = set()
    action_totals: dict[str, int] = {
        "confirm_external_write": 0,
        "persist_request": 0,
        "register_storage_adapter": 0,
        "renew_approval": 0,
    }
    renewal_required: set[str] = set()
    storage_missing: set[str] = set()
    missing_adapters: set[str] = set()
    external_followup: set[str] = set()

    for policy in policies:
        request_count = policy["request_count"]
        if not isinstance(request_count, int):
            raise click.ClickException(
                "lifecycle policy schema mismatch: request_count must be an integer"
            )
        unique_requests.update(
            cast(list[str], policy["renewal_required_request_hashes"])
        )
        unique_requests.update(
            cast(list[str], policy["storage_missing_request_hashes"])
        )
        unique_requests.update(
            cast(list[str], policy["missing_adapter_request_hashes"])
        )
        unique_requests.update(
            cast(list[str], policy["external_write_followup_request_hashes"])
        )
        policy_actions = policy.get("policy_action_counts")
        if not isinstance(policy_actions, dict):
            raise click.ClickException(
                "lifecycle policy schema mismatch: policy_action_counts is malformed"
            )
        for key in action_totals:
            value = policy_actions.get(key, 0)
            if not isinstance(value, int) or value < 0:
                raise click.ClickException(
                    "lifecycle policy schema mismatch: "
                    "policy_action_counts is malformed"
                )
            action_totals[key] += value
        renewal_required.update(
            cast(list[str], policy["renewal_required_request_hashes"])
        )
        storage_missing.update(
            cast(list[str], policy["storage_missing_request_hashes"])
        )
        missing_adapters.update(
            cast(list[str], policy["missing_adapter_request_hashes"])
        )
        external_followup.update(
            cast(list[str], policy["external_write_followup_request_hashes"])
        )

    dashboard_payload: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_lifecycle_multistore_dashboard_v1",
        "version": "1.0.0",
        "policy_count": len(policies),
        "policy_hashes": list(policy_hashes),
        "summary_hashes": list(summary_hashes),
        "aggregated_policy_action_counts": action_totals,
        "renewal_required_request_hashes": sorted(renewal_required),
        "storage_missing_request_hashes": sorted(storage_missing),
        "missing_adapter_request_hashes": sorted(missing_adapters),
        "external_write_followup_request_hashes": sorted(external_followup),
        "unique_flagged_request_count": len(unique_requests),
        "created_by": created_by,
    }
    dashboard_payload["dashboard_hash"] = _record_hash(dashboard_payload)
    click.echo(json.dumps(dashboard_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-multistore-drilldown")
@click.argument(
    "policy_json",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the cross-store drill-down.",
)
def plugins_lifecycle_multistore_drilldown(
    policy_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit deterministic per-store lifecycle queues with provenance hashes."""
    if not created_by:
        raise click.ClickException(
            "multi-store drilldown schema mismatch: created_by must be non-empty"
        )
    policies = tuple(
        _load_lifecycle_policy_report_payload(
            _load_json_file(path, artifact="lifecycle policy")
        )
        for path in policy_json
    )
    if not policies:
        raise click.ClickException(
            "multi-store drilldown requires at least one policy report"
        )
    policy_hashes = tuple(
        sorted(
            _require_sha256(policy["policy_hash"], "policy_hash")
            for policy in policies
        )
    )
    if len(set(policy_hashes)) != len(policy_hashes):
        raise click.ClickException("duplicate lifecycle policy hash")

    per_store: list[dict[str, object]] = []
    global_requests: set[str] = set()
    for policy in sorted(
        policies,
        key=lambda item: str(item["policy_hash"]),
    ):
        policy_hash = _require_sha256(policy["policy_hash"], "policy_hash")
        summary_hash = _require_sha256(policy["summary_hash"], "summary_hash")
        request_count = policy["request_count"]
        if not isinstance(request_count, int) or request_count < 1:
            raise click.ClickException(
                "lifecycle policy schema mismatch: "
                "request_count must be a positive integer"
            )
        status_counts = policy.get("status_counts")
        action_counts = policy.get("policy_action_counts")
        if not isinstance(status_counts, dict) or not isinstance(action_counts, dict):
            raise click.ClickException(
                "lifecycle policy schema mismatch: status/action counts are malformed"
            )
        store_payload: dict[str, object] = {
            "policy_hash": policy_hash,
            "summary_hash": summary_hash,
            "request_count": request_count,
            "status_counts": dict(cast(dict[str, int], status_counts)),
            "policy_action_counts": dict(cast(dict[str, int], action_counts)),
            "renewal_required_request_hashes": sorted(
                cast(list[str], policy["renewal_required_request_hashes"])
            ),
            "storage_missing_request_hashes": sorted(
                cast(list[str], policy["storage_missing_request_hashes"])
            ),
            "missing_adapter_request_hashes": sorted(
                cast(list[str], policy["missing_adapter_request_hashes"])
            ),
            "external_write_followup_request_hashes": sorted(
                cast(list[str], policy["external_write_followup_request_hashes"])
            ),
        }
        store_payload["store_hash"] = _record_hash(store_payload)
        global_requests.update(
            cast(list[str], store_payload["renewal_required_request_hashes"])
        )
        global_requests.update(
            cast(list[str], store_payload["storage_missing_request_hashes"])
        )
        global_requests.update(
            cast(list[str], store_payload["missing_adapter_request_hashes"])
        )
        global_requests.update(
            cast(
                list[str],
                store_payload["external_write_followup_request_hashes"],
            )
        )
        per_store.append(store_payload)

    drilldown_payload: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_lifecycle_multistore_drilldown_v1",
        "version": "1.0.0",
        "policy_count": len(per_store),
        "policy_hashes": list(policy_hashes),
        "stores": per_store,
        "global_flagged_request_hashes": sorted(global_requests),
        "global_flagged_request_count": len(global_requests),
        "created_by": created_by,
    }
    drilldown_payload["drilldown_hash"] = _record_hash(drilldown_payload)
    click.echo(json.dumps(drilldown_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-orchestration")
@click.argument(
    "drilldown_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the remediation plan.",
)
def plugins_lifecycle_remediation_orchestration(
    drilldown_json: Path,
    created_by: str,
) -> None:
    """Emit a deterministic, priority-ordered cross-store remediation plan."""
    if not created_by:
        raise click.ClickException(
            "remediation orchestration schema mismatch: created_by must be non-empty"
        )
    drilldown = _load_lifecycle_multistore_drilldown_payload(
        _load_json_file(drilldown_json, artifact="multi-store drilldown")
    )
    stores = cast(list[dict[str, object]], drilldown["stores"])
    actions: list[dict[str, object]] = []
    priority_map: dict[str, int] = {
        "renew_approval": 1,
        "persist_request": 2,
        "register_storage_adapter": 3,
        "confirm_external_write": 4,
    }
    for store in stores:
        store_hash = _require_sha256(store.get("store_hash"), "store_hash")
        policy_hash = _require_sha256(store.get("policy_hash"), "policy_hash")
        summary_hash = _require_sha256(store.get("summary_hash"), "summary_hash")
        for action_type, source_field in (
            ("renew_approval", "renewal_required_request_hashes"),
            ("persist_request", "storage_missing_request_hashes"),
            ("register_storage_adapter", "missing_adapter_request_hashes"),
            ("confirm_external_write", "external_write_followup_request_hashes"),
        ):
            request_hashes = cast(list[str], store[source_field])
            for request_hash in request_hashes:
                action: dict[str, object] = {
                    "action_type": action_type,
                    "priority": priority_map[action_type],
                    "request_hash": request_hash,
                    "store_hash": store_hash,
                    "policy_hash": policy_hash,
                    "summary_hash": summary_hash,
                }
                action["action_hash"] = _record_hash(action)
                actions.append(action)
    actions.sort(
        key=lambda item: (
            cast(int, item["priority"]),
            str(item["request_hash"]),
            str(item["store_hash"]),
            str(item["action_type"]),
        )
    )
    orchestration_payload: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_lifecycle_remediation_plan_v1",
        "version": "1.0.0",
        "drilldown_hash": _require_sha256(
            drilldown.get("drilldown_hash"),
            "drilldown_hash",
        ),
        "action_count": len(actions),
        "actions": actions,
        "created_by": created_by,
    }
    orchestration_payload["plan_hash"] = _record_hash(orchestration_payload)
    click.echo(json.dumps(orchestration_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-action-status")
@click.argument(
    "plan_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument("action_hash")
@click.option(
    "--state",
    type=click.Choice(["pending", "in_progress", "completed", "blocked"]),
    required=True,
    help="Execution state for the remediation action.",
)
@click.option(
    "--updated-by",
    required=True,
    help="Operator or deployment component updating the action state.",
)
@click.option(
    "--note",
    default="",
    show_default=False,
    help="Optional operator note for this state transition.",
)
def plugins_lifecycle_remediation_action_status(
    plan_json: Path,
    action_hash: str,
    state: str,
    updated_by: str,
    note: str,
) -> None:
    """Emit a deterministic remediation action status record."""
    if not updated_by:
        raise click.ClickException(
            "remediation action status schema mismatch: updated_by must be non-empty"
        )
    action_hash = _require_sha256(action_hash, "action_hash")
    plan = _load_lifecycle_remediation_plan_payload(
        _load_json_file(plan_json, artifact="remediation plan")
    )
    actions = cast(list[dict[str, object]], plan["actions"])
    selected: dict[str, object] | None = None
    for action in actions:
        if action["action_hash"] == action_hash:
            selected = action
            break
    if selected is None:
        raise click.ClickException(
            "action_hash is not part of the remediation plan"
        )
    status_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_action_status_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(plan["plan_hash"], "plan_hash"),
        "action_hash": action_hash,
        "request_hash": selected["request_hash"],
        "store_hash": selected["store_hash"],
        "action_type": selected["action_type"],
        "priority": selected["priority"],
        "state": state,
        "updated_by": updated_by,
        "note": note,
    }
    status_payload["status_hash"] = _record_hash(status_payload)
    click.echo(json.dumps(status_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-execution-dashboard")
@click.argument(
    "plan_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "status_json",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the execution dashboard.",
)
def plugins_lifecycle_remediation_execution_dashboard(
    plan_json: Path,
    status_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit a deterministic closed-loop dashboard for remediation execution."""
    if not created_by:
        raise click.ClickException(
            "remediation dashboard schema mismatch: created_by must be non-empty"
        )
    plan = _load_lifecycle_remediation_plan_payload(
        _load_json_file(plan_json, artifact="remediation plan")
    )
    statuses = tuple(
        _load_lifecycle_remediation_action_status_payload(
            _load_json_file(path, artifact="remediation action status")
        )
        for path in status_json
    )
    plan_hash = _require_sha256(plan["plan_hash"], "plan_hash")
    actions = cast(list[dict[str, object]], plan["actions"])
    action_by_hash = {
        _require_sha256(action["action_hash"], "action_hash"): action
        for action in actions
    }
    status_by_hash: dict[str, dict[str, object]] = {}
    for status_record in statuses:
        status_plan_hash = _require_sha256(status_record["plan_hash"], "plan_hash")
        if status_plan_hash != plan_hash:
            raise click.ClickException(
                "status plan_hash does not match remediation plan"
            )
        action_hash = _require_sha256(status_record["action_hash"], "action_hash")
        if action_hash not in action_by_hash:
            raise click.ClickException(
                "status action_hash is not part of remediation plan"
            )
        if action_hash in status_by_hash:
            raise click.ClickException(
                "duplicate remediation action status for action_hash"
            )
        status_by_hash[action_hash] = status_record
    state_counts = {
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
        "blocked": 0,
    }
    unresolved: list[str] = []
    resolved: list[str] = []
    execution_rows: list[dict[str, object]] = []
    for action in sorted(
        actions,
        key=lambda item: (
            cast(int, item["priority"]),
            str(item["request_hash"]),
            str(item["action_hash"]),
        ),
    ):
        action_hash = _require_sha256(action["action_hash"], "action_hash")
        status = status_by_hash.get(action_hash)
        if status is None:
            state = "pending"
            status_hash: str | None = None
            updated_by: str | None = None
            note = ""
        else:
            state = cast(str, status["state"])
            status_hash = _require_sha256(status["status_hash"], "status_hash")
            updated_by = str(status["updated_by"])
            note = str(status.get("note", ""))
        state_counts[state] += 1
        if state in {"completed"}:
            resolved.append(action_hash)
        else:
            unresolved.append(action_hash)
        execution_rows.append(
            {
                "action_hash": action_hash,
                "request_hash": action["request_hash"],
                "action_type": action["action_type"],
                "priority": action["priority"],
                "state": state,
                "status_hash": status_hash,
                "updated_by": updated_by,
                "note": note,
            }
        )
    dashboard_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_execution_dashboard_v1"
        ),
        "version": "1.0.0",
        "plan_hash": plan_hash,
        "action_count": len(actions),
        "state_counts": state_counts,
        "resolved_action_hashes": sorted(resolved),
        "unresolved_action_hashes": sorted(unresolved),
        "rows": execution_rows,
        "created_by": created_by,
    }
    dashboard_payload["execution_hash"] = _record_hash(dashboard_payload)
    click.echo(json.dumps(dashboard_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-deployment-handoff")
@click.argument(
    "execution_dashboard_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the deployment handoff.",
)
def plugins_lifecycle_remediation_deployment_handoff(
    execution_dashboard_json: Path,
    created_by: str,
) -> None:
    """Emit deterministic deployment handoff actions for unresolved remediation."""
    if not created_by:
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: "
            "created_by must be non-empty"
        )
    dashboard = _load_lifecycle_remediation_execution_dashboard_payload(
        _load_json_file(
            execution_dashboard_json,
            artifact="remediation execution dashboard",
        )
    )
    plan_hash = _require_sha256(dashboard.get("plan_hash"), "plan_hash")
    rows = cast(list[dict[str, object]], dashboard["rows"])
    unresolved_rows = [
        row for row in rows if row["state"] in {"pending", "in_progress", "blocked"}
    ]
    command_templates = {
        "renew_approval": (
            "spo plugins approve-execution-plan PLAN_JSON "
            "--operator-id OPERATOR_ID --approval-reference REF "
            "--approval-reason REASON"
        ),
        "persist_request": (
            "spo plugins persist-execution-request REQUEST_JSON OUTPUT_JSON "
            "--storage-uri STORAGE_URI --created-by DEPLOYMENT_COMPONENT"
        ),
        "register_storage_adapter": (
            "spo plugins storage-adapter-manifest REQUEST_JSON "
            "--storage-uri STORAGE_URI --storage-backend BACKEND "
            "--created-by DEPLOYMENT_COMPONENT"
        ),
        "confirm_external_write": (
            "Record external storage/API write completion and emit "
            "spo plugins lifecycle-remediation-action-status PLAN_JSON ACTION_HASH "
            "--state completed --updated-by DEPLOYMENT_COMPONENT"
        ),
    }
    handoff_actions: list[dict[str, object]] = []
    for row in sorted(
        unresolved_rows,
        key=lambda item: (
            cast(int, item["priority"]),
            str(item["action_hash"]),
        ),
    ):
        action_type = cast(str, row["action_type"])
        handoff_action: dict[str, object] = {
            "action_hash": row["action_hash"],
            "request_hash": row["request_hash"],
            "action_type": action_type,
            "priority": row["priority"],
            "state": row["state"],
            "deployment_command_template": command_templates[action_type],
        }
        handoff_action["handoff_action_hash"] = _record_hash(handoff_action)
        handoff_actions.append(handoff_action)
    handoff_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_deployment_handoff_v1"
        ),
        "version": "1.0.0",
        "plan_hash": plan_hash,
        "execution_hash": _require_sha256(
            dashboard.get("execution_hash"), "execution_hash"
        ),
        "unresolved_action_count": len(unresolved_rows),
        "handoff_actions": handoff_actions,
        "created_by": created_by,
    }
    handoff_payload["handoff_hash"] = _record_hash(handoff_payload)
    click.echo(json.dumps(handoff_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-queue")
@click.argument(
    "handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--window-start-epoch",
    required=True,
    type=int,
    help="Scheduler window start as Unix epoch seconds (UTC).",
)
@click.option(
    "--window-duration-seconds",
    default=3600,
    show_default=True,
    type=int,
    help="Scheduler execution window length in seconds.",
)
@click.option(
    "--created-by",
    required=True,
    help="Scheduler component creating the queue payload.",
)
def plugins_lifecycle_remediation_scheduler_queue(
    handoff_json: Path,
    window_start_epoch: int,
    window_duration_seconds: int,
    created_by: str,
) -> None:
    """Emit a deterministic scheduler queue from remediation deployment handoff."""
    if not created_by:
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: created_by must be non-empty"
        )
    if window_start_epoch < 0:
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: "
            "window_start_epoch must be non-negative"
        )
    if window_duration_seconds < 1:
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: "
            "window_duration_seconds must be positive"
        )
    handoff = _load_lifecycle_remediation_deployment_handoff_payload(
        _load_json_file(handoff_json, artifact="remediation deployment handoff")
    )
    actions = cast(list[dict[str, object]], handoff["handoff_actions"])
    if len(actions) > window_duration_seconds:
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: unresolved action count "
            "exceeds scheduler window duration"
        )
    queue_entries: list[dict[str, object]] = []
    for index, action in enumerate(
        sorted(
            actions,
            key=lambda item: (
                cast(int, item["priority"]),
                str(item["handoff_action_hash"]),
            ),
        )
    ):
        schedule_epoch = window_start_epoch + index
        entry: dict[str, object] = {
            "handoff_action_hash": action["handoff_action_hash"],
            "action_hash": action["action_hash"],
            "request_hash": action["request_hash"],
            "action_type": action["action_type"],
            "priority": action["priority"],
            "schedule_epoch": schedule_epoch,
            "scheduler_command_template": action["deployment_command_template"],
        }
        entry["entry_hash"] = _record_hash(entry)
        queue_entries.append(entry)
    queue_payload: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_lifecycle_remediation_scheduler_queue_v1",
        "version": "1.0.0",
        "plan_hash": _require_sha256(handoff.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            handoff.get("execution_hash"),
            "execution_hash",
        ),
        "handoff_hash": _require_sha256(handoff.get("handoff_hash"), "handoff_hash"),
        "window_start_epoch": window_start_epoch,
        "window_duration_seconds": window_duration_seconds,
        "queue_entry_count": len(queue_entries),
        "queue_entries": queue_entries,
        "created_by": created_by,
    }
    queue_payload["scheduler_hash"] = _record_hash(queue_payload)
    click.echo(json.dumps(queue_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-telemetry")
@click.argument(
    "scheduler_queue_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "action_status_json",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--as-of-epoch",
    required=True,
    type=int,
    help="Telemetry snapshot epoch seconds (UTC).",
)
@click.option(
    "--created-by",
    required=True,
    help="Scheduler component creating telemetry payload.",
)
def plugins_lifecycle_remediation_scheduler_telemetry(
    scheduler_queue_json: Path,
    action_status_json: tuple[Path, ...],
    as_of_epoch: int,
    created_by: str,
) -> None:
    """Emit deterministic operator telemetry for scheduler remediation queue."""
    if not created_by:
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: "
            "created_by must be non-empty"
        )
    if as_of_epoch < 0:
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: "
            "as_of_epoch must be non-negative"
        )
    queue = _load_lifecycle_remediation_scheduler_queue_payload(
        _load_json_file(scheduler_queue_json, artifact="remediation scheduler queue")
    )
    status_by_action_hash: dict[str, dict[str, object]] = {}
    for path in action_status_json:
        status = _load_lifecycle_remediation_action_status_payload(
            _load_json_file(path, artifact="remediation action status")
        )
        action_hash = status.action_hash
        if action_hash in status_by_action_hash:
            raise click.ClickException(
                "remediation scheduler telemetry schema mismatch: "
                "duplicate action status action_hash"
            )
        status_by_action_hash[action_hash] = {
            "state": status.state,
            "status_hash": status.status_hash,
            "updated_by": status.updated_by,
            "note": status.note,
        }

    queue_entries = cast(list[dict[str, object]], queue["queue_entries"])
    state_counts: dict[str, int] = {
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
        "blocked": 0,
        "overdue": 0,
    }
    rows: list[dict[str, object]] = []
    overdue_action_hashes: list[str] = []
    for entry in sorted(
        queue_entries,
        key=lambda item: (
            cast(int, item["schedule_epoch"]),
            cast(int, item["priority"]),
            str(item["action_hash"]),
        ),
    ):
        action_hash = _require_sha256(entry.get("action_hash"), "action_hash")
        schedule_epoch = cast(int, entry["schedule_epoch"])
        status_record = status_by_action_hash.get(action_hash)
        if status_record is None:
            state = "pending"
            status_hash: str | None = None
            updated_by: str | None = None
            note = ""
        else:
            state = cast(str, status_record["state"])
            status_hash = _require_sha256(status_record["status_hash"], "status_hash")
            updated_by = cast(str, status_record["updated_by"])
            note = cast(str, status_record["note"])
        overdue = state in {"pending", "in_progress", "blocked"} and (
            schedule_epoch < as_of_epoch
        )
        state_counts[state] += 1
        if overdue:
            state_counts["overdue"] += 1
            overdue_action_hashes.append(action_hash)
        row: dict[str, object] = {
            "entry_hash": entry["entry_hash"],
            "handoff_action_hash": entry["handoff_action_hash"],
            "action_hash": action_hash,
            "request_hash": entry["request_hash"],
            "action_type": entry["action_type"],
            "priority": entry["priority"],
            "schedule_epoch": schedule_epoch,
            "state": state,
            "overdue": overdue,
            "status_hash": status_hash,
            "updated_by": updated_by,
            "note": note,
        }
        rows.append(row)
    telemetry_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_telemetry_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(queue.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(queue.get("execution_hash"), "execution_hash"),
        "handoff_hash": _require_sha256(queue.get("handoff_hash"), "handoff_hash"),
        "scheduler_hash": _require_sha256(queue.get("scheduler_hash"), "scheduler_hash"),
        "as_of_epoch": as_of_epoch,
        "queue_entry_count": len(queue_entries),
        "state_counts": state_counts,
        "overdue_action_hashes": sorted(overdue_action_hashes),
        "rows": rows,
        "created_by": created_by,
    }
    telemetry_payload["telemetry_hash"] = _record_hash(telemetry_payload)
    click.echo(json.dumps(telemetry_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-adapter-handoff")
@click.argument(
    "scheduler_telemetry_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--adapter-name",
    required=True,
    help="External scheduler adapter name.",
)
@click.option(
    "--adapter-endpoint",
    required=True,
    help="External scheduler adapter endpoint identifier.",
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating adapter handoff payload.",
)
def plugins_lifecycle_remediation_scheduler_adapter_handoff(
    scheduler_telemetry_json: Path,
    adapter_name: str,
    adapter_endpoint: str,
    created_by: str,
) -> None:
    """Emit deterministic external scheduler adapter handoff payload."""
    if not created_by:
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: "
            "created_by must be non-empty"
        )
    if not adapter_name:
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: "
            "adapter_name must be non-empty"
        )
    if not adapter_endpoint:
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: "
            "adapter_endpoint must be non-empty"
        )
    telemetry = _load_lifecycle_remediation_scheduler_telemetry_payload(
        _load_json_file(
            scheduler_telemetry_json,
            artifact="remediation scheduler telemetry",
        )
    )
    rows = cast(list[dict[str, object]], telemetry["rows"])
    active_rows = [
        row for row in rows if cast(str, row["state"]) in {"pending", "in_progress", "blocked"}
    ]
    entries: list[dict[str, object]] = []
    for row in sorted(
        active_rows,
        key=lambda item: (
            cast(int, item["priority"]),
            cast(int, item["schedule_epoch"]),
            str(item["action_hash"]),
        ),
    ):
        entry: dict[str, object] = {
            "entry_hash": row["entry_hash"],
            "handoff_action_hash": row["handoff_action_hash"],
            "action_hash": row["action_hash"],
            "request_hash": row["request_hash"],
            "action_type": row["action_type"],
            "priority": row["priority"],
            "schedule_epoch": row["schedule_epoch"],
            "overdue": row["overdue"],
            "adapter_target": {
                "adapter_name": adapter_name,
                "adapter_endpoint": adapter_endpoint,
            },
            "acknowledgement_command_template": (
                "spo plugins lifecycle-remediation-scheduler-acknowledgement "
                "ADAPTER_HANDOFF_JSON ENTRY_HASH --state STATE "
                "--acknowledged-by OPERATOR --external-reference REF"
            ),
        }
        entry["adapter_entry_hash"] = _record_hash(entry)
        entries.append(entry)
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_adapter_handoff_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(telemetry.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(telemetry.get("execution_hash"), "execution_hash"),
        "telemetry_hash": _require_sha256(telemetry.get("telemetry_hash"), "telemetry_hash"),
        "adapter_name": adapter_name,
        "adapter_endpoint": adapter_endpoint,
        "entry_count": len(entries),
        "entries": entries,
        "created_by": created_by,
    }
    payload["adapter_handoff_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-acknowledgement")
@click.argument(
    "adapter_handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument("entry_hash")
@click.option(
    "--state",
    required=True,
    type=click.Choice(["in_progress", "completed", "blocked"]),
    help="External scheduler execution state.",
)
@click.option(
    "--acknowledged-by",
    required=True,
    help="Actor or component acknowledging execution.",
)
@click.option(
    "--external-reference",
    required=True,
    help="External scheduler job/task reference.",
)
@click.option(
    "--note",
    default="",
    show_default=True,
    help="Optional acknowledgement note.",
)
def plugins_lifecycle_remediation_scheduler_acknowledgement(
    adapter_handoff_json: Path,
    entry_hash: str,
    state: str,
    acknowledged_by: str,
    external_reference: str,
    note: str,
) -> None:
    """Emit deterministic acknowledgement artifact for adapter execution."""
    if not acknowledged_by:
        raise click.ClickException(
            "remediation scheduler acknowledgement schema mismatch: "
            "acknowledged_by must be non-empty"
        )
    if not external_reference:
        raise click.ClickException(
            "remediation scheduler acknowledgement schema mismatch: "
            "external_reference must be non-empty"
        )
    handoff = _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
        _load_json_file(
            adapter_handoff_json,
            artifact="remediation scheduler adapter handoff",
        )
    )
    normalized_entry_hash = _require_sha256(entry_hash, "entry_hash")
    entries = cast(list[dict[str, object]], handoff["entries"])
    matched = next(
        (entry for entry in entries if entry["adapter_entry_hash"] == normalized_entry_hash),
        None,
    )
    if matched is None:
        raise click.ClickException(
            "remediation scheduler acknowledgement schema mismatch: "
            "entry_hash not present in adapter handoff"
        )
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_acknowledgement_v1"
        ),
        "version": "1.0.0",
        "adapter_handoff_hash": _require_sha256(
            handoff.get("adapter_handoff_hash"), "adapter_handoff_hash"
        ),
        "telemetry_hash": _require_sha256(handoff.get("telemetry_hash"), "telemetry_hash"),
        "plan_hash": _require_sha256(handoff.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(handoff.get("execution_hash"), "execution_hash"),
        "adapter_entry_hash": normalized_entry_hash,
        "entry_hash": matched["entry_hash"],
        "action_hash": matched["action_hash"],
        "request_hash": matched["request_hash"],
        "state": state,
        "acknowledged_by": acknowledged_by,
        "external_reference": external_reference,
        "note": note,
    }
    payload["acknowledgement_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("revoke-execution-request")
@click.argument(
    "request_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--revoked-by",
    required=True,
    help="Operator or deployment component revoking the request.",
)
@click.option(
    "--revocation-reference",
    required=True,
    help="Reference for the revocation decision.",
)
@click.option(
    "--revocation-reason",
    required=True,
    help="Human reason for revoking the request.",
)
def plugins_revoke_execution_request(
    request_json: Path,
    revoked_by: str,
    revocation_reference: str,
    revocation_reason: str,
) -> None:
    """Emit a deterministic revocation artefact for an execution request."""
    request_payload = _load_json_file(request_json, artifact="request")
    request = _load_request_from_payload(request_payload)

    try:
        revocation = build_plugin_execution_request_revocation(
            request,
            revoked_by=revoked_by,
            revocation_reference=revocation_reference,
            revocation_reason=revocation_reason,
        )
    except (PermissionError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(revocation.audit_record, indent=2, sort_keys=True))


@plugins_group.command("revocation-list")
@click.argument(
    "revocation_json",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Deployment component creating the revocation list.",
)
def plugins_revocation_list(
    revocation_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit a deterministic aggregate revocation list."""
    revocations = tuple(
        _load_revocation_from_payload(_load_json_file(path, artifact="revocation"))
        for path in revocation_json
    )

    try:
        revocation_list = build_plugin_execution_request_revocation_list(
            revocations,
            created_by=created_by,
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(revocation_list.audit_record, indent=2, sort_keys=True))


@main.command("meta-transfer-manifest")
@click.argument(
    "audit_paths",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--audit-directory",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Nested audit-history directory to discover with --pattern.",
)
@click.option(
    "--pattern",
    default="**/*.jsonl",
    show_default=True,
    help="Glob pattern used with --audit-directory.",
)
@click.option("--min-records", default=1, show_default=True, type=int)
@click.option("--package-name", default="scpn-meta", show_default=True)
@click.option(
    "--import-target",
    default="scpn_phase_orchestrator.meta",
    show_default=True,
)
@click.option("--console-script", default="scpn-meta", show_default=True)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Write manifest JSON to a file instead of stdout.",
)
def meta_transfer_manifest(
    audit_paths: tuple[str, ...],
    audit_directory: str | None,
    pattern: str,
    min_records: int,
    package_name: str,
    import_target: str,
    console_script: str,
    output: str | None,
) -> None:
    """Emit a review-only meta-transfer package manifest from audit history."""
    if min_records < 1:
        raise click.ClickException("--min-records must be at least 1")
    if audit_directory is None and not audit_paths:
        raise click.ClickException(
            "provide one or more audit JSONL files or --audit-directory"
        )
    if audit_directory is not None and audit_paths:
        raise click.ClickException(
            "audit JSONL files and --audit-directory are mutually exclusive"
        )
    try:
        if audit_directory is not None:
            model = CrossDomainMetaTransfer.fit_audit_directory(
                audit_directory,
                pattern=pattern,
                min_records=min_records,
            )
        else:
            model = CrossDomainMetaTransfer.fit_audit_history(
                audit_paths,
                min_records=min_records,
            )
        manifest = model.to_package_manifest(
            package_name=package_name,
            import_target=import_target,
            console_script=console_script,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    text = json.dumps(manifest.to_audit_record(), indent=2, sort_keys=True) + "\n"
    if output is None:
        click.echo(text, nl=False)
        return
    Path(output).write_text(text, encoding="utf-8")
    click.echo(f"Meta-transfer package manifest written: {output}")


def _parse_checker_path_overrides(
    checker_paths: tuple[str, ...],
) -> dict[str, str | None]:
    overrides: dict[str, str | None] = {}
    for item in checker_paths:
        if "=" not in item:
            raise click.ClickException(
                "--checker-path entries must use executable=/path syntax"
            )
        executable, path = item.split("=", 1)
        executable = executable.strip()
        if not executable:
            raise click.ClickException("--checker-path executable must not be empty")
        overrides[executable] = path.strip() or None
    return overrides


@main.command("formal-export")
@click.argument("binding_spec", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Write formal model to a file instead of stdout",
)
@click.option("--module-name", default="spo_petri", help="Formal module name")
@click.option("--max-tokens", default=None, type=int, help="Token upper bound")
@click.option(
    "--export",
    "export_target",
    type=click.Choice(
        ["protocol", "protocol-tla", "policy", "policy-tla", "stl", "package"]
    ),
    default="protocol",
    show_default=True,
    help="Supervisor artefact to export",
)
@click.option(
    "--policy",
    "policy_path",
    default=None,
    type=click.Path(exists=True),
    help="Policy YAML path for --export policy/stl; defaults to sibling policy.yaml",
)
@click.option(
    "--include-checker-readiness",
    is_flag=True,
    help="Add non-executing PRISM/TLC executable-readiness records to package JSON",
)
@click.option(
    "--checker-path",
    "checker_paths",
    multiple=True,
    help=(
        "Deterministic checker resolver override for package readiness, formatted "
        "as executable=/path or executable= to force missing"
    ),
)
def formal_export(
    binding_spec: str,
    output: str | None,
    module_name: str,
    max_tokens: int | None,
    export_target: str,
    policy_path: str | None,
    include_checker_readiness: bool,
    checker_paths: tuple[str, ...],
) -> None:
    """Export supervisor artefacts for formal model checking."""
    if include_checker_readiness and export_target != "package":
        click.echo(
            "ERROR: --include-checker-readiness is only valid with --export package",
            err=True,
        )
        raise SystemExit(1)
    if checker_paths and not include_checker_readiness:
        click.echo(
            "ERROR: --checker-path requires --include-checker-readiness",
            err=True,
        )
        raise SystemExit(1)
    spec_path = Path(binding_spec)
    spec = load_binding_spec(spec_path)
    errors = validate_binding_spec(spec)
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)

    if export_target in {"policy", "policy-tla", "stl", "package"}:
        policy_file = (
            Path(policy_path)
            if policy_path is not None
            else spec_path.parent / "policy.yaml"
        )
        if not policy_file.exists():
            click.echo(f"ERROR: policy file not found: {policy_file}", err=True)
            raise SystemExit(1)
        if export_target == "stl":
            stl_specs = load_policy_stl_specs(policy_file)
            if not stl_specs:
                click.echo("ERROR: policy file contains no stl_monitors", err=True)
                raise SystemExit(1)
            export = export_stl_specs_prism(stl_specs, module_name=module_name)
            if output is None:
                click.echo(export.model, nl=False)
                return
            Path(output).write_text(export.model, encoding="utf-8")
            click.echo(f"PRISM model written: {output}")
            return
        rules = load_policy_rules(policy_file)
        if not rules:
            click.echo("ERROR: policy file contains no rules", err=True)
            raise SystemExit(1)
        if export_target == "package":
            if spec.protocol_net is None:
                click.echo("ERROR: binding spec has no protocol_net", err=True)
                raise SystemExit(1)
            net, marking = _petri_net_from_protocol(spec.protocol_net)
            petri_prism = export_petri_net_prism(
                net,
                marking,
                module_name=f"{module_name}_protocol",
                max_tokens=max_tokens,
            )
            petri_tla = export_petri_net_tla(
                net,
                marking,
                module_name=f"{module_name}_protocol_tla",
                max_tokens=max_tokens,
            )
            policy_prism = export_policy_rules_prism(
                rules,
                module_name=f"{module_name}_policy",
            )
            package = build_formal_verification_package(
                {
                    "protocol_prism": petri_prism,
                    "protocol_tla": petri_tla,
                    "policy_prism": policy_prism,
                },
                (
                    FormalSafetyProperty(
                        name="protocol_type_ok",
                        artifact_name="protocol_tla",
                        checker="tlc",
                        expression="Safety",
                        description="Protocol state variables remain bounded.",
                    ),
                    FormalSafetyProperty(
                        name="protocol_reachable_terminal",
                        artifact_name="protocol_prism",
                        checker="prism",
                        expression='P>=1 [ F "active_done" ]',
                        description="Protocol can reach the terminal place.",
                    ),
                    FormalSafetyProperty(
                        name="policy_rule_review",
                        artifact_name="policy_prism",
                        checker="prism",
                        expression="P>=0 [ F true ]",
                        description="Policy artefact is available for review.",
                    ),
                ),
                package_name=module_name,
            )
            payload = package.to_audit_record()
            if include_checker_readiness:
                payload["checker_availability"] = [
                    record.to_audit_record()
                    for record in audit_formal_checker_availability(
                        package,
                        executable_paths=_parse_checker_path_overrides(checker_paths)
                        if checker_paths
                        else None,
                    )
                ]
            text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            if output is None:
                click.echo(text, nl=False)
                return
            Path(output).write_text(text, encoding="utf-8")
            click.echo(f"Formal verification package written: {output}")
            return
        if export_target == "policy-tla":
            tla_export = export_policy_rules_tla(rules, module_name=module_name)
            if output is None:
                click.echo(tla_export.module, nl=False)
                return
            Path(output).write_text(tla_export.module, encoding="utf-8")
            click.echo(f"TLA+ model written: {output}")
            return
        export = export_policy_rules_prism(rules, module_name=module_name)
        if output is None:
            click.echo(export.model, nl=False)
            return
        Path(output).write_text(export.model, encoding="utf-8")
        click.echo(f"PRISM model written: {output}")
        return

    if spec.protocol_net is None:
        click.echo("ERROR: binding spec has no protocol_net", err=True)
        raise SystemExit(1)

    net, marking = _petri_net_from_protocol(spec.protocol_net)
    if export_target == "protocol-tla":
        tla_export = export_petri_net_tla(
            net,
            marking,
            module_name=module_name,
            max_tokens=max_tokens,
        )
        if output is None:
            click.echo(tla_export.module, nl=False)
            return
        Path(output).write_text(tla_export.module, encoding="utf-8")
        click.echo(f"TLA+ model written: {output}")
        return
    export = export_petri_net_prism(
        net,
        marking,
        module_name=module_name,
        max_tokens=max_tokens,
    )
    if output is None:
        click.echo(export.model, nl=False)
        return
    Path(output).write_text(export.model, encoding="utf-8")
    click.echo(f"PRISM model written: {output}")


def _policy_report_dict(report: PolicyDryRunReport) -> dict[str, object]:
    return {
        "steps": report.steps,
        "rules": list(report.rules),
        "fire_counts": report.fire_counts,
        "action_counts": report.action_counts,
        "unreachable_rules": list(report.unreachable_rules),
        "overlapping_steps": list(report.overlapping_steps),
        "action_collision_steps": list(report.action_collision_steps),
        "step_reports": [
            {
                "step": step.step,
                "regime": step.regime,
                "fired_rules": list(step.fired_rules),
                "actions": list(step.actions),
            }
            for step in report.step_reports
        ],
    }


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _float_list(value: object) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value if isinstance(item, int | float)]
    return []


def _float_value(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _int_value(value: object) -> int:
    if isinstance(value, int):
        return value
    return 0


def _count_dict(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        if isinstance(raw_count, int):
            counts[str(key)] = raw_count
    return counts


def _parse_dependency_locks(values: tuple[str, ...]) -> dict[str, str]:
    locks: dict[str, str] = {}
    for raw_value in values:
        label, separator, digest = raw_value.partition(":")
        if not label or not separator or not digest:
            raise click.ClickException(
                "--dependency-lock values must use '<label>:<digest>' format"
            )
        locks[label] = digest
    if not locks:
        raise click.ClickException("at least one --dependency-lock is required")
    return locks


def _write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _supervisor_default_scenario_config() -> dict[str, object]:
    return {
        "n_oscillators": 4,
        "phases": [0.0, 0.1, 2.7, 3.1],
        "omegas": [0.04, 0.03, -0.03, -0.04],
        "base_coupling_off_diagonal": 0.03,
        "good_mask": [1.0, 1.0, 0.0, 0.0],
        "bad_mask": [0.0, 0.0, 1.0, 1.0],
        "dt": 0.05,
        "inner_steps": 4,
        "horizon": 6,
    }


def _supervisor_float_list(record: dict[str, object], field: str) -> list[float]:
    value = record.get(field)
    if not isinstance(value, list) or not value:
        raise click.ClickException(f"scenario {field} must be a non-empty list")
    values: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise click.ClickException(f"scenario {field}[{index}] must be numeric")
        values.append(float(item))
    return values


def _supervisor_positive_float(record: dict[str, object], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise click.ClickException(f"scenario {field} must be a positive number")
    return float(value)


def _supervisor_positive_int(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise click.ClickException(f"scenario {field} must be a positive integer")
    return value


def _supervisor_scenario_config_from_record(
    record: dict[str, object],
) -> dict[str, object]:
    phases = _supervisor_float_list(record, "phases")
    n_oscillators = len(phases)
    normalized: dict[str, object] = {
        "n_oscillators": n_oscillators,
        "phases": phases,
        "omegas": _supervisor_float_list(record, "omegas"),
        "base_coupling_off_diagonal": _supervisor_positive_float(
            record,
            "base_coupling_off_diagonal",
        ),
        "good_mask": _supervisor_float_list(record, "good_mask"),
        "bad_mask": _supervisor_float_list(record, "bad_mask"),
        "dt": _supervisor_positive_float(record, "dt"),
        "inner_steps": _supervisor_positive_int(record, "inner_steps"),
        "horizon": _supervisor_positive_int(record, "horizon"),
    }
    for field in ("omegas", "good_mask", "bad_mask"):
        values = normalized[field]
        if not isinstance(values, list) or len(values) != n_oscillators:
            raise click.ClickException(
                f"scenario {field} length must match phases length"
            )
    return normalized


def _load_supervisor_scenario_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return _supervisor_default_scenario_config()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"invalid scenario JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException("scenario JSON must be an object")
    return _supervisor_scenario_config_from_record(payload)


@main.command("supervisor-baseline-experiment")
@click.option(
    "--scenario-json",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional scenario JSON overriding the built-in deterministic fixture.",
)
@click.option(
    "--config-json",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write deterministic experiment configuration JSON.",
)
@click.option(
    "--metrics-jsonl",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write one baseline comparison audit record per JSONL line.",
)
@click.option(
    "--summary-json",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write reproducible baseline summary table as JSON.",
)
@click.option(
    "--manifest-json",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional path for the reproducibility manifest JSON.",
)
@click.option(
    "--checkpoint-manifest",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Existing checkpoint manifest JSON to reference in reproducibility output.",
)
@click.option(
    "--plot-manifest",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Existing plot manifest JSON to reference in reproducibility output.",
)
@click.option("--git-sha", required=True, help="Git revision used for this run.")
@click.option(
    "--seed",
    "seeds",
    multiple=True,
    type=int,
    required=True,
    help="Non-negative deterministic seed; may be passed more than once.",
)
@click.option(
    "--dependency-lock",
    "dependency_locks",
    multiple=True,
    required=True,
    help="Dependency lock provenance as '<label>:<digest>'; may repeat.",
)
@click.option("--json-out", is_flag=True, help="Emit manifest JSON to stdout.")
@click.pass_context
def supervisor_baseline_experiment(
    ctx: click.Context,
    scenario_json: Path | None,
    config_json: Path,
    metrics_jsonl: Path,
    summary_json: Path,
    manifest_json: Path | None,
    checkpoint_manifest: Path | None,
    plot_manifest: Path | None,
    git_sha: str,
    seeds: tuple[int, ...],
    dependency_locks: tuple[str, ...],
    json_out: bool,
) -> None:
    """Materialise deterministic neural-supervisor baseline audit artifacts."""
    try:
        import jax
        import jax.numpy as jnp

        from scpn_phase_orchestrator.nn.supervisor import (
            DifferentiableSupervisorConfig,
            DifferentiableSupervisorPolicy,
            KuramotoSupervisorScenario,
            build_supervisor_baseline_report,
            build_supervisor_experiment_manifest,
            compare_supervisor_hand_tuned_baseline,
            compare_supervisor_random_baseline,
            compare_supervisor_static_baseline,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without NN deps
        raise click.ClickException(
            "supervisor baseline experiments require the optional NN/JAX stack"
        ) from exc

    dependency_lock = _parse_dependency_locks(dependency_locks)
    if any(seed < 0 for seed in seeds):
        raise click.ClickException("--seed values must be non-negative")

    scenario_config = _load_supervisor_scenario_config(scenario_json)
    n_oscillators = cast(int, scenario_config["n_oscillators"])
    phases = jnp.array(cast(list[float], scenario_config["phases"]))
    base_k = jnp.full(
        (n_oscillators, n_oscillators),
        cast(float, scenario_config["base_coupling_off_diagonal"]),
    )
    base_k = base_k.at[jnp.diag_indices(n_oscillators)].set(0.0)
    scenario = KuramotoSupervisorScenario(
        phases=phases,
        omegas=jnp.array(cast(list[float], scenario_config["omegas"])),
        base_K=base_k,
        good_mask=jnp.array(cast(list[float], scenario_config["good_mask"])),
        bad_mask=jnp.array(cast(list[float], scenario_config["bad_mask"])),
        dt=cast(float, scenario_config["dt"]),
        inner_steps=cast(int, scenario_config["inner_steps"]),
        horizon=cast(int, scenario_config["horizon"]),
    )
    config = DifferentiableSupervisorConfig(
        n_oscillators=n_oscillators,
        hidden_width=8,
        hidden_depth=1,
    )
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(seeds[0]))
    comparisons = (
        compare_supervisor_static_baseline(
            policy,
            scenario,
            comparison_label="cli_static_zero_action",
        ),
        compare_supervisor_random_baseline(
            policy,
            scenario,
            key=jax.random.PRNGKey(seeds[0] + 1),
            comparison_label="cli_bounded_random_action",
        ),
        compare_supervisor_hand_tuned_baseline(
            policy,
            scenario,
            comparison_label="cli_hand_tuned_supervisor_policy",
        ),
    )
    baseline_report = build_supervisor_baseline_report(
        comparisons,
        report_label="cli_supervisor_baseline_report",
    )
    baseline_record = baseline_report.to_audit_record()
    comparison_records = [comparison.to_audit_record() for comparison in comparisons]
    config_payload = {
        "proposal_type": "supervisor_baseline_experiment_config",
        "actuation_permitted": False,
        "policy_config": {
            "n_oscillators": config.n_oscillators,
            "hidden_width": config.hidden_width,
            "hidden_depth": config.hidden_depth,
            "n_layer_controls": config.n_layer_controls,
            "max_global_delta_K": config.max_global_delta_K,
            "max_global_delta_zeta": config.max_global_delta_zeta,
            "max_layer_delta_K": config.max_layer_delta_K,
            "control_energy_weight": config.control_energy_weight,
            "bad_sync_weight": config.bad_sync_weight,
            "smoothness_weight": config.smoothness_weight,
        },
        "scenario": scenario_config,
        "comparisons": [
            "cli_static_zero_action",
            "cli_bounded_random_action",
            "cli_hand_tuned_supervisor_policy",
        ],
    }
    summary_payload = {
        "proposal_type": "supervisor_baseline_summary_table",
        "comparison_count": len(comparison_records),
        "actuation_permitted": False,
        "metric_record_path": str(metrics_jsonl),
        "summary": baseline_record["summary"],
        "report_label": baseline_record["report_label"],
    }
    manifest = build_supervisor_experiment_manifest(
        baseline_report,
        command=ctx.command_path,
        git_sha=git_sha,
        dependency_lock=dependency_lock,
        device_info={
            "jax_default_backend": str(jax.default_backend()),
            "jax_enable_x64": str(getattr(jax.config, "jax_enable_x64", "unknown")),
        },
        seed_list=seeds,
        config_json_path=str(config_json),
        metrics_jsonl_path=str(metrics_jsonl),
        summary_table_path=str(summary_json),
        checkpoint_manifest_path=(
            str(checkpoint_manifest) if checkpoint_manifest is not None else None
        ),
        plot_manifest_path=str(plot_manifest) if plot_manifest is not None else None,
    )
    manifest_record = manifest.to_audit_record()

    metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    metrics_jsonl.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in comparison_records)
        + "\n",
        encoding="utf-8",
    )
    _write_json_file(config_json, config_payload)
    _write_json_file(summary_json, summary_payload)
    if manifest_json is not None:
        _write_json_file(manifest_json, manifest_record)
    if json_out:
        click.echo(json.dumps(manifest_record, indent=2, sort_keys=True))
    else:
        click.echo(f"Wrote supervisor config: {config_json}")
        click.echo(f"Wrote supervisor metrics: {metrics_jsonl}")
        click.echo(f"Wrote supervisor summary: {summary_json}")
        if manifest_json is not None:
            click.echo(f"Wrote supervisor manifest: {manifest_json}")


@main.command("policy-dry-run")
@click.argument("binding_spec", type=click.Path(exists=True))
@click.argument("audit_log", type=click.Path(exists=True))
@click.option(
    "--policy",
    "policy_path",
    default=None,
    type=click.Path(exists=True),
    help="Policy YAML path; defaults to binding spec sibling policy.yaml",
)
@click.option("--json-out", is_flag=True, help="Output JSON instead of text")
def policy_dry_run(
    binding_spec: str,
    audit_log: str,
    policy_path: str | None,
    json_out: bool,
) -> None:
    """Replay policy rules against an audit log without applying actuation."""
    spec_path = Path(binding_spec)
    spec = load_binding_spec(spec_path)
    errors = validate_binding_spec(spec)
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)

    policy_file = (
        Path(policy_path)
        if policy_path is not None
        else spec_path.parent / "policy.yaml"
    )
    if not policy_file.exists():
        click.echo(f"ERROR: policy file not found: {policy_file}", err=True)
        raise SystemExit(1)
    rules = load_policy_rules(policy_file)
    if not rules:
        click.echo("ERROR: policy file contains no rules", err=True)
        raise SystemExit(1)

    entries = ReplayEngine(audit_log).load()
    report = dry_run_policy_rules(
        rules,
        entries,
        good_layers=list(spec.objectives.good_layers),
        bad_layers=list(spec.objectives.bad_layers),
    )
    if json_out:
        click.echo(json.dumps(_policy_report_dict(report), indent=2, sort_keys=True))
        return

    click.echo(f"Steps: {report.steps}  Rules: {len(report.rules)}")
    click.echo("Rule fires:")
    for rule in report.rules:
        click.echo(f"  {rule}: {report.fire_counts.get(rule, 0)}")
    if report.unreachable_rules:
        click.echo()
        click.echo("Unreachable rules:")
        for rule in report.unreachable_rules:
            click.echo(f"  {rule}")
    if report.overlapping_steps:
        click.echo()
        click.echo(
            "Overlapping rule steps: "
            + ", ".join(str(step) for step in report.overlapping_steps)
        )
    if report.action_collision_steps:
        click.echo()
        click.echo(
            "Action collision steps: "
            + ", ".join(str(step) for step in report.action_collision_steps)
        )


@main.command()
@click.argument("binding_spec", type=click.Path(exists=True))
@click.option("--steps", default=100, type=int, help="Simulation steps")
@click.option("--audit", default=None, type=click.Path(), help="Audit log (JSONL)")
@click.option(
    "--audit-stream",
    default=None,
    type=click.Path(),
    help="Audit event stream (length-delimited protobuf)",
)
@click.option("--seed", default=42, type=int, help="RNG seed")
def run(
    binding_spec: str,
    steps: int,
    audit: str | None,
    audit_stream: str | None,
    seed: int,
) -> None:
    """Run simulation from a binding spec."""
    spec = load_binding_spec(Path(binding_spec))
    errors = validate_binding_spec(spec)
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)

    if spec.safety_tier != "research":
        raise click.ClickException(
            f"safety_tier={spec.safety_tier!r} is not enforced by the local "
            "runtime; use the formal export and certified controller pipeline "
            "before executing non-research specs"
        )
    binding_summary = resolved_binding_config(spec)
    for line in format_resolved_binding_config(binding_summary):
        click.echo(line)

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    if n_osc == 0:
        click.echo("ERROR: no oscillators defined in layers", err=True)
        raise SystemExit(1)

    builder = CouplingBuilder()
    amplitude_mode = spec.amplitude is not None
    sl_engine: StuartLandauEngine | None = None
    upde_engine: UPDEEngine | None = None
    mu: FloatArray | None = None

    if amplitude_mode:
        amp = spec.amplitude
        assert amp is not None  # nosec B101
        coupling = builder.build_with_amplitude(
            n_osc,
            spec.coupling.base_strength,
            spec.coupling.decay_alpha,
            amp.amp_coupling_strength,
            amp.amp_coupling_decay,
        )
        sl_engine = StuartLandauEngine(n_osc, dt=spec.sample_period_s)
        mu = np.full(n_osc, amp.mu)
    else:
        coupling = builder.build(
            n_osc,
            spec.coupling.base_strength,
            spec.coupling.decay_alpha,
        )
        upde_engine = UPDEEngine(n_osc, dt=spec.sample_period_s)
    event_bus = EventBus()
    boundary_observer = BoundaryObserver(spec.boundaries)
    boundary_observer.set_event_bus(event_bus)
    regime_manager = RegimeManager(event_bus=event_bus)
    channel_runtime = ChannelRuntimeExecutor.from_spec(spec)

    petri_adapter: PetriNetAdapter | None = None
    if spec.protocol_net is not None:
        net, marking = _petri_net_from_protocol(spec.protocol_net)
        petri_adapter = PetriNetAdapter(
            net,
            marking,
            spec.protocol_net.place_regime,
            event_bus=event_bus,
        )

    supervisor = SupervisorPolicy(regime_manager, petri_adapter=petri_adapter)

    # ActionProjector — derive bounds from domainpack actuators
    value_bounds: dict[str, tuple[float, float]] = {}
    for act in spec.actuators:
        if act.limits and len(act.limits) == 2:
            value_bounds[act.knob] = (act.limits[0], act.limits[1])
    projector = ActionProjector(
        rate_limits={"K": 0.1, "zeta": 0.2, "alpha": 0.1, "Psi": 0.5},
        value_bounds=value_bounds
        or {
            "K": (-0.5, 0.5),
            "zeta": (0.0, 0.5),
            "alpha": (-1.0, 1.0),
        },
    )
    prev_values: dict[str, float] = {"K": 0.0, "zeta": 0.0, "alpha": 0.0, "Psi": 0.0}

    # Policy rules from domainpack (optional)
    policy_engine: PolicyEngine | None = None
    spec_path = Path(binding_spec)
    policy_path = spec_path.parent / "policy.yaml"
    if policy_path.exists():
        rules = load_policy_rules(policy_path)
        if rules:
            policy_engine = PolicyEngine(rules)

    imprint_model = None
    imprint_state = None
    if spec.imprint_model is not None:
        imprint_model = ImprintModel(
            spec.imprint_model.decay_rate, spec.imprint_model.saturation
        )
        imprint_state = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)

    geo_constraints: list[GeometryConstraint] = []
    if spec.geometry_prior is not None:
        ct = spec.geometry_prior.constraint_type.lower()
        if "symmetric" in ct:
            geo_constraints.append(SymmetryConstraint())
        if "non_negative" in ct or "nonneg" in ct:
            geo_constraints.append(NonNegativeConstraint())

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, n_osc)
    omegas = np.array(spec.get_omegas(), dtype=np.float64)

    # Stuart-Landau state: (2N,) = [phases, amplitudes]
    if amplitude_mode and mu is not None:
        r_init = np.sqrt(np.maximum(mu, 0.0))
        sl_state = np.concatenate([phases, r_init])
        phases_history: list[FloatArray] = []
        amps_history: list[FloatArray] = []

    layer_osc_ranges: dict[int, list[int]] = {}
    osc_idx = 0
    for layer in spec.layers:
        n_layer = len(layer.oscillator_ids)
        layer_osc_ranges[layer.index] = list(range(osc_idx, osc_idx + n_layer))
        osc_idx += n_layer

    # Initialise drive parameters from spec.drivers
    zeta = max(
        (cfg.get("zeta", 0.0) for cfg in spec.drivers.all_channel_configs().values()),
        default=0.0,
    )
    zeta_ttl = 0
    psi_target = spec.drivers.physical.get("psi", 0.0)

    psi_driver: PhysicalDriver | InformationalDriver | SymbolicDriver | None = None
    if "frequency" in spec.drivers.physical:
        psi_driver = PhysicalDriver(
            frequency=spec.drivers.physical["frequency"],
            amplitude=spec.drivers.physical.get("amplitude", 1.0),
        )
    elif "cadence_hz" in spec.drivers.informational:
        psi_driver = InformationalDriver(
            cadence_hz=spec.drivers.informational["cadence_hz"],
        )
    elif "sequence" in spec.drivers.symbolic:
        psi_driver = SymbolicDriver(
            sequence=spec.drivers.symbolic["sequence"],
        )
    control_interval = max(1, round(spec.control_period_s / spec.sample_period_s))
    audit_logger = (
        AuditLogger(audit, event_stream=audit_stream)
        if audit
        else AuditLogger(
            Path(audit_stream).with_suffix(".jsonl"),
            event_stream=audit_stream,
        )
        if audit_stream
        else None
    )
    if audit_logger is not None:
        audit_logger.log_header(
            n_oscillators=n_osc,
            dt=spec.sample_period_s,
            seed=seed,
            amplitude_mode=amplitude_mode,
            binding_config=binding_summary,
        )
    try:
        for step_idx in range(steps):
            if zeta_ttl > 0:
                zeta_ttl -= 1
                if zeta_ttl == 0:
                    zeta = 0.0

            if psi_driver is not None:
                t = step_idx * spec.sample_period_s
                if isinstance(psi_driver, SymbolicDriver):
                    psi_target = psi_driver.compute(step_idx)
                else:
                    psi_target = psi_driver.compute(t)

            eff_knm = coupling.knm
            eff_alpha = coupling.alpha
            if imprint_model is not None and imprint_state is not None:
                eff_knm = imprint_model.modulate_coupling(eff_knm, imprint_state)
                eff_alpha = imprint_model.modulate_lag(eff_alpha, imprint_state)
            if geo_constraints:
                eff_knm = project_knm(eff_knm, geo_constraints)

            input_phases = phases.copy()
            logged_zeta = zeta
            logged_psi = psi_target
            if amplitude_mode and sl_engine is not None and mu is not None:
                assert coupling.knm_r is not None  # nosec B101
                eff_mu = mu
                if imprint_model is not None and imprint_state is not None:
                    eff_mu = imprint_model.modulate_mu(mu, imprint_state)
                sl_state = sl_engine.step(
                    sl_state,
                    omegas,
                    eff_mu,
                    eff_knm,
                    coupling.knm_r,
                    zeta,
                    psi_target,
                    eff_alpha,
                    # type ignore: amplitude_mode proves spec.amplitude is set here.
                    epsilon=spec.amplitude.epsilon,  # type: ignore[union-attr]
                )
                phases = sl_state[:n_osc]
                amplitudes = sl_state[n_osc:]
                phases_history.append(phases.copy())
                amps_history.append(amplitudes.copy())
            else:
                assert upde_engine is not None  # nosec B101
                phases = upde_engine.step(
                    phases, omegas, eff_knm, zeta, psi_target, eff_alpha
                )

            layer_states = []
            for layer in spec.layers:
                osc_ids = layer_osc_ranges[layer.index]
                if osc_ids:
                    r, psi_l = compute_order_parameter(phases[osc_ids])
                else:
                    r, psi_l = 0.0, 0.0
                ls_kwargs: dict = {"R": r, "psi": psi_l}
                if amplitude_mode:
                    layer_r = amplitudes[osc_ids] if osc_ids else np.array([])
                    if layer_r.size > 0:
                        ls_kwargs["mean_amplitude"] = float(np.mean(layer_r))
                        mean_r = float(np.mean(layer_r))
                        if mean_r > 0:
                            ls_kwargs["amplitude_spread"] = float(
                                np.std(layer_r) / mean_r
                            )
                layer_states.append(LayerState(**ls_kwargs))

            n_layers = len(spec.layers)
            cla = np.zeros((n_layers, n_layers))
            for li in range(n_layers):
                for lj in range(li + 1, n_layers):
                    ids_i = layer_osc_ranges[spec.layers[li].index]
                    ids_j = layer_osc_ranges[spec.layers[lj].index]
                    if ids_i and ids_j:
                        pi, pj = phases[ids_i], phases[ids_j]
                        min_len = min(len(pi), len(pj))
                        plv = compute_plv(pi[:min_len], pj[:min_len])
                        cla[li, lj] = plv
                        cla[lj, li] = plv

            runtime_execution = channel_runtime.execute(layer_states)
            executed_layer_states = list(runtime_execution.layers)

            mean_r_val = (
                float(np.mean([ls.R for ls in executed_layer_states]))
                if executed_layer_states
                else 0.0
            )
            state_kwargs: dict = {
                "layers": executed_layer_states,
                "cross_layer_alignment": cla,
                "stability_proxy": mean_r_val,
                "regime_id": regime_manager.current_regime.value,
            }
            if amplitude_mode:
                state_kwargs["mean_amplitude"] = float(np.mean(amplitudes))
                sub_count = int(np.sum(amplitudes < 0.1))
                state_kwargs["subcritical_fraction"] = (
                    sub_count / n_osc if n_osc > 0 else 0.0
                )
                if len(phases_history) >= 20:
                    recent_ph = np.array(phases_history[-20:])
                    recent_am = np.array(amps_history[-20:])
                    pac_vals = [
                        modulation_index(recent_ph[:, i], recent_am[:, i])
                        for i in range(n_osc)
                    ]
                    state_kwargs["pac_max"] = float(max(pac_vals))

            if imprint_state is not None:
                state_kwargs["imprint_mean"] = float(np.mean(imprint_state.m_k))

            obs_values: dict[str, float] = {"R": state_kwargs["stability_proxy"]}
            if amplitude_mode:
                obs_values["mean_amplitude"] = state_kwargs.get("mean_amplitude", 0.0)
                obs_values["pac_max"] = state_kwargs.get("pac_max", 0.0)
                obs_values["subcritical_fraction"] = state_kwargs.get(
                    "subcritical_fraction", 0.0
                )
            for i, ls in enumerate(executed_layer_states):
                obs_values[f"R_{i}"] = ls.R
            boundary_state = boundary_observer.observe(obs_values, step=step_idx)
            state_kwargs["boundary_violation_count"] = len(boundary_state.violations)
            upde_state = UPDEState(**state_kwargs)
            actions: list = []
            if step_idx % control_interval == 0:
                actions = supervisor.decide(
                    upde_state, boundary_state, petri_ctx=obs_values
                )

                if policy_engine is not None:
                    actions.extend(
                        policy_engine.evaluate(
                            regime_manager.current_regime,
                            upde_state,
                            spec.objectives.good_layers,
                            spec.objectives.bad_layers,
                        )
                    )

                actions = [
                    projector.project(a, prev_values.get(a.knob, 0.0)) for a in actions
                ]

            for act in actions:
                if act.knob == "zeta":
                    zeta = max(0.0, min(zeta + act.value, 0.5))
                    zeta_ttl = int(act.ttl_s / spec.sample_period_s)
                elif act.knob == "K":
                    if act.scope == "global":
                        coupling = CouplingState(
                            knm=coupling.knm * (1.0 + act.value),
                            alpha=coupling.alpha,
                            active_template=coupling.active_template,
                            knm_r=coupling.knm_r,
                        )
                    elif act.scope.startswith("layer_"):
                        idx = int(act.scope.split("_", 1)[1])
                        new_knm = coupling.knm.copy()
                        new_knm[idx, :] *= 1.0 + act.value
                        new_knm[:, idx] *= 1.0 + act.value
                        new_knm[idx, idx] = 0.0
                        coupling = CouplingState(
                            knm=new_knm,
                            alpha=coupling.alpha,
                            active_template=coupling.active_template,
                            knm_r=coupling.knm_r,
                        )
                elif act.knob == "Psi":
                    psi_target = act.value
                prev_values[act.knob] = act.value

            if imprint_model is not None and imprint_state is not None:
                exposure = np.array(
                    [
                        layer_states[i].R
                        for i, layer in enumerate(spec.layers)
                        for _ in layer.oscillator_ids
                    ]
                )
                imprint_state = imprint_model.update(
                    imprint_state, exposure, spec.sample_period_s
                )

            if audit_logger is not None:
                log_kwargs: dict = {
                    "phases": input_phases,
                    "omegas": omegas,
                    "knm": eff_knm,
                    "alpha": eff_alpha,
                    "zeta": logged_zeta,
                    "psi_drive": logged_psi,
                    "channel_runtime": runtime_execution.to_audit_record(),
                }
                if amplitude_mode:
                    log_kwargs["amplitudes"] = amplitudes
                    log_kwargs["mu"] = eff_mu
                    log_kwargs["knm_r"] = coupling.knm_r
                    # type ignore: amplitude_mode proves spec.amplitude is set here.
                    log_kwargs["epsilon"] = spec.amplitude.epsilon  # type: ignore[union-attr]
                audit_logger.log_step(
                    step_idx,
                    upde_state,
                    actions,
                    **log_kwargs,
                )

        # Final coherence
        good_phases = [
            phases[i]
            for idx in spec.objectives.good_layers
            for i in layer_osc_ranges.get(idx, [])
        ]
        bad_phases = [
            phases[i]
            for idx in spec.objectives.bad_layers
            for i in layer_osc_ranges.get(idx, [])
        ]

        r_good = (
            compute_order_parameter(np.array(good_phases))[0] if good_phases else 0.0
        )
        r_bad = compute_order_parameter(np.array(bad_phases))[0] if bad_phases else 0.0

        regime = regime_manager.current_regime.value
        msg = f"R_good={r_good:.4f}  R_bad={r_bad:.4f}  regime={regime}"
        if amplitude_mode:
            mean_a = float(np.mean(amplitudes))
            msg += f"  mean_amplitude={mean_a:.4f}"
        click.echo(msg)
    finally:
        if audit_logger is not None:
            for evt in event_bus.history:
                audit_logger.log_event(
                    evt.kind, {"step": evt.step, "detail": evt.detail}
                )
            audit_logger.close()


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option("--output", default=None, type=click.Path(), help="Output file")
@click.option("--verify", is_flag=True, help="Verify determinism via re-execution")
def replay(log_path: str, output: str | None, verify: bool) -> None:
    """Replay an audit log and print summary."""
    replay_engine = ReplayEngine(log_path)
    entries = replay_engine.load()
    step_data = [e for e in entries if "step" in e]
    event_data = [e for e in entries if "event" in e]
    click.echo(f"Steps logged: {len(step_data)}")
    click.echo(f"Events logged: {len(event_data)}")
    if step_data:
        last = step_data[-1]
        click.echo(f"Final regime: {last.get('regime', 'unknown')}")
        click.echo(f"Final stability: {last.get('stability', 0.0):.4f}")
    if verify:
        integrity_ok, n_integrity = ReplayEngine.verify_integrity(entries)
        if not integrity_ok:
            click.echo(
                f"ERROR: audit integrity FAILED after {n_integrity} records",
                err=True,
            )
            raise SystemExit(1)
        header = replay_engine.load_header(entries)
        if header is None:
            click.echo("ERROR: no header record in log", err=True)
            raise SystemExit(1)
        engine = replay_engine.build_engine(header)
        if isinstance(engine, StuartLandauEngine):
            passed, n = replay_engine.verify_determinism_sl_chained(engine, entries)
        else:
            passed, n = replay_engine.verify_determinism_chained(engine, entries)
        if passed:
            click.echo(f"Determinism verified: {n} transitions OK")
        else:
            click.echo(f"Determinism FAILED at transition {n}", err=True)
            raise SystemExit(1)


def _watch_line(event: AuditStreamEvent) -> str:
    payload = event.payload
    if event.event_type == "step":
        step = _int_value(payload.get("step"))
        regime = str(payload.get("regime", "unknown"))
        stability = _float_value(payload.get("stability"))
        return (
            f"#{event.sequence} step step={step} regime={regime} "
            f"stability={stability:.4f} hash={event.event_hash[:12]}"
        )
    if event.event_type == "header":
        n_osc = _int_value(payload.get("n_oscillators"))
        dt = _float_value(payload.get("dt"))
        return (
            f"#{event.sequence} header n_oscillators={n_osc} "
            f"dt={dt:.6g} hash={event.event_hash[:12]}"
        )
    step_value = payload.get("step")
    suffix = f" step={step_value}" if isinstance(step_value, int) else ""
    return f"#{event.sequence} {event.event_type}{suffix} hash={event.event_hash[:12]}"


@main.command()
@click.argument("stream_path", type=click.Path(exists=True))
@click.option(
    "--format",
    "stream_format",
    type=click.Choice(["protobuf"]),
    default="protobuf",
    show_default=True,
    help="Audit stream encoding.",
)
@click.option("--from-start", is_flag=True, help="Replay existing events first")
@click.option("--max-events", default=None, type=int, help="Stop after N events")
@click.option("--poll-interval", default=0.2, type=float, help="Tail poll interval")
def watch(
    stream_path: str,
    stream_format: str,
    from_start: bool,
    max_events: int | None,
    poll_interval: float,
) -> None:
    """Tail and replay the live audit event stream."""
    if max_events is not None and max_events < 1:
        click.echo("ERROR: --max-events must be >= 1", err=True)
        raise SystemExit(1)
    if poll_interval <= 0.0:
        click.echo("ERROR: --poll-interval must be positive", err=True)
        raise SystemExit(1)
    if stream_format != "protobuf":
        click.echo("ERROR: unsupported stream format", err=True)
        raise SystemExit(1)

    events: list[AuditStreamEvent] = []
    try:
        if from_start and max_events is None:
            events = read_event_stream(stream_path)
            for event in events:
                click.echo(_watch_line(event))
        else:
            for event in iter_event_stream(
                stream_path,
                from_start=from_start,
                poll_interval_s=poll_interval,
            ):
                events.append(event)
                click.echo(_watch_line(event))
                if max_events is not None and len(events) >= max_events:
                    break
    except ValueError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1) from exc

    ok, verified = verify_event_stream_integrity(events)
    status = "OK" if ok else "FAILED"
    click.echo(f"stream integrity: {status} ({verified} events)")
    if not ok:
        raise SystemExit(1)


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option("--json-out", is_flag=True, help="Output JSON instead of text")
def report(log_path: str, json_out: bool) -> None:
    """Generate coherence report from audit log."""
    import json as _json

    replay_engine = ReplayEngine(log_path)
    entries = replay_engine.load()
    steps = [e for e in entries if "step" in e and "layers" in e]

    if not steps:
        click.echo("ERROR: no step records in log", err=True)
        raise SystemExit(1)

    integrity_ok, n_verified = ReplayEngine.verify_integrity(entries)
    summary = build_audit_report_summary(
        entries,
        hash_chain_ok=integrity_ok,
        hash_chain_verified=n_verified,
    )

    if json_out:
        click.echo(_json.dumps(summary, indent=2))
        return

    n_steps = _int_value(summary["steps"])
    n_layers = _int_value(summary["layers"])
    layer_r_mean = _float_list(summary.get("layer_r_mean"))
    layer_r_final = _float_list(summary.get("layer_r_final"))
    regime_counts = _count_dict(summary.get("regime_counts"))
    action_counts = _count_dict(summary.get("action_counts"))

    click.echo(f"Steps: {n_steps}  Layers: {n_layers}")
    mode = "Stuart-Landau" if summary["amplitude_mode"] else "Kuramoto"
    click.echo(f"Mode: {mode}")
    click.echo(f"Final regime: {summary['final_regime']}")
    final_stability = _float_value(summary["final_stability"])
    click.echo(f"Final stability: {final_stability:.4f}")
    click.echo()
    for i in range(n_layers):
        click.echo(
            f"  L{i}: R_mean={layer_r_mean[i]:.4f}  R_final={layer_r_final[i]:.4f}"
        )
    channel_algebra = summary.get("channel_algebra")
    if isinstance(channel_algebra, dict):
        required = _string_list(channel_algebra.get("required_channels"))
        optional = _string_list(channel_algebra.get("optional_channels"))
        derived = _string_list(channel_algebra.get("derived_channels"))
        delayed = _string_list(channel_algebra.get("delayed_channels"))
        uncertain = _string_list(channel_algebra.get("uncertain_channels"))
        missing = _string_list(channel_algebra.get("missing_required_channels"))
        click.echo()
        click.echo(
            "Channel algebra: "
            f"required={len(required)} optional={len(optional)} "
            f"derived={len(derived)} delayed={len(delayed)} "
            f"uncertain={len(uncertain)}"
        )
        if missing:
            click.echo(f"  Missing required channels: {', '.join(missing)}")
    integrated_information = summary.get("integrated_information")
    if isinstance(integrated_information, dict):
        records = _int_value(integrated_information.get("records", 0))
        latest_phi = _float_value(integrated_information.get("latest_phi", 0.0))
        latest_normalised = _float_value(
            integrated_information.get("latest_normalised_phi", 0.0)
        )
        total_integration = _float_value(
            integrated_information.get("latest_total_integration", 0.0)
        )
        click.echo()
        click.echo(
            "Integrated information: "
            f"records={records} phi={latest_phi:.4f} "
            f"normalised_phi={latest_normalised:.4f} "
            f"total_integration={total_integration:.4f}"
        )
    click.echo()
    click.echo("Regime distribution:")
    for regime, count in sorted(regime_counts.items()):
        pct = 100.0 * count / n_steps
        click.echo(f"  {regime}: {count} ({pct:.1f}%)")
    if action_counts:
        click.echo()
        click.echo("Actions fired:")
        for knob, count in sorted(action_counts.items()):
            click.echo(f"  {knob}: {count}")
    click.echo()
    status = "OK" if integrity_ok else "FAILED"
    click.echo(f"Hash chain: {status} ({n_verified} records verified)")


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option("--markdown-out", default=None, type=click.Path(), help="Write Markdown")
@click.option("--pdf-out", default=None, type=click.Path(), help="Write text PDF")
@click.option("--max-actions", default=12, type=int, help="Maximum action explanations")
def explain(
    log_path: str,
    markdown_out: str | None,
    pdf_out: str | None,
    max_actions: int,
) -> None:
    """Generate a human-readable explanation report from an audit log."""
    from scpn_phase_orchestrator.reporting.explainability import (
        build_explainability_report,
        render_markdown,
        write_markdown,
        write_pdf,
    )

    if max_actions < 1:
        click.echo("ERROR: --max-actions must be >= 1", err=True)
        raise SystemExit(1)

    replay_engine = ReplayEngine(log_path)
    entries = replay_engine.load()
    try:
        explanation = build_explainability_report(entries, max_actions=max_actions)
    except ValueError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1) from exc

    wrote = False
    if markdown_out is not None:
        write_markdown(explanation, markdown_out)
        click.echo(f"Markdown report written: {markdown_out}")
        wrote = True
    if pdf_out is not None:
        write_pdf(explanation, pdf_out)
        click.echo(f"PDF report written: {pdf_out}")
        wrote = True
    if not wrote:
        click.echo(render_markdown(explanation), nl=False)


@main.group()
def queuewaves() -> None:
    """QueueWaves — real-time cascade failure detector."""


main.add_command(queuewaves)


@queuewaves.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8080, type=int)
def serve(config_path: str, host: str, port: int) -> None:
    """Start QueueWaves server."""
    from scpn_phase_orchestrator.apps.queuewaves.server import run_server

    run_server(config_path, host=host, port=port)


@queuewaves.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def check(config_path: str) -> None:
    """One-shot: scrape → analyze → exit 0 (ok) or 1 (anomalies)."""
    from pathlib import Path as _Path

    from scpn_phase_orchestrator.apps.queuewaves.config import load_config
    from scpn_phase_orchestrator.apps.queuewaves.detector import AnomalyDetector
    from scpn_phase_orchestrator.apps.queuewaves.pipeline import PhaseComputePipeline

    cfg = load_config(_Path(config_path))
    pipeline = PhaseComputePipeline(cfg)

    # Run a few ticks with empty buffers to initialise phases
    import numpy as _np

    rng = _np.random.default_rng(0)
    buffers = {svc.name: rng.standard_normal(cfg.buffer_length) for svc in cfg.services}
    snap = pipeline.tick(buffers)
    detector = AnomalyDetector(cfg.thresholds)
    anomalies = detector.detect(snap)

    click.echo(
        f"R_good={snap.r_good:.4f}  R_bad={snap.r_bad:.4f}  regime={snap.regime}"
    )
    if anomalies:
        for a in anomalies:
            click.echo(f"  [{a.severity}] {a.message}")
        raise SystemExit(1)
    click.echo("No anomalies detected.")


@main.command()
@click.argument("domain_name")
@click.option(
    "--llm",
    "use_llm",
    is_flag=True,
    help="Generate the binding spec from a natural-language description.",
)
@click.option(
    "--description",
    default=None,
    help="Natural-language domain description for --llm mode.",
)
@click.option(
    "--llm-response-json",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Offline JSON response file for deterministic LLM scaffold review.",
)
def scaffold(
    domain_name: str,
    use_llm: bool,
    description: str | None,
    llm_response_json: str | None,
) -> None:
    """Create a domainpack directory structure with template files."""
    if not re.match(r"^[a-zA-Z0-9_-]+$", domain_name):
        raise click.BadParameter(
            f"domain_name must match [a-zA-Z0-9_-]+, got {domain_name!r}"
        )
    base = Path(f"domainpacks/{domain_name}")
    if use_llm:
        if not description:
            raise click.BadParameter("--description is required with --llm")
        provider: LLMScaffoldProvider
        if llm_response_json:
            provider = StaticJSONScaffoldProvider(
                Path(llm_response_json).read_text(encoding="utf-8")
            )
        else:
            try:
                provider = configured_llm_scaffold_provider()
            except RuntimeError as exc:
                raise click.ClickException(str(exc)) from exc
        try:
            proposal = propose_domainpack_from_description(
                description,
                project_name=domain_name,
                provider=provider,
            )
        except (RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        base.mkdir(parents=True, exist_ok=True)
        (base / "binding_spec.yaml").write_text(
            proposal.yaml_text,
            encoding="utf-8",
        )
        readme = base / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# {domain_name} domainpack\n\n"
                "LLM-assisted domainpack scaffold. Review the generated "
                "binding_spec.yaml, llm_scaffold_audit.json, boundaries, "
                "actuators, and oscillator mappings before production use.\n",
                encoding="utf-8",
            )
        (base / "llm_scaffold_audit.json").write_text(
            json.dumps(proposal.to_audit_record(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        click.echo(f"Scaffolded LLM-assisted domainpack at {base}")
        return

    base.mkdir(parents=True, exist_ok=True)
    spec_file = base / "binding_spec.yaml"
    if not spec_file.exists():
        spec_file.write_text(
            f"name: {domain_name}\n"
            "version: '0.1.0'\n"
            "safety_tier: research\n"
            "sample_period_s: 0.01\n"
            "control_period_s: 0.1\n"
            "layers:\n"
            "  - name: default\n"
            "    index: 0\n"
            "    oscillator_ids: [osc_0]\n"
            "oscillator_families:\n"
            "  default:\n"
            "    channel: P\n"
            "    extractor_type: physical\n"
            "coupling:\n"
            "  base_strength: 0.45\n"
            "  decay_alpha: 0.3\n"
            "drivers:\n"
            "  physical: {}\n"
            "  informational: {}\n"
            "  symbolic: {}\n"
            "objectives:\n"
            "  good_layers: [0]\n"
            "  bad_layers: []\n"
            "boundaries: []\n"
            "actuators: []\n",
            encoding="utf-8",
        )
    readme = base / "README.md"
    if not readme.exists():
        readme.write_text(f"# {domain_name} domainpack\n", encoding="utf-8")
    click.echo(f"Scaffolded domainpack at {base}")


@main.command("generate")
@click.argument("intent")
@click.option(
    "--name",
    default="generated_domain",
    help="Generated domainpack name.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help=(
        "Directory for binding_spec.yaml, policy.yaml, README.md, "
        "review_notebook.ipynb, and audit.json."
    ),
)
@click.option(
    "--oscillators-per-layer",
    default=8,
    show_default=True,
    help="Oscillators assigned to each inferred layer.",
)
@click.option(
    "--dry-run-steps",
    default=8,
    show_default=True,
    help="Validation simulation steps before artefacts are emitted.",
)
def generate(
    intent: str,
    name: str,
    output_dir: str | None,
    oscillators_per_layer: int,
    dry_run_steps: int,
) -> None:
    """Generate reviewable binding artefacts from symbolic domain intent."""
    artefacts = compile_symbolic_binding(
        intent,
        name=name,
        oscillators_per_layer=oscillators_per_layer,
        dry_run_steps=dry_run_steps,
    )
    output_path = Path("domainpacks") / name if output_dir is None else Path(output_dir)
    artefacts.write_domainpack(output_path)
    click.echo(f"Generated domainpack at {output_path}")
    click.echo(f"schema_valid={artefacts.schema_valid}")
    click.echo(f"confidence={artefacts.audit_record['confidence']:.3f}")
    click.echo(f"retrieval_matches={len(artefacts.retrieval_evidence)}")
    click.echo(f"dry_run_R={artefacts.dry_run_order_parameter:.6f}")


@main.command()
@click.option(
    "--domain",
    default="minimal_domain",
    help="Domainpack to demo (default: minimal_domain).",
)
@click.option("--steps", default=100, help="Number of simulation steps.")
@click.option("--port", default=8000, help="Server port.")
def demo(domain: str, steps: int, port: int) -> None:
    """Run a self-contained demo: simulate + print live coherence."""
    domainpack_dir = Path(__file__).parent.parent.parent / "domainpacks"
    spec_path = domainpack_dir / domain / "binding_spec.yaml"
    if not spec_path.exists():
        # Try relative to cwd
        spec_path = Path("domainpacks") / domain / "binding_spec.yaml"
    if not spec_path.exists():
        available = sorted(
            d.name
            for d in (
                domainpack_dir if domainpack_dir.exists() else Path("domainpacks")
            ).iterdir()
            if d.is_dir() and (d / "binding_spec.yaml").exists()
        )
        click.echo(f"Domainpack '{domain}' not found.", err=True)
        click.echo(f"Available: {', '.join(available)}", err=True)
        raise SystemExit(1)

    spec = load_binding_spec(spec_path)
    click.echo(f"SPO Demo — {spec.name}")
    click.echo(f"  Oscillators: {sum(len(ly.oscillator_ids) for ly in spec.layers)}")
    click.echo(f"  Layers: {len(spec.layers)}")
    click.echo(f"  Steps: {steps}")
    click.echo("-" * 40)

    from scpn_phase_orchestrator.runtime.server import SimulationState

    sim = SimulationState(spec)
    for step in range(1, steps + 1):
        state = sim.step()
        if step % max(1, steps // 10) == 0 or step == steps:
            R = state["R_global"]
            regime = state["regime"]
            click.echo(f"  Step {step:>5d}: R={R:.3f} [{regime}]")

    click.echo("-" * 40)
    click.echo(f"Final R={state['R_global']:.3f}, regime={state['regime']}")
    click.echo("\nTo serve with full stack:")
    click.echo("  cd deploy && docker compose up")
    click.echo("  Open http://localhost:8000 (dashboard)")
    click.echo("  Open http://localhost:3000 (Grafana)")
    click.echo("  Open http://localhost:9090 (Prometheus)")
