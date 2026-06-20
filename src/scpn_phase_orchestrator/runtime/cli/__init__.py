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

import csv
import http.client
import io
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast
from urllib.parse import urlparse

import click
import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator import plugins as plugin_api
from scpn_phase_orchestrator.autotune.binding_proposal import (
    propose_binding_from_event_log,
    propose_binding_from_graph,
    propose_binding_from_time_series_csv,
)
from scpn_phase_orchestrator.binding import (
    compile_symbolic_binding,
    format_resolved_binding_config,
    load_binding_spec,
    resolved_binding_config,
    validate_binding_spec,
    validate_binding_spec_security,
)
from scpn_phase_orchestrator.coupling.infer import auto_coupling_estimation
from scpn_phase_orchestrator.meta import CrossDomainMetaTransfer
from scpn_phase_orchestrator.monitor.twin_confidence import (
    TwinConfidenceCalibrator,
    TwinConfidenceSummary,
    phase_order_divergence,
    score_twin_confidence,
    summarise_twin_confidence,
    twin_confidence_prometheus_text,
)
from scpn_phase_orchestrator.plugins import (
    PluginExecutionApproval,
    PluginExecutionPlan,
    PluginExecutionRequestRevocationList,
    PluginExecutionRequestStorageManifest,
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
from scpn_phase_orchestrator.runtime.chaos import (
    ChaosFault,
    ChaosSchedule,
    run_resilience_experiment,
)
from scpn_phase_orchestrator.runtime.cli._payloads import (
    _PLUGIN_KIND_OPTIONS,
    _find_capability,
    _find_discovered_plugin,
    _load_approval_from_payload,
    _load_json_file,
    _load_lifecycle_from_payload,
    _load_lifecycle_multistore_drilldown_payload,
    _load_lifecycle_policy_report_payload,
    _load_lifecycle_remediation_action_status_payload,
    _load_lifecycle_remediation_deployment_handoff_payload,
    _load_lifecycle_remediation_execution_dashboard_payload,
    _load_lifecycle_remediation_plan_payload,
    _load_lifecycle_remediation_scheduler_acknowledgement_payload,
    _load_lifecycle_remediation_scheduler_adapter_handoff_payload,
    _load_lifecycle_remediation_scheduler_queue_payload,
    _load_lifecycle_remediation_scheduler_telemetry_payload,
    _load_lifecycle_summary_from_payload,
    _load_plan_from_payload,
    _load_request_from_payload,
    _load_revocation_from_payload,
    _load_revocation_list_from_payload,
    _load_storage_adapter_from_payload,
    _load_storage_manifest_from_payload,
    _normalize_approved_target_hashes,
    _record_hash,
    _require_sha256,
)
from scpn_phase_orchestrator.runtime.doctor import (
    render_report,
    run_environment_diagnostics,
)
from scpn_phase_orchestrator.runtime.observability import RuntimeObservability
from scpn_phase_orchestrator.runtime.replay import ReplayEngine
from scpn_phase_orchestrator.runtime.simulation import petri_net_from_protocol, simulate
from scpn_phase_orchestrator.scaffold.llm import (
    LLMScaffoldProvider,
    StaticJSONScaffoldProvider,
    configured_llm_scaffold_provider,
    propose_domainpack_from_description,
)
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
from scpn_phase_orchestrator.supervisor.policy_diagnostics import (
    PolicyDryRunReport,
    dry_run_policy_rules,
)
from scpn_phase_orchestrator.supervisor.policy_rules import (
    load_policy_rules,
    load_policy_stl_specs,
)
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

FloatArray: TypeAlias = NDArray[np.float64]
_PHYSIONET_HEARTBEAT_URL = (
    "https://physionet.org/files/respiratory-heartrate-dataset/1.0.0/"
    "HRM_rawData/HRB/3.txt"
)
_PHYSIONET_HEARTBEAT_CITATION = (
    "Guy et al. (2024), Respiratory and heart rate monitoring dataset from "
    "aeration study, PhysioNet, doi:10.13026/e4dt-f689"
)


@click.group()
def main() -> None:
    """SCPN Phase Orchestrator CLI."""


@main.command()
@click.option("--json-out", is_flag=True, help="Output the readiness record as JSON.")
def doctor(json_out: bool) -> None:
    """Check environment readiness: interpreter, required deps, backends, extras.

    Exits non-zero when the interpreter is outside the supported window or a
    required runtime dependency is missing; missing optional accelerators
    (Rust/Julia/Go/Mojo) and feature extras are reported as warnings only.

    Parameters
    ----------
    json_out : bool
        Whether to print machine-readable JSON output.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
    report = run_environment_diagnostics()
    if json_out:
        click.echo(json.dumps(report.to_audit_record(), indent=2, sort_keys=True))
    else:
        for line in render_report(report):
            click.echo(line)
    if not report.ok:
        raise SystemExit(report.exit_code)


def _load_twin_confidence_ticks(
    path: Path,
) -> list[tuple[NDArray[np.float64], ...]]:
    """Load model/observed phase + order ticks from a JSONL file.

    Each non-blank line must be a JSON object with ``model_phases``,
    ``observed_phases``, ``model_order``, and ``observed_order`` arrays.

    Parameters
    ----------
    path : Path
        The JSONL file to read.

    Returns
    -------
    list[tuple[NDArray[np.float64], ...]]
        One ``(model_phases, observed_phases, model_order, observed_order)``
        tuple per tick.

    Raises
    ------
    click.ClickException
        If the file is empty, a line is malformed, or a tick is missing a field.
    """
    ticks: list[tuple[NDArray[np.float64], ...]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"{path}:{line_no}: malformed JSON") from exc
        if not isinstance(record, Mapping):
            raise click.ClickException(f"{path}:{line_no}: tick must be a JSON object")
        try:
            tick = tuple(
                np.asarray(record[field], dtype=np.float64)
                for field in (
                    "model_phases",
                    "observed_phases",
                    "model_order",
                    "observed_order",
                )
            )
        except KeyError as exc:
            raise click.ClickException(
                f"{path}:{line_no}: missing field {exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise click.ClickException(
                f"{path}:{line_no}: non-numeric tick field"
            ) from exc
        ticks.append(tick)
    if not ticks:
        raise click.ClickException(f"{path}: no ticks found")
    return ticks


def _twin_confidence_summary_lines(summary: TwinConfidenceSummary) -> list[str]:
    return [
        f"ticks scored:      {summary.tick_count}",
        f"worst status:      {summary.worst_status}",
        f"latest status:     {summary.latest_status}",
        f"mean confidence:   {summary.mean_confidence:.4f}",
        f"min confidence:    {summary.min_confidence:.4f}",
        f"latest confidence: {summary.latest_confidence:.4f}",
        (
            f"status counts:     healthy={summary.healthy_count} "
            f"warning={summary.warning_count} critical={summary.critical_count}"
        ),
    ]


@main.command(name="twin-confidence")
@click.option(
    "--calibration",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="JSONL of trusted nominal ticks used to fit the baseline.",
)
@click.option(
    "--observations",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="JSONL of ticks to score against the calibrated baseline.",
)
@click.option("--n-bins", default=36, type=int, help="Phase histogram bins.")
@click.option("--sensitivity", default=3.0, type=float, help="Confidence decay scale.")
@click.option("--warning-confidence", default=0.6, type=float)
@click.option("--critical-confidence", default=0.3, type=float)
@click.option("--band-z", default=3.0, type=float, help="Operating-band multiplier.")
@click.option("--json-out", is_flag=True, help="Emit the summary as JSON.")
@click.option("--prometheus", is_flag=True, help="Emit Prometheus exposition text.")
@click.option(
    "--fail-on-critical",
    is_flag=True,
    help="Exit non-zero when the worst scored status is critical.",
)
def twin_confidence(
    calibration: Path,
    observations: Path,
    n_bins: int,
    sensitivity: float,
    warning_confidence: float,
    critical_confidence: float,
    band_z: float,
    json_out: bool,
    prometheus: bool,
    fail_on_critical: bool,
) -> None:
    """Score digital-twin confidence over an observation stream.

    Fits a nominal baseline from the calibration JSONL, scores each observation
    tick (phase Jensen-Shannon divergence + order Wasserstein-1 against the
    baseline), and reports the operator-facing summary as human lines, JSON, or
    Prometheus text.

    Parameters
    ----------
    calibration : Path
        JSONL of trusted nominal ticks.
    observations : Path
        JSONL of ticks to score.
    n_bins : int
        Phase histogram bins.
    sensitivity : float
        Confidence decay scale.
    warning_confidence, critical_confidence : float
        Operator status thresholds.
    band_z : float
        Operating-band normal-quantile multiplier.
    json_out, prometheus : bool
        Output-format switches.
    fail_on_critical : bool
        Whether to exit non-zero on a critical worst status.

    Raises
    ------
    SystemExit
        If ``--fail-on-critical`` is set and the worst scored status is critical.
    """
    try:
        calibrator = TwinConfidenceCalibrator(band_z=band_z)
        for tick in _load_twin_confidence_ticks(calibration):
            calibrator.observe(phase_order_divergence(*tick, n_bins=n_bins))
        baseline = calibrator.baseline()
        scores = [
            score_twin_confidence(
                phase_order_divergence(*tick, n_bins=n_bins),
                baseline,
                sensitivity=sensitivity,
                warning_confidence=warning_confidence,
                critical_confidence=critical_confidence,
            )
            for tick in _load_twin_confidence_ticks(observations)
        ]
        summary = summarise_twin_confidence(scores)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if prometheus:
        click.echo(twin_confidence_prometheus_text(summary), nl=False)
    elif json_out:
        click.echo(json.dumps(summary.to_audit_record(), indent=2, sort_keys=True))
    else:
        for line in _twin_confidence_summary_lines(summary):
            click.echo(line)
    if fail_on_critical and summary.worst_status == "critical":
        raise SystemExit(2)


def _parse_chaos_fault(spec: str) -> ChaosFault:
    """Parse a ``kind:start:duration:magnitude`` fault specifier.

    Parameters
    ----------
    spec : str
        Colon-separated fault specifier.

    Returns
    -------
    ChaosFault
        The parsed fault.

    Raises
    ------
    click.ClickException
        If the specifier is malformed or the fault parameters are invalid.
    """
    parts = spec.split(":")
    if len(parts) != 4:
        raise click.ClickException(
            f"fault {spec!r} must be 'kind:start:duration:magnitude'"
        )
    kind, start, duration, magnitude = parts
    try:
        return ChaosFault(
            kind=kind,
            start_step=int(start),
            duration_steps=int(duration),
            magnitude=float(magnitude),
        )
    except (ValueError, TypeError) as exc:
        raise click.ClickException(f"invalid fault {spec!r}: {exc}") from exc


@main.command(name="chaos")
@click.argument("binding_spec", type=click.Path(exists=True))
@click.option(
    "--fault",
    "faults",
    multiple=True,
    required=True,
    help="Fault as 'kind:start:duration:magnitude'; repeatable.",
)
@click.option("--steps", default=200, type=int, help="Simulation steps.")
@click.option("--seed", default=42, type=int, help="Shared RNG seed.")
@click.option("--recovery-tolerance", default=0.05, type=float)
@click.option("--json-out", is_flag=True, help="Emit the result as JSON.")
@click.option(
    "--fail-unrecovered",
    is_flag=True,
    help="Exit non-zero when the perturbed run does not recover.",
)
def chaos(
    binding_spec: str,
    faults: tuple[str, ...],
    steps: int,
    seed: int,
    recovery_tolerance: float,
    json_out: bool,
    fail_unrecovered: bool,
) -> None:
    """Inject faults into a binding spec and score its resilience.

    Runs the spec once nominally and once with the injected fault schedule, then
    reports recovery time, peak coherence drop, stability-margin erosion, and the
    final deviation. The runs are review-only and never actuate.

    Parameters
    ----------
    binding_spec : str
        Path to the binding spec YAML.
    faults : tuple[str, ...]
        Fault specifiers ``kind:start:duration:magnitude``.
    steps : int
        Simulation steps.
    seed : int
        Shared RNG seed.
    recovery_tolerance : float
        Recovery tolerance on ``|nominal_R - perturbed_R|``.
    json_out : bool
        Whether to emit JSON.
    fail_unrecovered : bool
        Whether to exit non-zero when the run does not recover.

    Raises
    ------
    SystemExit
        If ``--fail-unrecovered`` is set and the perturbed run did not recover.
    """
    schedule = ChaosSchedule(faults=tuple(_parse_chaos_fault(item) for item in faults))
    try:
        result = run_resilience_experiment(
            load_binding_spec(Path(binding_spec)),
            schedule,
            steps=steps,
            seed=seed,
            recovery_tolerance=recovery_tolerance,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    metrics = result.metrics
    if json_out:
        click.echo(json.dumps(result.to_audit_record(), indent=2, sort_keys=True))
    else:
        click.echo(f"spec:              {result.spec_name}")
        click.echo(f"faults:            {len(schedule.faults)}")
        click.echo(f"recovered:         {metrics.recovered}")
        click.echo(f"recovery steps:    {metrics.recovery_steps}")
        click.echo(f"max coherence drop:{metrics.max_coherence_drop:.4f}")
        click.echo(f"margin erosion:    {metrics.stability_margin_erosion:.4f}")
        click.echo(f"final deviation:   {metrics.final_deviation:.4f}")
        click.echo(
            f"nominal/perturbed R: {result.nominal_final_r:.4f}"
            f" / {result.perturbed_final_r:.4f}"
        )
    if fail_unrecovered and not metrics.recovered:
        raise SystemExit(2)


@main.command()
@click.argument("binding_spec", type=click.Path(exists=True))
@click.option(
    "--security",
    "security_checks",
    is_flag=True,
    help="Run stricter security linting for production-facing binding specs.",
)
def validate(binding_spec: str, security_checks: bool) -> None:
    """Validate a binding specification file.

    Parameters
    ----------
    binding_spec : str
        Filesystem path to the binding-spec file.
    security_checks : bool
        Whether to run the stricter security validation pass.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
    spec = load_binding_spec(Path(binding_spec))
    errors = validate_binding_spec(spec)
    if security_checks:
        errors.extend(validate_binding_spec_security(spec))
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)
    click.echo("Valid")
    if security_checks:
        click.echo("Security checks passed")
    summary = resolved_binding_config(spec)
    for line in format_resolved_binding_config(summary):
        click.echo(line)


@main.command("inspect")
@click.argument("binding_spec", type=click.Path(exists=True))
@click.option("--json-out", is_flag=True, help="Output resolved summary as JSON")
def inspect_binding(binding_spec: str, json_out: bool) -> None:
    """Inspect resolved runtime choices for a binding spec.

    Parameters
    ----------
    binding_spec : str
        Filesystem path to the binding-spec file.
    json_out : bool
        Whether to print machine-readable JSON output.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
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
    """Propose a review-only binding spec from raw local source data.

    Parameters
    ----------
    source_kind : str
        Kind of raw source data (e.g. ``csv`` or ``events``).
    source_path : str
        Filesystem path to the raw source data.
    project_name : str
        Name for the generated project.
    sample_rate_hz : float | None
        Sampling rate in Hz, or ``None`` to infer.
    json_out : bool
        Whether to print machine-readable JSON output.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
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
    """Infer directed K_nm from phase time-series data.

    Parameters
    ----------
    source_path : str
        Filesystem path to the raw source data.
    orientation : str
        Time-series orientation (rows or columns).
    n_bins : int
        Number of histogram bins.
    threshold_quantile : float
        Quantile threshold for edge pruning.
    threshold_absolute : float | None
        Absolute threshold for edge pruning, or ``None``.
    normalisation : str
        Edge-weight normalisation mode.
    json_out : bool
        Whether to print machine-readable JSON output.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Print the discovered plugin marketplace catalogue as JSON.

    Parameters
    ----------
    include_incompatible : bool
        Whether to include incompatible plugins in the catalogue.
    rust_registry : bool
        Whether to include the Rust plugin registry.
    rust_runtime_handoff : bool
        Whether to include the Rust runtime handoff.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Emit a non-executing plan for a discovered plugin capability.

    Parameters
    ----------
    plugin_name : str
        Name of the plugin.
    kind : str
        Plugin capability kind.
    capability_name : str
        Name of the plugin capability.
    approved_target_hashes : tuple[str, ...]
        Approved target hashes for the capability.
    require_target_hash_approval : bool
        Whether target-hash approval is required.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Emit a deterministic operator approval artefact for a stored execution plan.

    Parameters
    ----------
    plan_json : Path
        Path to the plan JSON file.
    operator_id : str
        Identifier of the operator.
    approval_reference : str
        External approval reference.
    approval_reason : str
        Reason recorded with the approval.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Emit a deterministic execution request from a stored plan and approval.

    Parameters
    ----------
    plan_json : Path
        Path to the plan JSON file.
    approval_json : Path
        Path to the approval JSON file.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Persist a validated execution request as a local storage bundle.

    Parameters
    ----------
    request_json : Path
        Path to the request JSON file.
    output_path : Path
        Destination path for the artefact.
    storage_uri : str
        Storage URI for the request bundle.
    storage_backend : str
        Storage backend identifier.
    retention_policy : str
        Retention policy label.
    created_by : str
        Identifier of the creating actor.
    revoked_request_hashes : tuple[str, ...]
        Hashes of revoked execution requests.
    revocation_list_path : Path | None
        Filesystem path to the revocation list.
    overwrite : bool
        Whether to overwrite an existing artefact.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Emit a deterministic storage-adapter handoff manifest without writing.

    Parameters
    ----------
    request_json : Path
        Path to the request JSON file.
    storage_uri : str
        Storage URI for the request bundle.
    storage_backend : str
        Storage backend identifier.
    retention_policy : str
        Retention policy label.
    created_by : str
        Identifier of the creating actor.
    revoked_request_hashes : tuple[str, ...]
        Hashes of revoked execution requests.
    revocation_list_path : Path | None
        Filesystem path to the revocation list.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Emit an operator lifecycle status record for an execution request.

    Parameters
    ----------
    request_json : Path
        Path to the request JSON file.
    storage_bundle_path : Path | None
        Filesystem path to the storage bundle.
    revocation_list_path : Path | None
        Filesystem path to the revocation list.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Emit a deterministic batch summary for lifecycle-status records.

    Parameters
    ----------
    lifecycle_json : tuple[Path, ...]
        Path to the lifecycle JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Emit a deterministic lifecycle policy report for operator dashboards.

    Parameters
    ----------
    summary_json : Path
        Path to the summary JSON.
    storage_adapter_paths : tuple[Path, ...]
        Paths to the storage-adapter handoff manifests.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Emit a deterministic renewal/follow-up queue for lifecycle operations.

    Parameters
    ----------
    summary_json : Path
        Path to the summary JSON.
    policy_report_path : Path | None
        Filesystem path to the policy report.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Emit a deterministic aggregate dashboard across policy reports.

    Parameters
    ----------
    policy_json : tuple[Path, ...]
        Path to the policy JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
            _require_sha256(policy["policy_hash"], "policy_hash") for policy in policies
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
    """Emit deterministic per-store lifecycle queues with provenance hashes.

    Parameters
    ----------
    policy_json : tuple[Path, ...]
        Path to the policy JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
            _require_sha256(policy["policy_hash"], "policy_hash") for policy in policies
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
    """Emit a deterministic, priority-ordered cross-store remediation plan.

    Parameters
    ----------
    drilldown_json : Path
        Path to the drilldown JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Emit a deterministic remediation action status record.

    Parameters
    ----------
    plan_json : Path
        Path to the plan JSON file.
    action_hash : str
        Hash of the remediation action.
    state : str
        State label for the record.
    updated_by : str
        Identifier of the updating actor.
    note : str
        Free-form note recorded with the record.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
        raise click.ClickException("action_hash is not part of the remediation plan")
    status_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_action_status_v1"
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
    """Emit a deterministic closed-loop dashboard for remediation execution.

    Parameters
    ----------
    plan_json : Path
        Path to the plan JSON file.
    status_json : tuple[Path, ...]
        Path to the status JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
            "scpn_plugin_execution_request_lifecycle_remediation_execution_dashboard_v1"
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
    """Emit deterministic deployment handoff actions for unresolved remediation.

    Parameters
    ----------
    execution_dashboard_json : Path
        Path to the execution dashboard JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
            "scpn_plugin_execution_request_lifecycle_remediation_deployment_handoff_v1"
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
    """Emit a deterministic scheduler queue from remediation deployment handoff.

    Parameters
    ----------
    handoff_json : Path
        Path to the handoff JSON file.
    window_start_epoch : int
        Window start as a UNIX epoch.
    window_duration_seconds : int
        Window duration in seconds.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_queue_v1"
        ),
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
    """Emit deterministic operator telemetry for scheduler remediation queue.

    Parameters
    ----------
    scheduler_queue_json : Path
        Path to the scheduler queue JSON file.
    action_status_json : tuple[Path, ...]
        Path to the action status JSON file.
    as_of_epoch : int
        Reference time as a UNIX epoch.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
        action_hash = _require_sha256(status.get("action_hash"), "action_hash")
        if action_hash in status_by_action_hash:
            raise click.ClickException(
                "remediation scheduler telemetry schema mismatch: "
                "duplicate action status action_hash"
            )
        status_by_action_hash[action_hash] = {
            "state": status["state"],
            "status_hash": status["status_hash"],
            "updated_by": status.get("updated_by", ""),
            "note": status.get("note", ""),
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
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_telemetry_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(queue.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            queue.get("execution_hash"), "execution_hash"
        ),
        "handoff_hash": _require_sha256(queue.get("handoff_hash"), "handoff_hash"),
        "scheduler_hash": _require_sha256(
            queue.get("scheduler_hash"), "scheduler_hash"
        ),
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
    """Emit deterministic external scheduler adapter handoff payload.

    Parameters
    ----------
    scheduler_telemetry_json : Path
        Path to the scheduler telemetry JSON file.
    adapter_name : str
        Name of the external adapter.
    adapter_endpoint : str
        Endpoint of the external adapter.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
        row
        for row in rows
        if cast(str, row["state"]) in {"pending", "in_progress", "blocked"}
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
        "execution_hash": _require_sha256(
            telemetry.get("execution_hash"), "execution_hash"
        ),
        "telemetry_hash": _require_sha256(
            telemetry.get("telemetry_hash"), "telemetry_hash"
        ),
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
    """Emit deterministic acknowledgement artifact for adapter execution.

    Parameters
    ----------
    adapter_handoff_json : Path
        Path to the adapter handoff JSON file.
    entry_hash : str
        Hash of the handoff entry.
    state : str
        State label for the record.
    acknowledged_by : str
        Identifier of the acknowledged actor.
    external_reference : str
        External system reference.
    note : str
        Free-form note recorded with the record.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
        (
            entry
            for entry in entries
            if entry["adapter_entry_hash"] == normalized_entry_hash
        ),
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
        "telemetry_hash": _require_sha256(
            handoff.get("telemetry_hash"), "telemetry_hash"
        ),
        "plan_hash": _require_sha256(handoff.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            handoff.get("execution_hash"), "execution_hash"
        ),
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


@plugins_group.command("lifecycle-remediation-scheduler-acknowledgement-replay")
@click.argument(
    "adapter_handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "acknowledgement_json",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating acknowledgement replay manifest.",
)
def plugins_lifecycle_remediation_scheduler_acknowledgement_replay(
    adapter_handoff_json: Path,
    acknowledgement_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit deterministic replay manifest from scheduler acknowledgements.

    Parameters
    ----------
    adapter_handoff_json : Path
        Path to the adapter handoff JSON file.
    acknowledgement_json : tuple[Path, ...]
        Path to the acknowledgement JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler acknowledgement replay schema mismatch: "
            "created_by must be non-empty"
        )
    handoff = _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
        _load_json_file(
            adapter_handoff_json,
            artifact="remediation scheduler adapter handoff",
        )
    )
    handoff_hash = _require_sha256(
        handoff.get("adapter_handoff_hash"),
        "adapter_handoff_hash",
    )
    entries = cast(list[dict[str, object]], handoff["entries"])
    entry_by_adapter_hash: dict[str, dict[str, object]] = {}
    for entry in entries:
        adapter_entry_hash = _require_sha256(
            entry.get("adapter_entry_hash"),
            "adapter_entry_hash",
        )
        entry_by_adapter_hash[adapter_entry_hash] = entry

    replay_rows: list[dict[str, object]] = []
    seen_adapter_entry_hashes: set[str] = set()
    for path in acknowledgement_json:
        payload = _load_lifecycle_remediation_scheduler_acknowledgement_payload(
            _load_json_file(path, artifact="remediation scheduler acknowledgement")
        )
        payload_handoff_hash = _require_sha256(
            payload.get("adapter_handoff_hash"),
            "adapter_handoff_hash",
        )
        if payload_handoff_hash != handoff_hash:
            raise click.ClickException(
                "remediation scheduler acknowledgement replay schema mismatch: "
                "adapter_handoff_hash mismatch"
            )
        adapter_entry_hash = _require_sha256(
            payload.get("adapter_entry_hash"),
            "adapter_entry_hash",
        )
        if adapter_entry_hash in seen_adapter_entry_hashes:
            raise click.ClickException(
                "remediation scheduler acknowledgement replay schema mismatch: "
                "duplicate adapter_entry_hash acknowledgement"
            )
        handoff_entry = entry_by_adapter_hash.get(adapter_entry_hash)
        if handoff_entry is None:
            raise click.ClickException(
                "remediation scheduler acknowledgement replay schema mismatch: "
                "acknowledgement adapter_entry_hash missing from handoff"
            )
        seen_adapter_entry_hashes.add(adapter_entry_hash)
        replay_row: dict[str, object] = {
            "acknowledgement_hash": _require_sha256(
                payload.get("acknowledgement_hash"),
                "acknowledgement_hash",
            ),
            "adapter_entry_hash": adapter_entry_hash,
            "entry_hash": handoff_entry["entry_hash"],
            "action_hash": handoff_entry["action_hash"],
            "request_hash": handoff_entry["request_hash"],
            "state": payload["state"],
            "external_reference": payload["external_reference"],
            "acknowledged_by": payload["acknowledged_by"],
            "note": payload.get("note", ""),
        }
        replay_row["replay_row_hash"] = _record_hash(replay_row)
        replay_rows.append(replay_row)

    state_counts: dict[str, int] = {"in_progress": 0, "completed": 0, "blocked": 0}
    for row in replay_rows:
        state_counts[cast(str, row["state"])] += 1

    replay_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_acknowledgement_replay_v1"
        ),
        "version": "1.0.0",
        "adapter_handoff_hash": handoff_hash,
        "plan_hash": _require_sha256(handoff.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            handoff.get("execution_hash"), "execution_hash"
        ),
        "telemetry_hash": _require_sha256(
            handoff.get("telemetry_hash"), "telemetry_hash"
        ),
        "acknowledgement_count": len(replay_rows),
        "state_counts": state_counts,
        "rows": sorted(
            replay_rows,
            key=lambda item: (
                cast(str, item["state"]),
                cast(str, item["adapter_entry_hash"]),
            ),
        ),
        "created_by": created_by,
    }
    replay_payload["replay_hash"] = _record_hash(replay_payload)
    click.echo(json.dumps(replay_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-execution-dashboard")
@click.argument(
    "scheduler_telemetry_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "acknowledgement_replay_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating external scheduler execution dashboard.",
)
def plugins_lifecycle_remediation_scheduler_execution_dashboard(
    scheduler_telemetry_json: Path,
    acknowledgement_replay_json: Path,
    created_by: str,
) -> None:
    """Emit deterministic live execution dashboard across scheduler adapters.

    Parameters
    ----------
    scheduler_telemetry_json : Path
        Path to the scheduler telemetry JSON file.
    acknowledgement_replay_json : Path
        Path to the acknowledgement replay JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler execution dashboard schema mismatch: "
            "created_by must be non-empty"
        )
    telemetry = _load_lifecycle_remediation_scheduler_telemetry_payload(
        _load_json_file(
            scheduler_telemetry_json,
            artifact="remediation scheduler telemetry",
        )
    )
    replay = _load_json_file(
        acknowledgement_replay_json,
        artifact="remediation scheduler acknowledgement replay",
    )
    if replay.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_replay_v1"
    ):
        raise click.ClickException(
            "remediation scheduler execution dashboard schema mismatch: "
            "unexpected acknowledgement replay schema"
        )
    _require_sha256(replay.get("replay_hash"), "replay_hash")
    telemetry_hash = _require_sha256(telemetry.get("telemetry_hash"), "telemetry_hash")
    replay_telemetry_hash = _require_sha256(
        replay.get("telemetry_hash"), "telemetry_hash"
    )
    if replay_telemetry_hash != telemetry_hash:
        raise click.ClickException(
            "remediation scheduler execution dashboard schema mismatch: "
            "telemetry_hash mismatch"
        )
    replay_rows = cast(list[dict[str, object]], replay.get("rows", []))
    ack_state_by_action_hash: dict[str, str] = {}
    for row in replay_rows:
        action_hash = _require_sha256(row.get("action_hash"), "action_hash")
        state = row.get("state")
        if not isinstance(state, str) or state not in {
            "in_progress",
            "completed",
            "blocked",
        }:
            raise click.ClickException(
                "remediation scheduler execution dashboard schema mismatch: "
                "unsupported replay state"
            )
        if action_hash in ack_state_by_action_hash:
            raise click.ClickException(
                "remediation scheduler execution dashboard schema mismatch: "
                "duplicate replay action_hash"
            )
        ack_state_by_action_hash[action_hash] = state

    telemetry_rows = cast(list[dict[str, object]], telemetry["rows"])
    rows: list[dict[str, object]] = []
    dashboard_counts: dict[str, int] = {
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
        "blocked": 0,
        "overdue": 0,
    }
    for row in sorted(
        telemetry_rows,
        key=lambda item: (
            cast(int, item["priority"]),
            cast(int, item["schedule_epoch"]),
            str(item["action_hash"]),
        ),
    ):
        action_hash = _require_sha256(row.get("action_hash"), "action_hash")
        telemetry_state = cast(str, row["state"])
        ack_state = ack_state_by_action_hash.get(action_hash)
        effective_state = ack_state if ack_state is not None else telemetry_state
        if effective_state not in {"pending", "in_progress", "completed", "blocked"}:
            raise click.ClickException(
                "remediation scheduler execution dashboard schema mismatch: "
                "unsupported effective state"
            )
        overdue = bool(row["overdue"]) and effective_state != "completed"
        dashboard_counts[effective_state] += 1
        if overdue:
            dashboard_counts["overdue"] += 1
        output_row: dict[str, object] = {
            "entry_hash": row["entry_hash"],
            "action_hash": action_hash,
            "request_hash": row["request_hash"],
            "action_type": row["action_type"],
            "priority": row["priority"],
            "schedule_epoch": row["schedule_epoch"],
            "telemetry_state": telemetry_state,
            "acknowledgement_state": ack_state,
            "effective_state": effective_state,
            "overdue": overdue,
        }
        output_row["dashboard_row_hash"] = _record_hash(output_row)
        rows.append(output_row)

    dashboard_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_execution_dashboard_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(telemetry.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            telemetry.get("execution_hash"), "execution_hash"
        ),
        "handoff_hash": _require_sha256(telemetry.get("handoff_hash"), "handoff_hash"),
        "scheduler_hash": _require_sha256(
            telemetry.get("scheduler_hash"), "scheduler_hash"
        ),
        "telemetry_hash": telemetry_hash,
        "replay_hash": _require_sha256(replay.get("replay_hash"), "replay_hash"),
        "row_count": len(rows),
        "state_counts": dashboard_counts,
        "rows": rows,
        "created_by": created_by,
    }
    dashboard_payload["dashboard_hash"] = _record_hash(dashboard_payload)
    click.echo(json.dumps(dashboard_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-control-plan")
@click.argument(
    "scheduler_execution_dashboard_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator component creating interactive control plan artifact.",
)
def plugins_lifecycle_remediation_scheduler_control_plan(
    scheduler_execution_dashboard_json: Path,
    created_by: str,
) -> None:
    """Emit deterministic interactive control actions from scheduler dashboard.

    Parameters
    ----------
    scheduler_execution_dashboard_json : Path
        Path to the scheduler execution dashboard JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler control plan schema mismatch: "
            "created_by must be non-empty"
        )
    dashboard = _load_json_file(
        scheduler_execution_dashboard_json,
        artifact="remediation scheduler execution dashboard",
    )
    if dashboard.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_execution_dashboard_v1"
    ):
        raise click.ClickException(
            "remediation scheduler control plan schema mismatch: "
            "unexpected scheduler execution dashboard schema"
        )
    dashboard_hash = _require_sha256(dashboard.get("dashboard_hash"), "dashboard_hash")
    rows = dashboard.get("rows")
    if not isinstance(rows, list):
        raise click.ClickException(
            "remediation scheduler control plan schema mismatch: rows must be a list"
        )
    control_actions: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise click.ClickException(
                "remediation scheduler control plan schema mismatch: row must be object"
            )
        action_hash = _require_sha256(row.get("action_hash"), "action_hash")
        effective_state = row.get("effective_state")
        if effective_state not in {"pending", "in_progress", "completed", "blocked"}:
            raise click.ClickException(
                "remediation scheduler control plan schema mismatch: "
                "unsupported effective_state"
            )
        overdue = bool(row.get("overdue", False))
        if effective_state == "completed":
            action = "no_op"
            reason = "already_completed"
        elif effective_state == "blocked":
            action = "escalate"
            reason = "blocked_requires_operator_intervention"
        elif overdue:
            action = "expedite"
            reason = "overdue_action_requires_priority_bump"
        elif effective_state == "in_progress":
            action = "monitor"
            reason = "execution_in_progress_track_progress"
        else:
            action = "dispatch"
            reason = "ready_for_dispatch"
        control_row: dict[str, object] = {
            "action_hash": action_hash,
            "request_hash": _require_sha256(row.get("request_hash"), "request_hash"),
            "action_type": row.get("action_type"),
            "priority": row.get("priority"),
            "effective_state": effective_state,
            "overdue": overdue,
            "control_action": action,
            "reason": reason,
            "operator_command_template": (
                "spo plugins lifecycle-remediation-action-status PLAN_JSON ACTION_HASH "
                "--state STATE --updated-by OPERATOR --note NOTE"
            ),
        }
        control_row["control_row_hash"] = _record_hash(control_row)
        control_actions.append(control_row)
    control_counts: dict[str, int] = {
        "dispatch": 0,
        "monitor": 0,
        "expedite": 0,
        "escalate": 0,
        "no_op": 0,
    }
    for item in control_actions:
        control_counts[cast(str, item["control_action"])] += 1
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_control_plan_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(dashboard.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            dashboard.get("execution_hash"), "execution_hash"
        ),
        "dashboard_hash": dashboard_hash,
        "control_action_count": len(control_actions),
        "control_counts": control_counts,
        "control_actions": sorted(
            control_actions,
            key=lambda item: (
                cast(int, item["priority"]),
                cast(str, item["control_action"]),
                cast(str, item["action_hash"]),
            ),
        ),
        "created_by": created_by,
    }
    payload["control_plan_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-runbook")
@click.argument(
    "scheduler_control_plan_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "scheduler_adapter_handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator component creating scheduler runbook artifact.",
)
def plugins_lifecycle_remediation_scheduler_runbook(
    scheduler_control_plan_json: Path,
    scheduler_adapter_handoff_json: Path,
    created_by: str,
) -> None:
    """Emit deterministic operator runbook grouped by control action and adapter.

    Parameters
    ----------
    scheduler_control_plan_json : Path
        Path to the scheduler control plan JSON file.
    scheduler_adapter_handoff_json : Path
        Path to the scheduler adapter handoff JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler runbook schema mismatch: "
            "created_by must be non-empty"
        )
    control_plan = _load_json_file(
        scheduler_control_plan_json,
        artifact="remediation scheduler control plan",
    )
    if control_plan.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_control_plan_v1"
    ):
        raise click.ClickException(
            "remediation scheduler runbook schema mismatch: "
            "unexpected scheduler control plan schema"
        )
    _require_sha256(control_plan.get("control_plan_hash"), "control_plan_hash")
    adapter_handoff = _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
        _load_json_file(
            scheduler_adapter_handoff_json,
            artifact="remediation scheduler adapter handoff",
        )
    )
    plan_hash = _require_sha256(control_plan.get("plan_hash"), "plan_hash")
    adapter_plan_hash = _require_sha256(adapter_handoff.get("plan_hash"), "plan_hash")
    if plan_hash != adapter_plan_hash:
        raise click.ClickException(
            "remediation scheduler runbook schema mismatch: plan_hash mismatch"
        )
    control_actions = control_plan.get("control_actions")
    if not isinstance(control_actions, list):
        raise click.ClickException(
            "remediation scheduler runbook schema mismatch: "
            "control_actions must be a list"
        )
    adapter_entries = cast(list[dict[str, object]], adapter_handoff["entries"])
    adapter_by_action_hash: dict[str, dict[str, object]] = {}
    for entry in adapter_entries:
        action_hash = _require_sha256(entry.get("action_hash"), "action_hash")
        if action_hash in adapter_by_action_hash:
            raise click.ClickException(
                "remediation scheduler runbook schema mismatch: "
                "duplicate action_hash in adapter handoff"
            )
        adapter_by_action_hash[action_hash] = entry
    groups: dict[str, list[dict[str, object]]] = {
        "dispatch": [],
        "monitor": [],
        "expedite": [],
        "escalate": [],
        "no_op": [],
    }
    for action in control_actions:
        if not isinstance(action, dict):
            raise click.ClickException(
                "remediation scheduler runbook schema mismatch: control action must be "
                "object"
            )
        action_hash = _require_sha256(action.get("action_hash"), "action_hash")
        control_action = action.get("control_action")
        if control_action not in groups:
            raise click.ClickException(
                "remediation scheduler runbook schema mismatch: unsupported "
                "control_action"
            )
        adapter_entry = adapter_by_action_hash.get(action_hash)
        runbook_step: dict[str, object] = {
            "action_hash": action_hash,
            "request_hash": _require_sha256(action.get("request_hash"), "request_hash"),
            "control_action": control_action,
            "reason": action.get("reason"),
            "priority": action.get("priority"),
            "action_type": action.get("action_type"),
            "adapter_entry_hash": (
                adapter_entry.get("adapter_entry_hash")
                if adapter_entry is not None
                else None
            ),
            "adapter_name": (
                cast(dict[str, object], adapter_entry["adapter_target"]).get(
                    "adapter_name"
                )
                if adapter_entry is not None
                else None
            ),
            "adapter_endpoint": (
                cast(dict[str, object], adapter_entry["adapter_target"]).get(
                    "adapter_endpoint"
                )
                if adapter_entry is not None
                else None
            ),
            "acknowledgement_command_template": (
                adapter_entry.get("acknowledgement_command_template")
                if adapter_entry is not None
                else None
            ),
        }
        runbook_step["runbook_step_hash"] = _record_hash(runbook_step)
        groups[cast(str, control_action)].append(runbook_step)
    ordered_groups: list[dict[str, object]] = []
    for name in ("escalate", "expedite", "dispatch", "monitor", "no_op"):
        items = sorted(
            groups[name],
            key=lambda item: (
                cast(int, item["priority"]),
                cast(str, item["action_hash"]),
            ),
        )
        ordered_groups.append(
            {
                "control_action": name,
                "step_count": len(items),
                "steps": items,
            }
        )
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_runbook_v1"
        ),
        "version": "1.0.0",
        "plan_hash": plan_hash,
        "execution_hash": _require_sha256(
            control_plan.get("execution_hash"), "execution_hash"
        ),
        "control_plan_hash": _require_sha256(
            control_plan.get("control_plan_hash"),
            "control_plan_hash",
        ),
        "adapter_handoff_hash": _require_sha256(
            adapter_handoff.get("adapter_handoff_hash"),
            "adapter_handoff_hash",
        ),
        "group_count": len(ordered_groups),
        "groups": ordered_groups,
        "created_by": created_by,
    }
    payload["runbook_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-automation-profile")
@click.argument(
    "scheduler_runbook_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--profile-name",
    required=True,
    help="Automation profile name.",
)
@click.option(
    "--profile-version",
    required=True,
    help="Automation profile semantic version.",
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating automation profile artifact.",
)
def plugins_lifecycle_remediation_scheduler_automation_profile(
    scheduler_runbook_json: Path,
    profile_name: str,
    profile_version: str,
    created_by: str,
) -> None:
    """Emit deterministic adapter automation profile from scheduler runbook.

    Parameters
    ----------
    scheduler_runbook_json : Path
        Path to the scheduler runbook JSON file.
    profile_name : str
        Name of the automation profile.
    profile_version : str
        Version of the automation profile.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler automation profile schema mismatch: "
            "created_by must be non-empty"
        )
    if not profile_name:
        raise click.ClickException(
            "remediation scheduler automation profile schema mismatch: "
            "profile_name must be non-empty"
        )
    if not profile_version:
        raise click.ClickException(
            "remediation scheduler automation profile schema mismatch: "
            "profile_version must be non-empty"
        )
    runbook = _load_json_file(
        scheduler_runbook_json,
        artifact="remediation scheduler runbook",
    )
    if (
        runbook.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_scheduler_runbook_v1"
    ):
        raise click.ClickException(
            "remediation scheduler automation profile schema mismatch: "
            "unexpected scheduler runbook schema"
        )
    _require_sha256(runbook.get("runbook_hash"), "runbook_hash")
    groups = runbook.get("groups")
    if not isinstance(groups, list):
        raise click.ClickException(
            "remediation scheduler automation profile schema mismatch: "
            "groups must be a list"
        )
    automation_rules: list[dict[str, object]] = []
    for group in groups:
        if not isinstance(group, dict):
            raise click.ClickException(
                "remediation scheduler automation profile schema mismatch: "
                "group must be object"
            )
        control_action = group.get("control_action")
        if control_action not in {
            "dispatch",
            "monitor",
            "expedite",
            "escalate",
            "no_op",
        }:
            raise click.ClickException(
                "remediation scheduler automation profile schema mismatch: "
                "unsupported control_action"
            )
        steps = group.get("steps")
        if not isinstance(steps, list):
            raise click.ClickException(
                "remediation scheduler automation profile schema mismatch: "
                "steps must be list"
            )
        for step in steps:
            if not isinstance(step, dict):
                raise click.ClickException(
                    "remediation scheduler automation profile schema mismatch: "
                    "step must be object"
                )
            action_hash = _require_sha256(step.get("action_hash"), "action_hash")
            request_hash = _require_sha256(step.get("request_hash"), "request_hash")
            action_type = step.get("action_type")
            priority = step.get("priority")
            if not isinstance(action_type, str) or not action_type:
                raise click.ClickException(
                    "remediation scheduler automation profile schema mismatch: "
                    "action_type must be non-empty string"
                )
            if not isinstance(priority, int) or priority < 1:
                raise click.ClickException(
                    "remediation scheduler automation profile schema mismatch: "
                    "priority must be positive integer"
                )
            automation_mode = (
                "manual" if control_action in {"escalate", "no_op"} else "auto"
            )
            target_state = {
                "dispatch": "in_progress",
                "monitor": "in_progress",
                "expedite": "in_progress",
                "escalate": "blocked",
                "no_op": "completed",
            }[cast(str, control_action)]
            rule: dict[str, object] = {
                "control_action": control_action,
                "action_hash": action_hash,
                "request_hash": request_hash,
                "action_type": action_type,
                "priority": priority,
                "automation_mode": automation_mode,
                "target_state": target_state,
                "capture_command_template": (
                    "spo plugins "
                    "lifecycle-remediation-scheduler-acknowledgement-capture "
                    "AUTOMATION_PROFILE_JSON ADAPTER_HANDOFF_JSON ACTION_HASH "
                    "--external-reference REF --acknowledged-by OPERATOR "
                    "--captured-state STATE --note NOTE"
                ),
            }
            rule["automation_rule_hash"] = _record_hash(rule)
            automation_rules.append(rule)
    profile_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_automation_profile_v1"
        ),
        "version": "1.0.0",
        "profile_name": profile_name,
        "profile_version": profile_version,
        "plan_hash": _require_sha256(runbook.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            runbook.get("execution_hash"), "execution_hash"
        ),
        "runbook_hash": _require_sha256(runbook.get("runbook_hash"), "runbook_hash"),
        "automation_rule_count": len(automation_rules),
        "automation_rules": sorted(
            automation_rules,
            key=lambda item: (
                cast(int, item["priority"]),
                cast(str, item["control_action"]),
                cast(str, item["action_hash"]),
            ),
        ),
        "created_by": created_by,
    }
    profile_payload["automation_profile_hash"] = _record_hash(profile_payload)
    click.echo(json.dumps(profile_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-acknowledgement-capture")
@click.argument(
    "automation_profile_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "adapter_handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument("action_hash")
@click.option(
    "--external-reference",
    required=True,
    help="External scheduler run identifier.",
)
@click.option(
    "--acknowledged-by",
    required=True,
    help="Operator or adapter acknowledging execution.",
)
@click.option(
    "--captured-state",
    required=True,
    type=click.Choice(["in_progress", "completed", "blocked"]),
    help="Captured execution state.",
)
@click.option(
    "--note",
    default="",
    show_default=True,
    help="Optional capture note.",
)
def plugins_lifecycle_remediation_scheduler_acknowledgement_capture(
    automation_profile_json: Path,
    adapter_handoff_json: Path,
    action_hash: str,
    external_reference: str,
    acknowledged_by: str,
    captured_state: str,
    note: str,
) -> None:
    """Capture acknowledgement using automation profile and adapter handoff.

    Parameters
    ----------
    automation_profile_json : Path
        Path to the automation profile JSON file.
    adapter_handoff_json : Path
        Path to the adapter handoff JSON file.
    action_hash : str
        Hash of the remediation action.
    external_reference : str
        External system reference.
    acknowledged_by : str
        Identifier of the acknowledged actor.
    captured_state : str
        Captured acknowledgement state.
    note : str
        Free-form note recorded with the record.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not external_reference:
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "external_reference must be non-empty"
        )
    if not acknowledged_by:
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "acknowledged_by must be non-empty"
        )
    profile = _load_json_file(
        automation_profile_json,
        artifact="remediation scheduler automation profile",
    )
    if profile.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_automation_profile_v1"
    ):
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "unexpected automation profile schema"
        )
    _require_sha256(profile.get("automation_profile_hash"), "automation_profile_hash")
    normalized_action_hash = _require_sha256(action_hash, "action_hash")
    rules = profile.get("automation_rules")
    if not isinstance(rules, list):
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "automation_rules must be list"
        )
    rule = next(
        (
            item
            for item in rules
            if isinstance(item, dict)
            and item.get("action_hash") == normalized_action_hash
        ),
        None,
    )
    if not isinstance(rule, dict):
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "action_hash not present in automation profile"
        )
    target_state = cast(str, rule.get("target_state"))
    if (
        target_state != captured_state
        and cast(str, rule.get("automation_mode")) == "auto"
    ):
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "captured_state does not match auto target_state"
        )
    adapter_handoff = _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
        _load_json_file(
            adapter_handoff_json,
            artifact="remediation scheduler adapter handoff",
        )
    )
    if _require_sha256(profile.get("plan_hash"), "plan_hash") != _require_sha256(
        adapter_handoff.get("plan_hash"), "plan_hash"
    ):
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "plan_hash mismatch between automation profile and adapter handoff"
        )
    entries = cast(list[dict[str, object]], adapter_handoff["entries"])
    matched_entry = next(
        (
            entry
            for entry in entries
            if _require_sha256(entry.get("action_hash"), "action_hash")
            == normalized_action_hash
        ),
        None,
    )
    if matched_entry is None:
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "action_hash not present in adapter handoff"
        )
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_acknowledgement_capture_v1"
        ),
        "version": "1.0.0",
        "automation_profile_hash": _require_sha256(
            profile.get("automation_profile_hash"),
            "automation_profile_hash",
        ),
        "adapter_handoff_hash": _require_sha256(
            adapter_handoff.get("adapter_handoff_hash"),
            "adapter_handoff_hash",
        ),
        "plan_hash": _require_sha256(profile.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            profile.get("execution_hash"), "execution_hash"
        ),
        "action_hash": normalized_action_hash,
        "request_hash": _require_sha256(rule.get("request_hash"), "request_hash"),
        "adapter_entry_hash": _require_sha256(
            matched_entry.get("adapter_entry_hash"),
            "adapter_entry_hash",
        ),
        "captured_state": captured_state,
        "target_state": target_state,
        "automation_mode": rule.get("automation_mode"),
        "external_reference": external_reference,
        "acknowledged_by": acknowledged_by,
        "note": note,
    }
    payload["capture_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-retry-profile")
@click.argument(
    "automation_profile_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--max-attempts",
    default=3,
    show_default=True,
    type=int,
    help="Maximum retry attempts for eligible automated actions.",
)
@click.option(
    "--base-delay-seconds",
    default=30,
    show_default=True,
    type=int,
    help="Base delay in seconds before first retry.",
)
@click.option(
    "--backoff-multiplier",
    default=2.0,
    show_default=True,
    type=float,
    help="Retry backoff multiplier.",
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating retry profile artifact.",
)
def plugins_lifecycle_remediation_scheduler_retry_profile(
    automation_profile_json: Path,
    max_attempts: int,
    base_delay_seconds: int,
    backoff_multiplier: float,
    created_by: str,
) -> None:
    """Emit deterministic retry/backoff policy profile from automation profile.

    Parameters
    ----------
    automation_profile_json : Path
        Path to the automation profile JSON file.
    max_attempts : int
        Maximum number of retry attempts.
    base_delay_seconds : int
        Base retry delay in seconds.
    backoff_multiplier : float
        Exponential backoff multiplier.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler retry profile schema mismatch: "
            "created_by must be non-empty"
        )
    if max_attempts < 1:
        raise click.ClickException(
            "remediation scheduler retry profile schema mismatch: "
            "max_attempts must be positive"
        )
    if base_delay_seconds < 1:
        raise click.ClickException(
            "remediation scheduler retry profile schema mismatch: "
            "base_delay_seconds must be positive"
        )
    if backoff_multiplier < 1.0:
        raise click.ClickException(
            "remediation scheduler retry profile schema mismatch: "
            "backoff_multiplier must be >= 1.0"
        )
    profile = _load_json_file(
        automation_profile_json,
        artifact="remediation scheduler automation profile",
    )
    if profile.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_automation_profile_v1"
    ):
        raise click.ClickException(
            "remediation scheduler retry profile schema mismatch: "
            "unexpected automation profile schema"
        )
    automation_profile_hash = _require_sha256(
        profile.get("automation_profile_hash"),
        "automation_profile_hash",
    )
    rules = profile.get("automation_rules")
    if not isinstance(rules, list):
        raise click.ClickException(
            "remediation scheduler retry profile schema mismatch: "
            "automation_rules must be list"
        )
    retry_rules: list[dict[str, object]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            raise click.ClickException(
                "remediation scheduler retry profile schema mismatch: "
                "rule must be object"
            )
        action_hash = _require_sha256(rule.get("action_hash"), "action_hash")
        request_hash = _require_sha256(rule.get("request_hash"), "request_hash")
        automation_mode = rule.get("automation_mode")
        control_action = rule.get("control_action")
        if automation_mode not in {"auto", "manual"}:
            raise click.ClickException(
                "remediation scheduler retry profile schema mismatch: unsupported "
                "automation_mode"
            )
        if control_action not in {
            "dispatch",
            "monitor",
            "expedite",
            "escalate",
            "no_op",
        }:
            raise click.ClickException(
                "remediation scheduler retry profile schema mismatch: unsupported "
                "control_action"
            )
        policy_mode = (
            "retry_enabled"
            if automation_mode == "auto" and control_action in {"dispatch", "expedite"}
            else "retry_disabled"
        )
        retry_rule: dict[str, object] = {
            "action_hash": action_hash,
            "request_hash": request_hash,
            "automation_mode": automation_mode,
            "control_action": control_action,
            "target_state": rule.get("target_state"),
            "policy_mode": policy_mode,
            "max_attempts": max_attempts if policy_mode == "retry_enabled" else 0,
            "base_delay_seconds": (
                base_delay_seconds if policy_mode == "retry_enabled" else 0
            ),
            "backoff_multiplier": (
                backoff_multiplier if policy_mode == "retry_enabled" else 1.0
            ),
        }
        retry_rule["retry_rule_hash"] = _record_hash(retry_rule)
        retry_rules.append(retry_rule)
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_retry_profile_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(profile.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            profile.get("execution_hash"), "execution_hash"
        ),
        "automation_profile_hash": automation_profile_hash,
        "retry_rule_count": len(retry_rules),
        "retry_rules": sorted(
            retry_rules,
            key=lambda item: (
                cast(str, item["policy_mode"]),
                cast(str, item["action_hash"]),
            ),
        ),
        "created_by": created_by,
    }
    payload["retry_profile_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-retry-orchestration")
@click.argument(
    "retry_profile_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "acknowledgement_capture_json",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating retry orchestration artifact.",
)
def plugins_lifecycle_remediation_scheduler_retry_orchestration(
    retry_profile_json: Path,
    acknowledgement_capture_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit deterministic retry queue from captured acknowledgements and policy.

    Parameters
    ----------
    retry_profile_json : Path
        Path to the retry profile JSON file.
    acknowledgement_capture_json : tuple[Path, ...]
        Path to the acknowledgement capture JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler retry orchestration schema mismatch: "
            "created_by must be non-empty"
        )
    retry_profile = _load_json_file(
        retry_profile_json,
        artifact="remediation scheduler retry profile",
    )
    if retry_profile.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_retry_profile_v1"
    ):
        raise click.ClickException(
            "remediation scheduler retry orchestration schema mismatch: "
            "unexpected retry profile schema"
        )
    retry_profile_hash = _require_sha256(
        retry_profile.get("retry_profile_hash"),
        "retry_profile_hash",
    )
    retry_rules = retry_profile.get("retry_rules")
    if not isinstance(retry_rules, list):
        raise click.ClickException(
            "remediation scheduler retry orchestration schema mismatch: "
            "retry_rules must be list"
        )
    retry_rule_by_action_hash: dict[str, dict[str, object]] = {}
    for rule in retry_rules:
        if not isinstance(rule, dict):
            raise click.ClickException(
                "remediation scheduler retry orchestration schema mismatch: "
                "rule must be object"
            )
        action_hash = _require_sha256(rule.get("action_hash"), "action_hash")
        if action_hash in retry_rule_by_action_hash:
            raise click.ClickException(
                "remediation scheduler retry orchestration schema mismatch: duplicate "
                "rule action_hash"
            )
        retry_rule_by_action_hash[action_hash] = rule

    capture_by_action_hash: dict[str, dict[str, object]] = {}
    for path in acknowledgement_capture_json:
        capture = _load_json_file(
            path,
            artifact="remediation scheduler acknowledgement capture",
        )
        if capture.get("schema") != (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_capture_v1"
        ):
            raise click.ClickException(
                "remediation scheduler retry orchestration schema mismatch: "
                "unexpected acknowledgement capture schema"
            )
        action_hash = _require_sha256(capture.get("action_hash"), "action_hash")
        if action_hash in capture_by_action_hash:
            raise click.ClickException(
                "remediation scheduler retry orchestration schema mismatch: duplicate "
                "capture action_hash"
            )
        if _require_sha256(capture.get("plan_hash"), "plan_hash") != _require_sha256(
            retry_profile.get("plan_hash"), "plan_hash"
        ):
            raise click.ClickException(
                "remediation scheduler retry orchestration schema mismatch: plan_hash "
                "mismatch"
            )
        capture_by_action_hash[action_hash] = capture

    retry_entries: list[dict[str, object]] = []
    for action_hash, capture in sorted(capture_by_action_hash.items()):
        rule = retry_rule_by_action_hash.get(action_hash)
        if rule is None:
            raise click.ClickException(
                "remediation scheduler retry orchestration schema mismatch: "
                "capture action_hash missing from retry profile"
            )
        state = capture.get("captured_state")
        if state == "completed":
            continue
        if cast(str, rule["policy_mode"]) != "retry_enabled":
            continue
        max_attempts = cast(int, rule["max_attempts"])
        base_delay_seconds = cast(int, rule["base_delay_seconds"])
        backoff_multiplier = cast(float, rule["backoff_multiplier"])
        attempt = 1
        next_delay_seconds = int(
            base_delay_seconds * (backoff_multiplier ** (attempt - 1))
        )
        entry: dict[str, object] = {
            "action_hash": action_hash,
            "request_hash": _require_sha256(
                capture.get("request_hash"), "request_hash"
            ),
            "capture_hash": _require_sha256(
                capture.get("capture_hash"), "capture_hash"
            ),
            "capture_state": state,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "next_delay_seconds": next_delay_seconds,
            "external_reference": capture.get("external_reference"),
            "retry_command_template": (
                "spo plugins lifecycle-remediation-scheduler-acknowledgement-capture "
                "AUTOMATION_PROFILE_JSON ADAPTER_HANDOFF_JSON ACTION_HASH "
                "--external-reference REF --acknowledged-by OPERATOR "
                "--captured-state STATE --note NOTE"
            ),
        }
        entry["retry_entry_hash"] = _record_hash(entry)
        retry_entries.append(entry)
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_retry_orchestration_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(retry_profile.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            retry_profile.get("execution_hash"), "execution_hash"
        ),
        "retry_profile_hash": retry_profile_hash,
        "retry_entry_count": len(retry_entries),
        "retry_entries": sorted(
            retry_entries,
            key=lambda item: (
                cast(int, item["next_delay_seconds"]),
                cast(str, item["action_hash"]),
            ),
        ),
        "created_by": created_by,
    }
    payload["retry_orchestration_hash"] = _record_hash(payload)
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
    """Emit a deterministic revocation artefact for an execution request.

    Parameters
    ----------
    request_json : Path
        Path to the request JSON file.
    revoked_by : str
        Identifier of the revoking actor.
    revocation_reference : str
        External revocation reference.
    revocation_reason : str
        Reason recorded with the revocation.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Emit a deterministic aggregate revocation list.

    Parameters
    ----------
    revocation_json : tuple[Path, ...]
        Path to the revocation JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Emit a review-only meta-transfer package manifest from audit history.

    Parameters
    ----------
    audit_paths : tuple[str, ...]
        Audit-log paths to package.
    audit_directory : str | None
        Directory of audit logs, or ``None``.
    pattern : str
        Glob pattern for audit-log discovery.
    min_records : int
        Minimum number of records required.
    package_name : str
        Name for the emitted package.
    import_target : str
        Import target for the generated package.
    console_script : str
        Console-script entry-point name.
    output : str | None
        Destination path, or ``None`` for stdout.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Export supervisor artefacts for formal model checking.

    Parameters
    ----------
    binding_spec : str
        Filesystem path to the binding-spec file.
    output : str | None
        Destination path, or ``None`` for stdout.
    module_name : str
        Name of the emitted model-checker module.
    max_tokens : int | None
        Maximum token bound per place, or ``None``.
    export_target : str
        Formal export target (e.g. ``prism`` or ``tla``).
    policy_path : str | None
        Path to the policy YAML, or ``None``.
    include_checker_readiness : bool
        Whether to include external-checker readiness.
    checker_paths : tuple[str, ...]
        Paths to external model-checker executables.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
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
            net, marking = petri_net_from_protocol(spec.protocol_net)
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

    net, marking = petri_net_from_protocol(spec.protocol_net)
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
    """Materialise deterministic neural-supervisor baseline audit artifacts.

    Parameters
    ----------
    ctx : click.Context
        The Click invocation context.
    scenario_json : Path | None
        Path to the scenario JSON, or ``None``.
    config_json : Path
        Path to the configuration JSON.
    metrics_jsonl : Path
        Destination path for the metrics JSONL.
    summary_json : Path
        Path to the summary JSON.
    manifest_json : Path | None
        Path to the manifest JSON, or ``None``.
    checkpoint_manifest : Path | None
        Path to the checkpoint manifest, or ``None``.
    plot_manifest : Path | None
        Path to the plot manifest, or ``None``.
    git_sha : str
        Git commit SHA recorded with the experiment.
    seeds : tuple[int, ...]
        Seeds for the deterministic experiment runs.
    dependency_locks : tuple[str, ...]
        Dependency lock identifiers recorded with the run.
    json_out : bool
        Whether to print machine-readable JSON output.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Replay policy rules against an audit log without applying actuation.

    Parameters
    ----------
    binding_spec : str
        Filesystem path to the binding-spec file.
    audit_log : str
        Filesystem path to the audit log.
    policy_path : str | None
        Path to the policy YAML, or ``None``.
    json_out : bool
        Whether to print machine-readable JSON output.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
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
    """Run simulation from a binding spec.

    Parameters
    ----------
    binding_spec : str
        Filesystem path to the binding-spec file.
    steps : int
        Number of simulation steps to run.
    audit : str | None
        Destination audit-log path, or ``None``.
    audit_stream : str | None
        Destination audit-stream path, or ``None``.
    seed : int
        Seed for the deterministic RNG, or ``None``.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    ClickException
        If the inputs are invalid or the operation fails.
    """
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

    spec_path = Path(binding_spec)
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
    try:
        result = simulate(
            spec,
            steps=steps,
            seed=seed,
            policy_enabled=True,
            audit_logger=audit_logger,
            binding_spec_path=spec_path,
        )
        msg = (
            f"R_good={result.r_good:.4f}  "
            f"R_bad={result.r_bad:.4f}  "
            f"regime={result.final_regime}"
        )
        if result.mean_amplitude is not None:
            msg += f"  mean_amplitude={result.mean_amplitude:.4f}"
        click.echo(msg)
    except ValueError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1) from exc
    finally:
        if audit_logger is not None:
            audit_logger.close()


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option("--output", default=None, type=click.Path(), help="Output file")
@click.option("--verify", is_flag=True, help="Verify determinism via re-execution")
def replay(log_path: str, output: str | None, verify: bool) -> None:
    """Replay an audit log and print summary.

    Parameters
    ----------
    log_path : str
        Filesystem path to the audit log.
    output : str | None
        Destination path, or ``None`` for stdout.
    verify : bool
        Whether to verify hash-chain integrity.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
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
    """Tail and replay the live audit event stream.

    Parameters
    ----------
    stream_path : str
        Filesystem path to the audit event stream.
    stream_format : str
        Audit stream format.
    from_start : bool
        Whether to replay from the start of the stream.
    max_events : int | None
        Maximum number of events to read, or ``None``.
    poll_interval : float
        Poll interval in seconds.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
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
    """Generate coherence report from audit log.

    Parameters
    ----------
    log_path : str
        Filesystem path to the audit log.
    json_out : bool
        Whether to print machine-readable JSON output.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
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
    """Generate a human-readable explanation report from an audit log.

    Parameters
    ----------
    log_path : str
        Filesystem path to the audit log.
    markdown_out : str | None
        Destination Markdown path, or ``None``.
    pdf_out : str | None
        Destination PDF path, or ``None``.
    max_actions : int
        Maximum number of actions to include.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
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


@main.command("digital-twin-observability-bundle")
@click.argument(
    "operator_evidence_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--scheduler-dashboard-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional scheduler execution dashboard JSON for replay linkage.",
)
@click.option(
    "--scheduler-replay-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional scheduler acknowledgement replay JSON for replay linkage.",
)
@click.option(
    "--metric-prefix",
    default="spo",
    show_default=True,
    help="Prometheus metric prefix for rendered observability text.",
)
@click.option(
    "--created-by",
    required=True,
    help="Operator component creating the observability bundle artifact.",
)
def digital_twin_observability_bundle(
    operator_evidence_json: Path,
    scheduler_dashboard_json: Path | None,
    scheduler_replay_json: Path | None,
    metric_prefix: str,
    created_by: str,
) -> None:
    """Bundle digital-twin Prometheus telemetry with replay linkage evidence.

    Parameters
    ----------
    operator_evidence_json : Path
        Path to the operator-evidence JSON.
    scheduler_dashboard_json : Path | None
        Path to the scheduler dashboard JSON, or ``None``.
    scheduler_replay_json : Path | None
        Path to the scheduler replay JSON, or ``None``.
    metric_prefix : str
        Prefix applied to emitted metric names.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "digital-twin observability bundle schema mismatch: "
            "created_by must be non-empty"
        )
    evidence = _load_json_file(
        operator_evidence_json,
        artifact="digital-twin operator evidence",
    )
    observability = RuntimeObservability(metric_prefix=metric_prefix)
    try:
        prometheus_text = observability.digital_twin_prometheus_text(evidence)
    except ValueError as exc:
        raise click.ClickException(
            f"digital-twin observability bundle schema mismatch: {exc}"
        ) from exc

    replay_linkage: dict[str, object] = {
        "scheduler_dashboard_present": scheduler_dashboard_json is not None,
        "scheduler_replay_present": scheduler_replay_json is not None,
        "scheduler_row_count": 0,
        "scheduler_overdue_count": 0,
        "scheduler_blocked_count": 0,
        "scheduler_completed_count": 0,
        "scheduler_replay_count": 0,
        "scheduler_replay_blocked_count": 0,
        "scheduler_replay_completed_count": 0,
        "scheduler_dashboard_hash": None,
        "scheduler_replay_hash": None,
    }

    if scheduler_dashboard_json is not None:
        dashboard = _load_json_file(
            scheduler_dashboard_json,
            artifact="remediation scheduler execution dashboard",
        )
        if dashboard.get("schema") != (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_execution_dashboard_v1"
        ):
            raise click.ClickException(
                "digital-twin observability bundle schema mismatch: "
                "unexpected scheduler dashboard schema"
            )
        dashboard_hash = _require_sha256(
            dashboard.get("dashboard_hash"), "dashboard_hash"
        )
        rows = dashboard.get("rows")
        if not isinstance(rows, list):
            raise click.ClickException(
                "digital-twin observability bundle schema mismatch: "
                "scheduler dashboard rows must be list"
            )
        blocked_count = 0
        completed_count = 0
        overdue_count = 0
        for row in rows:
            if not isinstance(row, dict):
                raise click.ClickException(
                    "digital-twin observability bundle schema mismatch: "
                    "scheduler dashboard row must be object"
                )
            state = row.get("effective_state")
            if state == "blocked":
                blocked_count += 1
            if state == "completed":
                completed_count += 1
            if bool(row.get("overdue", False)):
                overdue_count += 1
        replay_linkage["scheduler_row_count"] = len(rows)
        replay_linkage["scheduler_overdue_count"] = overdue_count
        replay_linkage["scheduler_blocked_count"] = blocked_count
        replay_linkage["scheduler_completed_count"] = completed_count
        replay_linkage["scheduler_dashboard_hash"] = dashboard_hash

    if scheduler_replay_json is not None:
        replay = _load_json_file(
            scheduler_replay_json,
            artifact="remediation scheduler acknowledgement replay",
        )
        if replay.get("schema") != (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_replay_v1"
        ):
            raise click.ClickException(
                "digital-twin observability bundle schema mismatch: "
                "unexpected scheduler replay schema"
            )
        replay_hash = _require_sha256(replay.get("replay_hash"), "replay_hash")
        replay_rows = replay.get("rows")
        if not isinstance(replay_rows, list):
            raise click.ClickException(
                "digital-twin observability bundle schema mismatch: "
                "scheduler replay rows must be list"
            )
        replay_blocked = 0
        replay_completed = 0
        for row in replay_rows:
            if not isinstance(row, dict):
                raise click.ClickException(
                    "digital-twin observability bundle schema mismatch: "
                    "scheduler replay row must be object"
                )
            state = row.get("state")
            if state == "blocked":
                replay_blocked += 1
            if state == "completed":
                replay_completed += 1
        replay_linkage["scheduler_replay_count"] = len(replay_rows)
        replay_linkage["scheduler_replay_blocked_count"] = replay_blocked
        replay_linkage["scheduler_replay_completed_count"] = replay_completed
        replay_linkage["scheduler_replay_hash"] = replay_hash

    accepted_count = evidence.get("accepted_count", 0)
    rejected_count = evidence.get("rejected_count", 0)
    if not isinstance(accepted_count, int) or isinstance(accepted_count, bool):
        raise click.ClickException(
            "digital-twin observability bundle schema mismatch: "
            "accepted_count must be an integer"
        )
    if not isinstance(rejected_count, int) or isinstance(rejected_count, bool):
        raise click.ClickException(
            "digital-twin observability bundle schema mismatch: "
            "rejected_count must be an integer"
        )

    bundle_payload: dict[str, object] = {
        "schema": "scpn_digital_twin_observability_bundle_v1",
        "version": "1.0.0",
        "contract_hash": _require_sha256(
            evidence.get("contract_hash"), "contract_hash"
        ),
        "status": str(evidence.get("status")),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "prometheus_metric_prefix": metric_prefix,
        "prometheus_text": prometheus_text,
        "replay_linkage": replay_linkage,
        "created_by": created_by,
    }
    bundle_payload["bundle_hash"] = _record_hash(bundle_payload)
    click.echo(json.dumps(bundle_payload, indent=2, sort_keys=True))


@main.command("digital-twin-grafana-dashboard-pack")
@click.argument(
    "observability_bundle_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--adapter-family",
    required=True,
    help="Adapter family label (for example: rest, grpc, kafka, hardware).",
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating Grafana dashboard pack artifact.",
)
def digital_twin_grafana_dashboard_pack(
    observability_bundle_json: Path,
    adapter_family: str,
    created_by: str,
) -> None:
    """Emit deterministic Grafana dashboard pack from observability bundle.

    Parameters
    ----------
    observability_bundle_json : Path
        Path to the observability bundle JSON file.
    adapter_family : str
        Adapter family label.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "digital-twin grafana dashboard pack schema mismatch: "
            "created_by must be non-empty"
        )
    if not adapter_family:
        raise click.ClickException(
            "digital-twin grafana dashboard pack schema mismatch: "
            "adapter_family must be non-empty"
        )
    bundle = _load_json_file(
        observability_bundle_json,
        artifact="digital-twin observability bundle",
    )
    if bundle.get("schema") != "scpn_digital_twin_observability_bundle_v1":
        raise click.ClickException(
            "digital-twin grafana dashboard pack schema mismatch: "
            "unexpected observability bundle schema"
        )
    bundle_hash = _require_sha256(bundle.get("bundle_hash"), "bundle_hash")
    contract_hash = _require_sha256(bundle.get("contract_hash"), "contract_hash")
    metric_prefix = bundle.get("prometheus_metric_prefix")
    if not isinstance(metric_prefix, str) or not metric_prefix:
        raise click.ClickException(
            "digital-twin grafana dashboard pack schema mismatch: "
            "prometheus_metric_prefix must be non-empty string"
        )
    panels = [
        {
            "title": "Sync Acceptance Ratio",
            "kind": "timeseries",
            "query_template": (
                f"sum({metric_prefix}_digital_twin_sync_accepted_total"
                f'{{contract_hash="{contract_hash}"}}) / '
                f"(sum({metric_prefix}_digital_twin_sync_accepted_total"
                f'{{contract_hash="{contract_hash}"}}) + '
                f"sum({metric_prefix}_digital_twin_sync_rejected_total"
                f'{{contract_hash="{contract_hash}"}}))'
            ),
            "unit": "percentunit",
        },
        {
            "title": "Twin Residual Max",
            "kind": "timeseries",
            "query_template": (
                f'{metric_prefix}_digital_twin_max_abs_residual{{contract_hash="{contract_hash}"}}'
            ),
            "unit": "none",
        },
        {
            "title": "Unhealthy Adapter Count",
            "kind": "stat",
            "query_template": (
                f'{metric_prefix}_digital_twin_unhealthy_adapter_count{{contract_hash="{contract_hash}"}}'
            ),
            "unit": "short",
        },
        {
            "title": "Twin Mismatch Reasons",
            "kind": "barchart",
            "query_template": (
                f"sum by (reason) "
                f"({metric_prefix}_digital_twin_mismatch_reason_count"
                f'{{contract_hash="{contract_hash}"}})'
            ),
            "unit": "short",
        },
        {
            "title": "Scheduler Overdue Actions",
            "kind": "stat",
            "query_template": "linked_bundle.replay_linkage.scheduler_overdue_count",
            "unit": "short",
        },
    ]
    for panel in panels:
        panel["panel_hash"] = _record_hash(panel)
    payload: dict[str, object] = {
        "schema": "scpn_digital_twin_grafana_dashboard_pack_v1",
        "version": "1.0.0",
        "adapter_family": adapter_family,
        "contract_hash": contract_hash,
        "observability_bundle_hash": bundle_hash,
        "panel_count": len(panels),
        "panels": panels,
        "created_by": created_by,
    }
    payload["dashboard_pack_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@main.command("digital-twin-live-deployment-playbook")
@click.argument(
    "observability_bundle_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "grafana_dashboard_pack_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--environment-name",
    required=True,
    help="Deployment environment name (for example: prod-eu-west).",
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating live deployment playbook artifact.",
)
def digital_twin_live_deployment_playbook(
    observability_bundle_json: Path,
    grafana_dashboard_pack_json: Path,
    environment_name: str,
    created_by: str,
) -> None:
    """Emit deterministic live deployment playbook from observability artifacts.

    Parameters
    ----------
    observability_bundle_json : Path
        Path to the observability bundle JSON file.
    grafana_dashboard_pack_json : Path
        Path to the Grafana dashboard-pack JSON.
    environment_name : str
        Target environment name.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "created_by must be non-empty"
        )
    if not environment_name:
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "environment_name must be non-empty"
        )
    bundle = _load_json_file(
        observability_bundle_json,
        artifact="digital-twin observability bundle",
    )
    if bundle.get("schema") != "scpn_digital_twin_observability_bundle_v1":
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "unexpected observability bundle schema"
        )
    dashboard_pack = _load_json_file(
        grafana_dashboard_pack_json,
        artifact="digital-twin grafana dashboard pack",
    )
    if dashboard_pack.get("schema") != "scpn_digital_twin_grafana_dashboard_pack_v1":
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "unexpected grafana dashboard pack schema"
        )
    bundle_hash = _require_sha256(bundle.get("bundle_hash"), "bundle_hash")
    dashboard_linked_bundle_hash = _require_sha256(
        dashboard_pack.get("observability_bundle_hash"),
        "observability_bundle_hash",
    )
    if bundle_hash != dashboard_linked_bundle_hash:
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "observability_bundle_hash mismatch"
        )
    replay_linkage = bundle.get("replay_linkage")
    if not isinstance(replay_linkage, dict):
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "replay_linkage must be object"
        )
    overdue = replay_linkage.get("scheduler_overdue_count")
    blocked = replay_linkage.get("scheduler_blocked_count")
    if not isinstance(overdue, int) or overdue < 0:
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "scheduler_overdue_count must be non-negative integer"
        )
    if not isinstance(blocked, int) or blocked < 0:
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "scheduler_blocked_count must be non-negative integer"
        )
    rollout_gate = (
        "blocked" if blocked > 0 else ("degraded" if overdue > 0 else "ready")
    )
    steps = [
        {
            "id": "publish-metrics",
            "description": (
                "Expose Prometheus text from digital-twin observability bundle."
            ),
            "command_template": (
                "spo digital-twin-observability-bundle EVIDENCE_JSON "
                "--created-by OPERATOR"
            ),
        },
        {
            "id": "publish-dashboards",
            "description": "Deploy Grafana dashboard pack for adapter family.",
            "command_template": (
                "spo digital-twin-grafana-dashboard-pack OBS_BUNDLE_JSON "
                "--adapter-family FAMILY --created-by OPERATOR"
            ),
        },
        {
            "id": "verify-scheduler-health",
            "description": "Review overdue/blocked scheduler telemetry linkage.",
            "command_template": (
                "Inspect replay_linkage.scheduler_overdue_count and "
                "replay_linkage.scheduler_blocked_count in observability bundle"
            ),
        },
    ]
    for step in steps:
        step["step_hash"] = _record_hash(step)
    payload: dict[str, object] = {
        "schema": "scpn_digital_twin_live_deployment_playbook_v1",
        "version": "1.0.0",
        "environment_name": environment_name,
        "contract_hash": _require_sha256(bundle.get("contract_hash"), "contract_hash"),
        "observability_bundle_hash": bundle_hash,
        "dashboard_pack_hash": _require_sha256(
            dashboard_pack.get("dashboard_pack_hash"),
            "dashboard_pack_hash",
        ),
        "rollout_gate": rollout_gate,
        "step_count": len(steps),
        "steps": steps,
        "created_by": created_by,
    }
    payload["playbook_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@main.group()
def queuewaves() -> None:
    """QueueWaves — real-time cascade failure detector."""


main.add_command(queuewaves)


@queuewaves.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8080, type=int)
def serve(config_path: str, host: str, port: int) -> None:
    """Start QueueWaves server.

    Parameters
    ----------
    config_path : str
        Filesystem path to the configuration file.
    host : str
        Host interface to bind.
    port : int
        Port to bind.
    """
    from scpn_phase_orchestrator.apps.queuewaves.server import run_server

    run_server(config_path, host=host, port=port)


@queuewaves.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def check(config_path: str) -> None:
    """One-shot: scrape → analyse → exit 0 (ok) or 1 (anomalies).

    Parameters
    ----------
    config_path : str
        Filesystem path to the configuration file.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
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
    """Create a domainpack directory structure with template files.

    Parameters
    ----------
    domain_name : str
        Name of the domainpack to scaffold.
    use_llm : bool
        Whether to use the LLM-assisted scaffolder.
    description : str | None
        Domain description text, or ``None``.
    llm_response_json : str | None
        Path to a cached LLM response JSON, or ``None``.

    Raises
    ------
    BadParameter
        If a CLI argument is invalid.
    ClickException
        If the inputs are invalid or the operation fails.
    """
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
    """Generate reviewable binding artefacts from symbolic domain intent.

    Parameters
    ----------
    intent : str
        Symbolic domain intent text.
    name : str
        The span or resource name.
    output_dir : str | None
        Destination directory, or ``None``.
    oscillators_per_layer : int
        Number of oscillators per generated layer.
    dry_run_steps : int
        Number of integration steps for the embedded dry run.
    """
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
@click.option(
    "--dataset",
    default=None,
    help="Real-data demo dataset alias/path/URL. Use heartbeat.csv for PhysioNet HRB.",
)
@click.option(
    "--target",
    default="coherence",
    type=click.Choice(["coherence"]),
    help="Review target for real-data demo.",
)
@click.option("--steps", default=100, help="Number of simulation steps.")
@click.option("--port", default=8000, help="Server port.")
def demo(domain: str, dataset: str | None, target: str, steps: int, port: int) -> None:
    """Run a self-contained demo: simulate + print live coherence.

    Parameters
    ----------
    domain : str
        Domain label.
    dataset : str | None
        Dataset name, or ``None``.
    target : str
        Target metric or channel.
    steps : int
        Number of simulation steps to run.
    port : int
        Port to bind.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
    if dataset is not None:
        _run_real_data_demo(dataset=dataset, target=target, steps=steps, port=port)
        return

    domainpack_dir = Path(__file__).parent.parent.parent / "domainpacks"
    spec_path = _contained_domainpack_spec(domainpack_dir, domain)
    if not spec_path.exists():
        # Try relative to cwd.
        spec_path = _contained_domainpack_spec(Path("domainpacks"), domain)
    if not spec_path.exists():
        listing_root = (
            domainpack_dir if domainpack_dir.exists() else Path("domainpacks")
        )
        available = sorted(
            d.name
            for d in listing_root.iterdir()
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


def _run_real_data_demo(*, dataset: str, target: str, steps: int, port: int) -> None:
    if steps < 1:
        raise click.BadParameter("steps must be positive")
    if target != "coherence":
        raise click.BadParameter("only target=coherence is supported")
    csv_text, source = _load_demo_dataset(dataset)
    proposal = propose_binding_from_time_series_csv(
        csv_text,
        sample_rate_hz=None,
        project_name="heartbeat_coherence_demo",
    )
    record = cast(dict[str, Any], proposal.to_audit_record())
    binding_record = cast(dict[str, Any], record["binding"])
    source_record = cast(dict[str, Any], record["source"])
    metadata_record = cast(dict[str, Any], record["metadata"])
    runtime_record = cast(dict[str, Any], record["runtime"])
    provenance = cast(dict[str, Any], binding_record["provenance"])
    click.echo("SPO Real-Data Demo — heartbeat coherence")
    click.echo(f"  Dataset: {dataset}")
    click.echo(f"  Source: {source}")
    click.echo(f"  Citation: {_PHYSIONET_HEARTBEAT_CITATION}")
    click.echo(f"  Target: {target}")
    click.echo(f"  Rows used: {source_record['sample_count']}")
    click.echo(f"  Sample rate: {provenance['sample_rate_hz']:.6g} Hz")
    click.echo(f"  Inferred channels: {', '.join(binding_record['inferred_channels'])}")
    click.echo(f"  Proposal mode: {metadata_record['proposal_mode']}")
    click.echo(f"  Replay status: {runtime_record['replay_status']}")
    click.echo(f"  Initial R: {runtime_record['R']:.6f}")
    click.echo(f"  Initial K: {runtime_record['K']:.6f}")
    click.echo("-" * 40)
    click.echo("Review-only binding YAML:")
    click.echo(proposal.binding.yaml_text, nl=False)
    click.echo("-" * 40)
    click.echo("Dashboard/replay path:")
    click.echo(
        "  spo auto-bind time-series-csv heartbeat.csv "
        "--project-name heartbeat_coherence_demo --json-out"
    )
    click.echo(f"  spo demo --dataset heartbeat.csv --target coherence --steps {steps}")
    click.echo("  cd deploy && docker compose up")
    click.echo(f"  Open http://localhost:{port} (dashboard)")


def _load_demo_dataset(dataset: str) -> tuple[str, str]:
    if dataset == "heartbeat.csv":
        raw = _download_text(_PHYSIONET_HEARTBEAT_URL, max_bytes=512_000)
        return _normalise_heartbeat_csv(raw, max_rows=256), _PHYSIONET_HEARTBEAT_URL
    path = Path(dataset)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8"), str(path)
    if dataset.startswith(("https://", "http://")):
        return _download_text(dataset, max_bytes=512_000), dataset
    raise click.BadParameter(
        "dataset must be heartbeat.csv, an existing local CSV path, or an http(s) URL"
    )


def _download_text(url: str, *, max_bytes: int) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise click.ClickException("dataset URL must be an absolute HTTPS URL")
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    connection = http.client.HTTPSConnection(
        parsed.hostname,
        parsed.port,
        timeout=20,
    )
    try:
        connection.request(
            "GET",
            path,
            headers={"User-Agent": "scpn-phase-orchestrator-demo/1"},
        )
        response = connection.getresponse()
        if response.status < 200 or response.status >= 300:
            raise click.ClickException(
                f"demo dataset download failed with HTTP {response.status}"
            )
        payload = response.read(max_bytes + 1)
    finally:
        connection.close()
    if len(payload) > max_bytes:
        raise click.ClickException("demo dataset is too large")
    return payload.decode("utf-8")


def _normalise_heartbeat_csv(raw: str, *, max_rows: int) -> str:
    reader = csv.DictReader(io.StringIO(raw))
    required = {"rr_ms", "hr_bpm"}
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        raise click.ClickException("heartbeat dataset must include rr_ms and hr_bpm")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["time", "rr_ms", "hr_bpm"])
    writer.writeheader()
    samples: list[tuple[float, float]] = []
    for row in reader:
        rr_ms = _finite_csv_float(row.get("rr_ms"), "rr_ms")
        hr_bpm = _finite_csv_float(row.get("hr_bpm"), "hr_bpm")
        samples.append((rr_ms, hr_bpm))
        if len(samples) >= max_rows:
            break
    if len(samples) < 3:
        raise click.ClickException("heartbeat dataset must contain at least 3 rows")
    sample_period_s = float(np.median([rr_ms for rr_ms, _hr_bpm in samples])) / 1000.0
    if not np.isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise click.ClickException("heartbeat dataset has invalid RR interval timing")
    for index, (rr_ms, hr_bpm) in enumerate(samples):
        writer.writerow(
            {
                "time": f"{index * sample_period_s:.6f}",
                "rr_ms": f"{rr_ms:.9g}",
                "hr_bpm": f"{hr_bpm:.9g}",
            }
        )
    return output.getvalue()


def _finite_csv_float(value: object, field: str) -> float:
    if not isinstance(value, str):
        raise click.ClickException(f"heartbeat dataset has non-numeric {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(
            f"heartbeat dataset has non-numeric {field}"
        ) from exc
    if not np.isfinite(number):
        raise click.ClickException(f"heartbeat dataset has non-finite {field}")
    return number


def _contained_domainpack_spec(domainpack_root: Path, domain: str) -> Path:
    if not isinstance(domain, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", domain):
        raise click.BadParameter("domain must match [A-Za-z0-9_-]+")
    root = domainpack_root.resolve()
    spec_path = (root / domain / "binding_spec.yaml").resolve()
    try:
        spec_path.relative_to(root)
    except ValueError as exc:
        raise click.BadParameter("domain resolves outside domainpack root") from exc
    return spec_path
