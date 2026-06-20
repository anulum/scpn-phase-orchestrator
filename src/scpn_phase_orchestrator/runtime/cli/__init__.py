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
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import click
import numpy as np

from scpn_phase_orchestrator.autotune.binding_proposal import (
    propose_binding_from_time_series_csv,
)
from scpn_phase_orchestrator.binding import (
    compile_symbolic_binding,
    format_resolved_binding_config,
    load_binding_spec,
    resolved_binding_config,
    validate_binding_spec,
)
from scpn_phase_orchestrator.meta import CrossDomainMetaTransfer
from scpn_phase_orchestrator.reporting.summary import build_audit_report_summary
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.runtime.audit_stream import (
    AuditStreamEvent,
    iter_event_stream,
    read_event_stream,
    verify_event_stream_integrity,
)
from scpn_phase_orchestrator.runtime.cli import binding as binding
from scpn_phase_orchestrator.runtime.cli import diagnostics as diagnostics
from scpn_phase_orchestrator.runtime.cli import monitoring as monitoring
from scpn_phase_orchestrator.runtime.cli import plugins as plugins
from scpn_phase_orchestrator.runtime.cli._app import (
    _PHYSIONET_HEARTBEAT_CITATION,
    _PHYSIONET_HEARTBEAT_URL,
    main,
)
from scpn_phase_orchestrator.runtime.cli._payloads import (
    _load_json_file,
    _record_hash,
    _require_sha256,
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
