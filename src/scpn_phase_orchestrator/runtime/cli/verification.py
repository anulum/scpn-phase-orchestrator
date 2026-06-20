# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI formal-export and policy verification commands

"""Command-line entry point for validation, replay, export, and review workflows.

The CLI wraps public SPO APIs behind explicit commands for binding validation,
inspection, auto-binding proposals, coupling estimation, formal export, replay,
plugin catalogs, scaffolding, and selected runtime utilities. Commands validate
local inputs and emit text or JSON review artifacts; they do not push commits,
start network services, or perform live actuation unless an explicit subcommand
is invoked for that runtime path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import click

from scpn_phase_orchestrator.binding import (
    load_binding_spec,
    validate_binding_spec,
)
from scpn_phase_orchestrator.runtime.cli._app import (
    main,
)
from scpn_phase_orchestrator.runtime.replay import ReplayEngine
from scpn_phase_orchestrator.runtime.simulation import petri_net_from_protocol
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
