# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio deployment package builders

"""Export manifest, deployment readiness/package, and command-table builders."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256

from scpn_phase_orchestrator.studio.workflow import (
    ExportManifest,
    StudioProjectState,
)

from ._shared import (
    _blocked_target,
    _deployment_blocked_reasons,
    _require_non_empty_text,
    _require_sequence,
)


def build_export_manifests(
    *,
    project_name: str,
    binding_yaml: str,
    audit_payload: Mapping[str, object],
    validation_errors: Sequence[str],
) -> tuple[ExportManifest, ...]:
    """Build review-only export manifests for Studio.

    Parameters
    ----------
    project_name : str
        Name of the project.
    binding_yaml : str
        The binding spec serialised as YAML.
    audit_payload : Mapping[str, object]
        The audit payload mapping.
    validation_errors : Sequence[str]
        Binding validation error messages.

    Returns
    -------
    tuple[ExportManifest, ...]
        Review-only export manifests for Studio.
    """
    deploy_warnings = disabled_export_reasons(validation_errors)
    audit_export_payload = {
        **dict(audit_payload),
        "enabled": not deploy_warnings,
        "disabled_reasons": list(deploy_warnings),
    }
    audit_json = json.dumps(audit_export_payload, sort_keys=True, indent=2)
    docker_payload = json.dumps(
        {
            "project_name": project_name,
            "image": "scpn-phase-orchestrator:local",
            "command": "spo run binding_spec.yaml --audit audit.jsonl",
            "enabled": not deploy_warnings,
            "disabled_reasons": list(deploy_warnings),
        },
        sort_keys=True,
        indent=2,
    )
    wasm_payload = json.dumps(
        {
            "project_name": project_name,
            "target": "wasm_review_manifest",
            "enabled": not deploy_warnings,
            "disabled_reasons": list(deploy_warnings),
        },
        sort_keys=True,
        indent=2,
    )
    return (
        ExportManifest.review_artifact(
            target_kind="binding_spec",
            file_name="binding_spec.yaml",
            payload=binding_yaml,
            command="spo run binding_spec.yaml --audit audit.jsonl",
            warnings=deploy_warnings,
        ),
        ExportManifest.review_artifact(
            target_kind="audit_summary",
            file_name="spo_studio_audit.json",
            payload=audit_json,
            command="spo audit summary spo_studio_audit.json",
            warnings=deploy_warnings,
        ),
        ExportManifest.review_artifact(
            target_kind="docker_manifest",
            file_name="docker_manifest.json",
            payload=docker_payload,
            command="docker compose config",
            warnings=deploy_warnings,
        ),
        ExportManifest.review_artifact(
            target_kind="wasm_manifest",
            file_name="wasm_manifest.json",
            payload=wasm_payload,
            command="spo export wasm --manifest wasm_manifest.json",
            warnings=deploy_warnings,
        ),
    )


def build_deployment_readiness(
    project_state: StudioProjectState,
) -> dict[str, object]:
    """Return target-specific deployment readiness guidance for Studio.

    Parameters
    ----------
    project_state : StudioProjectState
        The Studio project state.

    Returns
    -------
    dict[str, object]
        Target-specific deployment readiness guidance for Studio.
    """
    blocked_reasons = _deployment_blocked_reasons(project_state.exports)
    if blocked_reasons:
        return {
            "project_name": project_state.project_name,
            "overall_status": "blocked",
            "operator_next_step": "fix binding validation errors",
            "targets": [
                _blocked_target("docker", blocked_reasons),
                _blocked_target("wasm", blocked_reasons),
                _blocked_target("hardware", blocked_reasons),
            ],
        }

    return {
        "project_name": project_state.project_name,
        "overall_status": "review_ready",
        "operator_next_step": "review target-specific packaging",
        "targets": [
            {
                "target": "docker",
                "status": "ready",
                "required_artifacts": [
                    "binding_spec.yaml",
                    "spo_studio_audit.json",
                    "docker_manifest.json",
                ],
                "commands": [
                    "docker compose config",
                    "docker build -t scpn-phase-orchestrator:local .",
                    "docker run --rm -v $PWD:/workspace "
                    "scpn-phase-orchestrator:local "
                    "spo run binding_spec.yaml --audit audit.jsonl",
                ],
                "operator_action": "run docker manifest review before packaging",
            },
            {
                "target": "wasm",
                "status": "ready",
                "required_artifacts": [
                    "binding_spec.yaml",
                    "spo_studio_audit.json",
                    "wasm_manifest.json",
                ],
                "commands": [
                    "cd spo-kernel && wasm-pack build crates/spo-wasm "
                    "--target web --out-dir ../../../docs/wasm-pkg",
                ],
                "operator_action": "review browser-safe replay constraints",
            },
            {
                "target": "hardware",
                "status": "postponed",
                "required_artifacts": [
                    "binding_spec.yaml",
                    "spo_studio_audit.json",
                    "verified_hardware_target_evidence",
                ],
                "commands": [],
                "operator_action": "attach verified hardware-target evidence",
            },
        ],
    }


def build_deployment_package(
    project_state: StudioProjectState,
) -> dict[str, object]:
    """Return a deterministic deployment package manifest for Studio.

    Parameters
    ----------
    project_state : StudioProjectState
        The Studio project state.

    Returns
    -------
    dict[str, object]
        A deterministic deployment package manifest for Studio.
    """
    readiness = build_deployment_readiness(project_state)
    targets = _readiness_targets(readiness)
    blocked_reasons = _deployment_blocked_reasons(project_state.exports)
    return {
        "package_kind": "studio_deployment_package",
        "project_name": project_state.project_name,
        "overall_status": readiness["overall_status"],
        "ready_targets": [
            target["target"] for target in targets if target["status"] == "ready"
        ],
        "postponed_targets": [
            target["target"] for target in targets if target["status"] == "postponed"
        ],
        "blocked_targets": [
            target["target"] for target in targets if target["status"] == "blocked"
        ],
        "blocked_reasons": list(blocked_reasons),
        "required_artifacts": _unique_artifacts(targets),
        "export_artifacts": [
            {
                "target_kind": manifest.target_kind,
                "file_name": manifest.file_name,
                "payload_sha256": manifest.payload_sha256,
                "safety_posture": manifest.safety_posture,
                "warnings": list(manifest.warnings),
            }
            for manifest in project_state.exports
        ],
        "commands": list(build_command_table(project_state)),
        "safety_gates": [
            "local replay completed",
            (
                "binding validation blocked"
                if blocked_reasons
                else "binding validation passed"
            ),
            "live actuation disabled",
            "hardware output requires verified evidence",
        ],
        "readiness": readiness,
    }


def build_service_process_manifest(
    project_state: StudioProjectState,
) -> dict[str, object]:
    """Return localhost-only service process packaging for Studio deployment.

    Parameters
    ----------
    project_state : StudioProjectState
        The Studio project state.

    Returns
    -------
    dict[str, object]
        Localhost-only service process packaging for Studio deployment.
    """
    blocked_reasons = _deployment_blocked_reasons(project_state.exports)
    if blocked_reasons:
        return {
            "manifest_kind": "studio_service_process_manifest",
            "project_name": project_state.project_name,
            "overall_status": "blocked",
            "execution_mode": "operator_invoked",
            "network_opened": False,
            "actuation_permitted": False,
            "hardware_write_permitted": False,
            "host_bind": "127.0.0.1",
            "compose_file": "spo_studio_services.compose.yaml",
            "services": [],
            "blocked_reasons": list(blocked_reasons),
            "required_artifacts": [],
            "compose_yaml": "",
            "compose_yaml_sha256": "",
        }

    services = _studio_service_processes()
    compose_yaml = _render_service_compose_yaml(services)
    return {
        "manifest_kind": "studio_service_process_manifest",
        "project_name": project_state.project_name,
        "overall_status": "operator_ready",
        "execution_mode": "operator_invoked",
        "network_opened": False,
        "actuation_permitted": False,
        "hardware_write_permitted": False,
        "host_bind": "127.0.0.1",
        "compose_file": "spo_studio_services.compose.yaml",
        "services": services,
        "blocked_reasons": [],
        "required_artifacts": [
            "binding_spec.yaml",
            "spo_studio_audit.json",
            "docker_manifest.json",
            "owned_connector_runtime.json",
        ],
        "operator_commands": [
            "docker compose -f spo_studio_services.compose.yaml config",
            "docker compose -f spo_studio_services.compose.yaml up spo-studio-ui",
        ],
        "compose_yaml": compose_yaml,
        "compose_yaml_sha256": sha256(compose_yaml.encode("utf-8")).hexdigest(),
    }


def build_package_materialisation_plan(
    project_state: StudioProjectState,
) -> dict[str, object]:
    """Return ordered, operator-invoked package materialisation commands.

    Parameters
    ----------
    project_state : StudioProjectState
        The Studio project state.

    Returns
    -------
    dict[str, object]
        Ordered, operator-invoked package materialisation commands.
    """
    package = build_deployment_package(project_state)
    command_rows = build_command_table(project_state)
    commands = [
        {
            "step": index,
            "target": _require_non_empty_text(row.get("target"), "target"),
            "command": _require_non_empty_text(row.get("command"), "command"),
            "status": _require_non_empty_text(row.get("status"), "status"),
            "requires_operator": True,
            "writes_artifact": _materialisation_command_writes_artifact(
                row.get("command")
            ),
        }
        for index, row in enumerate(command_rows, 1)
    ]
    readiness = build_deployment_readiness(project_state)
    targets = _readiness_targets(readiness)
    return {
        "plan_kind": "studio_package_materialisation_plan",
        "project_name": project_state.project_name,
        "overall_status": package["overall_status"],
        "execution_mode": "operator_invoked",
        "network_opened": False,
        "hardware_write_permitted": False,
        "commands": commands,
        "blocked_targets": list(
            _require_sequence(package.get("blocked_targets"), "blocked_targets")
        ),
        "blocked_reasons": list(
            _require_sequence(package.get("blocked_reasons"), "blocked_reasons")
        ),
        "postponed_targets": [
            {
                "target": target["target"],
                "reason": _require_non_empty_text(
                    target.get("operator_action"),
                    "operator_action",
                ),
            }
            for target in targets
            if target["status"] == "postponed"
        ],
        "required_artifacts": list(
            _require_sequence(package.get("required_artifacts"), "required_artifacts")
        ),
        "safety_gates": list(
            _require_sequence(package.get("safety_gates"), "safety_gates")
        ),
    }


def build_operator_checklist(
    project_state: StudioProjectState,
) -> tuple[dict[str, object], ...]:
    """Return beginner-friendly ordered deployment steps for Studio.

    Parameters
    ----------
    project_state : StudioProjectState
        The Studio project state.

    Returns
    -------
    tuple[dict[str, object], ...]
        Beginner-friendly ordered deployment steps for Studio.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    readiness = build_deployment_readiness(project_state)
    validation_blocked = readiness["overall_status"] == "blocked"
    steps: list[dict[str, object]] = [
        {
            "step": 1,
            "title": "Run local replay",
            "status": (
                "complete"
                if project_state.runtime.replay_status == "completed"
                else "blocked"
            ),
            "detail": project_state.runtime.replay_status,
        },
        {
            "step": 2,
            "title": "Validate binding",
            "status": "blocked" if validation_blocked else "complete",
            "detail": (
                "; ".join(_deployment_blocked_reasons(project_state.exports))
                if validation_blocked
                else "binding validation passed"
            ),
        },
    ]
    for target in _require_sequence(readiness.get("targets"), "targets"):
        if not isinstance(target, Mapping):
            raise ValueError("readiness targets must be mappings")
        target_name = _require_non_empty_text(target.get("target"), "target")
        status = _require_non_empty_text(target.get("status"), "status")
        operator_action = _require_non_empty_text(
            target.get("operator_action"),
            "operator_action",
        )
        blocked_detail = "; ".join(
            str(reason)
            for reason in _require_sequence(
                target.get("blocked_reasons", ()),
                "blocked_reasons",
            )
        )
        steps.append(
            {
                "step": len(steps) + 1,
                "title": f"Review {target_name} packaging",
                "target": target_name,
                "status": status,
                "detail": blocked_detail or operator_action,
            }
        )
    return tuple(steps)


def build_command_table(
    project_state: StudioProjectState,
) -> tuple[dict[str, object], ...]:
    """Return copyable deployment-review commands for ready targets.

    Parameters
    ----------
    project_state : StudioProjectState
        The Studio project state.

    Returns
    -------
    tuple[dict[str, object], ...]
        Copyable deployment-review commands for ready targets.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    readiness = build_deployment_readiness(project_state)
    rows: list[dict[str, object]] = []
    for target in _require_sequence(readiness.get("targets"), "targets"):
        if not isinstance(target, Mapping):
            raise ValueError("readiness targets must be mappings")
        status = _require_non_empty_text(target.get("status"), "status")
        if status == "blocked":
            continue
        target_name = _require_non_empty_text(target.get("target"), "target")
        commands = target.get("commands", ())
        if isinstance(commands, str | bytes) or not isinstance(commands, Sequence):
            raise ValueError("target commands must be a sequence of strings")
        for index, command in enumerate(commands, 1):
            rows.append(
                {
                    "target": target_name,
                    "command_index": index,
                    "command": _require_non_empty_text(command, "command"),
                    "status": status,
                }
            )
    return tuple(rows)


def disabled_export_reasons(validation_errors: Sequence[str]) -> tuple[str, ...]:
    """Return reasons deploy-like exports must stay review-only.

    Parameters
    ----------
    validation_errors : Sequence[str]
        Binding validation error messages.

    Returns
    -------
    tuple[str, ...]
        Reasons deploy-like exports must stay review-only.
    """
    errors = tuple(str(error) for error in validation_errors)
    if not errors:
        return ()
    return (
        "binding validation must pass before deploy manifests are enabled",
        *errors,
    )


def _readiness_targets(
    readiness: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Validate the readiness targets, each carrying a target name and status."""
    raw_targets = readiness.get("targets", ())
    if isinstance(raw_targets, str | bytes) or not isinstance(raw_targets, Sequence):
        raise ValueError("readiness targets must be a sequence")
    targets: list[dict[str, object]] = []
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, Mapping):
            raise ValueError(f"readiness targets[{index}] must be a mapping")
        targets.append(
            {
                **dict(raw_target),
                "target": _require_non_empty_text(raw_target.get("target"), "target"),
                "status": _require_non_empty_text(raw_target.get("status"), "status"),
            }
        )
    return tuple(targets)


def _unique_artifacts(targets: Sequence[Mapping[str, object]]) -> list[str]:
    """Return the de-duplicated required artifacts across all readiness targets."""
    artifacts: list[str] = []
    for target in targets:
        required = target.get("required_artifacts", ())
        if isinstance(required, str | bytes) or not isinstance(required, Sequence):
            raise ValueError("required_artifacts must be a sequence")
        for artifact in required:
            name = _require_non_empty_text(artifact, "required_artifact")
            if name not in artifacts:
                artifacts.append(name)
    return artifacts


def _studio_service_processes() -> list[dict[str, object]]:
    """Return the review-only studio service definitions (no network, no actuation)."""
    validate_binding_command = (
        "python -m scpn_phase_orchestrator.runtime.cli validate binding_spec.yaml"
    )
    return [
        {
            "name": "spo-studio-ui",
            "image": "scpn-phase-orchestrator:local",
            "command": (
                "streamlit run tools/spo_studio.py "
                "--server.address 127.0.0.1 --server.port 8501"
            ),
            "ports": ["127.0.0.1:8501:8501"],
            "profiles": ["studio"],
            "healthcheck": validate_binding_command,
            "network_opened": False,
            "actuation_permitted": False,
        },
        {
            "name": "spo-binding-validator",
            "image": "scpn-phase-orchestrator:local",
            "command": validate_binding_command,
            "ports": [],
            "profiles": ["validation"],
            "healthcheck": validate_binding_command,
            "network_opened": False,
            "actuation_permitted": False,
        },
        {
            "name": "spo-connector-boundary",
            "image": "scpn-phase-orchestrator:local",
            "command": validate_binding_command,
            "ports": [],
            "profiles": ["connector-boundary-review"],
            "healthcheck": validate_binding_command,
            "network_opened": False,
            "actuation_permitted": False,
        },
    ]


def _render_service_compose_yaml(services: Sequence[Mapping[str, object]]) -> str:
    """Render the studio services as a read-only docker-compose YAML document."""
    lines = ["services:"]
    for service in services:
        name = _require_non_empty_text(service.get("name"), "service.name")
        image = _require_non_empty_text(service.get("image"), "image")
        command = json.dumps(_require_non_empty_text(service.get("command"), "command"))
        lines.extend(
            [
                f"  {name}:",
                f"    image: {image}",
                "    working_dir: /workspace",
                "    volumes:",
                "      - .:/workspace:ro",
                f"    command: {command}",
            ]
        )
        ports = service.get("ports", ())
        if isinstance(ports, Sequence) and not isinstance(ports, str | bytes) and ports:
            lines.append("    ports:")
            for port in ports:
                port_text = json.dumps(_require_non_empty_text(port, "port"))
                lines.append(f"      - {port_text}")
        profiles = service.get("profiles", ())
        if (
            isinstance(profiles, Sequence)
            and not isinstance(profiles, str | bytes)
            and profiles
        ):
            lines.append("    profiles:")
            for profile in profiles:
                lines.append(
                    f"      - {json.dumps(_require_non_empty_text(profile, 'profile'))}"
                )
        healthcheck = json.dumps(
            _require_non_empty_text(service.get("healthcheck"), "healthcheck")
        )
        lines.extend(
            [
                "    healthcheck:",
                f'      test: ["CMD-SHELL", {healthcheck}]',
                "      interval: 30s",
                "      timeout: 10s",
                "      retries: 3",
            ]
        )
    return "\n".join(lines) + "\n"


def _materialisation_command_writes_artifact(command: object) -> bool:
    """Return whether the command builds or writes a materialised artefact."""
    command_text = _require_non_empty_text(command, "command")
    return any(
        marker in command_text
        for marker in (
            "docker build",
            "docker run",
            "wasm-pack build",
        )
    )
