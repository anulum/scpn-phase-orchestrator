# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio deployment command execution tests

"""The commands Studio hands to an operator run against the shipped CLI.

Studio's export manifests, deployment readiness, and service compose file list
commands an operator copies and runs. These tests execute each command that
does not need a container runtime or a Rust toolchain: the CLI arguments a
``docker run`` or compose service passes to the image entrypoint are run
through the real ``spo`` click group in a workspace holding the exported
files, and the audit review command is run with the interpreter.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.studio.ui_helpers import (
    StudioKnobState,
    binding_spec_project_state,
    build_command_table,
    build_deployment_readiness,
    build_runtime_snapshot,
    build_service_process_manifest,
)
from scpn_phase_orchestrator.studio.workflow import StudioProjectState

ROOT = Path(__file__).resolve().parents[1]
IMAGE_TAG = "scpn-phase-orchestrator:local"


def _project_state() -> StudioProjectState:
    """Return a validated Studio project state for the minimal domainpack."""
    knobs = StudioKnobState(K=1.0)
    return binding_spec_project_state(
        project_name="minimal_domain",
        spec_path=ROOT / "domainpacks/minimal_domain/binding_spec.yaml",
        knobs=knobs,
        runtime=build_runtime_snapshot(
            final_state={
                "R_global": 0.72,
                "regime": "nominal",
                "layers": [{"name": "layer-a", "R": 0.7}],
            },
            knobs=knobs,
            replay_status="completed",
        ),
    )


def _materialise_exports(state: StudioProjectState, workspace: Path) -> None:
    """Write every exported artefact into ``workspace`` under its file name."""
    for manifest in state.exports:
        (workspace / manifest.file_name).write_text(manifest.payload, encoding="utf-8")


def _image_entrypoint_code() -> str:
    """Return the ``python -c`` program of the Dockerfile entrypoint."""
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        if line.startswith("ENTRYPOINT "):
            argv = json.loads(line.removeprefix("ENTRYPOINT "))
            assert argv[:2] == ["python", "-c"]
            return str(argv[2])
    raise AssertionError("Dockerfile has no ENTRYPOINT")


def _image_arguments(command: str) -> list[str]:
    """Return the arguments a ``docker run`` command passes after the image."""
    tokens = shlex.split(command)
    assert tokens[:2] == ["docker", "run"]
    assert tokens[tokens.index("-v") + 1] == "$PWD:/workspace"
    assert tokens[tokens.index("-w") + 1] == "/workspace"
    return tokens[tokens.index(IMAGE_TAG) + 1 :]


def test_no_command_names_a_missing_spo_subcommand() -> None:
    """Every ``spo <subcommand>`` Studio lists is a registered CLI command."""
    state = _project_state()
    commands = [manifest.command for manifest in state.exports]
    commands += [str(row["command"]) for row in build_command_table(state)]

    spo_commands = [shlex.split(text) for text in commands if text.startswith("spo ")]
    assert all(tokens[1] in main.commands for tokens in spo_commands), spo_commands


def test_docker_target_lists_no_compose_file_it_does_not_export() -> None:
    """The docker target builds and runs the image; it reads no compose file."""
    readiness = build_deployment_readiness(_project_state())
    docker = readiness["targets"][0]

    assert docker["target"] == "docker"
    assert [shlex.split(text)[:2] for text in docker["commands"]] == [
        ["docker", "build"],
        ["docker", "run"],
    ]


def test_docker_run_arguments_run_the_exported_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The image arguments of ``docker run`` replay the exported binding spec."""
    state = _project_state()
    _materialise_exports(state, tmp_path)
    monkeypatch.chdir(tmp_path)
    run_command = next(
        str(row["command"])
        for row in build_command_table(state)
        if str(row["command"]).startswith("docker run")
    )

    result = CliRunner().invoke(main, _image_arguments(run_command))

    assert result.exit_code == 0, result.output
    assert (tmp_path / "audit.jsonl").stat().st_size > 0


def test_audit_review_command_reads_the_exported_audit(tmp_path: Path) -> None:
    """The audit export's review command parses the exported JSON."""
    state = _project_state()
    _materialise_exports(state, tmp_path)
    audit = next(m for m in state.exports if m.target_kind == "audit_summary")
    tokens = shlex.split(audit.command)
    assert tokens[:3] == ["python", "-m", "json.tool"]

    result = subprocess.run(
        [sys.executable, *tokens[1:]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["enabled"] is True


def test_compose_validator_services_validate_the_exported_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validator service commands and healthchecks run the image CLI."""
    state = _project_state()
    _materialise_exports(state, tmp_path)
    monkeypatch.chdir(tmp_path)
    compose = yaml.safe_load(build_service_process_manifest(state)["compose_yaml"])
    entrypoint_code = _image_entrypoint_code()
    runner = CliRunner()

    for name in ("spo-binding-validator", "spo-connector-boundary"):
        service = compose["services"][name]
        command = runner.invoke(main, shlex.split(service["command"]))
        assert command.exit_code == 0, (name, command.output)

        kind, shell = service["healthcheck"]["test"]
        assert kind == "CMD-SHELL"
        interpreter, flag, code, *arguments = shlex.split(shell)
        assert (interpreter, flag, code) == ("python", "-c", entrypoint_code)
        check = runner.invoke(main, arguments)
        assert check.exit_code == 0, (name, check.output)
