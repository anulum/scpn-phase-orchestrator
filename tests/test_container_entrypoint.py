# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — container entrypoint tests

"""The container image's entrypoint runs the ``spo`` CLI.

Docker appends the ``docker run`` arguments to the exec-form ``ENTRYPOINT``.
These tests read the entrypoint and the default command from the shipped
``Dockerfile`` and run that exact argv, in the repository root that mirrors the
image's ``/app`` working directory. The production stage installs the project
into ``/usr/local`` and sets no ``PYTHONPATH``; the repository ``src`` tree is
the package the builder stage installs. No container runtime is needed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"


def _production_stage() -> list[str]:
    """Return the Dockerfile lines of the final (production) stage."""
    lines = DOCKERFILE.read_text(encoding="utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("FROM ")]
    assert starts, "Dockerfile has no FROM stage"
    return lines[starts[-1] :]


def _entrypoint() -> list[str]:
    """Return the exec-form ENTRYPOINT argv of the production stage."""
    entries = [line for line in _production_stage() if line.startswith("ENTRYPOINT ")]
    assert len(entries) == 1, entries
    argv = json.loads(entries[0].removeprefix("ENTRYPOINT "))
    assert isinstance(argv, list)
    assert all(isinstance(part, str) for part in argv)
    return argv


def _default_command() -> list[str]:
    """Return the exec-form default ``CMD`` argv of the production stage."""
    entries = [line for line in _production_stage() if line.startswith("CMD ")]
    assert len(entries) == 1, entries
    argv = json.loads(entries[0].removeprefix("CMD "))
    assert isinstance(argv, list)
    assert all(isinstance(part, str) for part in argv)
    return argv


def _image_pythonpath() -> str:
    """Return the repository source root that mirrors the installed package.

    The production stage copies the builder's ``/install`` prefix into
    ``/usr/local`` and sets no ``PYTHONPATH``. The builder installs the project
    itself, so the repository ``src`` tree is the same package.
    """
    stage = _production_stage()
    assert not [line for line in stage if line.startswith("ENV PYTHONPATH=")]
    assert "COPY --from=python-builder /install /usr/local" in stage
    assert "--no-build-isolation /wheels/*.whl ." in DOCKERFILE.read_text(
        encoding="utf-8"
    )
    return str(ROOT / "src")


def _run_image_argv(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the container entrypoint with ``args`` as Docker would append them."""
    argv = _entrypoint()
    assert argv[0] == "python"
    env = {**os.environ, "PYTHONPATH": _image_pythonpath()}
    return subprocess.run(
        [sys.executable, *argv[1:], *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_entrypoint_imports_the_console_script_target() -> None:
    """The entrypoint imports the same callable as the ``spo`` console script."""
    argv = _entrypoint()
    match = re.fullmatch(r"from ([\w.]+) import (\w+); \2\(\)", argv[-1])
    assert match is not None, argv
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    script = pyproject["project"]["scripts"]["spo"]

    assert f"{match.group(1)}:{match.group(2)}" == script


def test_help_argument_prints_cli_usage() -> None:
    """``docker run <image> --help`` prints the CLI usage."""
    result = _run_image_argv("--help")

    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout
    assert "validate" in result.stdout


def test_default_command_serves_a_packaged_spec_with_the_kernel_required() -> None:
    """The default ``CMD`` starts ``spo serve`` on a spec the image ships."""
    argv = _default_command()

    assert argv[0] == "serve"
    assert (ROOT / argv[1]).is_file()
    assert "--require-kernel" in argv
    assert argv[argv.index("--port") + 1] == "8000"
    assert "EXPOSE 8000" in _production_stage()

    help_result = _run_image_argv("serve", "--help")
    assert help_result.returncode == 0, help_result.stderr
    for option in (part for part in argv[2:] if part.startswith("--")):
        assert option in help_result.stdout


def test_documented_run_command_executes_a_domainpack() -> None:
    """``docker run <image> run domainpacks/...`` runs the packaged spec."""
    result = _run_image_argv(
        "run", "domainpacks/minimal_domain/binding_spec.yaml", "--steps", "3"
    )

    assert result.returncode == 0, result.stderr
